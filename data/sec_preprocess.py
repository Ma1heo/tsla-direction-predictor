"""
SEC Filing Structural Preprocessor
====================================
Reads extracted sections from sec_filings.db, applies deterministic
preprocessing to create structured "packets" ready for LLM classification.

Each packet contains:
  - lead_sentences: first 3-5 sentences (management leads with what matters)
  - numerical_sentences: every sentence containing concrete numbers
  - forward_looking_sentences: sentences signalling future direction
  - comparison_sentences: sentences with explicit period-over-period comparisons
  - section_length_delta: change in section length vs. prior filing of same type
  - metadata: form type, filing date, section name, 8-K item code

Usage:
    python data/sec_preprocess.py              # process all filings
    python data/sec_preprocess.py --test 3     # test on N filings per type
"""

import re
import sys
import json
import sqlite3
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "data" else SCRIPT_DIR
DATA_RAW = PROJECT_ROOT / "data" / "raw"
SEC_DB_PATH = DATA_RAW / "sec_filings.db"

# ---------------------------------------------------------------------------
# Which sections to process (the ones that matter for trading signals)
# ---------------------------------------------------------------------------
RELEVANT_SECTIONS = {
    # 10-K
    "MD&A",
    # 8-K event items
    "2.02 - Results of Operations (Earnings)",
    "8.01 - Other Events",
    "7.01 - Regulation FD Disclosure",
    "1.01 - Entry into Material Agreement",
    "5.02 - Departure/Election of Directors/Officers",
    "5.07 - Shareholder Vote Submission",
    "2.01 - Completion of Acquisition/Disposition",
    # Earnings exhibit (extracted from full-submission.txt)
    "Shareholder Letter (EX-99.1)",
}


# ═══════════════════════════════════════════════════════════════════════════
# TEXT CLEANING (applied before sentence splitting)
# ═══════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Clean raw section text before any processing."""
    # Normalize non-breaking spaces
    text = text.replace("\xa0", " ")
    # Fix tag-split "I\nTEM" → "ITEM"
    text = re.sub(r"\bI\n(TEM)\b", r"I\1", text)
    # Fix concatenated sentences (missing space: "2023.We" → "2023. We")
    text = re.sub(r"\.([A-Z])", r". \1", text)
    # Strip standalone page numbers on their own line
    text = re.sub(r"\n\d{1,3}\n", "\n", text)
    # Remove signatures block and everything after
    sig_match = re.search(r"\bSIGNATURES?\b", text)
    if sig_match:
        text = text[:sig_match.start()]
    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════
# SENTENCE SPLITTING
# ═══════════════════════════════════════════════════════════════════════════

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")

ABBREV_ENDINGS = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Inc|Corp|Ltd|Co|vs|No|nos|approx|U\.S|i\.e|e\.g)\.$"
)

# SEC legal boilerplate — any sentence matching this is noise
BOILERPLATE_PATTERN = re.compile(
    r"""
    securities\s+exchange\s+act             |
    shall\s+not\s+be\s+deemed\s+"?filed     |
    incorporated\s+(?:herein\s+)?by\s+reference |
    pursuant\s+to\s+(?:the\s+requirements|item|rule|section) |
    attached\s+hereto\s+as\s+exhibit        |
    hereunto\s+duly\s+authorized            |
    form\s+8-k                              |
    current\s+report\s+on\s+form            |
    registrant\s+has\s+duly\s+caused        |
    commission\s+file\s+(?:number|no)       |
    for\s+purposes\s+of\s+section\s+18      |
    exhibit\s+\d+\.\d+\s+(?:to|of)\s+this   |
    filed\s+as\s+exhibit                    |
    filed\s+with\s+the\s+(?:sec|securities) |
    registration\s+statement
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Table-like content detector (excessive newlines = table rows)
TABLE_PATTERN = re.compile(r"(?:\n.*?){5,}")


def is_table_content(text: str) -> bool:
    """Detect if text is a table rendered as newline-separated values."""
    newline_count = text.count("\n")
    # Only flag as table if high density of newlines relative to text length
    # (real paragraphs can have 3-5 newlines across 500+ chars)
    if newline_count >= 8:
        return True
    if newline_count >= 5 and len(text) < 300:
        return True
    # Lines that are mostly numbers/dollar signs = table row
    lines = text.split("\n")
    numeric_lines = sum(1 for l in lines if re.match(r"^[\d\s\-,$%.()]+$", l.strip()) and len(l.strip()) > 0)
    return numeric_lines >= 4


def is_boilerplate(text: str) -> bool:
    """Check if a sentence is SEC legal boilerplate."""
    return bool(BOILERPLATE_PATTERN.search(text))


def split_sentences(text: str) -> list[str]:
    """Split cleaned text into filtered sentences."""
    raw = SENTENCE_SPLIT.split(text)

    # Rejoin fragments caused by abbreviation false-splits
    merged = []
    for frag in raw:
        if merged and ABBREV_ENDINGS.search(merged[-1]):
            merged[-1] = merged[-1] + " " + frag
        else:
            merged.append(frag)

    # Filter
    result = []
    for s in merged:
        s = s.strip()
        if len(s) < 30:
            continue
        if re.match(r"^[\d\s\-,$%.()]+$", s):  # pure numbers/table row
            continue
        if is_table_content(s):
            continue
        if is_boilerplate(s):
            continue
        if s.startswith("Table of Contents"):
            continue
        if s.startswith("See accompanying notes"):
            continue
        # Strip leading sub-section markers like "(c)\n"
        s = re.sub(r"^\([a-z]\)\s*\n?\s*", "", s)
        if len(s) < 30:
            continue
        result.append(s)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# EXTRACTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

# Boilerplate phrases for lead sentence filtering (in addition to global filter)
LEAD_BOILERPLATE = [
    "the following discussion",
    "should be read in conjunction",
    "refer to item",
    "refer to part",
    "for further discussion",
    "for discussion related",
    "as used in this",
    "this section contains forward-looking",
    "management's discussion and analysis",
    "which was filed with the",
    "posted its",  # "Tesla posted its Q3 update on its website"
    "by posting its",
    "released its financial results",
    "does not purport to be complete",
    "qualified in its entirety by reference",
    "foregoing description",
    "supplemental indenture",
    "the credit agreement amendment",
    "warehouse administrative agent",
    "group agent",
    "re-filing of such exhibits",
]


def extract_lead_sentences(sentences: list[str], n: int = 5) -> list[str]:
    """First N meaningful sentences — management leads with what matters."""
    leads = []
    for s in sentences:
        s_lower = s.lower()
        if any(bp in s_lower for bp in LEAD_BOILERPLATE):
            continue
        leads.append(s)
        if len(leads) >= n:
            break
    return leads


# --- Numerical claims ---
NUMBER_PATTERN = re.compile(
    r"""
    \$[\d,.]+[BMK]?                     |   # dollar amounts
    \d+\.?\d*\s*%                       |   # percentages
    \d{1,3}(?:,\d{3})+                 |   # large numbers: 1,234,567
    \d+\.?\d*\s*(?:billion|million|thousand|GWh|MWh|kWh|units|vehicles)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Numbers that are noise (legal citations, par values, file numbers)
NOISE_NUMBER_PATTERN = re.compile(
    r"""
    par\s+value                         |
    file\s+no                           |
    section\s+\d+                       |
    rule\s+\d+                          |
    \$0\.00\d                           |   # par value $0.001
    333-\d+                             |   # SEC file numbers
    ^\d{1,2}[,.]?\d*\n                      # standalone vote counts
    """,
    re.IGNORECASE | re.VERBOSE,
)


def extract_numerical_sentences(sentences: list[str]) -> list[str]:
    """Sentences containing concrete, meaningful numerical claims."""
    results = []
    for s in sentences:
        if NUMBER_PATTERN.search(s) and not NOISE_NUMBER_PATTERN.search(s):
            results.append(s)
    return results


# --- Forward-looking language ---
FORWARD_PATTERNS = re.compile(
    r"""
    \bwe\s+(?:expect|anticipate|believe|intend|plan|project|estimate|aim|target)\b |
    \bwe\s+are\s+focused\s+on\b                |
    \bgoing\s+forward\b                         |
    \boutlook\b                                 |
    \bguidance\b                                |
    \bforecast\b                                |
    \bwe\s+will\b                               |
    \bnext\s+(?:quarter|year|phase|step)\b      |
    \bramp(?:ing)?\b                            |
    \bexpansion\b                               |
    \bpipeline\b                                |
    \bscal(?:e|ing|able)\b                      |
    \bupcoming\b                                |
    \blong[\s-]term\b                           |
    \bnear[\s-]term\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Safe harbor disclaimer — false positive filter
SAFE_HARBOR_PATTERN = re.compile(
    r"""
    forward[\s-]looking\s+statements?\s+(?:that\s+are\s+subject|within\s+the\s+meaning) |
    actual\s+results\s+may\s+differ\s+materially |
    subject\s+to\s+risks\s+and\s+uncertainties  |
    we\s+(?:caution|urge)\s+(?:you|investors|readers) |
    safe\s+harbor                               |
    wish\s+him\s+(?:well|the\s+best)\s+in\s+his\s+future
    """,
    re.IGNORECASE | re.VERBOSE,
)


def extract_forward_looking(sentences: list[str]) -> list[str]:
    """Sentences with genuine forward-looking business language (no disclaimers)."""
    results = []
    for s in sentences:
        if FORWARD_PATTERNS.search(s) and not SAFE_HARBOR_PATTERN.search(s):
            results.append(s)
    return results


# --- Comparison/change language (NEW) ---
COMPARISON_PATTERN = re.compile(
    r"""
    \bcompared\s+to\b                           |
    \bversus\b                                  |
    \bvs\.?\b                                   |
    \byear[\s-]over[\s-]year\b                  |
    \bquarter[\s-]over[\s-]quarter\b            |
    \bperiod[\s-]over[\s-]period\b              |
    \bprior\s+(?:year|quarter|period)\b         |
    \b(?:increased|decreased|declined|grew|fell|rose|dropped|surged|plunged)\b |
    \b(?:up|down)\s+\d+                         |
    \bfavorable\s+change\b                      |
    \bunfavorable\s+change\b                    |
    \brepresenting\s+(?:a|an)\s+(?:increase|decrease|change)\b |
    \bimproved?\b                               |
    \bdeteriorat(?:ed|ion|ing)\b                |
    \bheadwinds?\b                              |
    \btailwinds?\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def extract_comparison_sentences(sentences: list[str]) -> list[str]:
    """Sentences with explicit period-over-period comparisons or directional claims."""
    return [s for s in sentences if COMPARISON_PATTERN.search(s)]


# ═══════════════════════════════════════════════════════════════════════════
# SECTION LENGTH DELTA
# ═══════════════════════════════════════════════════════════════════════════

def compute_length_deltas(filings: list[dict]) -> dict[str, dict]:
    """Compute char_count delta vs. prior filing of same form_type/section_name."""
    groups = defaultdict(list)
    for f in filings:
        key = (f["form_type"], f["section_name"])
        groups[key].append(f)

    deltas = {}
    for key, group in groups.items():
        group.sort(key=lambda x: x["filing_date"])
        for i, f in enumerate(group):
            lookup = f"{f['accession_number']}|{f['section_name']}"
            if i == 0:
                deltas[lookup] = {"delta_chars": None, "delta_pct": None}
            else:
                prev_len = group[i - 1]["char_count"]
                curr_len = f["char_count"]
                delta = curr_len - prev_len
                delta_pct = (delta / prev_len * 100) if prev_len > 0 else None
                deltas[lookup] = {
                    "delta_chars": delta,
                    "delta_pct": round(delta_pct, 1) if delta_pct else None,
                }
    return deltas


# ═══════════════════════════════════════════════════════════════════════════
# 8-K ITEM CODE
# ═══════════════════════════════════════════════════════════════════════════

def extract_8k_item_code(section_name: str) -> str | None:
    """Extract the numeric item code from an 8-K section name."""
    m = re.match(r"(\d+\.\d+)", section_name)
    return m.group(1) if m else None


# ═══════════════════════════════════════════════════════════════════════════
# BUILD PACKET
# ═══════════════════════════════════════════════════════════════════════════

def build_packet(filing: dict, length_deltas: dict) -> dict:
    """Create a structured packet from a single filing section."""
    text = clean_text(filing["section_text"])
    sentences = split_sentences(text)

    lead = extract_lead_sentences(sentences, n=5)
    numerical = extract_numerical_sentences(sentences)
    forward = extract_forward_looking(sentences)
    comparison = extract_comparison_sentences(sentences)

    lookup = f"{filing['accession_number']}|{filing['section_name']}"
    delta = length_deltas.get(lookup, {"delta_chars": None, "delta_pct": None})

    # Flag low-content sections
    is_substantive = len(lead) >= 1 and filing["char_count"] > 200

    packet = {
        "accession_number": filing["accession_number"],
        "form_type": filing["form_type"],
        "filing_date": filing["filing_date"],
        "section_name": filing["section_name"],
        "is_substantive": is_substantive,
        "total_sentences": len(sentences),
        "total_chars": filing["char_count"],
        "length_delta_chars": delta["delta_chars"],
        "length_delta_pct": delta["delta_pct"],
        "lead_sentences": lead,
        "numerical_sentences": numerical[:15],
        "forward_looking_sentences": forward[:10],
        "comparison_sentences": comparison[:15],
        "num_numerical": len(numerical),
        "num_forward_looking": len(forward),
        "num_comparison": len(comparison),
    }

    if filing["form_type"] == "8-K":
        packet["item_code"] = extract_8k_item_code(filing["section_name"])

    return packet


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def load_filings(db_path: Path, limit_per_type: int | None = None) -> list[dict]:
    """Load relevant filings from the database."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    placeholders = ",".join("?" for _ in RELEVANT_SECTIONS)
    query = f"""
        SELECT f.accession_number, f.form_type, f.filing_date,
               s.section_name, s.section_text, s.char_count
        FROM sections s
        JOIN filings f ON s.filing_id = f.id
        WHERE s.section_name IN ({placeholders})
        ORDER BY f.form_type, f.filing_date
    """
    rows = conn.execute(query, list(RELEVANT_SECTIONS)).fetchall()
    conn.close()

    filings = [dict(r) for r in rows]

    if limit_per_type:
        limited = []
        counts = defaultdict(int)
        for f in filings:
            key = f["form_type"]
            if counts[key] < limit_per_type:
                limited.append(f)
                counts[key] += 1
        return limited

    return filings


def main():
    test_mode = None
    if "--test" in sys.argv:
        idx = sys.argv.index("--test")
        test_mode = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 3

    print("=" * 60)
    print("SEC FILING STRUCTURAL PREPROCESSOR")
    print("=" * 60)

    filings = load_filings(SEC_DB_PATH, limit_per_type=test_mode)
    print(f"\nLoaded {len(filings)} sections to process")

    all_filings = load_filings(SEC_DB_PATH)
    deltas = compute_length_deltas(all_filings)

    packets = []
    for f in filings:
        packet = build_packet(f, deltas)
        packets.append(packet)

    # Print results
    for p in packets:
        print(f"\n{'─' * 60}")
        print(f"  {p['form_type']} | {p['filing_date']} | {p['section_name']}")
        print(f"  Substantive: {p['is_substantive']} | Sentences: {p['total_sentences']} | Chars: {p['total_chars']:,}")
        if p['length_delta_pct'] is not None:
            print(f"  Length delta vs prior: {p['length_delta_chars']:+,} chars ({p['length_delta_pct']:+.1f}%)")
        print(f"  Numerical: {p['num_numerical']} | Forward: {p['num_forward_looking']} | Comparison: {p['num_comparison']}")

        if p["lead_sentences"]:
            print(f"\n  LEAD SENTENCES:")
            for i, s in enumerate(p["lead_sentences"], 1):
                display = s[:200] + "..." if len(s) > 200 else s
                print(f"    {i}. {display}")

        if p["numerical_sentences"]:
            print(f"\n  NUMERICAL (first 5):")
            for i, s in enumerate(p["numerical_sentences"][:5], 1):
                display = s[:200] + "..." if len(s) > 200 else s
                print(f"    {i}. {display}")

        if p["comparison_sentences"]:
            print(f"\n  COMPARISON (first 5):")
            for i, s in enumerate(p["comparison_sentences"][:5], 1):
                display = s[:200] + "..." if len(s) > 200 else s
                print(f"    {i}. {display}")

        if p["forward_looking_sentences"]:
            print(f"\n  FORWARD-LOOKING (first 3):")
            for i, s in enumerate(p["forward_looking_sentences"][:3], 1):
                display = s[:200] + "..." if len(s) > 200 else s
                print(f"    {i}. {display}")

    # Save packets
    if not test_mode:
        out_path = DATA_RAW / "sec_preprocessed_packets.json"
        with open(out_path, "w") as f:
            json.dump(packets, f, indent=2, default=str)
        print(f"\n\nSaved {len(packets)} packets → {out_path}")
    else:
        print(f"\n\n[TEST MODE] {len(packets)} sections. Run without --test to save.")


if __name__ == "__main__":
    main()
