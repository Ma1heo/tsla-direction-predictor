"""
SEC Filing Feature Engineering Pipeline
========================================
Preprocesses extracted SEC filing sections into structured packets,
classifies them with a local LLM (Qwen 2.5 7B via Ollama), and
outputs a date-aligned daily signal table.

Usage:
    python data/sec_feature_engineering.py

Requires:
    - Ollama running locally with qwen2.5:7b pulled
    - sec_filings.db populated by SEC_data_collection.py
"""

import re
import json
import sqlite3
import csv
import requests
from pathlib import Path

import ollama

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "data" else SCRIPT_DIR
DATA_RAW = PROJECT_ROOT / "data" / "raw"

SEC_DB_PATH = DATA_RAW / "sec_filings.db"
DAILY_SIGNALS_CSV = DATA_RAW / "sec_daily_signals.csv"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL = "qwen2.5:7b"
COMPANY_NAME = "Matheo Menges"
EMAIL = "mengesmatheo@gmail.com"
CIK = "0001318605"

# Sections worth analyzing (the rest is noise)
RELEVANT_SECTIONS = {
    "MD&A",
    "Shareholder Letter (EX-99.1)",
    "2.02 - Results of Operations (Earnings)",
    "5.02 - Departure/Election of Directors/Officers",
    "7.01 - Regulation FD Disclosure",
    "8.01 - Other Events",
    "1.01 - Entry into Material Agreement",
}

SYSTEM_PROMPT = (
    "You are an equity research analyst specializing in Tesla (TSLA). "
    "You evaluate SEC filing excerpts and assess their likely impact on "
    "the stock price from an investor's perspective. You look past the "
    "corporate language and evaluate the underlying business reality. "
    "You respond with ONLY valid JSON. Never add text outside the JSON object."
)

# ---------------------------------------------------------------------------
# Sentence extraction patterns
# ---------------------------------------------------------------------------
RE_NUMERICAL = re.compile(
    r'[\$%]'
    r'|\d[\d,]*\.?\d*\s*(?:million|billion|percent|%|units|vehicles|deliveries|kwh|gwh|mwh)',
    re.IGNORECASE,
)
RE_COMPARISON = re.compile(
    r'compared\s+to|increased?\s+(?:from|by)|decreased?\s+(?:from|by)'
    r'|year[\s-]*over[\s-]*year|quarter[\s-]*over[\s-]*quarter'
    r'|prior\s+(?:year|quarter|period)|sequentially',
    re.IGNORECASE,
)
RE_FORWARD = re.compile(
    r'we\s+(?:expect|plan|anticipate|believe|intend|project|estimate|aim|target)'
    r'|going\s+forward|outlook|forward[\s-]*looking|guidance'
    r'|we\s+will|we\s+are\s+(?:confident|optimistic|committed)',
    re.IGNORECASE,
)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences. Handles common abbreviations."""
    # Protect common abbreviations
    text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Inc|Corp|Ltd|vs|etc|approx)\.',
                  r'\1<DOT>', text, flags=re.IGNORECASE)
    # Split on sentence-ending punctuation followed by space + capital
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    # Restore dots
    return [p.replace('<DOT>', '.').strip() for p in parts if p.strip()]


# ═══════════════════════════════════════════════════════════════════════════
# STEP 0: FIX FILING DATES
# ═══════════════════════════════════════════════════════════════════════════

def fetch_filing_dates() -> dict[str, str]:
    """Fetch exact filing dates from SEC EDGAR submissions API.
    Returns {accession_number: filing_date} mapping."""
    headers = {"User-Agent": f"{COMPANY_NAME} {EMAIL}"}
    url = f"https://data.sec.gov/submissions/CIK{CIK}.json"

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    date_map = {}
    recent = data["filings"]["recent"]
    for acc, date in zip(recent["accessionNumber"], recent["filingDate"]):
        # Normalize accession: SEC API uses dashes, our DB uses dashes
        date_map[acc] = date

    # Fetch additional pages if they exist
    for extra in data["filings"].get("files", []):
        extra_url = f"https://data.sec.gov/submissions/{extra['name']}"
        resp2 = requests.get(extra_url, headers=headers, timeout=30)
        resp2.raise_for_status()
        extra_data = resp2.json()
        for acc, date in zip(extra_data["accessionNumber"], extra_data["filingDate"]):
            date_map[acc] = date

    return date_map


def update_filing_dates(conn: sqlite3.Connection):
    """Update filings table with exact dates from SEC API."""
    print("Fetching exact filing dates from SEC EDGAR API...")
    date_map = fetch_filing_dates()

    rows = conn.execute("SELECT id, accession_number FROM filings").fetchall()
    updated = 0
    for fid, acc in rows:
        exact_date = date_map.get(acc)
        if exact_date:
            conn.execute("UPDATE filings SET filing_date = ? WHERE id = ?",
                         (exact_date, fid))
            updated += 1

    conn.commit()
    print(f"  Updated {updated}/{len(rows)} filings with exact dates")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def build_packet(section_text: str, form_type: str, section_name: str,
                 filing_date: str, length_delta_pct: float | None) -> dict:
    """Build a structured packet from raw section text."""
    sentences = split_sentences(section_text)

    # Lead sentences (first 5)
    lead = sentences[:5]

    # Categorized sentences (skip first 5 to avoid duplication)
    numerical = [s for s in sentences[5:] if RE_NUMERICAL.search(s)][:10]
    comparison = [s for s in sentences[5:] if RE_COMPARISON.search(s)][:8]
    forward = [s for s in sentences[5:] if RE_FORWARD.search(s)][:8]

    # Extract 8-K item code if present
    event_code = None
    code_match = re.match(r'^(\d+\.\d+)\s*-', section_name)
    if code_match:
        event_code = code_match.group(1)

    return {
        "form_type": form_type,
        "filing_date": filing_date,
        "section_name": section_name,
        "event_code": event_code,
        "length_delta_pct": length_delta_pct,
        "lead_sentences": "\n".join(lead) if lead else "(none)",
        "numerical_sentences": "\n".join(numerical) if numerical else "(none)",
        "comparison_sentences": "\n".join(comparison) if comparison else "(none)",
        "forward_looking_sentences": "\n".join(forward) if forward else "(none)",
    }


def compute_length_deltas(conn: sqlite3.Connection) -> dict[tuple[int, str], float]:
    """Compute section length % change vs. prior filing of the same type+section."""
    rows = conn.execute("""
        SELECT f.id, f.form_type, f.filing_date, s.section_name, s.char_count
        FROM sections s
        JOIN filings f ON f.id = s.filing_id
        ORDER BY f.form_type, s.section_name, f.filing_date
    """).fetchall()

    deltas = {}
    prev = {}
    for fid, form, date, section, chars in rows:
        key = (form, section)
        if key in prev and prev[key] > 0:
            delta = ((chars - prev[key]) / prev[key]) * 100
            deltas[(fid, section)] = round(delta, 1)
        prev[key] = chars

    return deltas


def format_prompt(packet: dict) -> str:
    """Format the user prompt for the LLM."""
    delta_str = f"{packet['length_delta_pct']}%" if packet['length_delta_pct'] is not None else "N/A (first filing)"

    return f"""TESLA (TSLA) SEC FILING — INVESTOR IMPACT ANALYSIS

Filing type: {packet['form_type']}
Filing date: {packet['filing_date']}
Section: {packet['section_name']}
Section length change vs prior filing: {delta_str}

=== OPENING STATEMENTS ===
{packet['lead_sentences']}

=== KEY NUMBERS ===
{packet['numerical_sentences']}

=== PERIOD COMPARISONS ===
{packet['comparison_sentences']}

=== FORWARD-LOOKING STATEMENTS ===
{packet['forward_looking_sentences']}

You are an equity analyst. Evaluate this filing from an INVESTOR perspective — not the tone of the writing, but the actual business impact. Corporate filings use careful language; look past the spin.

Scoring rules for "impact_score" (integer from -5 to +5):
  -5 = severely negative (major losses, SEC enforcement, existential risk)
  -4 = very negative (significant earnings miss, large restructuring/layoffs)
  -3 = negative (revenue decline, margin compression, executive departure)
  -2 = mildly negative (cost increases, modest misses, minor legal issues)
  -1 = slightly negative (cautious guidance, small operational setbacks)
   0 = truly neutral (routine procedural filings, no material information)
  +1 = slightly positive (modest operational improvements)
  +2 = mildly positive (revenue growth, new agreements, minor beats)
  +3 = positive (strong earnings, expanding margins, significant deals)
  +4 = very positive (record results, major strategic wins, strong guidance)
  +5 = extremely positive (transformative events, massive beats, new markets)

A layoff or restructuring is NEGATIVE even if framed positively.
A routine procedural filing (indemnification, Reg FD pointer) is 0.
An earnings stub that just says "see Exhibit 99.1" with no data is 0.
Record deliveries and revenue growth is POSITIVE regardless of cautious language.

Respond with this exact JSON format:
{{"impact_score":0,"materiality":"high|medium|low|procedural","topic":"earnings|production|margins|guidance|personnel|legal|deal|restructuring|product|other","confidence":"high|medium|low","summary":"one sentence investor-focused summary"}}"""


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: LLM CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

VALID_CATEGORICAL = {
    "materiality": {"high", "medium", "low", "procedural"},
    "confidence": {"high", "medium", "low"},
    "topic": {"earnings", "production", "margins", "guidance", "personnel",
              "legal", "deal", "restructuring", "product", "other"},
}


def classify_packet(packet: dict) -> dict | None:
    """Send a packet to Ollama and parse the JSON response."""
    prompt = format_prompt(packet)

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0},
            format="json",
        )
        raw = response["message"]["content"].strip()
        result = json.loads(raw)

        # Validate impact_score: must be integer in [-5, +5]
        score = result.get("impact_score", 0)
        if isinstance(score, (int, float)):
            result["impact_score"] = max(-5, min(5, int(round(score))))
        else:
            result["impact_score"] = 0

        # Validate categorical fields
        for field, valid in VALID_CATEGORICAL.items():
            if field in result and result[field] not in valid:
                result[field] = "other" if field == "topic" else list(valid)[0]

        # Ensure summary exists
        if "summary" not in result or not result["summary"]:
            result["summary"] = "No summary provided"

        return result

    except (json.JSONDecodeError, KeyError) as e:
        print(f"    JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"    Ollama error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: STORE RESULTS
# ═══════════════════════════════════════════════════════════════════════════

def init_classifications_table(conn: sqlite3.Connection):
    """Drop and recreate the classifications table with new schema."""
    conn.execute("DROP TABLE IF EXISTS classifications")
    conn.execute("""
        CREATE TABLE classifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filing_id INTEGER NOT NULL,
            section_name TEXT NOT NULL,
            impact_score INTEGER,
            materiality TEXT,
            confidence TEXT,
            topic TEXT,
            summary TEXT,
            event_code TEXT,
            lead_sentences TEXT,
            numerical_sentences TEXT,
            forward_looking_sentences TEXT,
            length_delta_pct REAL,
            FOREIGN KEY (filing_id) REFERENCES filings(id)
        )
    """)
    conn.commit()


def store_classification(conn: sqlite3.Connection, filing_id: int,
                         packet: dict, classification: dict):
    """Store one classification result."""
    conn.execute("""
        INSERT INTO classifications
            (filing_id, section_name, impact_score, materiality,
             confidence, topic, summary, event_code,
             lead_sentences, numerical_sentences, forward_looking_sentences,
             length_delta_pct)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        filing_id,
        packet["section_name"],
        classification.get("impact_score"),
        classification.get("materiality"),
        classification.get("confidence"),
        classification.get("topic"),
        classification.get("summary"),
        packet.get("event_code"),
        packet["lead_sentences"],
        packet["numerical_sentences"],
        packet["forward_looking_sentences"],
        packet.get("length_delta_pct"),
    ))


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: EXPORT DAILY SIGNALS
# ═══════════════════════════════════════════════════════════════════════════

def export_daily_signals(conn: sqlite3.Connection, csv_path: Path):
    """Export one row per filing-section-date to CSV."""
    rows = conn.execute("""
        SELECT f.filing_date, f.form_type, f.accession_number,
               c.section_name, c.event_code,
               c.impact_score, c.materiality, c.confidence,
               c.topic, c.summary, c.length_delta_pct
        FROM classifications c
        JOIN filings f ON f.id = c.filing_id
        ORDER BY f.filing_date, f.form_type, c.section_name
    """).fetchall()

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "date", "form_type", "accession_number", "section_name",
            "event_code", "impact_score", "materiality", "confidence",
            "topic", "summary", "length_delta_pct"
        ])
        writer.writerows(rows)

    print(f"Daily signals CSV: {len(rows)} rows → {csv_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("SEC FILING FEATURE ENGINEERING")
    print("=" * 60)

    conn = sqlite3.connect(str(SEC_DB_PATH))

    # Step 0: Fix filing dates
    print("\n[0/4] Updating filing dates...")
    update_filing_dates(conn)

    # Step 1: Preprocess
    print("\n[1/4] Building structured packets...")
    init_classifications_table(conn)

    # Clear previous classifications (idempotent re-runs)
    conn.execute("DELETE FROM classifications")
    conn.commit()

    # Get length deltas
    deltas = compute_length_deltas(conn)

    # Load relevant sections
    rows = conn.execute("""
        SELECT f.id, f.form_type, f.filing_date, s.section_name, s.section_text
        FROM sections s
        JOIN filings f ON f.id = s.filing_id
        WHERE s.section_name IN ({})
        ORDER BY f.filing_date
    """.format(",".join("?" * len(RELEVANT_SECTIONS))),
        list(RELEVANT_SECTIONS)
    ).fetchall()

    print(f"  {len(rows)} relevant sections to process")

    # Step 2 & 3: Classify and store
    print(f"\n[2/4] Classifying with {MODEL} via Ollama...")
    success = 0
    fail = 0

    for i, (fid, form, date, section, text) in enumerate(rows):
        delta = deltas.get((fid, section))
        packet = build_packet(text, form, section, date, delta)
        classification = classify_packet(packet)

        if classification:
            store_classification(conn, fid, packet, classification)
            success += 1
            score = classification.get("impact_score", "?")
            materiality = classification.get("materiality", "?")
            topic = classification.get("topic", "?")
            score_str = f"{score:+d}" if isinstance(score, int) else str(score)
            print(f"  [{i+1}/{len(rows)}] {date} {form} {section[:40]:<40s} "
                  f"→ score={score_str} mat={materiality} topic={topic}")
        else:
            fail += 1
            print(f"  [{i+1}/{len(rows)}] ✗ {date} {form} {section[:40]}")

    conn.commit()

    # Step 4: Export
    print(f"\n[3/4] Exporting daily signals...")
    export_daily_signals(conn, DAILY_SIGNALS_CSV)

    # Summary
    print(f"\n[4/4] Summary")
    total = conn.execute("SELECT COUNT(*) FROM classifications").fetchone()[0]
    print(f"  Classifications stored: {total}")
    print(f"  Successes: {success}, Failures: {fail}")

    # Quick distribution check
    for field in ["impact_score", "materiality", "topic"]:
        dist = conn.execute(f"""
            SELECT {field}, COUNT(*) FROM classifications
            GROUP BY {field} ORDER BY COUNT(*) DESC
        """).fetchall()
        print(f"  {field}: {dict(dist)}")

    # Impact score statistics
    stats = conn.execute("""
        SELECT AVG(impact_score), MIN(impact_score), MAX(impact_score),
               SUM(CASE WHEN impact_score < 0 THEN 1 ELSE 0 END),
               SUM(CASE WHEN impact_score = 0 THEN 1 ELSE 0 END),
               SUM(CASE WHEN impact_score > 0 THEN 1 ELSE 0 END)
        FROM classifications
    """).fetchone()
    print(f"  impact_score: mean={stats[0]:.2f}, min={stats[1]}, max={stats[2]}")
    print(f"    negative(<0): {stats[3]}, zero: {stats[4]}, positive(>0): {stats[5]}")

    conn.close()
    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
