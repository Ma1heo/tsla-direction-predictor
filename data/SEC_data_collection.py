"""
SEC Filing Data Collection Prototype for TSLA
==============================================
Downloads 10-K, 8-K, and DEF 14A filings from SEC EDGAR for the last 10 years,
extracts key text sections, and stores results in SQLite + summary CSV.

Usage:
    python data/SEC_data_collection.py
"""

import re
import sys
import sqlite3
import csv
from pathlib import Path
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Project paths (mirror src/helpers.py)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "data" else SCRIPT_DIR
DATA_RAW = PROJECT_ROOT / "data" / "raw"

SEC_DOWNLOAD_DIR = DATA_RAW / "sec_filings"
SEC_DB_PATH = DATA_RAW / "sec_filings.db"
SEC_SUMMARY_CSV = DATA_RAW / "sec_filings_summary.csv"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
COMPANY_NAME = "Matheo Menges"
EMAIL = "mengesmatheo@gmail.com"
TICKER = "TSLA"
FORMS = ["10-K", "8-K", "DEF 14A"]
AFTER_DATE = "2016-01-01"
BEFORE_DATE = "2026-02-17"

# 10-K sections we want to extract (Item number → friendly name)
TENK_SECTIONS = {
    "1":  "Business",
    "1a": "Risk Factors",
    "1b": "Unresolved Staff Comments",
    "7":  "MD&A",
    "7a": "Market Risk",
}


# ═══════════════════════════════════════════════════════════════════════════
# 1. DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════

def download_filings():
    """Download SEC filings using sec_edgar_downloader."""
    from sec_edgar_downloader import Downloader

    SEC_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    dl = Downloader(COMPANY_NAME, EMAIL, str(SEC_DOWNLOAD_DIR))

    for form in FORMS:
        print(f"Downloading {form} filings for {TICKER}...")
        try:
            n = dl.get(
                form,
                TICKER,
                after=AFTER_DATE,
                before=BEFORE_DATE,
                include_amends=False,
                download_details=True,
            )
            print(f"  → {n} {form} filings downloaded")
        except Exception as e:
            print(f"  ✗ Error downloading {form}: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. DISCOVER FILES ON DISK
# ═══════════════════════════════════════════════════════════════════════════

def discover_filings():
    """Walk the download directory and return a list of filing dicts."""
    filings = []
    base = SEC_DOWNLOAD_DIR / "sec-edgar-filings" / TICKER

    if not base.exists():
        print(f"No filings found at {base}")
        return filings

    for form_dir in sorted(base.iterdir()):
        if not form_dir.is_dir():
            continue
        form_type = form_dir.name  # e.g. "10-K", "8-K", "DEF 14A"

        for acc_dir in sorted(form_dir.iterdir()):
            if not acc_dir.is_dir():
                continue
            accession = acc_dir.name

            # Prefer primary-document.html, fall back to full-submission.txt
            primary = None
            for f in acc_dir.iterdir():
                if f.name.startswith("primary-document"):
                    primary = f
                    break
            if primary is None:
                full_sub = acc_dir / "full-submission.txt"
                if full_sub.exists():
                    primary = full_sub

            if primary is None:
                continue

            filings.append({
                "form_type": form_type,
                "accession_number": accession,
                "file_path": primary,
            })

    print(f"Discovered {len(filings)} filing documents on disk")
    return filings


# ═══════════════════════════════════════════════════════════════════════════
# 3. HTML → CLEAN TEXT
# ═══════════════════════════════════════════════════════════════════════════

def html_to_text(html: str) -> str:
    """Convert HTML to clean readable text using BeautifulSoup + lxml."""
    from bs4 import XMLParsedAsHTMLWarning
    import warnings
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

    soup = BeautifulSoup(html, "lxml")
    # Remove non-content elements
    for tag in soup(["script", "style", "meta", "link", "header", "footer"]):
        tag.decompose()
    # Remove inline XBRL wrapper tags but keep their text
    for ix_tag in soup.find_all(re.compile(r"^ix:")):
        ix_tag.unwrap()
    text = soup.get_text(separator="\n", strip=True)
    # Fix tag-split "I\nTEM" → "ITEM" (common in 2016-2020 EDGAR HTML)
    text = re.sub(r"\bI\n(TEM)\b", r"I\1", text)
    # Normalize non-breaking spaces to regular spaces
    text = text.replace("\xa0", " ")
    # Fix concatenated sentences (missing space after period before capital)
    text = re.sub(r"\.([A-Z])", r". \1", text)
    # Strip standalone page numbers
    text = re.sub(r"\n\d{1,3}\n", "\n", text)
    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def read_filing_html(file_path: Path) -> str:
    """Read a filing file. Handles full-submission.txt SGML extraction."""
    raw = file_path.read_text(encoding="utf-8", errors="replace")

    if file_path.name == "full-submission.txt":
        # Extract the primary document (TYPE = 10-K, 8-K, etc.) from SGML
        doc_match = re.search(
            r"<DOCUMENT>\s*<TYPE>(10-K|8-K|DEF 14A|DEFA14A).*?<TEXT>(.*?)</TEXT>",
            raw,
            re.DOTALL | re.IGNORECASE,
        )
        if doc_match:
            return doc_match.group(2)
        # Fallback: return raw (might be plain text filing)
        return raw

    return raw


# ═══════════════════════════════════════════════════════════════════════════
# 4. SECTION EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def parse_10k(text: str) -> dict[str, str]:
    """Extract key sections from a 10-K filing's plain text."""
    sections = {}

    # Find all "Item N" headers — accounts for varying formats:
    #   "Item 1.", "ITEM 1A.", "Item 1A -", "ITEM 7. MANAGEMENT'S ..."
    item_pattern = re.compile(
        r"(?:^|\n)\s*(?:ITEM|Item)\s+(\d+[A-Za-z]?)\s*[.:\-–—]\s*(.*)",
        re.MULTILINE,
    )
    matches = list(item_pattern.finditer(text))

    if not matches:
        # Some filings use "PART II" style — try a looser pattern
        item_pattern_loose = re.compile(
            r"(?:^|\n)\s*(?:ITEM|Item)\s+(\d+[A-Za-z]?)\b\s*(.*)",
            re.MULTILINE,
        )
        matches = list(item_pattern_loose.finditer(text))

    for i, m in enumerate(matches):
        item_num = m.group(1).lower()

        if item_num not in TENK_SECTIONS:
            continue

        start = m.end()
        # End at the next Item header, or 50k chars max
        end = matches[i + 1].start() if i + 1 < len(matches) else start + 50000
        section_text = text[start:end].strip()

        # Skip if suspiciously short (probably a table-of-contents reference)
        if len(section_text) < 500:
            continue

        # Keep a reasonable max length
        if len(section_text) > 100000:
            section_text = section_text[:100000] + "\n[...truncated...]"

        section_name = TENK_SECTIONS[item_num]
        # Keep the longest match (real section, not TOC entry)
        if section_name not in sections or len(section_text) > len(sections[section_name]):
            sections[section_name] = section_text

    return sections


def parse_8k(text: str) -> dict[str, str]:
    """Extract event items from an 8-K filing."""
    sections = {}

    # 8-K items follow pattern: "Item 2.02" or "ITEM 7.01"
    item_pattern = re.compile(
        r"(?:^|\n)\s*(?:ITEM|Item)\s+(\d+\.\d+)\s*[.:\-–—]?\s*(.*)",
        re.MULTILINE,
    )
    matches = list(item_pattern.finditer(text))

    # Known 8-K item descriptions
    item_8k_names = {
        "1.01": "Entry into Material Agreement",
        "1.02": "Termination of Material Agreement",
        "2.01": "Completion of Acquisition/Disposition",
        "2.02": "Results of Operations (Earnings)",
        "2.03": "Creation of Direct Financial Obligation",
        "2.05": "Costs of Exit/Disposal Activities",
        "2.06": "Material Impairments",
        "3.01": "Delisting Notice",
        "3.03": "Material Modification to Rights",
        "4.01": "Changes in Certifying Accountant",
        "4.02": "Non-Reliance on Financial Statements",
        "5.02": "Departure/Election of Directors/Officers",
        "5.03": "Amendments to Articles/Bylaws",
        "5.07": "Shareholder Vote Submission",
        "7.01": "Regulation FD Disclosure",
        "8.01": "Other Events",
        "9.01": "Financial Statements and Exhibits",
    }

    if not matches:
        # If no items found, store the whole text as a single section
        if len(text.strip()) > 100:
            sections["Full Text"] = text.strip()[:50000]
        return sections

    for i, m in enumerate(matches):
        item_num = m.group(1)
        item_name = item_8k_names.get(item_num, m.group(2).strip() or f"Item {item_num}")

        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else start + 20000
        section_text = text[start:end].strip()

        # Skip exhibits-only sections and very short ones
        if item_num == "9.01" or len(section_text) < 50:
            continue

        sections[f"{item_num} - {item_name}"] = section_text[:50000]

    return sections


def parse_def14a(text: str) -> dict[str, str]:
    """Extract key sections from a DEF 14A (proxy statement)."""
    sections = {}

    # Proxy statements have recognizable section headers
    proxy_headers = [
        ("Executive Compensation", re.compile(
            r"(?i)(?:^|\n)\s*(EXECUTIVE\s+COMPENSATION|COMPENSATION\s+DISCUSSION\s+AND\s+ANALYSIS)",
            re.MULTILINE,
        )),
        ("Director Compensation", re.compile(
            r"(?i)(?:^|\n)\s*(DIRECTOR\s+COMPENSATION)",
            re.MULTILINE,
        )),
        ("Corporate Governance", re.compile(
            r"(?i)(?:^|\n)\s*(CORPORATE\s+GOVERNANCE)",
            re.MULTILINE,
        )),
        ("Board of Directors", re.compile(
            r"(?i)(?:^|\n)\s*(BOARD\s+OF\s+DIRECTORS|ELECTION\s+OF\s+DIRECTORS)",
            re.MULTILINE,
        )),
        ("Shareholder Proposals", re.compile(
            r"(?i)(?:^|\n)\s*(SHAREHOLDER\s+PROPOSALS?|STOCKHOLDER\s+PROPOSALS?)",
            re.MULTILINE,
        )),
    ]

    for name, pattern in proxy_headers:
        match = pattern.search(text)
        if match:
            start = match.end()
            section_text = text[start:start + 30000].strip()
            if len(section_text) > 200:
                sections[name] = section_text

    # Always store full proxy text (truncated) as fallback
    if not sections and len(text.strip()) > 500:
        sections["Full Proxy Text"] = text.strip()[:80000]

    return sections


# ═══════════════════════════════════════════════════════════════════════════
# 5. PARSE ALL FILINGS
# ═══════════════════════════════════════════════════════════════════════════

def extract_exhibit_99(filing_dir: Path) -> str | None:
    """Extract Exhibit 99.1 (Shareholder Letter) from full-submission.txt.
    Returns clean text or None if not found."""
    full_sub = filing_dir / "full-submission.txt"
    if not full_sub.exists():
        return None

    raw = full_sub.read_text(encoding="utf-8", errors="replace")
    # Find the EX-99.1 document block
    ex_match = re.search(
        r"<DOCUMENT>\s*<TYPE>EX-99\.1.*?<TEXT>(.*?)</TEXT>",
        raw,
        re.DOTALL | re.IGNORECASE,
    )
    if not ex_match:
        return None

    html = ex_match.group(1)
    text = html_to_text(html)

    # Skip if too short (probably just a cover page reference)
    if len(text.strip()) < 200:
        return None

    return text.strip()[:80000]


def parse_filing(filing: dict) -> dict:
    """Read and parse a single filing, returning sections."""
    html = read_filing_html(filing["file_path"])
    text = html_to_text(html)

    form = filing["form_type"]
    if form == "10-K":
        sections = parse_10k(text)
    elif form == "8-K":
        sections = parse_8k(text)
        # For 8-K filings with 2.02 (earnings), try to extract the Shareholder Letter
        has_earnings = any("2.02" in s for s in sections)
        if has_earnings or "2.02" in text[:500]:
            exhibit_text = extract_exhibit_99(filing["file_path"].parent)
            if exhibit_text:
                sections["Shareholder Letter (EX-99.1)"] = exhibit_text
    elif form in ("DEF 14A", "DEFA14A"):
        sections = parse_def14a(text)
    else:
        sections = {"Full Text": text[:50000]} if len(text) > 100 else {}

    return {
        **filing,
        "sections": sections,
        "text_length": len(text),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6. SQLITE STORAGE
# ═══════════════════════════════════════════════════════════════════════════

def init_db(db_path: Path) -> sqlite3.Connection:
    """Create SQLite database with schema."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS filings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            form_type TEXT NOT NULL,
            accession_number TEXT UNIQUE NOT NULL,
            filing_date TEXT,
            file_path TEXT,
            text_length INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filing_id INTEGER NOT NULL,
            section_name TEXT NOT NULL,
            section_text TEXT,
            char_count INTEGER,
            FOREIGN KEY (filing_id) REFERENCES filings(id)
        )
    """)
    conn.commit()
    return conn


def extract_filing_date(accession: str) -> str:
    """Try to extract an approximate filing date from the accession number.
    Accession format: CIK-YY-NNNNNN → year is positions after first dash."""
    parts = accession.split("-")
    if len(parts) >= 2 and parts[1].isdigit():
        yy = int(parts[1])
        year = 2000 + yy if yy < 50 else 1900 + yy
        return f"{year}"
    return "unknown"


def store_results(parsed_filings: list[dict], db_path: Path):
    """Store parsed filings into SQLite."""
    conn = init_db(db_path)

    for pf in parsed_filings:
        filing_date = extract_filing_date(pf["accession_number"])

        # Upsert filing
        conn.execute("""
            INSERT OR REPLACE INTO filings (form_type, accession_number, filing_date, file_path, text_length)
            VALUES (?, ?, ?, ?, ?)
        """, (
            pf["form_type"],
            pf["accession_number"],
            filing_date,
            str(pf["file_path"]),
            pf["text_length"],
        ))

        filing_id = conn.execute(
            "SELECT id FROM filings WHERE accession_number = ?",
            (pf["accession_number"],)
        ).fetchone()[0]

        # Delete old sections for this filing (in case of re-run)
        conn.execute("DELETE FROM sections WHERE filing_id = ?", (filing_id,))

        for section_name, section_text in pf["sections"].items():
            conn.execute("""
                INSERT INTO sections (filing_id, section_name, section_text, char_count)
                VALUES (?, ?, ?, ?)
            """, (filing_id, section_name, section_text, len(section_text)))

    conn.commit()
    total_sections = conn.execute("SELECT COUNT(*) FROM sections").fetchone()[0]
    total_filings = conn.execute("SELECT COUNT(*) FROM filings").fetchone()[0]
    print(f"\nDatabase: {total_filings} filings, {total_sections} sections → {db_path}")
    conn.close()


# ═══════════════════════════════════════════════════════════════════════════
# 7. SUMMARY CSV EXPORT
# ═══════════════════════════════════════════════════════════════════════════

def export_summary(db_path: Path, csv_path: Path):
    """Export a summary CSV for quick inspection."""
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("""
        SELECT f.form_type, f.accession_number, f.filing_date,
               s.section_name, s.char_count
        FROM filings f
        JOIN sections s ON s.filing_id = f.id
        ORDER BY f.form_type, f.filing_date, s.section_name
    """).fetchall()
    conn.close()

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["form_type", "accession_number", "filing_date", "section_name", "char_count"])
        writer.writerows(rows)

    print(f"Summary CSV: {len(rows)} rows → {csv_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import sys
    reparse_only = "--reparse" in sys.argv

    print("=" * 60)
    print("SEC FILING DATA COLLECTION — TSLA")
    print("=" * 60)

    # Step 1: Download (skip if --reparse)
    if not reparse_only:
        print("\n[1/4] Downloading filings from SEC EDGAR...")
        download_filings()
    else:
        print("\n[1/4] Skipping download (--reparse mode)")

    # Step 2: Discover
    print("\n[2/4] Discovering downloaded files...")
    filings = discover_filings()
    if not filings:
        print("No filings found. Exiting.")
        return

    # Step 3: Parse
    print(f"\n[3/4] Parsing {len(filings)} filings...")
    parsed = []
    for i, filing in enumerate(filings):
        try:
            result = parse_filing(filing)
            n_sections = len(result["sections"])
            print(f"  [{i+1}/{len(filings)}] {filing['form_type']} {filing['accession_number']} "
                  f"→ {n_sections} sections, {result['text_length']:,} chars")
            parsed.append(result)
        except Exception as e:
            print(f"  [{i+1}/{len(filings)}] ✗ {filing['form_type']} {filing['accession_number']}: {e}")

    # Step 4: Store
    print(f"\n[4/4] Storing results...")
    store_results(parsed, SEC_DB_PATH)
    export_summary(SEC_DB_PATH, SEC_SUMMARY_CSV)

    # Quick stats
    sections_found = sum(len(p["sections"]) for p in parsed)
    empty = sum(1 for p in parsed if not p["sections"])
    print(f"\n{'=' * 60}")
    print(f"DONE: {len(parsed)} filings parsed, {sections_found} total sections extracted")
    if empty:
        print(f"  ⚠ {empty} filings had no extractable sections")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
