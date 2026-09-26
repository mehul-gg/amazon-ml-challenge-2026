"""Stage 1 — Normalisation (country-agnostic).

Cleans business_name and business_address into forms that are comparable across
sources and countries: strips accents/scripts noise, expands abbreviations,
isolates the legal suffix, extracts postal codes / house numbers, and strips
placeholder tokens seen in real data (<NULL>, N/A, bracket noise around
suffixes, embedded phone numbers) — see experiments.md "Noise patterns observed".

Design notes (per CLAUDE.md revision after Day 1 EDA):
- Every dataframe-level function is vectorized (pandas .str / .map over ~5-10M
  rows), never a Python loop with per-row dict/Series lookups.
- Country is used only as a hard filter elsewhere (blocking); nothing here
  hard-codes {US, India} — French legal forms/street words are included
  alongside US/Indian ones since test adds an unseen country.

Usage (library): from normalize import normalize_source
Usage (CLI, smoke test): python src/normalize.py --data ../../student_resource/dataset
"""
from __future__ import annotations

import argparse
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Reference lists (our own domain rules — not external lookups; disclosed in
# the methodology doc per the challenge's fair-play rules).
# ---------------------------------------------------------------------------

# Legal-suffix tokens (after lowercasing/punctuation stripping), longest-first
# so e.g. "private limited" matches before "limited". Suffix is stored
# separately as a feature and stripped from the "core" name used for fuzzy
# matching, since two records can differ only in whether they include it.
LEGAL_SUFFIXES = sorted(
    [
        "private limited", "pvt ltd", "pvt limited", "private ltd",
        "limited liability company", "limited liability partnership",
        "llp", "llc", "l l c", "ltd", "limited", "inc", "incorporated",
        "corp", "corporation", "co", "company", "plc",
        "sa", "sas", "sarl", "eurl", "sci",  # France
        "gmbh", "ag",  # occasionally seen in international records
    ],
    key=len,
    reverse=True,
)
_SUFFIX_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in LEGAL_SUFFIXES) + r")\b\.?\s*$"
)

# Address abbreviation expansion (word-boundary, applied after lowercasing).
# US/India/France street + unit words combined — country-agnostic on purpose.
ADDRESS_ABBREV = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "dr": "drive", "ln": "lane",
    "hwy": "highway", "pkwy": "parkway", "ct": "court", "pl": "place",
    "sq": "square", "ter": "terrace", "cir": "circle", "apt": "apartment",
    "fl": "floor", "flr": "floor", "bldg": "building", "no": "number",
    "opp": "opposite", "nr": "near", "ch": "chemin",  # France
    "marg": "marg", "nagar": "nagar",  # kept as-is, common Indian address words
}
_ADDR_ABBREV_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in ADDRESS_ABBREV) + r")\b"
)

# Placeholder / junk tokens seen in real data (e.g. literal "<NULL>" inline in
# an address — see experiments.md sample matches) — stripped, not treated as content.
PLACEHOLDER_PATTERN = re.compile(
    r"\b(null|n/?a|none|unknown|nil|undefined)\b|<[^>]*>|\[|\]", re.IGNORECASE
)

# Landmark phrases in addresses — flagged as a feature, not stripped (they
# still carry weak locality signal even though they're not a formal address part).
LANDMARK_PATTERN = re.compile(r"\b(near|opp(?:osite)?|behind|beside|next to)\b", re.IGNORECASE)

# Postal code: generic 5-or-6-digit run near the end of the string. US ZIP and
# French CP are 5 digits, Indian PIN is 6 — deliberately generic (country is
# an open set; France must work without a France-specific rule).
POSTAL_PATTERN = re.compile(r"\b(\d{5,6})\b")

# House/unit number: a leading numeric token (with an optional letter suffix
# like "12A" or trailing unit like "1795") — first number in the string.
NUMBER_PATTERN = re.compile(r"\b(\d+[a-zA-Z]?)\b")

# Phone-number-like tokens embedded in the name field (e.g. "Chordia + Partners
# - 7306204978") — 7+ consecutive digits, optionally with separators.
PHONE_IN_NAME_PATTERN = re.compile(r"[\s\-,]*(?:\+?\d[\d\-\s]{6,}\d)\s*$")

# Strip a specific ASCII punctuation set rather than "anything not \w": Python's
# \w does NOT include Unicode combining-mark categories (Mn/Mc), so a "[^\w...]"
# pattern silently strips vowel signs from Devanagari and other scripts too
# (confirmed: it turned "होटल" into "ह टल", eating the "ो" vowel sign, category
# Mc). An explicit punctuation set leaves every script's letters/marks intact.
_PUNCT_CHARS = ".,;:!?\"'()[]{}/\\|_+=<>~`^*@#$%-"
_PUNCT_PATTERN = re.compile("[" + re.escape(_PUNCT_CHARS) + "]")
_WS_PATTERN = re.compile(r"\s+")


def strip_accents(s: str) -> str:
    """NFKD-normalise and drop *Latin* accent marks (e -> e). Needed for France.

    Only strips combining marks in the U+0300-U+036F "Combining Diacritical
    Marks" block (the block Latin accents decompose into). Devanagari vowel
    signs (matras) and other scripts' combining marks live in different
    Unicode blocks and are category "Mn" too — blindly stripping all
    category-Mn combining marks corrupts non-Latin text entirely (confirmed:
    it turned "होटल एंटरप्राइजेज लिमिटेड" into garbled fragments). Country is
    an open set including France, but the noisy sources also contain Indian
    transliterated names that must stay intact for the embedding retriever.
    """
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not (unicodedata.combining(c) and 0x0300 <= ord(c) <= 0x036F)
    )


def basic_clean(s: str) -> str:
    """Lowercase, strip accents, drop placeholders/brackets, collapse punctuation/whitespace."""
    if not s:
        return ""
    s = strip_accents(s)
    s = s.lower()
    s = PLACEHOLDER_PATTERN.sub(" ", s)
    s = s.replace("&", " and ")
    s = _PUNCT_PATTERN.sub(" ", s)
    s = _WS_PATTERN.sub(" ", s).strip()
    return s


NAME_FIELDS = ("name_clean", "name_core", "legal_suffix", "name_has_phone")
ADDRESS_FIELDS = ("address_clean", "postal_code", "house_number", "has_landmark")


def normalize_name(raw: str) -> dict:
    """Split a business name into a cleaned full form, a suffix-stripped core, and the suffix."""
    if not raw:
        return {"name_clean": "", "name_core": "", "legal_suffix": "", "name_has_phone": False}

    # Strip an embedded phone number from the raw text before cleaning (kept as a flag,
    # not used for text similarity — see experiments.md "phone numbers embedded in name").
    has_phone = bool(PHONE_IN_NAME_PATTERN.search(raw))
    text = PHONE_IN_NAME_PATTERN.sub("", raw)

    clean = basic_clean(text)
    m = _SUFFIX_PATTERN.search(clean)
    if m:
        suffix = m.group(1)
        core = clean[: m.start()].strip()
    else:
        suffix = ""
        core = clean
    return {"name_clean": clean, "name_core": core, "legal_suffix": suffix, "name_has_phone": has_phone}


def normalize_name_tuple(raw: str) -> tuple:
    d = normalize_name(raw)
    return tuple(d[f] for f in NAME_FIELDS)


def normalize_address(raw: str) -> dict:
    """Clean an address, expand abbreviations, and pull out postal code / house number / landmark flag."""
    if not raw:
        return {"address_clean": "", "postal_code": "", "house_number": "", "has_landmark": False}

    has_landmark = bool(LANDMARK_PATTERN.search(raw))
    postal = ""
    pm = POSTAL_PATTERN.search(raw)
    if pm:
        postal = pm.group(1)
    house = ""
    nm = NUMBER_PATTERN.search(raw)
    if nm:
        house = nm.group(1).lower()

    clean = basic_clean(raw)
    clean = _ADDR_ABBREV_PATTERN.sub(lambda m: ADDRESS_ABBREV[m.group(1)], clean)
    clean = _WS_PATTERN.sub(" ", clean).strip()
    return {
        "address_clean": clean,
        "postal_code": postal,
        "house_number": house,
        "has_landmark": has_landmark,
    }


def normalize_address_tuple(raw: str) -> tuple:
    d = normalize_address(raw)
    return tuple(d[f] for f in ADDRESS_FIELDS)


def normalize_source(df: pd.DataFrame) -> pd.DataFrame:
    """Normalisation of one source file, single-pass per column group.

    Memory note: an earlier version mapped each row to a dict, then re-scanned
    that Series of ~5M dict objects once per field (8 total passes) to pull
    fields out via `.map(lambda d: d[key])`. At this dataset's scale that held
    millions of Python dict objects alive at once and caused an out-of-memory
    crash on a plain 40MB allocation (real cause: overall memory pressure, not
    that allocation itself — see experiments.md). Fixed by mapping each column
    to a tuple once, then building all fields for that group in a single
    `pd.DataFrame(list_of_tuples, ...)` call — one pass, no dict objects.
    """
    import gc

    name_cols = pd.DataFrame(
        df["business_name"].map(normalize_name_tuple).tolist(), columns=list(NAME_FIELDS), index=df.index
    )
    addr_cols = pd.DataFrame(
        df["business_address"].map(normalize_address_tuple).tolist(), columns=list(ADDRESS_FIELDS), index=df.index
    )
    out = pd.concat([df, name_cols, addr_cols], axis=1)
    del name_cols, addr_cols
    gc.collect()
    return out


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def normalize_and_cache(raw_path: Path, cache_path: Path, columns: list | None = None) -> pd.DataFrame:
    """Normalize a source file and cache the result to parquet (artifacts/), unless already cached.
    `columns` restricts a cache-hit read to just those columns (parquet supports column
    pushdown, so unneeded columns — notably the full raw business_name/business_address text —
    are never loaded into memory at all, not loaded-then-dropped)."""
    if cache_path.exists():
        return pd.read_parquet(cache_path, columns=columns)
    t0 = time.time()
    df = read_tsv(raw_path)
    df = normalize_source(df)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"  normalized {raw_path.name}: {len(df):,} rows in {time.time()-t0:.1f}s -> {cache_path}")
    return df


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Smoke-test normalisation on a few rows and time a full source file.")
    ap.add_argument("--data", default="../../student_resource/dataset")
    ap.add_argument("--artifacts", default="../../artifacts")
    ap.add_argument("--full", action="store_true", help="Also normalize+cache train_source1 to test throughput.")
    args = ap.parse_args()
    data_dir = Path(args.data)

    samples = [
        "Sharma Traders Private Limited",
        "SHARMA TRADERS",
        "Chordia + Pagnters - 7306204978",
        "Obsidian, [[LLC]]",
        "Korbrixx D.B.A. Obsidian, LLC",
        "होटल एंटरप्राइजेज लिमिटेड",
    ]
    print("=== name normalisation smoke test ===")
    for s in samples:
        print(f"{s!r:55} -> {normalize_name(s)}")

    addr_samples = [
        "12, MG Road, Near SBI ATM, Pune 411001",
        "728 A Quail Avenue, Fl Ground Floor, Geneva, IA",
        "Wz-187C Shop No.13, 14 Kh. No.47 S/F. Vikaspuri Budhela Village Behind Oxford School, Delhi",
        "939389287",  # deliberately garbage to confirm no crash
        "33466 WARWICK HILLS ROAD, <NULL>, YUCAIPA, CA",
        "",
    ]
    print("\n=== address normalisation smoke test ===")
    for s in addr_samples:
        print(f"{s!r:80} -> {normalize_address(s)}")

    if args.full:
        print("\n=== throughput test: train_source1 ===")
        art = Path(args.artifacts)
        normalize_and_cache(data_dir / "train" / "train_source1.tsv", art / "norm_train_source1.parquet")
