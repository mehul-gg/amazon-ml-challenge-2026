"""Stage 3a — Pairwise features for (S1, candidate) pairs.

Reads a candidate_pairs.tsv (from blocking.py) plus the normalized source
parquets, computes similarity features for every (S1, candidate) pair, and
(for train) attaches the ground-truth label. Output feeds train.py.

Memory note: this is written to run alongside a possibly-still-running,
memory-heavy blocking.py background job (confirmed as little as ~2GB free
RAM in that situation during Day 1 development) — every source frame is
loaded with only the needed columns and immediately filtered down to just
the entity_ids referenced by the candidate pairs, before any feature
computation, same discipline as blocking.py.

Usage:
  python src/features.py --data ../../student_resource/dataset --artifacts ../../artifacts \
      --candidates ../../artifacts/candidate_pairs_train_sample50k.tsv --split train \
      --out ../../artifacts/features_train_sample50k.parquet
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz

from metric import parse_id_list

# Only the columns compute_features() actually reads. Originally included business_name,
# business_address, country and name_clean too — none of which any feature uses — which
# meant rec.reindex() (called twice per batch, once per side of the pair) was needlessly
# carrying full raw-text columns through a 43.4M-row operation. Confirmed as the direct
# cause of a MemoryError at full train scale (see experiments.md); trimming this list is
# the fix, same "column pushdown" principle as blocking.py's BLOCKING_COLS.
FULL_COLS = [
    "entity_id", "name_core", "legal_suffix", "name_has_phone",
    "address_clean", "postal_code", "house_number", "has_landmark",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_candidate_pairs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df["candidate_entity_id"] = df["candidate_entity_ids"].map(parse_id_list)
    df = df.explode("candidate_entity_id")
    df = df.loc[df["candidate_entity_id"].notna() & (df["candidate_entity_id"] != ""),
                ["source1_entity_id", "candidate_entity_id"]]
    return df.reset_index(drop=True)


def load_candidate_pairs_from_sources(artifacts_dir: Path, split: str, extra_modes: list) -> pd.DataFrame:
    """Alternative to load_candidate_pairs() for full-scale runs: builds the pairs frame
    directly from blocking.py's candidate_scores_{split}.parquet plus each
    {mode}_match_pairs_{split}.parquet (already flat, one row per pair), instead of writing a
    comma-joined candidate_pairs.tsv and then exploding it back apart. That round-trip is pure
    waste and, at merged4 scale (45.2M pairs), the explode()'s internal reindex/concat crashed
    with an ArrayMemoryError -- a smaller-scale version of the exact same problem
    exact_match_candidates.py's --measure-recall-gain already warns about and avoids (see its
    docstring: 'NOT the candidate_pairs.tsv -- exploding that comma-joined TSV crashed')."""
    parts = [pd.read_parquet(
        artifacts_dir / f"candidate_scores_{split}.parquet",
        columns=["source1_entity_id", "candidate_entity_id"],
    )]
    for mode in extra_modes:
        path = artifacts_dir / f"{mode}_match_pairs_{split}.parquet"
        if path.exists():
            parts.append(pd.read_parquet(path))
    combined = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    # Dedup via integer-encoded keys, not drop_duplicates() on the raw string columns --
    # confirmed at this exact row count (~60M) to sometimes thrash under memory pressure
    # (paging, not a hard crash) badly enough to take 15+ minutes instead of under a minute.
    # This mirrors the same fix already applied to blocking.py's measure_recall and to the
    # nospace-pass recall measurement earlier in this project.
    s1_codes, s1_uniques = pd.factorize(combined["source1_entity_id"])
    cand_codes, cand_uniques = pd.factorize(combined["candidate_entity_id"])
    n_cand = len(cand_uniques)
    keys = s1_codes.astype(np.int64) * n_cand + cand_codes.astype(np.int64)
    _, first_idx = np.unique(keys, return_index=True)
    first_idx = np.sort(first_idx)
    combined = combined.iloc[first_idx].reset_index(drop=True)
    return combined


def load_filtered(parquet_path: Path, ids: set) -> pd.DataFrame:
    """Load only FULL_COLS, filtered to `ids` via pyarrow predicate pushdown — filtering
    happens in pyarrow's C++ layer during the scan, so rows outside `ids` are never
    materialized as pandas/Python objects at all. Needed here specifically because this
    script is designed to run alongside a memory-heavy blocking.py background job (as
    little as ~2.4GB free RAM was observed during Day 1 development) — a plain
    "load everything, then .isin() filter" would briefly hold the full ~670MB (on-disk;
    several GB as Python string objects) source frame in memory first, which doesn't fit."""
    return pd.read_parquet(
        parquet_path, columns=FULL_COLS, engine="pyarrow",
        filters=[("entity_id", "in", list(ids))],
    )


def _acronym(name_core: str) -> str:
    """First letter of each token, e.g. 'international business machines' -> 'ibm'."""
    return "".join(tok[0] for tok in name_core.split() if tok)


def _norm_len(s: str) -> int:
    return len(s) if isinstance(s, str) else 0


def _compute_features_batch(pairs: pd.DataFrame, rec: pd.DataFrame) -> pd.DataFrame:
    """One batch's worth of the actual feature computation — see compute_features for why
    this is called in chunks rather than once over the whole dataset."""
    a = rec.reindex(pairs["source1_entity_id"]).reset_index(drop=True)
    b = rec.reindex(pairs["candidate_entity_id"]).reset_index(drop=True)
    out = pairs.reset_index(drop=True).copy()

    # --- Fuzzy string similarity (rapidfuzz) on core name and address ---
    # .map over paired Series via zip is the practical way to call a non-vectorized C
    # function (rapidfuzz) per row; still O(n) simple work per row, same cost class as the
    # normalize.py functions that were already confirmed to run fine at multi-million scale.
    name_a, name_b = a["name_core"].fillna(""), b["name_core"].fillna("")
    out["f_name_ratio"] = [fuzz.ratio(x, y) for x, y in zip(name_a, name_b)]
    out["f_name_partial_ratio"] = [fuzz.partial_ratio(x, y) for x, y in zip(name_a, name_b)]
    out["f_name_token_sort"] = [fuzz.token_sort_ratio(x, y) for x, y in zip(name_a, name_b)]
    out["f_name_token_set"] = [fuzz.token_set_ratio(x, y) for x, y in zip(name_a, name_b)]

    addr_a, addr_b = a["address_clean"].fillna(""), b["address_clean"].fillna("")
    out["f_addr_ratio"] = [fuzz.ratio(x, y) for x, y in zip(addr_a, addr_b)]
    out["f_addr_token_set"] = [fuzz.token_set_ratio(x, y) for x, y in zip(addr_a, addr_b)]
    out["f_addr_either_empty"] = (addr_a == "") | (addr_b == "")

    # --- Token overlap (Jaccard) on core name ---
    def jaccard(x: str, y: str) -> float:
        sx, sy = set(x.split()), set(y.split())
        if not sx or not sy:
            return 0.0
        return len(sx & sy) / len(sx | sy)

    out["f_name_jaccard"] = [jaccard(x, y) for x, y in zip(name_a, name_b)]

    # --- Name structure ---
    out["f_suffix_a"] = a["legal_suffix"].fillna("")
    out["f_suffix_b"] = b["legal_suffix"].fillna("")
    out["f_suffix_both_present"] = (out["f_suffix_a"] != "") & (out["f_suffix_b"] != "")
    out["f_suffix_match"] = out["f_suffix_both_present"] & (out["f_suffix_a"] == out["f_suffix_b"])
    out["f_suffix_one_missing"] = (out["f_suffix_a"] == "") != (out["f_suffix_b"] == "")
    out.drop(columns=["f_suffix_a", "f_suffix_b"], inplace=True)

    out["f_name_contains"] = [
        (x in y or y in x) if x and y else False for x, y in zip(name_a, name_b)
    ]
    acro_a = name_a.map(_acronym)
    acro_b = name_b.map(_acronym)
    out["f_acronym_match"] = (acro_a == b["name_core"].fillna("").str.replace(" ", "")) | (
        acro_b == a["name_core"].fillna("").str.replace(" ", "")
    )

    out["f_name_has_phone_a"] = a["name_has_phone"].fillna(False).astype(bool)
    out["f_name_has_phone_b"] = b["name_has_phone"].fillna(False).astype(bool)

    # --- Address evidence: postal code, house number, landmark ---
    postal_a, postal_b = a["postal_code"].fillna(""), b["postal_code"].fillna("")
    out["f_postal_both_present"] = (postal_a != "") & (postal_b != "")
    out["f_postal_match"] = out["f_postal_both_present"] & (postal_a == postal_b)
    out["f_postal_conflict"] = out["f_postal_both_present"] & (postal_a != postal_b)

    house_a, house_b = a["house_number"].fillna(""), b["house_number"].fillna("")
    out["f_house_both_present"] = (house_a != "") & (house_b != "")
    out["f_house_match"] = out["f_house_both_present"] & (house_a == house_b)
    out["f_house_conflict"] = out["f_house_both_present"] & (house_a != house_b)

    out["f_landmark_a"] = a["has_landmark"].fillna(False).astype(bool)
    out["f_landmark_b"] = b["has_landmark"].fillna(False).astype(bool)

    # --- Lengths (raw signal, cheap, lets the model learn e.g. very short-name risk) ---
    out["f_name_len_a"] = name_a.map(_norm_len)
    out["f_name_len_b"] = name_b.map(_norm_len)
    out["f_name_len_diff"] = (out["f_name_len_a"] - out["f_name_len_b"]).abs()

    return out


def _build_gt_key_set(gt_path: Path, scope: set) -> set:
    """A Python set of combined 'source1_entity_id\\x01candidate_entity_id' strings for every
    true match — built ONCE (this set is small enough: 7.6M keys at full train scale) and
    reused across every batch's label lookup, so labeling never needs the full feature set
    and the full ground truth in memory together (see compute_and_write_features)."""
    gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    gt = gt[gt["source1_entity_id"].isin(scope)]
    # Flatten (s1, {matches}) rows into one key per match without pandas' explode() -- its
    # internal reindex/concat path has now crashed with an ArrayMemoryError twice this session
    # under transient memory pressure, even at this comparatively small (~2.2M-row) scale. A
    # plain Python loop over match sets is fast here since most are tiny (1-5 items).
    keys = [
        f"{s1}\x01{cand}"
        for s1, matches in zip(gt["source1_entity_id"], gt["matched_entity_ids"].map(parse_id_list))
        for cand in matches
        if cand
    ]
    return set(keys)


def _downcast_features(feat: pd.DataFrame) -> pd.DataFrame:
    """Store feature columns in compact dtypes instead of pandas' float64/int64 defaults —
    rapidfuzz ratio/score features are 0-100 (or 0-1 for jaccard), already low-precision, so
    float32 loses nothing that matters; string-length columns and the 0/1 label don't need
    64 bits either. At full train scale (43.4M rows) this roughly halves the size of every
    downstream in-memory feature matrix, which is what several MemoryErrors in
    train.py/decide.py traced back to (see experiments.md) — better fixed once here than
    patched in every consumer."""
    for col in feat.columns:
        if feat[col].dtype == np.float64:
            feat[col] = feat[col].astype(np.float32)
        elif feat[col].dtype == np.int64 and col != "label":
            feat[col] = feat[col].astype(np.int32)
    if "label" in feat.columns:
        feat["label"] = feat["label"].astype(np.int8)
    return feat


def compute_and_write_features(
    pairs: pd.DataFrame, rec: pd.DataFrame, out_path: Path, gt_path: Path | None,
    batch_size: int = 2_000_000,
) -> int:
    """Computes features (and, if `gt_path` is given, the label) in batches and streams each
    batch straight to `out_path` via a single incrementally-written parquet file — never
    holding more than one batch's results in memory.

    This replaced an earlier version that computed all batches into a Python list and
    concatenated them at the end: at full train scale (43.4M pairs) that final concat alone
    needed as much memory as the un-batched version would have, defeating the point of
    batching in the first place (confirmed: MemoryError in pd.concat after all 22 batches'
    worth of computation had already succeeded — see experiments.md). Streaming to disk
    avoids ever materializing the full result at all.
    """
    gt_keys = None
    if gt_path is not None:
        log("Building ground-truth key set (once, reused across all batches)...")
        gt_keys = _build_gt_key_set(gt_path, set(pairs["source1_entity_id"]))
        log(f"  {len(gt_keys):,} ground-truth pairs in scope")

    log(f"Computing features for {len(pairs):,} pairs in batches of {batch_size:,}, "
        f"streaming to {out_path}...")
    writer = None
    n_written = 0
    n_positive = 0
    try:
        for start in range(0, len(pairs), batch_size):
            batch = pairs.iloc[start : start + batch_size]
            feat = _compute_features_batch(batch, rec)
            if gt_keys is not None:
                key = feat["source1_entity_id"] + "\x01" + feat["candidate_entity_id"]
                feat["label"] = key.isin(gt_keys).astype(int)
                n_positive += int(feat["label"].sum())
            feat = _downcast_features(feat)
            table = pa.Table.from_pandas(feat, preserve_index=False)
            if writer is None:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(str(out_path), table.schema)
            writer.write_table(table)
            n_written += len(feat)
            del feat, table
            gc.collect()
            log(f"  {min(start + batch_size, len(pairs)):,} / {len(pairs):,} pairs done")
    finally:
        if writer is not None:
            writer.close()

    if gt_keys is not None:
        log(f"Wrote {n_written:,} rows, {n_positive:,} positives ({n_positive / max(n_written, 1):.2%})")
    else:
        log(f"Wrote {n_written:,} rows")
    return n_written


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../student_resource/dataset")
    ap.add_argument("--artifacts", default="../../artifacts")
    ap.add_argument("--candidates", default=None, help="candidate_pairs.tsv from blocking.py "
                     "(explode-based; fine at smaller scale, crashed at 45M+ pairs).")
    ap.add_argument("--extra-modes", nargs="*", default=None,
                     help="Build pairs directly from candidate_scores_{split}.parquet + these "
                          "{mode}_match_pairs_{split}.parquet files instead of --candidates -- "
                          "avoids the comma-join/explode round-trip entirely. Mutually "
                          "exclusive with --candidates.")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if (args.candidates is None) == (args.extra_modes is None):
        raise SystemExit("Pass exactly one of --candidates or --extra-modes.")

    data_dir = Path(args.data)
    artifacts_dir = Path(args.artifacts)

    if args.candidates is not None:
        log(f"Loading candidate pairs from {args.candidates}...")
        pairs = load_candidate_pairs(Path(args.candidates))
    else:
        log(f"Building candidate pairs directly from parquet sources (modes: {args.extra_modes})...")
        pairs = load_candidate_pairs_from_sources(artifacts_dir, args.split, args.extra_modes)
    log(f"  {len(pairs):,} candidate pairs, {pairs['source1_entity_id'].nunique():,} S1 entities")

    s1_ids = set(pairs["source1_entity_id"])
    cand_ids = set(pairs["candidate_entity_id"])
    is_s2 = {c for c in cand_ids if c.startswith("S2-")}
    is_s3 = {c for c in cand_ids if c.startswith("S3-")}
    log(f"  referenced candidates: {len(is_s2):,} from S2, {len(is_s3):,} from S3")

    log("Loading + filtering S1...")
    s1 = load_filtered(artifacts_dir / f"norm_{args.split}_source1.parquet", s1_ids)
    log("Loading + filtering S2...")
    s2 = load_filtered(artifacts_dir / f"norm_{args.split}_source2.parquet", is_s2)
    log("Loading + filtering S3...")
    s3 = load_filtered(artifacts_dir / f"norm_{args.split}_source3.parquet", is_s3)
    rec = pd.concat([s1, s2, s3], ignore_index=True).set_index("entity_id")
    del s1, s2, s3
    gc.collect()
    log(f"Combined filtered record lookup: {len(rec):,} rows")

    gt_path = (data_dir / "train" / "train_ground_truth.tsv") if args.split == "train" else None
    out_path = Path(args.out)
    compute_and_write_features(pairs, rec, out_path, gt_path)
    del rec
    gc.collect()


if __name__ == "__main__":
    main()
