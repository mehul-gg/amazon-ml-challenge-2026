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
from rapidfuzz import fuzz

from metric import parse_id_list

FULL_COLS = [
    "entity_id", "country", "business_name", "business_address",
    "name_clean", "name_core", "legal_suffix", "name_has_phone",
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


def compute_features(pairs: pd.DataFrame, rec: pd.DataFrame) -> pd.DataFrame:
    """rec: entity_id-indexed frame with FULL_COLS (minus entity_id) for every S1 and candidate
    referenced in `pairs`. Returns pairs with feature columns appended."""
    log(f"Computing features for {len(pairs):,} pairs...")
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

    log(f"Done. {out.shape[1] - 2} feature columns.")
    return out


def attach_labels(features: pd.DataFrame, gt_path: Path) -> pd.DataFrame:
    # Filter ground truth to just the S1s actually present BEFORE exploding — the full
    # ground truth explodes to 7.6M pairs; scoping first keeps this proportional to the
    # sample size instead of the full dataset (same fix as blocking.py's measure_recall).
    scope = set(features["source1_entity_id"])
    gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    gt = gt[gt["source1_entity_id"].isin(scope)]
    match_lists = gt["matched_entity_ids"].map(parse_id_list)
    true_pairs = gt.assign(candidate_entity_id=match_lists).explode("candidate_entity_id")
    true_pairs = true_pairs.loc[
        true_pairs["candidate_entity_id"].notna() & (true_pairs["candidate_entity_id"] != ""),
        ["source1_entity_id", "candidate_entity_id"],
    ].copy()
    true_pairs["label"] = 1
    features = features.merge(
        true_pairs, on=["source1_entity_id", "candidate_entity_id"], how="left"
    )
    features["label"] = features["label"].fillna(0).astype(int)
    return features


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../student_resource/dataset")
    ap.add_argument("--artifacts", default="../../artifacts")
    ap.add_argument("--candidates", required=True, help="candidate_pairs.tsv from blocking.py")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data_dir = Path(args.data)
    artifacts_dir = Path(args.artifacts)

    log(f"Loading candidate pairs from {args.candidates}...")
    pairs = load_candidate_pairs(Path(args.candidates))
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

    features = compute_features(pairs, rec)
    del rec
    gc.collect()

    if args.split == "train":
        log("Attaching ground-truth labels...")
        features = attach_labels(features, data_dir / "train" / "train_ground_truth.tsv")
        log(f"  positives: {features['label'].sum():,} / {len(features):,} "
            f"({features['label'].mean():.2%})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)
    log(f"Wrote {len(features):,} rows x {features.shape[1]} cols to {out_path}")


if __name__ == "__main__":
    main()
