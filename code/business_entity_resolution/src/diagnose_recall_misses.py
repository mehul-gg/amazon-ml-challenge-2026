"""Diagnostic: sample real ground-truth pairs that the current full candidate set (blocking.py
+ exact + sorted-token passes) still misses, with their actual name/address text, so the next
supplementary blocking pass targets a real pattern instead of a guess.

Usage:
  python src/diagnose_recall_misses.py --artifacts ../../artifacts --n-samples 40
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from metric import parse_id_list


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="../../artifacts")
    ap.add_argument("--gt", default="../../student_resource/dataset/train/train_ground_truth.tsv")
    ap.add_argument("--n-samples", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts)

    log("Loading ground truth...")
    gt = pd.read_csv(args.gt, sep="\t", dtype=str, keep_default_na=False)
    match_lists = gt["matched_entity_ids"].map(parse_id_list)
    gt_pairs = gt.assign(candidate_entity_id=match_lists).explode("candidate_entity_id")
    gt_pairs = gt_pairs.loc[
        gt_pairs["candidate_entity_id"].notna() & (gt_pairs["candidate_entity_id"] != ""),
        ["source1_entity_id", "candidate_entity_id"],
    ].reset_index(drop=True)
    n_true = len(gt_pairs)
    log(f"  {n_true:,} ground-truth pairs")

    log("Loading full existing candidate set (blocking + exact + sorted)...")
    existing = pd.read_parquet(
        artifacts_dir / "candidate_scores_train.parquet",
        columns=["source1_entity_id", "candidate_entity_id"],
    )
    parts = [existing]
    for mode in ("exact", "sorted"):
        p = artifacts_dir / f"{mode}_match_pairs_train.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p))
    existing_pairs = pd.concat(parts, ignore_index=True).drop_duplicates()
    log(f"  {len(existing_pairs):,} unique existing candidate pairs")

    log("Finding missed ground-truth pairs (integer-encoded comparison)...")
    all_s1 = pd.concat([existing_pairs["source1_entity_id"], gt_pairs["source1_entity_id"]], ignore_index=True)
    s1_codes, _ = pd.factorize(all_s1)
    all_cand = pd.concat([existing_pairs["candidate_entity_id"], gt_pairs["candidate_entity_id"]], ignore_index=True)
    cand_codes, _ = pd.factorize(all_cand)
    n_cand = cand_codes.max() + 1
    n_have = len(existing_pairs)
    have_key = s1_codes[:n_have].astype(np.int64) * n_cand + cand_codes[:n_have].astype(np.int64)
    gt_key = s1_codes[n_have:].astype(np.int64) * n_cand + cand_codes[n_have:].astype(np.int64)
    have_key_sorted = np.sort(np.unique(have_key))
    is_hit = np.isin(gt_key, have_key_sorted)
    n_hit = int(is_hit.sum())
    log(f"  Recall: {n_hit:,} / {n_true:,} = {n_hit / n_true:.4%}")

    missed = gt_pairs.loc[~is_hit].reset_index(drop=True)
    log(f"  {len(missed):,} missed pairs. Sampling {args.n_samples}...")
    sample = missed.sample(n=min(args.n_samples, len(missed)), random_state=args.seed)

    log("Loading name/address text for sampled misses...")
    s1_ids = set(sample["source1_entity_id"])
    cand_ids = set(sample["candidate_entity_id"])
    is_s2 = {c for c in cand_ids if c.startswith("S2-")}
    is_s3 = {c for c in cand_ids if c.startswith("S3-")}
    cols = ["entity_id", "country", "name_core", "address_clean"]
    s1_rec = pd.read_parquet(
        artifacts_dir / "norm_train_source1.parquet", columns=cols,
        filters=[("entity_id", "in", list(s1_ids))],
    )
    s2_rec = pd.read_parquet(
        artifacts_dir / "norm_train_source2.parquet", columns=cols,
        filters=[("entity_id", "in", list(is_s2))],
    ) if is_s2 else pd.DataFrame(columns=cols)
    s3_rec = pd.read_parquet(
        artifacts_dir / "norm_train_source3.parquet", columns=cols,
        filters=[("entity_id", "in", list(is_s3))],
    ) if is_s3 else pd.DataFrame(columns=cols)
    rec = pd.concat([s1_rec, s2_rec, s3_rec], ignore_index=True).set_index("entity_id")

    print("\n===== SAMPLED MISSED GROUND-TRUTH PAIRS =====")
    for _, row in sample.iterrows():
        s1_id, cand_id = row["source1_entity_id"], row["candidate_entity_id"]
        if s1_id not in rec.index or cand_id not in rec.index:
            continue
        a, b = rec.loc[s1_id], rec.loc[cand_id]
        print(f"\n[{a['country']}] {s1_id} <-> {cand_id}")
        print(f"  name:    {a['name_core']!r:60}  |  {b['name_core']!r}")
        print(f"  address: {a['address_clean']!r:60}  |  {b['address_clean']!r}")


if __name__ == "__main__":
    main()
