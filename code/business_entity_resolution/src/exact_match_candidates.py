"""Supplementary blocking pass: exact core-name matches, uncapped by max_df.

Real recall-miss cases inspected from train ground truth (see experiments.md) showed
entities like 'krishna power' -> 'krishna power' (identical strings!) and 'united agro' ->
'united agro' still missing from candidates — the phrase is common enough that even an
EXACT match gets dropped by the rarest-token selection in blocking.py. An exact full-name
match is a strong, self-limiting signal (it can never produce more candidates than however
many records actually share that exact string) and deserves to never be filtered by
frequency the way individual tokens are.

This is a separate, much cheaper pass than blocking.py's inverted-index build: group all
records by (country, name_core) and union every S2/S3 in a group with every S1 in the same
group — no token explosion, no per-entity rarest-key selection, just a groupby.

Usage:
  python src/exact_match_candidates.py --artifacts ../../artifacts --split train
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from metric import parse_id_list


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compute_exact_match_pairs(artifacts_dir: Path, split: str, max_group_size: int = 50) -> pd.DataFrame:
    """Returns columns [source1_entity_id, candidate_entity_id] — every (S1, candidate) pair
    sharing an exact, non-empty (country, name_core). Vectorized merge per country, not a
    Python loop over groups — the earlier version's `groupby(...)` + per-group Python loop
    would have repeated the exact slow-loop-over-millions-of-rows pattern that caused
    problems throughout blocking.py's development.

    `max_group_size` caps how many total records (S1+candidates) may share one exact name
    before that name is dropped from this pass entirely — a first, uncapped run on train
    produced 75.2M pairs (nearly double the entire properly-blocked candidate set of 43.4M)
    from just 1.97M S1 entities. The per-S1 group-size distribution showed why: median 4
    (genuine near-duplicates, e.g. "krishna power"), but a max of 1,359 — a handful of very
    generic names (short/common words surviving normalization) were dominating the total.
    Capping mirrors blocking.py's max_df concept, just applied to whole-name groups instead
    of individual tokens."""
    cols = ["entity_id", "country", "name_core"]
    s1_full = pd.read_parquet(artifacts_dir / f"norm_{split}_source1.parquet", columns=cols)
    s2_full = pd.read_parquet(artifacts_dir / f"norm_{split}_source2.parquet", columns=cols)
    s3_full = pd.read_parquet(artifacts_dir / f"norm_{split}_source3.parquet", columns=cols)
    cand_full = pd.concat([s2_full, s3_full], ignore_index=True)
    del s2_full, s3_full
    s1_full = s1_full[s1_full["name_core"] != ""]
    cand_full = cand_full[cand_full["name_core"] != ""]

    results = []
    countries = sorted(set(s1_full["country"]) & set(cand_full["country"]))
    for country in countries:
        t0 = time.time()
        s1_c = s1_full[s1_full["country"] == country][["entity_id", "name_core"]]
        cand_c = cand_full[cand_full["country"] == country][["entity_id", "name_core"]]

        # Drop over-generic names BEFORE the merge: count total (S1+candidate) occurrences
        # per name_core in this country, exclude any name over the cap. Cheap (value_counts
        # on two already-small, country-scoped columns), and keeps the merge itself small.
        name_counts = pd.concat([s1_c["name_core"], cand_c["name_core"]]).value_counts()
        ok_names = name_counts[name_counts <= max_group_size].index
        s1_c = s1_c[s1_c["name_core"].isin(ok_names)]
        cand_c = cand_c[cand_c["name_core"].isin(ok_names)]

        merged = s1_c.merge(cand_c, on="name_core", suffixes=("_s1", "_cand"))
        merged = merged.rename(columns={"entity_id_s1": "source1_entity_id", "entity_id_cand": "candidate_entity_id"})
        log(f"  Country={country}: {len(merged):,} exact-match pairs in {time.time()-t0:.1f}s "
            f"(after dropping names shared by >{max_group_size} records)")
        results.append(merged[["source1_entity_id", "candidate_entity_id"]])

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(
        columns=["source1_entity_id", "candidate_entity_id"]
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="../../artifacts")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--out", default=None)
    ap.add_argument("--measure-recall-gain", action="store_true",
                     help="Train only: report how much this ADDS to existing blocking recall.")
    ap.add_argument("--gt", default="../../student_resource/dataset/train/train_ground_truth.tsv")
    ap.add_argument("--existing-candidates", default=None,
                     help="Existing candidate_scores_{split}.parquet (flat id columns, from "
                          "blocking.py) to compare against for --measure-recall-gain. NOT the "
                          "candidate_pairs.tsv — exploding that comma-joined TSV at full scale "
                          "crashed with MemoryError; the parquet already has one row per pair.")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts)
    log(f"Computing exact-match candidates for {args.split}...")
    pairs = compute_exact_match_pairs(artifacts_dir, args.split)

    out_path = Path(args.out) if args.out else artifacts_dir / f"exact_match_pairs_{args.split}.parquet"
    pairs.to_parquet(out_path, index=False)
    log(f"Wrote {len(pairs):,} exact-match pairs to {out_path}")

    if args.measure_recall_gain:
        if not args.existing_candidates:
            raise SystemExit("--measure-recall-gain needs --existing-candidates")
        log("Loading ground truth...")
        gt = pd.read_csv(args.gt, sep="\t", dtype=str, keep_default_na=False)
        match_lists = gt["matched_entity_ids"].map(parse_id_list)
        gt_pairs = gt.assign(candidate_entity_id=match_lists).explode("candidate_entity_id")
        gt_pairs = gt_pairs.loc[
            gt_pairs["candidate_entity_id"].notna() & (gt_pairs["candidate_entity_id"] != ""),
            ["source1_entity_id", "candidate_entity_id"],
        ]
        n_true = len(gt_pairs)
        log(f"  {n_true:,} ground-truth pairs total")

        log("Loading existing candidates (flat parquet, no explode needed)...")
        existing_pairs = pd.read_parquet(
            args.existing_candidates, columns=["source1_entity_id", "candidate_entity_id"]
        )

        # Integer-encoded comparison (same fix as blocking.py's measure_recall — a string
        # merge/set at this row count crashed before).
        import numpy as np

        def hit_rate(have_pairs: pd.DataFrame, label: str) -> float:
            all_s1 = pd.concat([have_pairs["source1_entity_id"], gt_pairs["source1_entity_id"]], ignore_index=True)
            s1_codes, _ = pd.factorize(all_s1)
            all_cand = pd.concat([have_pairs["candidate_entity_id"], gt_pairs["candidate_entity_id"]], ignore_index=True)
            cand_codes, cand_uniques = pd.factorize(all_cand)
            n_cand = len(cand_uniques)
            n_have = len(have_pairs)
            have_key = s1_codes[:n_have].astype(np.int64) * n_cand + cand_codes[:n_have].astype(np.int64)
            gt_key = s1_codes[n_have:].astype(np.int64) * n_cand + cand_codes[n_have:].astype(np.int64)
            have_key_sorted = np.sort(np.unique(have_key))
            n_hit = int(np.isin(gt_key, have_key_sorted).sum())
            recall = n_hit / n_true if n_true else 0.0
            print(f"{label}: {n_hit:,} / {n_true:,} = {recall:.4%}")
            return recall

        r_before = hit_rate(existing_pairs, "Existing blocking only")
        combined_pairs = pd.concat([existing_pairs, pairs], ignore_index=True)
        r_after = hit_rate(combined_pairs, "Existing + exact-match")
        print(f"\nRecall gain from exact-match pass: +{(r_after - r_before):.4%}")


if __name__ == "__main__":
    main()
