"""Merge blocking.py's token/bigram candidates with any number of
exact_match_candidates.py supplementary passes (exact-name, sorted-token/reorder, ...) into
one candidate_pairs.tsv, in the exact required submission format.

Works entirely from flat parquet sources (candidate_scores_{split}.parquet plus whichever
{mode}_match_pairs_{split}.parquet files exist) — never re-explodes the comma-joined TSV,
which crashed with MemoryError at full scale more than once during this project (see
experiments.md).

Usage:
  python src/merge_candidates.py --artifacts ../../artifacts --split train \
      --extra-modes exact sorted \
      --out ../../artifacts/candidate_pairs_train_merged.tsv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="../../artifacts")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--extra-modes", nargs="*", default=["exact", "sorted"],
                     help="Which {mode}_match_pairs_{split}.parquet files to union in, on top "
                          "of blocking.py's token/bigram candidates. Default: exact + sorted "
                          "(word-order-independent, catches reordering).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts)

    log("Loading existing (token/bigram) candidates...")
    existing = pd.read_parquet(
        artifacts_dir / f"candidate_scores_{args.split}.parquet",
        columns=["source1_entity_id", "candidate_entity_id"],
    )
    log(f"  {len(existing):,} pairs")

    parts = [existing]
    for mode in args.extra_modes:
        path = artifacts_dir / f"{mode}_match_pairs_{args.split}.parquet"
        extra = pd.read_parquet(path)
        log(f"  +{mode}: {len(extra):,} pairs ({path.name})")
        parts.append(extra)

    combined = pd.concat(parts, ignore_index=True).drop_duplicates()
    del parts, existing
    log(f"  {len(combined):,} combined unique pairs")

    log("Loading full S1 scope...")
    all_s1 = pd.read_parquet(
        artifacts_dir / f"norm_{args.split}_source1.parquet", columns=["entity_id"]
    )["entity_id"]
    log(f"  {len(all_s1):,} total S1 entities")

    log("Grouping into comma-separated candidate lists...")
    grouped = combined.groupby("source1_entity_id")["candidate_entity_id"].apply(
        lambda ids: ",".join(sorted(set(ids)))
    )
    result = pd.DataFrame({"source1_entity_id": all_s1})
    result = result.merge(grouped.rename("candidate_entity_ids"), on="source1_entity_id", how="left")
    result["candidate_entity_ids"] = result["candidate_entity_ids"].fillna("")

    assert result["source1_entity_id"].is_unique
    assert len(result) == len(all_s1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, sep="\t", index=False)
    n_with = (result["candidate_entity_ids"] != "").sum()
    log(f"Wrote {len(result):,} rows to {out_path} ({n_with:,} with candidates, "
        f"{len(result) - n_with:,} with none)")


if __name__ == "__main__":
    main()
