"""Sample a large-but-manageable subset of S1 entities from an already-computed full
features parquet, without ever loading the full file (uses pyarrow predicate pushdown, the
same pattern as features.py's load_filtered — see its docstring for why this matters at
this project's scale).

Why this exists: training LightGBM on all 43.4M rows of features_train_full.parquet hit a
native access-violation crash inside LightGBM's C library after several memory fixes (see
experiments.md) — a sign we're at this machine's practical ceiling for a single GroupKFold
run over the full train set. Training on a large sample (several hundred thousand S1
entities, not all 2.2M) reuses the full-scale blocking/features work already done, stays
comfortably within memory territory already proven reliable, and gives a substantially
stronger model than the original 50k-S1 development sample.

Usage:
  python src/sample_features.py --features ../../artifacts/features_train_full.parquet \
      --n-s1 500000 --out ../../artifacts/features_train_sample500k.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--n-s1", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    log(f"Reading just the source1_entity_id column from {args.features}...")
    id_col = pq.read_table(args.features, columns=["source1_entity_id"]).to_pandas(
        strings_to_categorical=True
    )["source1_entity_id"]
    all_s1 = id_col.cat.categories.to_numpy()
    log(f"  {len(all_s1):,} unique S1 entities available")

    rng = np.random.default_rng(args.seed)
    n = min(args.n_s1, len(all_s1))
    sampled = rng.choice(all_s1, size=n, replace=False)
    log(f"Sampling {n:,} S1 entities (seed={args.seed})...")

    log("Reading the sampled rows via predicate pushdown (never loads the full file)...")
    table = pq.read_table(
        args.features, filters=[("source1_entity_id", "in", sampled.tolist())]
    )
    log(f"  {table.num_rows:,} rows for the sampled entities")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)
    log(f"Wrote {table.num_rows:,} rows to {out_path}")


if __name__ == "__main__":
    main()
