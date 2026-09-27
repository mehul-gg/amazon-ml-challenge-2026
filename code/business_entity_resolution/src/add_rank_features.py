"""Stage 3b — Rank/context features, added as a post-process on top of an existing
features_{split}.parquet from features.py.

The original plan flagged "context/rank features" (a candidate's rank within its S1's
shortlist, the reverse rank, and the score gap to the best candidate) as "usually the
strongest feature group" -- but they were never implemented. Every feature features.py
computes is local to one (S1, candidate) pair; it has no way to express "is this the BEST
candidate for this S1, or one of five mediocre ones?", which is exactly the kind of signal
that lets a classifier separate a true match from a plausible-but-wrong one when several
candidates all look similar in isolation.

This is a separate pass (not folded into features.py) because it is a genuinely different
kind of computation -- a global groupby/rank over the whole candidate set, not a per-pair
function -- and because it's much cheaper to add on top of an already-computed features
file than to redo the (already slow) rapidfuzz pass from scratch.

Memory approach: computing the rank arrays only needs 2 ID columns + 3 small float/bool
columns, not the full ~28-column feature matrix. ID columns are factorized to int32 codes
and immediately dropped (each raw string column costs several GB at 35-45M rows -- this
project has hit MemoryError from exactly that pattern before, see experiments.md), and the
rank computation itself is done with numpy array tricks (bincount, searchsorted) instead of
pandas groupby, which stays well under a GB even at full scale. The final rewrite -- adding
the new columns onto the original file -- is streamed batch-by-batch via
pyarrow.iter_batches, same discipline as features.py's ParquetWriter streaming, so the full
string-heavy original table is never held in memory at once either.

Usage:
  python src/add_rank_features.py --features ../../artifacts/features_train_merged2.parquet \
      --out ../../artifacts/features_train_merged3.parquet
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _rank_within_group(group_codes: np.ndarray, score: np.ndarray, n_groups: int):
    """For each row, returns (rank_1based_best_first, group_size, best_score_in_group), all
    aligned to the ORIGINAL row order of `group_codes`/`score`.

    group_codes must be int codes in [0, n_groups); score is the value to rank by
    (higher = better/rank 1). Implemented with a single sort + numpy array tricks instead of
    pandas groupby -- bincount/searchsorted are pure-C vectorized ops with no per-group
    Python overhead, which is what let this stay fast at 44.7M rows."""
    order = np.lexsort((-score, group_codes))  # sort by group asc, then score desc
    sorted_groups = group_codes[order]
    sorted_scores = score[order]

    group_sizes_by_id = np.bincount(group_codes, minlength=n_groups)
    group_size_sorted = group_sizes_by_id[sorted_groups]

    # First index (in sorted order) where each group starts, since sorted_groups is
    # non-decreasing -- the standard searchsorted trick for run-start offsets.
    group_start_idx = np.searchsorted(sorted_groups, np.arange(n_groups))
    best_score_sorted = sorted_scores[group_start_idx[sorted_groups]]

    # Rank within group (1-based, 1=best): position minus that group's start position + 1.
    rank_sorted = (np.arange(len(sorted_groups)) - group_start_idx[sorted_groups] + 1).astype(np.int32)

    # Scatter back from sorted order to original row order.
    n = len(group_codes)
    rank = np.empty(n, dtype=np.int32)
    group_size = np.empty(n, dtype=np.int32)
    best_score = np.empty(n, dtype=np.float32)
    rank[order] = rank_sorted
    group_size[order] = group_size_sorted
    best_score[order] = best_score_sorted
    return rank, group_size, best_score


def compute_rank_arrays(features_path: Path) -> dict[str, np.ndarray]:
    log("Loading source1_entity_id and factorizing...")
    s1_raw = pd.read_parquet(features_path, columns=["source1_entity_id"])["source1_entity_id"]
    s1_codes, s1_uniques = pd.factorize(s1_raw)
    n_s1 = len(s1_uniques)
    n = len(s1_codes)
    del s1_raw, s1_uniques
    gc.collect()
    log(f"  {n:,} rows, {n_s1:,} distinct S1 entities")

    log("Loading candidate_entity_id and factorizing...")
    cand_raw = pd.read_parquet(features_path, columns=["candidate_entity_id"])["candidate_entity_id"]
    cand_codes, cand_uniques = pd.factorize(cand_raw)
    n_cand = len(cand_uniques)
    del cand_raw, cand_uniques
    gc.collect()
    log(f"  {n_cand:,} distinct candidates")

    log("Building ranking proxy score from existing similarity features...")
    sim = pd.read_parquet(
        features_path, columns=["f_name_token_set", "f_addr_token_set", "f_addr_either_empty"]
    )
    name_score = sim["f_name_token_set"].to_numpy(dtype=np.float32)
    addr_score = sim["f_addr_token_set"].to_numpy(dtype=np.float32)
    addr_missing = sim["f_addr_either_empty"].to_numpy(dtype=bool)
    del sim
    gc.collect()
    # When either address is missing/empty, addr_token_set is not meaningful evidence for
    # ranking (see features.py's f_addr_either_empty) -- fall back to name-only in that case.
    proxy = np.where(addr_missing, name_score, 0.7 * name_score + 0.3 * addr_score).astype(np.float32)
    del name_score, addr_score, addr_missing
    gc.collect()

    log("Ranking candidates within each S1's shortlist (forward direction)...")
    rank_in_s1, group_size_s1, best_score_s1 = _rank_within_group(s1_codes.astype(np.int32), proxy, n_s1)
    reverse_rank_in_s1 = (group_size_s1 - rank_in_s1 + 1).astype(np.int32)
    score_gap_to_best_in_s1 = (best_score_s1 - proxy).astype(np.float32)
    del s1_codes
    gc.collect()

    log("Ranking S1s competing for each candidate (reverse direction)...")
    rank_for_cand, group_size_cand, best_score_cand = _rank_within_group(cand_codes.astype(np.int32), proxy, n_cand)
    score_gap_to_best_for_cand = (best_score_cand - proxy).astype(np.float32)
    del cand_codes, proxy
    gc.collect()

    return {
        "f_rank_in_s1": rank_in_s1,
        "f_reverse_rank_in_s1": reverse_rank_in_s1,
        "f_group_size_s1": group_size_s1,
        "f_score_gap_to_best_in_s1": score_gap_to_best_in_s1,
        "f_rank_for_cand": rank_for_cand,
        "f_group_size_cand": group_size_cand,
        "f_score_gap_to_best_for_cand": score_gap_to_best_for_cand,
    }


def rewrite_with_rank_features(
    features_path: Path, rank_arrays: dict[str, np.ndarray], out_path: Path, batch_size: int = 2_000_000
) -> int:
    log(f"Streaming rewrite with {len(rank_arrays)} new columns -> {out_path}...")
    pf = pq.ParquetFile(features_path)
    writer = None
    n_written = 0
    try:
        for batch in pf.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch])
            n = table.num_rows
            for name, arr in rank_arrays.items():
                table = table.append_column(name, pa.array(arr[n_written : n_written + n]))
            if writer is None:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(str(out_path), table.schema)
            writer.write_table(table)
            n_written += n
            log(f"  {n_written:,} rows written")
    finally:
        if writer is not None:
            writer.close()
    return n_written


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="features_{split}.parquet from features.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    features_path = Path(args.features)
    rank_arrays = compute_rank_arrays(features_path)
    n_written = rewrite_with_rank_features(features_path, rank_arrays, Path(args.out))
    log(f"Done: {n_written:,} rows, {len(rank_arrays)} new rank/context features added")


if __name__ == "__main__":
    main()
