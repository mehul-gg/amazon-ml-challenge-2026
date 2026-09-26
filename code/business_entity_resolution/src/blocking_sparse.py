"""Stage 2 (speed variant) — same blocking algorithm as blocking.py, reimplemented with
scipy sparse matrix multiplication instead of a hand-rolled Python dict-accumulation loop.

Why this exists: blocking.py's `_score_s1_batch` is a Python-interpreter loop over dict
lookups, which measured at ~2h/country for the full train set (see experiments.md). The
IDF-weighted "sum of shared-key weights" score is exactly a sparse matrix product
(Q @ C.T, where Q is the S1-side selected-keys matrix with IDF weights and C is the
candidate-side keys matrix with 1s), which scipy computes in compiled C rather than the
Python interpreter. Same algorithm, same key-selection logic (rarest top_r_tokens + postal,
max_df safety net) — reused directly from blocking.py, not reimplemented — so results should
match blocking.py's up to floating-point/tie-breaking differences. ONLY the scoring/join
mechanism changes.

**Not yet validated against blocking.py's known-correct 56.2% recall on the 50k sample.**
Do not use for a real run until that check passes — see the __main__ self-test.

Usage (self-test against the 50k sample once resources are free):
  python src/blocking_sparse.py --validate
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from blocking import BLOCKING_COLS, build_keys, load_normalized_trimmed, write_candidate_pairs, measure_recall
from normalize import normalize_and_cache


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _build_candidate_matrix(s2_c: pd.DataFrame, s3_c: pd.DataFrame, max_df: int):
    """Returns (C, cand_ids, vocab, df_series). C is CSR (candidates x vocab), entries=1.
    Same max_df safety-net semantics as blocking.py._build_inverted_index."""
    s2_keys = build_keys(s2_c)
    s3_keys = build_keys(s3_c)
    cand_keys = pd.concat([s2_keys, s3_keys], ignore_index=True)
    del s2_keys, s3_keys
    gc.collect()

    df_counts = cand_keys.groupby("key", observed=True).size()
    extreme = set(df_counts[df_counts > max_df].index)
    if extreme:
        cand_keys = cand_keys[~cand_keys["key"].isin(extreme)]
        df_counts = df_counts[~df_counts.index.isin(extreme)]

    vocab = {k: i for i, k in enumerate(df_counts.index)}
    cand_ids = cand_keys["entity_id"].unique()
    cand_row = {cid: i for i, cid in enumerate(cand_ids)}

    row_idx = cand_keys["entity_id"].map(cand_row).to_numpy()
    col_idx = cand_keys["key"].map(vocab).to_numpy()
    data = np.ones(len(row_idx), dtype=np.float32)
    C = sp.csr_matrix((data, (row_idx, col_idx)), shape=(len(cand_ids), len(vocab)))
    return C, cand_ids, vocab, df_counts


def _score_country_sparse(
    s1_c: pd.DataFrame, C: sp.csr_matrix, cand_ids: np.ndarray, vocab: dict, df_counts: pd.Series,
    k: int, top_r_tokens: int,
) -> pd.DataFrame:
    """Same key-selection as blocking.py._score_s1_batch (rarest top_r_tokens + postal), but
    scores via a single sparse matrix product instead of a per-entity Python accumulation loop."""
    s1_keys_all = build_keys(s1_c)
    s1_keys_all["df"] = s1_keys_all["key"].map(df_counts)
    s1_keys_all = s1_keys_all.dropna(subset=["df"])
    if s1_keys_all.empty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "score"])

    is_postal = s1_keys_all["key"].str.startswith("P:")
    postal_keys = s1_keys_all[is_postal]
    word_keys = s1_keys_all[~is_postal].sort_values(["entity_id", "df"])
    word_keys = word_keys.groupby("entity_id", observed=True).head(top_r_tokens)
    selected = pd.concat([word_keys, postal_keys], ignore_index=True)
    del s1_keys_all, word_keys, postal_keys

    s1_ids = selected["entity_id"].unique()
    s1_row = {sid: i for i, sid in enumerate(s1_ids)}
    row_idx = selected["entity_id"].map(s1_row).to_numpy()
    col_idx = selected["key"].map(vocab).to_numpy()
    is_postal_arr = selected["key"].str.startswith("P:").to_numpy()
    weight = np.where(is_postal_arr, 5.0, 1.0 / selected["df"].to_numpy()).astype(np.float32)
    Q = sp.csr_matrix((weight, (row_idx, col_idx)), shape=(len(s1_ids), len(vocab)))
    del selected

    # The core operation: sparse matrix product replaces the Python dict-accumulation loop.
    scores = (Q @ C.T).tocsr()
    del Q

    results = []
    indptr, indices, data = scores.indptr, scores.indices, scores.data
    for i in range(scores.shape[0]):
        start, end = indptr[i], indptr[i + 1]
        if start == end:
            continue
        row_idx_local = indices[start:end]
        row_val = data[start:end]
        if len(row_val) > k:
            top = np.argpartition(-row_val, k - 1)[:k]
        else:
            top = np.arange(len(row_val))
        top = top[np.argsort(-row_val[top])]
        eid = s1_ids[i]
        for t in top:
            results.append((eid, cand_ids[row_idx_local[t]], row_val[t]))

    return pd.DataFrame(results, columns=["source1_entity_id", "candidate_entity_id", "score"])


def compute_candidates_sparse(
    s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame, k: int, max_df: int, top_r_tokens: int = 3,
) -> pd.DataFrame:
    countries = sorted(set(s1["country"]) | set(s2["country"]) | set(s3["country"]))
    results = []
    for country in countries:
        t0 = time.time()
        s1_c = s1[s1["country"] == country]
        if s1_c.empty:
            continue
        s2_c = s2[s2["country"] == country]
        s3_c = s3[s3["country"] == country]
        log(f"Country={country}: s1={len(s1_c):,} s2={len(s2_c):,} s3={len(s3_c):,}")
        C, cand_ids, vocab, df_counts = _build_candidate_matrix(s2_c, s3_c, max_df)
        log(f"  candidate matrix built: {C.shape}, nnz={C.nnz:,}")
        res = _score_country_sparse(s1_c, C, cand_ids, vocab, df_counts, k=k, top_r_tokens=top_r_tokens)
        log(f"  -> {len(res):,} candidate pairs in {time.time() - t0:.1f}s")
        results.append(res)
        del s1_c, s2_c, s3_c, C, cand_ids, vocab, df_counts, res
        gc.collect()

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(
        columns=["source1_entity_id", "candidate_entity_id", "score"]
    )


def _validate_against_sample(data_dir: Path, artifacts_dir: Path) -> None:
    """Self-test: run the sparse approach on the same 50k-S1 sample blocking.py already
    validated (56.24% recall, 0 zero-candidate exact-match failures — see experiments.md),
    and check it lands on the same recall. Only a same-ballpark recall check, not a
    byte-for-byte pair comparison — tie-breaking on equal scores can legitimately differ
    between the two implementations without indicating a bug."""
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    log("Loading normalized sources (from cache)...")
    s1_full = load_normalized_trimmed(data_dir, artifacts_dir, "train", 1)
    s2 = load_normalized_trimmed(data_dir, artifacts_dir, "train", 2)
    s3 = load_normalized_trimmed(data_dir, artifacts_dir, "train", 3)

    sample_ids = pd.read_csv(
        artifacts_dir / "candidate_pairs_train_sample50k.tsv", sep="\t", dtype=str, keep_default_na=False
    )["source1_entity_id"]
    s1 = s1_full[s1_full["entity_id"].isin(set(sample_ids))]
    log(f"Validating on {len(s1):,} S1 rows (same sample blocking.py used)")

    pairs = compute_candidates_sparse(s1, s2, s3, k=20, max_df=50_000, top_r_tokens=3)
    log(f"Total pairs: {len(pairs):,}")

    out_path = artifacts_dir / "candidate_pairs_train_sparse_validation.tsv"
    write_candidate_pairs(pairs, s1["entity_id"], out_path)
    measure_recall(pairs, data_dir / "train" / "train_ground_truth.tsv", s1["entity_id"])
    print("\nCompare this recall to blocking.py's known 56.24% on the same sample "
          "(experiments.md) — should match closely. If it doesn't, there's a real bug in "
          "this rewrite and it should NOT be used for a real run.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../student_resource/dataset")
    ap.add_argument("--artifacts", default="../../artifacts")
    ap.add_argument("--validate", action="store_true", help="Run the self-test against the known 50k sample.")
    args = ap.parse_args()
    if args.validate:
        _validate_against_sample(Path(args.data), Path(args.artifacts))
    else:
        print("Run with --validate first — this module is not yet confirmed correct.")
