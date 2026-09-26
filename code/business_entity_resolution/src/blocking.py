"""Stage 2 — Blocking / candidate generation (Day 1: token-blocking, no embeddings yet).

Per the plan revision after Day 1 EDA (see CLAUDE.md / experiments.md):
- Country is a HARD filter (100% of train ground-truth matches share country).
- Day 1 blocking is cheap and CPU-only: an inverted-index ("token blocking")
  over word-pair bigrams / word tokens of the normalized core name, plus
  postal code as an exact key.
- Multilingual embeddings + FAISS (for transliteration/typo-heavy cases this
  approach misses) are a Day 2 addition, run as a separate retriever whose
  output gets unioned into candidate_pairs.tsv.

Algorithm (revised — see experiments.md "pandas-merge join too slow at scale"):
  1. Build blocking keys per record: adjacent-word bigrams ("B:tok1_tok2"),
     word tokens ("W:token"), and postal_code ("P:code"), within-country.
  2. Build the candidate (S2+S3) side into a Python dict inverted index,
     {key: [candidate_ids]}, once per country.
  3. For each S1, pick its own top-R rarest keys (by candidate-side document
     frequency — bigrams are naturally rarer than either constituent word, so
     they get picked automatically) plus postal code if present, then look
     those keys up DIRECTLY in the dict and accumulate an IDF-weighted score
     per candidate — no pandas DataFrame join.
     A first version of this step used `pandas.merge()` to join an S1-side key
     table against the candidate-side key table. That materializes the full
     matched row set as a DataFrame for every batch, which measured at ~40+
     hours extrapolated to the full 2.2M-row train set — completely
     infeasible, even though the actual *set* of (S1, candidate) pairs touched
     was already small (rarest-token selection was working correctly). Direct
     dict lookups do the same logical work without ever building a DataFrame
     for the join, and are the current approach.
  4. Keep the top-K candidates per S1 by that score.

Usage:
  python src/blocking.py --split train --data ../../student_resource/dataset --measure-recall
  python src/blocking.py --split test  --data ../../student_resource/dataset --out ../../output/candidate_pairs.tsv
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from metric import parse_id_list
from normalize import normalize_and_cache

# Columns blocking actually needs — trimmed immediately after loading so we're
# not holding the full normalized frame (raw text, all normalized fields) for
# all three sources in memory simultaneously. This was the direct cause of an
# out-of-memory crash while normalizing source3 with s1+s2 already resident
# (see experiments.md); the deeper fix is in normalize.py, this trims the
# blast radius further specifically for the blocking stage.
BLOCKING_COLS = ["entity_id", "country", "name_core", "postal_code"]

MIN_TOKEN_LEN = 2


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _adjacent_bigrams(tokens: list) -> list:
    return [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]


def build_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Return a long (entity_id, country, key) frame: one row per (record, blocking key).

    Three key types, cheapest/most-selective first in practice (the S1 rarest-token
    selection in _join_s1_batch sorts by document frequency, so bigrams — naturally far
    rarer than either constituent word — get picked automatically when available, without
    any special-casing there):
      - "B:tok1_tok2" — adjacent word-pair bigrams of the core name. Confirmed on real data
        to be dramatically more selective than single words (e.g. "shree"=25,836 and
        "traders"=44,851 occurrences in India individually, but the pair together is far
        rarer) — this is what makes the join tractable at this corpus size; unigrams alone
        made full-scale blocking run in tens of *hours* (measured), not minutes.
      - "W:tok" — single word tokens, fallback for names with <2 usable tokens.
      - "P:code" — exact postal code.

    Vectorized throughout: str.split + explode/.map, never a Python loop with per-row
    dict/Series lookups (the pattern that caused earlier crashes — see experiments.md).
    """
    base = df[["entity_id", "country", "name_core", "postal_code"]].copy()
    core_tokens = base["name_core"].str.split()

    # Word-token keys from the core name.
    words = base[["entity_id", "country"]].copy()
    words["key"] = core_tokens
    words = words.explode("key")
    words["key"] = words["key"].str.strip()
    words = words[words["key"].str.len() >= MIN_TOKEN_LEN]
    words["key"] = "W:" + words["key"]

    # Adjacent-word-pair bigram keys — see docstring; far more selective than unigrams.
    bigrams = base[["entity_id", "country"]].copy()
    bigrams["key"] = core_tokens.map(_adjacent_bigrams)
    bigrams = bigrams.explode("key")
    bigrams = bigrams.dropna(subset=["key"])
    bigrams["key"] = "B:" + bigrams["key"]

    # Postal-code exact key.
    postal = base.loc[base["postal_code"] != "", ["entity_id", "country"]].copy()
    postal["key"] = "P:" + base.loc[base["postal_code"] != "", "postal_code"]

    keys = pd.concat([words, bigrams, postal], ignore_index=True)
    return keys.drop_duplicates()


def load_normalized_trimmed(data_dir: Path, artifacts_dir: Path, split: str, source: int) -> pd.DataFrame:
    """Load only the columns blocking needs. On a cache hit this uses parquet column
    pushdown (normalize_and_cache's `columns` arg), so the full raw business_name/
    business_address text — the bulk of each file's size — is never read into memory at
    all, not loaded and then dropped (the previous version's approach)."""
    raw = data_dir / split / f"{split}_source{source}.tsv"
    cache = artifacts_dir / f"norm_{split}_source{source}.parquet"
    return normalize_and_cache(raw, cache, columns=BLOCKING_COLS)


def _build_inverted_index(s2_c: pd.DataFrame, s3_c: pd.DataFrame, max_df: int) -> tuple[dict, dict]:
    """Build the candidate-side (S2+S3) inverted index for one country, once, reused across all
    S1 rows. Returns (postings, df): postings maps key -> list of candidate entity_ids;
    df maps key -> len(postings[key]). Plain Python dicts, not a DataFrame — see module
    docstring for why (a DataFrame join here measured ~40+ hours extrapolated to full scale).

    Built via numpy sort + np.split, not `groupby(...).apply(list)`: profiling showed the
    groupby-apply version spending ~56s of a ~94s total on this step alone for just India
    (2.3M groups, each incurring pandas' per-group Python-function-call overhead even though
    the work per group — wrapping an array as a list — is trivial). Sorting once and splitting
    the resulting numpy array at group boundaries does the same job in vectorized C code.
    """
    s2_keys = build_keys(s2_c)
    s3_keys = build_keys(s3_c)
    cand_keys = pd.concat([s2_keys, s3_keys], ignore_index=True)
    del s2_keys, s3_keys
    gc.collect()

    cand_keys = cand_keys.sort_values("key", kind="mergesort")
    keys_arr = cand_keys["key"].to_numpy()
    ids_list = cand_keys["entity_id"].tolist()  # one bulk conversion, not one per group
    del cand_keys
    uniq_keys, start_idx = np.unique(keys_arr, return_index=True)
    start_idx = start_idx.tolist()
    end_idx = start_idx[1:] + [len(ids_list)]

    postings: dict = {}
    df: dict = {}
    for key, s, e in zip(uniq_keys.tolist(), start_idx, end_idx):
        n = e - s
        if n <= max_df:
            postings[key] = ids_list[s:e]  # cheap Python-list slice, no numpy/Python boundary crossing
            df[key] = n
    return postings, df


def _score_s1_batch(s1_batch: pd.DataFrame, postings: dict, df: dict, k: int, top_r_tokens: int) -> pd.DataFrame:
    """Score one batch of S1 rows against the (already-built) inverted index via direct dict
    lookups — no DataFrame join. See module docstring for the rationale and the key-selection
    fix (rarest top_r_tokens per S1, not a global cutoff — see experiments.md "max_df bug":
    a flat cutoff dropped every key, and so all candidates, for names made of moderately-common
    words, e.g. "shree traders" in India, losing even exact-name matches).
    """
    s1_keys_all = build_keys(s1_batch)
    s1_keys_all["df"] = s1_keys_all["key"].map(df)
    s1_keys_all = s1_keys_all.dropna(subset=["df"])  # keys absent from the candidate index can't match
    if s1_keys_all.empty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "score"])
    s1_keys_all["df"] = s1_keys_all["df"].astype(np.int64)

    is_postal = s1_keys_all["key"].str.startswith("P:")
    postal_keys = s1_keys_all[is_postal]
    word_keys = s1_keys_all[~is_postal].sort_values(["entity_id", "df"])
    word_keys = word_keys.groupby("entity_id", observed=True).head(top_r_tokens)
    selected = pd.concat([word_keys, postal_keys], ignore_index=True).sort_values("entity_id")
    del s1_keys_all, word_keys, postal_keys

    ent_ids = selected["entity_id"].to_numpy()
    keys_arr = selected["key"].to_numpy()
    is_postal_arr = selected["key"].str.startswith("P:").to_numpy()
    df_arr = selected["df"].to_numpy()
    n = len(ent_ids)

    # IDF-style weighting: a shared rare key (small df) counts for much more than a shared
    # common key (large df), so candidates matching on distinctive tokens outrank ones that
    # only share a generic word (postal-code keys get a flat high weight — an exact, unambiguous
    # signal). Plain shared-key counting can't distinguish these and left top-K effectively
    # arbitrary whenever a common key was the only option.
    # Safety cap: with top_r_tokens=3 word/bigram keys + 1 postal key, each allowed up to
    # max_df postings, one entity's `scores` dict could in the worst case (near-zero overlap
    # between the four keys' postings) reach ~4*max_df entries. That combination crashed a
    # full-scale run with a genuine MemoryError after ~15h of wall-clock time (see
    # experiments.md) — rare enough not to show up in 5k/50k-row samples, but real at 2.2M
    # rows. Once an entity's own accumulator hits this many candidates it already has far more
    # than the top-k it needs, so we stop merging further keys for it rather than let a single
    # pathological entity's dict grow unbounded.
    MAX_SCORES_PER_ENTITY = 100_000

    results = []
    idx = 0
    while idx < n:
        eid = ent_ids[idx]
        j = idx
        scores: dict = {}
        while j < n and ent_ids[j] == eid:
            if len(scores) < MAX_SCORES_PER_ENTITY:
                key = keys_arr[j]
                ids = postings.get(key)
                if ids:
                    w = 5.0 if is_postal_arr[j] else 1.0 / df_arr[j]
                    for cid in ids:
                        scores[cid] = scores.get(cid, 0.0) + w
            j += 1
        if scores:
            top = sorted(scores.items(), key=lambda item: -item[1])[:k]
            results.extend((eid, cid, sc) for cid, sc in top)
        idx = j

    return pd.DataFrame(results, columns=["source1_entity_id", "candidate_entity_id", "score"])


def _candidates_for_country(
    s1_c: pd.DataFrame, s2_c: pd.DataFrame, s3_c: pd.DataFrame, k: int, max_df: int, top_r_tokens: int,
    s1_chunk_size: int = 50_000,
) -> pd.DataFrame:
    """Build the inverted index once for this country, then score S1 rows in batches
    (batching here is just to cap peak memory of the per-batch results list, not a
    correctness requirement like it was for the old DataFrame-join approach)."""
    postings, df = _build_inverted_index(s2_c, s3_c, max_df)

    batches = []
    for start in range(0, len(s1_c), s1_chunk_size):
        batch = s1_c.iloc[start : start + s1_chunk_size]
        batches.append(_score_s1_batch(batch, postings, df, k=k, top_r_tokens=top_r_tokens))
        gc.collect()
    del postings, df
    gc.collect()

    return pd.concat(batches, ignore_index=True) if batches else pd.DataFrame(
        columns=["source1_entity_id", "candidate_entity_id", "score"]
    )


def compute_candidates(
    s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame, k: int, max_df: int, top_r_tokens: int = 3,
    s1_chunk_size: int = 20_000, checkpoint_dir: Path | None = None,
) -> pd.DataFrame:
    """Return columns [source1_entity_id, candidate_entity_id, score] — top-k per S1.

    Processes one country at a time (per the plan: country is a hard blocking
    filter, confirmed 100% of train matches share a country — see
    experiments.md). This is also what keeps memory bounded: building the
    candidate index across all countries at once tried to explode a 10M+-row
    frame in a single step and crashed (see experiments.md) — the fix isn't
    just "free memory sooner", it's "never materialize the cross-country
    union in the first place".

    If `checkpoint_dir` is given, each country's result is saved to (and, on a
    later run, loaded from) its own parquet file there. A full run over this
    much data takes hours; without this, any crash mid-run — and this project
    has hit several real crashes at full scale (memory, a suspended laptop
    inflating wall-clock time) — meant redoing already-finished countries
    from scratch. Delete the relevant checkpoint file to force a country to
    be recomputed.
    """
    countries = sorted(set(s1["country"]) | set(s2["country"]) | set(s3["country"]))
    results = []
    for country in countries:
        ckpt_path = checkpoint_dir / f"_country_{country}.parquet" if checkpoint_dir else None
        if ckpt_path and ckpt_path.exists():
            res = pd.read_parquet(ckpt_path)
            log(f"Country={country}: loaded {len(res):,} candidate pairs from checkpoint {ckpt_path}")
            results.append(res)
            continue

        t0 = time.time()
        s1_c = s1[s1["country"] == country]
        if s1_c.empty:
            continue
        s2_c = s2[s2["country"] == country]
        s3_c = s3[s3["country"] == country]
        log(f"Country={country}: s1={len(s1_c):,} s2={len(s2_c):,} s3={len(s3_c):,}")
        res = _candidates_for_country(
            s1_c, s2_c, s3_c, k=k, max_df=max_df, top_r_tokens=top_r_tokens, s1_chunk_size=s1_chunk_size
        )
        log(f"  -> {len(res):,} candidate pairs in {time.time()-t0:.1f}s")
        if ckpt_path:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            res.to_parquet(ckpt_path, index=False)
            log(f"  checkpointed to {ckpt_path}")
        results.append(res)
        del s1_c, s2_c, s3_c, res
        gc.collect()

    top_k = pd.concat(results, ignore_index=True) if results else pd.DataFrame(
        columns=["source1_entity_id", "candidate_entity_id", "score"]
    )
    log(f"Total candidate pairs across all countries: {len(top_k):,}")
    return top_k


def write_candidate_pairs(pairs: pd.DataFrame, all_s1_ids: pd.Series, out_path: Path) -> None:
    grouped = pairs.groupby("source1_entity_id")["candidate_entity_id"].apply(
        lambda ids: ",".join(sorted(set(ids)))
    )
    result = pd.DataFrame({"source1_entity_id": all_s1_ids})
    result = result.merge(grouped.rename("candidate_entity_ids"), on="source1_entity_id", how="left")
    result["candidate_entity_ids"] = result["candidate_entity_ids"].fillna("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, sep="\t", index=False)
    log(f"Wrote {len(result):,} rows to {out_path}")


def measure_recall(pairs: pd.DataFrame, gt_path: Path, scope_s1_ids: pd.Series) -> None:
    """Recall over the S1 entities actually queried (`scope_s1_ids`) — NOT the full ground
    truth file. In a --sample-s1 debug run only a subset of S1s were ever queried; comparing
    against the full ground truth would make nearly everything look like a miss even though
    those S1s were simply never blocked at all. This bug produced a bogus 0.06% recall on the
    first run of this function — see experiments.md."""
    gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    gt = gt[gt["source1_entity_id"].isin(set(scope_s1_ids))]
    match_lists = gt["matched_entity_ids"].map(parse_id_list)
    gt_pairs = gt.assign(candidate_entity_id=match_lists).explode("candidate_entity_id")
    gt_pairs = gt_pairs.loc[
        gt_pairs["candidate_entity_id"].notna() & (gt_pairs["candidate_entity_id"] != ""),
        ["source1_entity_id", "candidate_entity_id"],
    ]
    n_true = len(gt_pairs)
    log(f"Ground-truth matched pairs within scope ({len(scope_s1_ids):,} S1s queried): {n_true:,}")

    have = pairs[["source1_entity_id", "candidate_entity_id"]].drop_duplicates()
    hit = gt_pairs.merge(have.assign(_hit=True), on=["source1_entity_id", "candidate_entity_id"], how="left")
    n_hit = hit["_hit"].notna().sum()
    recall = n_hit / n_true if n_true else 0.0
    print(f"\n===== BLOCKING RECALL =====")
    print(f"Recall (true matches found in candidate set): {n_hit:,} / {n_true:,} = {recall:.4%}")

    per_s1_candidates = pairs.groupby("source1_entity_id").size()
    print(f"Avg candidates per S1 (of S1s with >=1 candidate): {per_s1_candidates.mean():.1f}")
    s1_with_matches = set(gt.loc[match_lists.map(len) > 0, "source1_entity_id"])
    s1_with_candidates = set(pairs["source1_entity_id"])
    n_zero = len(s1_with_matches - s1_with_candidates)
    print(f"S1s (within scope) with a true match but zero blocking candidates at all: "
          f"{n_zero:,} / {len(s1_with_matches):,}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../student_resource/dataset")
    ap.add_argument("--artifacts", default="../../artifacts")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--max-df", type=int, default=50_000,
                     help="Candidate-side safety-net cap on key document frequency (per country). Trades off "
                          "recall (too low drops all keys for common-vocabulary names, e.g. 'shree traders' in "
                          "India both exceed 20,000) against runtime (too high reintroduces expensive common-key "
                          "joins — 300,000 measured ~2.85x slower for the same sample). 50,000 is a calibrated "
                          "middle ground; see experiments.md for the recall/runtime tradeoff data.")
    ap.add_argument("--top-r-tokens", type=int, default=3,
                     help="Per S1, use only its N rarest word-token keys (plus postal code if present).")
    ap.add_argument("--s1-chunk-size", type=int, default=50_000,
                     help="Score S1 rows against the candidate index in batches of this size, just to "
                          "cap peak memory of the accumulated results list (not a join-size safety net "
                          "anymore — scoring is direct dict lookups now, see module docstring).")
    ap.add_argument("--measure-recall", action="store_true", help="Only valid for --split train.")
    ap.add_argument("--out", default=None, help="Where to write candidate_pairs.tsv.")
    ap.add_argument("--sample-s1", type=int, default=None, help="Debug: only block this many S1 rows.")
    args = ap.parse_args()

    data_dir = Path(args.data)
    artifacts_dir = Path(args.artifacts)

    log(f"Loading + normalizing {args.split} sources (cached to {artifacts_dir})...")
    s1 = load_normalized_trimmed(data_dir, artifacts_dir, args.split, 1)
    s2 = load_normalized_trimmed(data_dir, artifacts_dir, args.split, 2)
    s3 = load_normalized_trimmed(data_dir, artifacts_dir, args.split, 3)
    log(f"Loaded s1={len(s1):,} s2={len(s2):,} s3={len(s3):,}")

    if args.sample_s1:
        s1 = s1.sample(n=min(args.sample_s1, len(s1)), random_state=0)
        log(f"Debug sampling: using {len(s1):,} S1 rows")

    # Checkpointing only applies to full (non-sampled) runs — a debug --sample-s1 run is cheap
    # enough not to need it, and checkpointing it would risk a stale sampled checkpoint being
    # mistaken for a full-run one later.
    checkpoint_dir = None if args.sample_s1 else artifacts_dir / f"blocking_checkpoints_{args.split}"
    pairs = compute_candidates(
        s1, s2, s3, k=args.k, max_df=args.max_df, top_r_tokens=args.top_r_tokens,
        s1_chunk_size=args.s1_chunk_size, checkpoint_dir=checkpoint_dir,
    )

    out_path = Path(args.out) if args.out else artifacts_dir / f"candidate_pairs_{args.split}.tsv"
    write_candidate_pairs(pairs, s1["entity_id"], out_path)

    # Persist the raw scored pairs too (score/rank), not just the id-list TSV — Stage 3
    # feature engineering wants the blocking score and each candidate's rank as "context"
    # features (per the plan, usually the strongest feature group), and rank can't be
    # recovered from the official TSV format (candidate_entity_ids there is written sorted
    # alphabetically, not by rank, to match the spec's format).
    scores_path = artifacts_dir / f"candidate_scores_{args.split}.parquet"
    pairs_ranked = pairs.sort_values(["source1_entity_id", "score"], ascending=[True, False]).copy()
    pairs_ranked["rank"] = pairs_ranked.groupby("source1_entity_id").cumcount()
    pairs_ranked.to_parquet(scores_path, index=False)
    log(f"Wrote scored candidate pairs (with rank) to {scores_path}")

    if args.measure_recall:
        if args.split != "train":
            raise SystemExit("--measure-recall requires --split train (needs ground truth).")
        measure_recall(pairs, data_dir / "train" / "train_ground_truth.tsv", s1["entity_id"])


if __name__ == "__main__":
    main()
