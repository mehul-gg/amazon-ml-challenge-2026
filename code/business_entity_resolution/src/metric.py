"""Macro-averaged F0.5 scorer, matching the Amazon ML Challenge 2026 evaluation.

F0.5 is computed per Source 1 entity and averaged over all Source 1 entities.
  - no true matches, empty prediction      -> 1.0
  - no true matches, any prediction        -> 0.0
  - has true matches, empty prediction     -> 0.0
"""
from __future__ import annotations

import pandas as pd

BETA = 0.5


def parse_id_list(value) -> set[str]:
    """Turn a comma-separated ID cell (possibly NaN/empty) into a set of IDs."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    return {x.strip() for x in str(value).split(",") if x.strip()}


def entity_fbeta(true: set[str], pred: set[str], beta: float = BETA) -> float:
    """F-beta for a single Source 1 entity, including the singleton rules."""
    if not true:
        return 1.0 if not pred else 0.0
    tp = len(true & pred)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(true)
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def macro_fbeta(truth: dict[str, set[str]], pred: dict[str, set[str]], beta: float = BETA) -> float:
    """Average per-entity F-beta over every Source 1 entity present in `truth`."""
    if not truth:
        return 0.0
    return sum(entity_fbeta(t, pred.get(s1, set()), beta) for s1, t in truth.items()) / len(truth)


def score_breakdown(truth: dict[str, set[str]], pred: dict[str, set[str]]) -> dict[str, float]:
    """Overall score plus the split between singletons and entities with matches."""
    single = {k: v for k, v in truth.items() if not v}
    multi = {k: v for k, v in truth.items() if v}
    return {
        "overall": macro_fbeta(truth, pred),
        "singletons": macro_fbeta(single, pred),
        "with_matches": macro_fbeta(multi, pred),
        "n_singletons": len(single),
        "n_with_matches": len(multi),
    }


def load_mapping(path: str, id_col: str = "source1_entity_id", list_col: str | None = None) -> dict[str, set[str]]:
    """Load a ground-truth or submission TSV into {s1_id: set(matched_ids)}."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if list_col is None:
        list_col = [c for c in df.columns if c != id_col][0]
    return {row[id_col]: parse_id_list(row[list_col]) for _, row in df.iterrows()}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Score a matching_results.tsv against ground truth.")
    ap.add_argument("--truth", required=True)
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()
    truth = load_mapping(args.truth)
    pred = load_mapping(args.pred)
    for k, v in score_breakdown(truth, pred).items():
        print(f"{k:>15}: {v:.4f}" if isinstance(v, float) else f"{k:>15}: {v}")
