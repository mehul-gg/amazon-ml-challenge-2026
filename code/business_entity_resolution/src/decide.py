"""Stage 4 — Decision layer. Applies the trained model (an ensemble of train.py's saved
per-fold boosters — see train.py's save_models) to a feature set, enforces exclusivity,
applies the tuned threshold, and writes the two required submission files.

matching_results.tsv is the only one scored; candidate_pairs.tsv is copied through unchanged
from blocking.py's own output — it's already in the exact required format and IS the
candidate set the model scored, which is exactly what the spec asks for.

Usage:
  python src/decide.py \
      --features ../../artifacts/features_test.parquet \
      --candidate-pairs ../../artifacts/candidate_pairs_test.tsv \
      --model-dir ../../artifacts/models_full \
      --out-dir ../../output
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_features_parquet(path) -> pd.DataFrame:
    """Same fix as train.py's read_features_parquet — plain pd.read_parquet() crashed with
    a MemoryError at full scale because the id columns as plain `object` dtype means one
    Python string object per row. strings_to_categorical=True stores each unique id once."""
    return pq.read_table(path).to_pandas(strings_to_categorical=True)


def load_models(model_dir: Path) -> tuple[list, list, float]:
    meta = json.loads((model_dir / "meta.json").read_text())
    models = [lgb.Booster(model_file=str(model_dir / f"fold_{i}.txt")) for i in range(meta["n_folds"])]
    return models, meta["feature_cols"], meta["best_threshold"]


def predict_ensemble(df: pd.DataFrame, models: list, feat_cols: list) -> np.ndarray:
    """Average all fold models' predictions — standard bagging. Every fold's model was
    trained on ~80% of train and validated on the rest; none of them ever saw test data,
    so averaging all 5 for test inference carries no leakage risk (see train.py's
    train_oof docstring)."""
    X = df[feat_cols].astype(np.float32)
    preds = np.zeros(len(df), dtype=np.float64)
    for model in models:
        preds += model.predict(X, num_iteration=model.best_iteration) / len(models)
    return preds


def apply_exclusivity(df: pd.DataFrame, prob_col: str = "prob") -> pd.DataFrame:
    """Each candidate goes to its single best-scoring S1 only (confirmed hard rule —
    0/7,638,365 violations in the real ground truth, see experiments.md). Same
    implementation as train.py's apply_exclusivity, applied here to test predictions."""
    idx = df.groupby("candidate_entity_id", observed=True)[prob_col].idxmax()
    out = df.copy()
    keep = out.index.isin(set(idx))
    out.loc[~keep, prob_col] = -1.0
    return out


def write_matching_results(df: pd.DataFrame, all_s1_ids: pd.Series, threshold: float, out_path: Path,
                            prob_col: str = "prob") -> None:
    passed = df[df[prob_col] > threshold]
    grouped = passed.groupby("source1_entity_id", observed=True)["candidate_entity_id"].apply(
        lambda ids: ",".join(sorted(set(ids)))
    )
    result = pd.DataFrame({"source1_entity_id": all_s1_ids})
    result = result.merge(grouped.rename("matched_entity_ids"), on="source1_entity_id", how="left")
    result["matched_entity_ids"] = result["matched_entity_ids"].fillna("")
    # One row per test S1, no duplicates — guard the hard submission rules before writing.
    assert result["source1_entity_id"].is_unique, "duplicate source1_entity_id rows — would be rejected"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, sep="\t", index=False)
    n_matched = (result["matched_entity_ids"] != "").sum()
    log(f"Wrote {len(result):,} rows to {out_path} ({n_matched:,} with at least one match, "
        f"{len(result) - n_matched:,} predicted singletons)")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--candidate-pairs", required=True,
                     help="candidate_pairs.tsv from blocking.py — supplies the full S1 scope "
                          "(including zero-candidate S1s) and is copied through as the "
                          "required output/candidate_pairs.tsv unchanged.")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threshold", type=float, default=None,
                     help="Override the threshold saved by train.py.")
    args = ap.parse_args()

    log(f"Loading models from {args.model_dir}...")
    models, feat_cols, saved_threshold = load_models(Path(args.model_dir))
    threshold = args.threshold if args.threshold is not None else saved_threshold
    log(f"Using threshold tau={threshold:.4f} "
        f"({'override' if args.threshold is not None else 'from training'})")

    log(f"Loading features from {args.features}...")
    df = read_features_parquet(args.features)
    log(f"  {len(df):,} candidate pairs, {df['source1_entity_id'].nunique():,} S1 entities with candidates")

    df["prob"] = predict_ensemble(df, models, feat_cols)
    df = apply_exclusivity(df, prob_col="prob")

    all_s1 = pd.read_csv(args.candidate_pairs, sep="\t", dtype=str, keep_default_na=False)["source1_entity_id"]
    log(f"Full S1 scope (from candidate_pairs.tsv, includes zero-candidate S1s): {len(all_s1):,} entities")

    out_dir = Path(args.out_dir)
    write_matching_results(df, all_s1, threshold, out_dir / "matching_results.tsv")

    cand_out = out_dir / "candidate_pairs.tsv"
    shutil.copy(args.candidate_pairs, cand_out)
    log(f"Copied {args.candidate_pairs} -> {cand_out}")

    print("\nNext: run utils/validate_submission.py before uploading — see README.md.")


if __name__ == "__main__":
    main()
