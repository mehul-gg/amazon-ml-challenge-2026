"""Stage 3b — Train LightGBM on pairwise features, get out-of-fold probabilities,
apply exclusivity, and sweep the decision threshold against the EXACT competition
metric (macro F0.5, from metric.py) — not a proxy like AUC or pairwise F1.

GroupKFold by source1_entity_id: every pair for one S1 lands in the same fold, so
no S1's pairs leak between train and validation.

Exclusivity (confirmed as a hard, zero-violation rule on 7.6M ground-truth pairs —
see experiments.md): each candidate goes to its single best-scoring S1 only, applied
once on raw OOF probability, before threshold selection.

Usage:
  python src/train.py --features ../../artifacts/features_train_sample50k.parquet \
      --gt ../../student_resource/dataset/train/train_ground_truth.tsv \
      --out-oof ../../artifacts/oof_train_sample50k.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from metric import parse_id_list, score_breakdown

FEATURE_PREFIX = "f_"
LGB_PARAMS = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.05,
    num_leaves=31,
    min_data_in_leaf=50,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=5,
    verbosity=-1,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def train_oof(df: pd.DataFrame, feat_cols: list, n_splits: int = 5, seed: int = 0) -> tuple:
    """Returns (oof, importances, models) — `models` is the list of per-fold LightGBM
    boosters. These are what decide.py needs for inference on test data: OOF predictions
    alone (what this function used to return) are only useful for evaluating train, since
    every train pair already has a fold assignment. Test pairs have none, so predicting on
    them means calling .predict() on saved models directly, averaged across folds (a
    standard bagging ensemble — every fold's model saw ~80% of train, none of it saw any
    test data, so there's no leakage risk in using all 5)."""
    X = df[feat_cols].astype(np.float32)
    y = df["label"].to_numpy()
    groups = df["source1_entity_id"].to_numpy()

    oof = np.zeros(len(df), dtype=np.float64)
    importances = np.zeros(len(feat_cols), dtype=np.float64)
    models = []
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        log(f"Fold {fold + 1}/{n_splits}: train={len(tr_idx):,} val={len(va_idx):,} "
            f"(val positives={y[va_idx].sum():,})")
        train_set = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx])
        val_set = lgb.Dataset(X.iloc[va_idx], label=y[va_idx], reference=train_set)
        params = dict(LGB_PARAMS, seed=seed + fold)
        model = lgb.train(
            params, train_set, num_boost_round=500,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )
        oof[va_idx] = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
        importances += model.feature_importance(importance_type="gain")
        models.append(model)
    return oof, importances / n_splits, models


def save_models(models: list, feat_cols: list, best_threshold: float, model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    for i, model in enumerate(models):
        model.save_model(str(model_dir / f"fold_{i}.txt"), num_iteration=model.best_iteration)
    meta = {"feature_cols": feat_cols, "best_threshold": best_threshold, "n_folds": len(models)}
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"Saved {len(models)} fold models + meta.json to {model_dir}")


def apply_exclusivity(df: pd.DataFrame, prob_col: str = "oof_prob") -> pd.DataFrame:
    """Each candidate goes to its single best-scoring S1 only (confirmed hard rule —
    0/7,638,365 violations in the real ground truth, see experiments.md)."""
    idx = df.groupby("candidate_entity_id")[prob_col].idxmax()
    out = df.copy()
    keep = out.index.isin(set(idx))
    out.loc[~keep, prob_col] = -1.0  # excluded candidates can never pass any positive threshold
    return out


def predictions_at_threshold(df: pd.DataFrame, all_s1_ids: list, threshold: float, prob_col: str = "oof_prob") -> dict:
    pred = {s1: set() for s1 in all_s1_ids}
    passed = df[df[prob_col] > threshold]
    for s1, group in passed.groupby("source1_entity_id", observed=True):
        pred[s1] = set(group["candidate_entity_id"])
    return pred


def load_ground_truth_scoped(gt_path: Path, scope: set) -> dict:
    """Ground truth restricted to `scope`, with singletons (no match, or entirely absent
    from the GT file's own scope) defaulting to an empty set — needed for a fair macro
    F0.5: a singleton scores 1.0 for an empty prediction, so it must be represented."""
    gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    gt = gt[gt["source1_entity_id"].isin(scope)]
    truth = {row.source1_entity_id: parse_id_list(row.matched_entity_ids) for row in gt.itertuples()}
    for s1 in scope:
        truth.setdefault(s1, set())
    return truth


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out-oof", required=True)
    ap.add_argument("--model-dir", default=None, help="Where to save fold models + meta.json for decide.py.")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    log(f"Loading {args.features}...")
    df = pd.read_parquet(args.features)
    feat_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
    log(f"Loaded {len(df):,} pairs, {len(feat_cols)} features, {df['label'].sum():,} positives "
        f"({df['label'].mean():.2%})")

    oof, importances, models = train_oof(df, feat_cols, n_splits=args.folds)
    df["oof_prob"] = oof

    imp = pd.Series(importances, index=feat_cols).sort_values(ascending=False)
    print("\n===== FEATURE IMPORTANCE (gain, avg over folds) =====")
    print(imp.to_string())

    df_excl = apply_exclusivity(df)
    all_s1_ids = df["source1_entity_id"].unique().tolist()
    truth = load_ground_truth_scoped(Path(args.gt), set(all_s1_ids))
    log(f"Scoring against ground truth for {len(all_s1_ids):,} S1 entities "
        f"({sum(1 for v in truth.values() if not v):,} singletons in scope)")

    print("\n===== THRESHOLD SWEEP (OOF macro F0.5, after exclusivity) =====")
    best = None
    for thr in np.arange(0.10, 0.95, 0.05):
        pred = predictions_at_threshold(df_excl, all_s1_ids, thr)
        scores = score_breakdown(truth, pred)
        print(f"tau={thr:.2f}  overall={scores['overall']:.4f}  "
              f"singletons={scores['singletons']:.4f} (n={scores['n_singletons']})  "
              f"with_matches={scores['with_matches']:.4f} (n={scores['n_with_matches']})")
        if best is None or scores["overall"] > best[1]["overall"]:
            best = (thr, scores)

    print(f"\nBest threshold: tau={best[0]:.2f} -> overall OOF macro F0.5 = {best[1]['overall']:.4f}")
    print(f"  (blocking-recall ceiling applies: this sample's blocking recall caps the "
          f"achievable score — see experiments.md)")

    if args.model_dir:
        save_models(models, feat_cols, float(best[0]), Path(args.model_dir))

    out_path = Path(args.out_oof)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[["source1_entity_id", "candidate_entity_id", "oof_prob", "label"]].to_parquet(out_path, index=False)
    log(f"Wrote OOF predictions to {out_path}")


if __name__ == "__main__":
    main()
