"""Step 0 — EDA. Answers the questions the pipeline design depends on:
sizes, singleton rate, S2 vs S3 share, exclusivity of S2/S3 IDs, country agreement.

Fully vectorized (no per-pair Python loops) and reads every file exactly once —
needed at this dataset's scale (millions of S1/S2/S3 records, several million
ground-truth pairs, ~15 GB RAM box).

Usage: python src/eda.py --data ../../student_resource/dataset
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from metric import parse_id_list


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_tsv(path: Path) -> pd.DataFrame:
    t0 = time.time()
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if "country" in df.columns:
        df["country"] = df["country"].astype("category")
    log(f"  loaded {path.name}: {len(df):,} rows in {time.time()-t0:.1f}s")
    return df


def print_source_stats(name: str, df: pd.DataFrame) -> None:
    print(f"{name}: {len(df):,} rows | columns={list(df.columns)}")
    print("  country counts:", dict(Counter(df["country"])))
    for col in ("business_name", "business_address"):
        empty = (df[col].str.strip() == "").mean()
        print(f"  {col}: mean len {df[col].str.len().mean():.1f}, empty {empty:.2%}")


def main(data_dir: Path) -> None:
    # ---- TEST: stats only, freed immediately, never held alongside train ----
    print("\n===== TEST =====")
    for s in (1, 2, 3):
        df = read_tsv(data_dir / "test" / f"test_source{s}.tsv")
        print_source_stats(f"source{s}", df)
        del df
        gc.collect()

    # ---- TRAIN: read each file exactly once, reuse for every downstream check ----
    print("\n===== TRAIN =====")
    train = data_dir / "train"
    s1 = read_tsv(train / "train_source1.tsv")
    print_source_stats("source1", s1)
    s2 = read_tsv(train / "train_source2.tsv")
    print_source_stats("source2", s2)
    s3 = read_tsv(train / "train_source3.tsv")
    print_source_stats("source3", s3)
    gt = read_tsv(train / "train_ground_truth.tsv")
    log(f"All train files loaded: s1={len(s1):,} s2={len(s2):,} s3={len(s3):,} gt={len(gt):,}")

    print("\n===== GROUND TRUTH =====")
    match_lists = gt["matched_entity_ids"].map(parse_id_list)
    sizes = Counter(match_lists.map(len))
    print(f"S1 in GT: {len(gt):,} / S1 rows: {len(s1):,}")
    print("matches per S1:", dict(sorted(sizes.items())))
    n_singleton = sizes[0]
    print(f"singleton rate: {n_singleton / len(gt):.2%}")
    log("Computed match-count distribution.")

    # Long table of (source1_entity_id, matched_entity_id) pairs — vectorized explode, not a Python loop.
    pairs = gt.assign(matched_entity_id=match_lists).explode("matched_entity_id")
    pairs = pairs.loc[pairs["matched_entity_id"].notna() & (pairs["matched_entity_id"] != ""),
                       ["source1_entity_id", "matched_entity_id"]]
    print(f"matched pairs total: {len(pairs):,}")
    is_s2 = pairs["matched_entity_id"].str.startswith("S2-")
    print(f"  from S2: {is_s2.sum():,} | from S3: {(~is_s2).sum():,}")
    del is_s2
    log("Exploded ground truth into pairs.")

    owners = pairs["matched_entity_id"].value_counts()
    print(f"IDs matched to >1 S1 (exclusivity violations): {(owners > 1).sum():,} / {len(owners):,} unique matched IDs")
    for name, df in (("S2", s2), ("S3", s3)):
        frac = df["entity_id"].isin(owners.index).mean()
        print(f"{name} records matched to some S1: {frac:.2%}")
    log("Computed exclusivity check.")

    # Single indexed frame reused for country lookup AND the sample-match printout below.
    rec = pd.concat(
        [s1[["entity_id", "business_name", "business_address", "country"]],
         s2[["entity_id", "business_name", "business_address", "country"]],
         s3[["entity_id", "business_name", "business_address", "country"]]],
        ignore_index=True,
    )
    del s2, s3
    gc.collect()
    rec = rec.drop_duplicates("entity_id").set_index("entity_id")
    log(f"Built combined record lookup with {len(rec):,} entities.")

    pairs_country_a = pairs["source1_entity_id"].map(rec["country"])
    pairs_country_b = pairs["matched_entity_id"].map(rec["country"])
    same = (pairs_country_a == pairs_country_b).mean()
    print(f"matched pairs with same country label: {same:.2%}")
    del pairs_country_a, pairs_country_b
    log("Computed country agreement.")

    s1_country = s1.set_index("entity_id")["country"]
    is_singleton = match_lists.map(len) == 0
    by_country = pd.DataFrame({"country": gt["source1_entity_id"].map(s1_country), "singleton": is_singleton})
    rate = by_country.groupby("country", observed=True)["singleton"].mean()
    print("singleton rate by country:", {k: f"{v:.2%}" for k, v in rate.items()})
    log("Computed singleton rate by country.")

    print("\n===== SAMPLE MATCHES =====")
    sample_keys = gt.loc[match_lists.map(len) > 0, "source1_entity_id"].head(15)
    gt_by_id = gt.set_index("source1_entity_id")["matched_entity_ids"]
    for k in sample_keys:
        row = rec.loc[k]
        print(f"\n{k}: {row.business_name} | {row.business_address} [{row.country}]")
        for m in sorted(parse_id_list(gt_by_id.loc[k])):
            r2 = rec.loc[m]
            print(f"   {m}: {r2.business_name} | {r2.business_address} [{r2.country}]")
    log("Done.")


if __name__ == "__main__":
    # Windows consoles default to cp1252; source data has non-ASCII (accents, mojibake). Force UTF-8 stdout.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../student_resource/dataset")
    main(Path(ap.parse_args().data))
