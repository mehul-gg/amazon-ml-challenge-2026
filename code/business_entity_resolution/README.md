# Business Entity Resolution — Amazon ML Challenge 2026

Matches business records across three noisy sources (S1 = clean reference, S2/S3 = noisy) to
find every S2/S3 record describing the same real business as each S1 entity. Full problem
context, rules and design rationale: [`../../PROJECT_OVERVIEW.md`](../../PROJECT_OVERVIEW.md).
Detailed experiment log, every bug found and fixed, and why each design decision was made:
[`../../experiments.md`](../../experiments.md).

## Status (Day 1, 25-26 Sep 2026)

Pipeline stages 0-3 are built and validated end-to-end on a 50,000-S1 sample:
**normalize → block (token/bigram-based) → pairwise features → LightGBM**, giving an
**OOF macro F0.5 of 0.6632** (τ=0.50) using the exact competition metric. A full-scale
(2.2M-row) train blocking run is in progress — see `experiments.md` for the current status
and a from-scratch account of the scaling problems hit and fixed along the way.

Not yet built: the test-set run, the decision layer's final output writer (`decide.py`),
and the Day 2 embedding retriever. See `PROJECT_OVERVIEW.md` §11 for the full timeline.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Reproduce end-to-end

Run every command from this directory (`code/business_entity_resolution/`). Each stage caches
its output to `../../artifacts/`, so re-running a stage after the first time is fast (or skips
entirely — delete the relevant cache file to force a recompute).

### 1. Sanity-check the scoring metric

```bash
python tests/test_metric.py
```
Confirms `metric.py` reproduces the spec's worked example (0.714) and the singleton scoring rules.

### 2. Explore the data

```bash
python src/eda.py --data ../../student_resource/dataset
```
Prints dataset sizes, singleton rate, exclusivity and country-agreement checks, and sample
matched pairs. Read this before touching the code — several of its findings (exclusivity is a
hard rule, country blocking is safe, matches skew 2-5 per S1 not 0-1) directly shaped the design.

### 3. Normalize a source file (smoke test)

```bash
python src/normalize.py --data ../../student_resource/dataset --full
```
Runs a smoke test on known tricky strings (Devanagari, bracket noise, embedded phone numbers,
`<NULL>` placeholders) and times a full-file normalization pass.

### 4. Block — generate candidate pairs

```bash
python src/blocking.py --data ../../student_resource/dataset --artifacts ../../artifacts \
    --split train --out ../../artifacts/candidate_pairs_train.tsv --measure-recall
```
Country-scoped inverted-index blocking (bigram + word-token + postal-code keys, IDF-weighted
scoring, each S1 restricted to its own rarest keys). `--measure-recall` only works with
`--split train` (needs ground truth). Useful flags for a quick check before a full run:
`--sample-s1 5000` (debug on a small S1 sample). **Full-scale run is slow — see the "Known
limitations" section below before running this on the full train or test set.**

For `--split test`, drop `--measure-recall` (no test ground truth exists) and point `--out` at
`../../output/candidate_pairs.tsv` for the actual submission artifact.

### 5. Compute pairwise features

```bash
python src/features.py --data ../../student_resource/dataset --artifacts ../../artifacts \
    --candidates ../../artifacts/candidate_pairs_train.tsv --split train \
    --out ../../artifacts/features_train.parquet
```
26 features per (S1, candidate) pair: rapidfuzz string similarity on name/address, postal/house
number agreement, legal-suffix agreement, token Jaccard, acronym/containment checks, and more.
Attaches the ground-truth `label` column automatically when `--split train`.

### 6. Train and evaluate

```bash
python src/train.py --features ../../artifacts/features_train.parquet \
    --gt ../../student_resource/dataset/train/train_ground_truth.tsv \
    --out-oof ../../artifacts/oof_train.parquet
```
LightGBM, 5-fold GroupKFold by `source1_entity_id`, exclusivity enforced on raw OOF probability,
then a threshold sweep against the exact macro F0.5 metric. Prints feature importances and the
full sweep table.

### 7. Generate the submission (not yet built)

`decide.py` and `run_pipeline.py` are planned next — they'll apply the trained model to test-set
features, enforce exclusivity, apply the chosen threshold, and write
`output/matching_results.tsv` + `output/candidate_pairs.tsv` in the exact required format, then
call `utils/validate_submission.py`.

## Layout

```
src/
  metric.py     macro F0.5 scorer (+ CLI to score any prediction file against ground truth)
  eda.py        Stage 0: exploratory data analysis
  normalize.py  Stage 1: text cleaning, legal-suffix isolation, postal/house-number extraction
  blocking.py   Stage 2: candidate generation (dict-based inverted index; current default)
  blocking_sparse.py  Stage 2 (experimental): same algorithm, scipy sparse-matrix scoring —
                built for speed, NOT YET VALIDATED against blocking.py's known-correct recall.
                Do not use for a real run until it passes its own --validate self-test.
  features.py   Stage 3a: pairwise similarity features
  train.py      Stage 3b: LightGBM + GroupKFold + threshold sweep against macro F0.5
tests/
  test_metric.py
```

## Known limitations (Day 1)

- **Blocking is slow at full scale and only single-threaded.** Full train (2.2M S1) takes
  ~4.3h of real compute (India ~2h, US ~2.3h), calibrated from a 50K-S1 sample. `blocking.py`
  checkpoints each country's result to `artifacts/blocking_checkpoints_{split}/`, so a crash
  or interrupted run only loses the in-progress country, not completed ones. See
  `experiments.md` for the full debugging history (several real crashes: a max_df bug that
  dropped exact-match candidates, three separate memory crashes, one caused by Windows virtual
  memory exhaustion unrelated to this code). `blocking_sparse.py` is a from-scratch rewrite of
  just the scoring step using scipy sparse matrix multiplication, expected to be meaningfully
  faster — validate it against the known 56.24% recall on the 50K sample before trusting it.
- **Recall is ~56% on the 50K-sample calibration** (single retriever — token/bigram blocking
  only). This is expected to be a Day 1 floor, not the final number: the plan's design is a
  *union* of retrievers, and the Day 2 multilingual-embedding retriever specifically targets
  what token blocking structurally can't reach (cross-script transliteration, heavy typos).
- **OOF macro F0.5 of 0.6632 was measured on the 50K-S1 sample**, not the full train set — expect
  this to change (likely improve) once trained on the full data.
- Do not run `blocking.py` (or any full-scale script here) concurrently with another
  memory-heavy process on a machine with ~16GB RAM — this caused at least one real crash
  during Day 1 development. Check free RAM/virtual memory first if in doubt.
