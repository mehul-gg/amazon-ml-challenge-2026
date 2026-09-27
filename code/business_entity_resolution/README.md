# Business Entity Resolution — Amazon ML Challenge 2026

Matches business records across three noisy sources (S1 = clean reference, S2/S3 = noisy) to
find every S2/S3 record describing the same real business as each S1 entity. Full problem
context, rules and design rationale: [`../../PROJECT_OVERVIEW.md`](../../PROJECT_OVERVIEW.md).
Detailed experiment log — every bug found and fixed, every recall measurement, and why each
design decision was made: [`../../experiments.md`](../../experiments.md).

## Status (27 Sep 2026)

Full pipeline built and validated end-to-end at full scale (train ~2.2M S1, test ~1.7M S1):
**normalize → block (token/bigram + 5 supplementary exact-key passes) → pairwise features →
rank/context features → LightGBM → decision layer with exclusivity**. Current best:
**OOF macro F0.5 ≈ 0.79** (τ=0.60, tuned LightGBM, 500k-S1 training sample) — see
`experiments.md` for the exact current number and the full progression from the 0.6636
baseline. `output/matching_results.tsv` and `output/candidate_pairs.tsv` are generated and
pass `utils/validate_submission.py --check-ids`.

Not built (scoped and judged infeasible in the time available): a multilingual embedding
retriever for the remaining cross-script transliteration recall gap — see
`experiments.md`'s embedding-feasibility entry (~12.6h CPU compute for the full dataset).

## Setup

```bash
python -m pip install -r requirements.txt
```

## Reproduce end-to-end

Run every command from this directory (`code/business_entity_resolution/`). Each stage caches
its output to `../../artifacts/`, so re-running a stage after the first time is fast (or skips
entirely — delete the relevant cache file to force a recompute). Commands below use `train` as
the example split; repeat with `--split test` for the test set (drop `--gt`/label-dependent
flags, which don't apply since no test ground truth exists).

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
matched pairs. Its findings (exclusivity is a hard rule, country-scoped blocking is safe,
matches skew 2-5 per S1 not 0-1) directly shaped the design below.

### 3. Normalize the source files

```bash
python src/normalize.py --data ../../student_resource/dataset --full
```
Cleans names and addresses: strips accents/scripts noise (Latin only — non-Latin scripts like
Devanagari pass through untouched, deliberately), expands street abbreviations, canonicalizes
state names to their two-letter abbreviation (both Latin and native-script full names — a
hand-written domain dictionary, not an external lookup), strips leading zeros from house
numbers, isolates the legal suffix, and extracts postal code / house number / landmark flags.
Caches `norm_{split}_source{1,2,3}.parquet` to `artifacts/`.

### 4. Block — generate candidate pairs

**4a. Main pass** — country-scoped inverted-index blocking (bigram + word-token + postal-code
keys, IDF-weighted scoring, each S1 restricted to its own rarest keys):
```bash
python src/blocking.py --data ../../student_resource/dataset --artifacts ../../artifacts \
    --split train --measure-recall
```
This is the slow stage (hours at full scale) — it checkpoints each country's result to
`artifacts/blocking_checkpoints_{split}/`, so a crash only loses the in-progress country.

**4b. Five supplementary exact-key passes** — cheap (seconds to low minutes each), found by
repeatedly sampling real ground-truth misses and targeting the actual noise pattern behind
each one (see `experiments.md`):
```bash
python src/exact_match_candidates.py --artifacts ../../artifacts --split train --key-mode exact
python src/exact_match_candidates.py --artifacts ../../artifacts --split train --key-mode sorted
python src/exact_match_candidates.py --artifacts ../../artifacts --split train --key-mode nospace
python src/exact_match_candidates.py --artifacts ../../artifacts --split train --key-mode exact --key-column address_clean
python src/exact_match_candidates.py --artifacts ../../artifacts --split train --key-mode sorted --key-column address_clean
```
`--measure-recall-gain --existing-candidates <candidate_scores_{split}.parquet>` reports each
pass's own incremental recall contribution before you commit to using it.

**4c. Merge into one candidate set:**
```bash
python src/merge_candidates.py --artifacts ../../artifacts --split train \
    --out ../../artifacts/candidate_pairs_train.tsv
```
Unions the main pass with all five supplementary passes (its `--extra-modes` default) and
writes the exact submission-format TSV. For `--split test`, point `--out` at
`../../output/candidate_pairs.tsv` for the actual submission artifact.

### 5. Diagnose remaining misses (train only — no ground truth on test)

```bash
python src/diagnose_recall_misses.py --artifacts ../../artifacts --n-samples 40
```
Samples real ground-truth pairs the current candidate set still misses, with their actual
name/address text side by side. This is how every supplementary blocking pass above and the
address-normalization fix in `normalize.py` were found — read this before assuming any
particular blocking fix is worth building next.

### 6. Compute pairwise features

```bash
python src/features.py --data ../../student_resource/dataset --artifacts ../../artifacts \
    --extra-modes exact sorted nospace address_clean_exact address_clean_sorted \
    --split train --out ../../artifacts/features_train.parquet
```
26 features per (S1, candidate) pair: rapidfuzz string similarity on name/address, postal/house
number agreement, legal-suffix agreement, token Jaccard, acronym/containment checks, and more.
Attaches the ground-truth `label` column automatically when `--split train`. `--extra-modes`
builds the pairs frame directly from the flat per-mode parquet files (bypassing a comma-joined
TSV explode, which crashes at this row count) — use `--candidates <tsv>` instead only for a
smaller, sampled candidate file.

### 7. Add rank/context features

```bash
python src/add_rank_features.py --features ../../artifacts/features_train.parquet \
    --out ../../artifacts/features_train_ranked.parquet
```
Adds 7 features: each candidate's rank and score gap to the best candidate competing for the
same S1 (and, more informative in practice, the reverse — each S1's rank among all S1s
competing for the same candidate). The single strongest feature group by LightGBM gain.

### 8. (Train only) Sample for tractable training

```bash
python src/sample_features.py --features ../../artifacts/features_train_ranked.parquet \
    --n-s1 500000 --out ../../artifacts/features_train_sample500k.parquet
```
Samples N S1 entities' worth of rows via parquet predicate pushdown (never loads the full
file). Training on more than ~1M S1 hit a native crash inside LightGBM's C library on this
16GB-RAM machine; 500k S1 (~10-11M pairs) trains reliably and, empirically, doubling it to 1M
S1 only improved OOF F0.5 by +0.0008 — data volume was not the bottleneck.

### 9. Train and evaluate

```bash
python src/train.py --features ../../artifacts/features_train_sample500k.parquet \
    --gt ../../student_resource/dataset/train/train_ground_truth.tsv \
    --out-oof ../../artifacts/oof_train.parquet --model-dir ../../artifacts/models \
    --num-leaves 63 --min-data-in-leaf 30 --learning-rate 0.04
```
LightGBM, 5-fold GroupKFold by `source1_entity_id`, exclusivity enforced on raw OOF probability,
then a threshold sweep against the exact macro F0.5 metric. Prints feature importances and the
full sweep table, saves the 5 fold models + chosen threshold to `--model-dir` for `decide.py`.

### 10. Generate the submission

```bash
python src/decide.py --features ../../artifacts/features_test_ranked.parquet \
    --candidate-pairs ../../artifacts/candidate_pairs_test.tsv \
    --model-dir ../../artifacts/models --out-dir ../../output
```
Applies the trained 5-fold ensemble (averaged — standard bagging, no leakage risk since no
fold saw test data) to test-set features, enforces exclusivity, applies the tuned threshold,
and writes `output/matching_results.tsv` + copies `candidate_pairs.tsv` through unchanged.

### 11. Validate before uploading

```bash
python ../../student_resource/utils/validate_submission.py --test-dir ../../student_resource/dataset/test --check-ids
```
Must print `PASS` before every leaderboard upload (5/day limit).

## Layout

```
src/
  metric.py                  macro F0.5 scorer (+ CLI to score any prediction file)
  eda.py                     Stage 0: exploratory data analysis
  normalize.py               Stage 1: text cleaning, state-name canonicalization, suffix/
                              postal/house-number extraction
  blocking.py                Stage 2a: main candidate generation (inverted index)
  blocking_sparse.py         Stage 2a (experimental, unused): scipy sparse-matrix rewrite,
                              never validated against blocking.py's known-correct recall
  exact_match_candidates.py  Stage 2b: 5 supplementary exact-key blocking passes
  merge_candidates.py        Stage 2c: unions all blocking passes into one candidate set
  diagnose_recall_misses.py  Diagnostic: samples real misses to find the next fix
  features.py                Stage 3a: pairwise similarity features
  add_rank_features.py       Stage 3b: rank/context features (post-process, global groupby)
  sample_features.py         Stage 3c (train only): samples a tractable training subset
  train.py                   Stage 4a: LightGBM + GroupKFold + threshold sweep
  decide.py                  Stage 4b: applies trained model, exclusivity, writes submission
tests/
  test_metric.py
```

## Known limitations / hard-won lessons

- **Blocking (stage 2a) is slow at full scale and only single-threaded** — hours per split.
  Checkpoints per country to `artifacts/blocking_checkpoints_{split}/`, so a crash only loses
  the in-progress country. Every other stage runs in minutes at full scale.
- **Never run `pandas.explode()` or `np.unique()` on a >40M-row array on a ~16GB-RAM machine
  without care.** Both hit real crashes repeatedly this project (ArrayMemoryError,
  MemoryError, and once a native segfault) despite several GB of free RAM reported at the
  time — this machine's constraint is virtual-memory *commit* and fragmentation, not raw
  physical RAM. Fixes used throughout this codebase: integer-encode string ID columns before
  any set/dedup operation, prefer sort-based dedup (`.sort()` + boolean diff) over
  `np.unique()`, and pass `assume_unique=True` to `np.isin()` when the inputs are already
  deduped (it otherwise calls `np.unique()` internally regardless).
- **Training beyond ~1M S1 hit a native access-violation crash inside LightGBM's C library**
  even with a memmap-backed feature array. We stopped chasing this and train on a 500k-S1
  sample instead — confirmed empirically that more data past this point isn't the bottleneck.
- **The multilingual-embedding retriever (Day 2 plan) was never built.** Benchmarked at ~533
  texts/sec on CPU (no GPU on this machine); full-dataset embedding was estimated at ~12.6h —
  judged infeasible given the remaining time, in favor of the cheap blocking fixes above,
  which delivered larger score gains per hour of work.
