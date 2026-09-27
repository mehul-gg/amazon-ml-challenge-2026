# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We built a blocking-plus-classifier entity resolution pipeline: several cheap, complementary
candidate-generation passes (inverted-index token/bigram blocking, plus six exact-key
supplementary passes) feed a LightGBM classifier trained with rank/context features, followed
by a hard exclusivity constraint at decision time. Our core innovation was a disciplined,
evidence-driven loop — repeatedly sampling real ground-truth misses, categorizing the actual
noise pattern behind them, and adding the cheapest fix that targets it — which took our
out-of-fold macro F0.5 from 0.6636 to **0.8278** without ever needing embeddings or external
data.

---

## 2. Methodology

### 2.1 Problem Analysis

Day-1 EDA on the real data (not the spec's "~1M records" placeholder) found the true scale is
~5x larger than assumed (train S1 2.21M / S2 5.03M / S3 5.29M), and confirmed two assumptions
as exact facts we could treat as hard rules: **country agreement** (100% of the 7,638,365
ground-truth matched pairs share a country) and **exclusivity** (0 of those matched IDs belong
to more than one S1). Reading real matched and near-miss pairs surfaced the noise patterns that
shaped every later decision:
- Transliteration between Latin and native Indian scripts (Devanagari, Telugu, Bengali,
  Kannada, Gujarati, Malayalam) for both business names and, separately, state names in
  addresses.
- Word reordering in both names and addresses (identical token sets, different order).
- Squashed/no-space names, often from scraped web listings, sometimes with a trailing
  domain-like word ("creativesystems com").
- State name vs. two-letter abbreviation mismatches ("uttar pradesh" vs "up") and leading
  zeros on house numbers ("001297" vs "1297") — both purely cosmetic differences that still
  broke exact-string blocking.
- Genuine typos, phone numbers embedded in the name field, and literal `<NULL>` placeholder
  tokens inline in text.
- Singletons are rare (5.58% of train), and matches-per-S1 peak at 2-5 — recall on multi-way
  match sets matters more than the singleton/non-singleton call.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (with a hard, metric-aware decision layer).
**Core Innovation:** An evidence-driven blocking-improvement loop, not a single clever trick —
sample real misses, find the actual noise pattern, add the cheapest targeted fix, measure the
real recall gain, repeat. Five of our six supplementary blocking passes and a critical
normalization fix were all found this way, and together delivered a larger score improvement
than any single modeling choice (hyperparameter tuning, more training data, or the rank
features).

---

## 3. Candidate Generation (Blocking)

Candidates for each S1 entity are the union of an inverted-index pass plus six cheap
supplementary exact-key passes, all scoped within-country (safe per the 100%-agreement fact
above) and all capped by a `max_group_size` to avoid a handful of pathologically generic
strings (e.g. a business named "meridian" shared by ~1,900 unrelated real records) blowing up
the candidate count.

- **Blocking keys used:**
  1. **Token/bigram inverted index** on `name_core` — rarest-token selection per S1 (not a
     global frequency cutoff, which we found silently dropped exact matches), IDF-weighted
     scoring, plus a flat postal-code key.
  2. **Exact name match** — catches identical `name_core` strings the token index still
     dropped due to frequency capping.
  3. **Sorted-token name match** — catches pure word reordering (adjacent-token bigrams have
     zero overlap when word order differs, even if every word matches).
  4. **Squashed-name match** — strips whitespace (and one trailing domain-suffix word like
     "com") to catch names that appear run-together in one source.
  5. **Exact address match** — catches cases where the name is unrecoverable (heavy
     transliteration or typos) but the address matches exactly; addresses are far more
     specific than short business names, so this carries almost no false-positive risk.
  6. **Sorted-token address match** — the address analogue of (3): the same key function
     applied to `address_clean` instead of `name_core`.
- **Candidate pairs generated:** ~46.8M (train), ~36.5M (test) after merging all six passes
  and deduplicating.
- **How we ensured true matches were not lost:** measured *exact* recall against the real
  ground truth after every single change (never estimated from a sample once real data was
  available), starting at 56.10% (token/bigram alone) and reaching **74.61%** after all six
  passes plus two normalization fixes in `normalize.py` (state-name canonicalization —
  covering both Latin abbreviations and native-script full names in Devanagari, Telugu,
  Bengali, Kannada, Tamil and Gujarati — plus leading-zero stripping on house numbers), which
  let the existing exact-address and sorted-address passes catch far more for free without any
  new blocking key. Every supplementary pass's incremental gain was measured in isolation
  before being folded in, so we could tell which fixes were worth their recompute cost — the
  address-based fixes alone (2 blocking passes + 2 normalization fixes) accounted for more than
  half of the total recall improvement (56.10% -> 74.61%).

---

## 4. Matching Model

**Features used** (33 total):
- **Name features:** rapidfuzz ratio/partial-ratio/token-sort/token-set on `name_core`,
  Jaccard token overlap, acronym match, legal-suffix agreement, substring containment, name
  length and length difference, phone-number-in-name flag.
- **Address features:** rapidfuzz ratio/token-set on `address_clean`, postal-code
  match/conflict, house-number match/conflict, landmark-phrase flags, either-side-empty flag.
- **Rank/context features** (the strongest single group by LightGBM gain): each candidate's
  rank and score gap to the best candidate among all candidates competing for the *same S1*
  (forward direction), and — more informative in practice — each S1's rank and score gap
  among all S1s competing for the *same candidate* (reverse direction, directly relevant to
  the exclusivity rule below).

**Model type:** LightGBM (binary classification), 5-fold `GroupKFold` grouped by
`source1_entity_id` (never split a single S1's candidates across train/validation folds),
tuned to `num_leaves=63`, `min_data_in_leaf=30`, `learning_rate=0.04`. Trained on a
negative-downsampled 500,000-S1 subsample of the full candidate set (~10-11M pairs) rather
than the full ~2.2M-S1 / ~47M-pair set, after confirming empirically that doubling the sample
size (500K → 1M S1) improved OOF F0.5 by only +0.0008 — training data volume was not the
bottleneck once the feature set stabilized.

**Threshold selection method:** grid search over τ in the OOF predictions using the exact
competition macro F0.5 scorer (not AUC or pairwise F1), reported separately for singletons and
multi-match entities. **Decision layer:** after scoring, each candidate whose probability
exceeds τ is kept, then **exclusivity is enforced as a hard rule** — each S2/S3 record is
assigned to only its single highest-scoring S1 (dropping it from every other S1's list) — since
0 of 7,638,365 real ground-truth matches ever violate this.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), OOF, exact competition metric:** progressed from **0.6636** (baseline:
  token/bigram blocking only, untuned model) to **0.8278** across nine changes (see table), a
  **+24.7% relative improvement**, entirely without embeddings or external data.

  | Change | OOF macro F0.5 | Blocking recall |
  |---|---|---|
  | Baseline (token/bigram blocking, untuned) | 0.6636 | 56.10% |
  | + exact-name-match pass | 0.6762 | 57.75% |
  | + sorted-token-name pass | 0.6840 | 59.02% |
  | + rank/context features | 0.6903 | 59.02% |
  | + LightGBM hyperparameter tuning | 0.6916 | 59.02% |
  | + squashed-name (nospace) pass | 0.7059 | 61.47% |
  | + exact-address-match pass | 0.7501 | 65.72% |
  | + address normalization fix (state abbrev, leading zeros) | 0.7894 | 69.99% |
  | + sorted-address-token pass + native-script state names | **0.8278** | **74.61%** |

- **Common false positives (wrong merges):** near-generic short names ("family specialists",
  "northwind") that a naive uncapped exact-match blocking pass would have generated in bulk —
  we deliberately excluded these via `max_group_size` after confirming raising it would flood
  the candidate set with unrelated businesses that happen to share a common name, which the
  precision-weighted F0.5 metric punishes heavily.
- **Common false negatives (missed matches):** transliteration into non-Latin scripts for the
  business name itself (no cheap fix without embeddings — this remains our largest known
  unaddressed recall gap), and severe scrambled typos (e.g. "lutz rexford" vs "lutz rfexdo")
  that exact-key blocking structurally cannot catch.

---

## 6. Conclusion

Our biggest lesson was that **cheap, targeted, evidence-driven fixes beat the more
sophisticated levers we expected to matter most**: three of our four largest single-change
score improvements came from supplementary exact-key blocking passes and a data-normalization
fix, found only by repeatedly sampling and reading real misses — not from the LightGBM tuning
or additional training data we initially assumed would move the needle most. Given more time,
the clearest next lever is a multilingual embedding retriever to address the remaining
transliteration gap, which we scoped but judged infeasible within this deadline (~12.6h of CPU
compute for the full dataset on our hardware).

---

## Appendix

### A. Code Artefacts

Full, runnable code ships under `code/business_entity_resolution/` (`src/`, `README.md`,
`requirements.txt`). Pipeline stages, run in order from `code/business_entity_resolution/`:

1. `python src/normalize.py --data ../../student_resource/dataset --full` — normalizes all
   three sources for a split, caching to `artifacts/norm_{split}_source{1,2,3}.parquet`.
2. `python src/blocking.py ...` — the main token/bigram inverted-index blocking pass.
3. `python src/exact_match_candidates.py --key-mode {exact,sorted,nospace} --key-column
   {name_core,address_clean} ...` — each of the five supplementary blocking passes (run once
   per key-mode/key-column combination used).
4. `python src/merge_candidates.py --extra-modes exact sorted nospace address_clean_exact
   address_clean_sorted ...` — unions all six passes into the final candidate set.
5. `python src/features.py --extra-modes ... ...` — computes the 26 pairwise similarity
   features and (for train) attaches ground-truth labels.
6. `python src/add_rank_features.py ...` — adds the 7 rank/context features.
7. `python src/sample_features.py ...` (train only) — samples a training subset.
8. `python src/train.py --num-leaves 63 --min-data-in-leaf 30 --learning-rate 0.04 ...` —
   trains the 5-fold LightGBM ensemble, sweeps the decision threshold, saves models.
9. `python src/decide.py ...` — applies the trained ensemble to test, enforces exclusivity,
   writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
10. `python ../../student_resource/utils/validate_submission.py --check-ids` — validates the
    submission format before upload.

`metric.py` reimplements the exact competition macro F0.5 scorer and is checked against the
spec's worked example (0.714) in `tests/test_metric.py`.

### B. Additional Results

The full experiment log — every fix, every crash and its root cause, every measured recall
number, and the reasoning behind each design decision — is in `experiments.md` at the project
root. It documents the complete debugging arc (memory-scaling fixes at 40-70M-row scale on a
16GB-RAM machine, a normalization bug that corrupted Devanagari text, and the evidence-driven
process that produced each of the six blocking passes above) in far more detail than fits here.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
