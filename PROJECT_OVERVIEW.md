# Amazon ML Challenge 2026: Business Entity Resolution
### Team project overview: problem, solution, plan and workflow

> **Status (26 Sep 2026, Day 2):** the full pipeline (normalise → block → features → train) is built and validated end-to-end on a 50,000-entity sample — **OOF macro F0.5 = 0.6632** using the exact competition metric. A full-scale (2.2M-row) train blocking run is in progress; see [§0b](#0b-day-1-implementation--scaling-the-pipeline-to-real-data-size) and `experiments.md` for the detailed, honest account of the scaling problems hit and fixed along the way — several real bugs, not just tuning. Not yet built: the test-set run, the final decision/output layer, and the Day 2 embedding retriever.
> **Challenge window:** 25 Sep 2026 00:00 IST to 27 Sep 2026 23:59 IST. **5 leaderboard submissions per day** (15 total).

---

## 0. Day 1 EDA — what we actually found

The dataset is **much bigger than we assumed**, and the ground truth behaves differently than our starting assumptions in a couple of important ways. Full numbers and command are in [`experiments.md`](experiments.md); raw dump in `artifacts/eda_full_output_day1.txt`.

| | S1 | S2 | S3 |
|---|---|---|---|
| **Train** | 2,206,821 | 5,034,616 | 5,285,603 |
| **Test** | 1,732,544 | 4,887,273 | 5,082,316 |

Train countries: US 60% / India 40%. **Test adds France: US 38% / India 47% / France 15%** (~259k French S1 entities — a meaningful chunk of the leaderboard, not an edge case).

**Three findings that change how we should build this:**
1. **Singletons are rare (5.58%)**, not the "correctly predicting no-match matters a lot so expect a lot of them" mental model we started with. 94.4% of S1 entities have a real match. More importantly, **matches per S1 peaks at 2–5** (roughly bell-shaped: 0→123k, 1→119k, 2→375k, 3→531k *(mode)*, 4→484k, 5→322k, tapering to 6–11). Most of our score comes from getting **multi-way match sets** right, not from the singleton/non-singleton call alone — recall on the full match set matters more than we initially weighted it.
2. **The exclusivity assumption is confirmed perfectly**: across 7,638,365 ground-truth pairs, **zero** S2/S3 IDs are matched to more than one S1. Our plan's "assign each S2/S3 record to its single best-scoring S1 only" step is safe to enforce as a **hard constraint**, not just a soft filter — this should be a meaningful precision win essentially for free.
3. **Matched pairs always share a country label (100.00%)** in the training data. Blocking can safely search **within each country only**, cutting the candidate space by 2–3x. (We can't verify this directly for France since test has no labels, but it's reasonable to expect it holds given the data design — worth a sanity check once we have predictions.)

Other useful facts: ~73–75% of S2/S3 records have *some* true match (the rest are genuine non-matches we must correctly reject — this is where precision is won or lost); ~3.3% of S2/S3 addresses are empty (never in S1), so name-only matching is a routine path, not a rare edge case.

**Noise, confirmed from real examples** (see `experiments.md` for the full annotated list): transliteration is severe and real (`Hotel Enterprises Limited` ↔ `होटल एंटरप्राइजेज लिमिटेड`, zero shared characters — only a multilingual embedding retriever can catch this); typos (`Warwick`→`Warwik`); word reordering (`Orellana Investments LLC`→`LLC Orellana Invsmbens`); literal placeholder tokens like `<NULL>` inline in addresses; bracket noise around legal suffixes (`Obsidian, [[LLC]]`); DBA/trade names (`Korbrixx D.B.A. Obsidian, LLC`); and phone numbers embedded directly in the name field. All of this confirms the planned approach (§6–7) is the right shape — it just sharpens the normalisation rules we need to write.

---

## 0b. Day 1 implementation: scaling the pipeline to real data size

Stages 0-3 are now built and running against real data:
[`metric.py`](code/business_entity_resolution/src/metric.py),
[`eda.py`](code/business_entity_resolution/src/eda.py),
[`normalize.py`](code/business_entity_resolution/src/normalize.py),
[`blocking.py`](code/business_entity_resolution/src/blocking.py),
[`features.py`](code/business_entity_resolution/src/features.py),
[`train.py`](code/business_entity_resolution/src/train.py). Full narrative, every bug, and the
reasoning behind every design decision: [`experiments.md`](experiments.md). This section is the
short version.

**First real result**: on a 50,000-S1 sample, LightGBM with 5-fold GroupKFold, exclusivity
enforced, and a threshold swept against the exact metric gives **OOF macro F0.5 = 0.6632**
(τ=0.50; singletons 0.87, with-matches 0.65). This is capped by that sample's blocking recall
(56.2%) and by training on only ~2% of the train set — both should improve once the full-scale
run below completes and the model retrains on all of it.

**Getting there took real debugging, not just development** — worth knowing about because the
same failure modes will resurface in later stages if we're not careful:
- Normalisation initially **corrupted non-Latin script** (Devanagari) through two independent
  bugs in the accent/punctuation-stripping code — caught by a smoke test *before* it touched
  real data, not after.
- Blocking's first version used a flat frequency cutoff to drop "too common" words, which at
  this corpus size (millions of records) dropped **every key for names made of moderately
  common words** — even exact string matches got zero candidates. Fixed by giving each entity
  its own rarest-available keys instead of a global cutoff (recall 27% → 56%).
- The blocking join, first written with `pandas.merge()`, measured at **~40+ hours extrapolated
  to full scale** — rebuilt as direct Python dict lookups (~9x faster), then further optimised
  (numpy-based index construction) once profiling showed where the remaining time was going.
- A full-scale run correctly identified an actual Windows **virtual-memory exhaustion** issue
  (not a code bug) using `Get-CimInstance Win32_OperatingSystem`'s `FreeVirtualMemory`, which
  diverged sharply from `FreePhysicalMemory` — a reminder that "free RAM" alone isn't a
  reliable signal on a loaded machine.
- Multi-hour unattended runs get their wall-clock timing badly inflated by laptop sleep
  (Windows suspends processes cleanly, but `time.time()`-based elapsed time still counts the
  suspended duration as elapsed) — cross-checking `Get-Process`'s cumulative CPU time is a more
  trustworthy signal than wall-clock alone for a long background job.
- Blocking now **checkpoints each country's result to disk** as soon as it's computed, so a
  crash mid-run (which has happened — a rare per-entity memory edge case only visible at full
  scale) no longer means redoing already-finished countries from scratch.

**Current numbers** (50,000-S1 sample, will be superseded by the full run): blocking recall
56.24%, avg 19.7 candidates/S1, 0 zero-candidate exact-match failures. Full train (2.2M S1) is
estimated at ~4.3h of real compute (India ~2h, US ~2.3h) with the current single-threaded
implementation — an experimental scipy-sparse-matrix rewrite (`blocking_sparse.py`) targets a
4-6x speedup on the scoring step specifically, but is not yet validated against the known-correct
recall number, so it isn't used for a real run yet.

---

## Table of contents
0. [Day 1 EDA — what we actually found](#0-day-1-eda--what-we-actually-found)
0b. [Day 1 implementation: scaling the pipeline to real data size](#0b-day-1-implementation--scaling-the-pipeline-to-real-data-size)
1. [The problem in one paragraph](#1-the-problem-in-one-paragraph)
2. [The data](#2-the-data)
3. [What we must submit](#3-what-we-must-submit)
4. [Rules and constraints (read these)](#4-rules-and-constraints-read-these)
5. [How we are scored and why it matters](#5-how-we-are-scored-and-why-it-matters)
6. [Our solution: overview](#6-our-solution-overview)
7. [Our solution: stage by stage](#7-our-solution-stage-by-stage)
8. [Handling France (the unseen country)](#8-handling-france-the-unseen-country)
9. [Scalability and hardware](#9-scalability-and-hardware)
10. [Feasibility and risks](#10-feasibility-and-risks)
11. [3-day timeline and task split](#11-3-day-timeline-and-task-split)
12. [Repository layout and how to run](#12-repository-layout-and-how-to-run)
13. [Submission workflow (checklist)](#13-submission-workflow-checklist)
14. [Final deliverables package](#14-final-deliverables-package)
15. [Glossary](#15-glossary)
16. [FAQ](#16-faq)

---

## 1. The problem in one paragraph

We get lists of businesses from **three independent sources**. Each record has only a **name, an address and a country**, and **no IDs are shared across sources**. **Source 1 (S1)** is a clean, deduplicated reference list, so every S1 row is a different real business. **Source 2 (S2)** and **Source 3 (S3)** are noisy. For **every S1 business**, we have to find **all S2/S3 records that describe the same real business**. That can be **zero** (a "singleton"), **one** or **many** records. This task is called **Entity Resolution (ER)**, also known as record linkage.

**A made-up example:**

| Source | ID | Name | Address | Country |
|---|---|---|---|---|
| S1 | S1-00001 | Sharma Traders Private Limited | 12, MG Road, Near SBI ATM, Pune 411001 | India |
| S2 | S2-00047 | Sharma Traders Pvt Ltd | 12 M.G. Rd, Pune | India |
| S3 | S3-00812 | SHARMA TRADERS | Shop 12, Mahatma Gandhi Road, Pune-411001 | India |
| S2 | S2-00193 | Sharma Textiles Pvt Ltd | 14 MG Road, Pune | India |

The correct answer for `S1-00001` is `S2-00047,S3-00812`. `S2-00193` looks similar but is a **different business**, and matching it would be a false merge, which costs a lot under this metric.

---

## 2. The data

All files are **tab-separated (`.tsv`)**. Addresses and ID lists contain commas, so always read them with `sep="\t"`:

```python
pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
```

| File | Contents |
|---|---|
| `dataset/train/train_source1.tsv` | S1 training records (deduplicated reference) |
| `dataset/train/train_source2.tsv` | S2 training records |
| `dataset/train/train_source3.tsv` | S3 training records |
| `dataset/train/train_ground_truth.tsv` | `source1_entity_id`, `matched_entity_ids` (comma-separated, empty = no match) |
| `dataset/test/test_source{1,2,3}.tsv` | Test records with **no labels**. We predict matches for **every** test S1 |

**Columns in each source file:**
- `entity_id` has a prefix of `S1-`, `S2-` or `S3-`, which tells you the source.
- `business_name`
- `business_address`
- `country`

**Countries:** training data has **US and India**. The **test set adds France**, which has no training examples.

**Noise we should expect:**

| Field | Types of noise |
|---|---|
| Name | Abbreviations (Corp/Corporation, Pvt/Private, Ltd/Limited); legal suffix present or missing; trade/DBA names; `&` vs `and`; punctuation; word order swapped; typos; transliteration (e.g. Hindi→English spellings) |
| Address | Abbreviations (Rd/Road, St/Street); transliteration; missing parts (no PIN/ZIP, no state); landmark addresses ("Near SBI ATM"); different house-number formats; parts in a different order |

There is **no test ground truth**. We measure ourselves on a held-out part of the training data using the same metric.

---

## 3. What we must submit

### 3.1 `output/matching_results.tsv`: scored on the leaderboard
```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```
- Exactly **one row per test S1 entity**, including France.
- Leave the list **empty** when there's no match.
- IDs are comma-separated with no quotes and **no duplicates**.
- Only **S2/S3 IDs that exist in the test set**.

### 3.2 `output/candidate_pairs.tsv`: not scored, but audited
```
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812,S3-00999
```
- This is the **final candidate set our model scored**: the last filtering stage right before the ML model.
- Organisers use it to measure our **blocking recall** (how many true matches we kept) and **reduction ratio** (how many pairs we skipped).
- Every ID in `matching_results.tsv` **must also appear here**.

### 3.3 Validate before every upload
```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```
It must print **PASS**. A rejected file still uses up one of our 5 daily submissions.
Note: the spec uses both `matched_entity_id` and `matched_entity_ids` as the column name. **Trust whatever the validator accepts.**

---

## 4. Rules and constraints (read these)

| Rule | What it means for us |
|---|---|
| ❌ **No external data or lookups** | No geocoding APIs, no business registries, no web scraping, no extra datasets. Breaking this means **instant disqualification**. |
| ✅ Pretrained models are allowed | Only if **MIT or Apache-2.0 licensed** and **≤ 8 billion parameters**. **Check the model card licence before using any model.** |
| Country is an open set | Never hard-code, filter or one-hot encode `{US, India}`. France must work, and every French S1 must appear in the output. |
| One login per person | Use one laptop per participant. Simultaneous logins can terminate the session. |
| 5 submissions per day | 15 total. Log every one (see [§13](#13-submission-workflow-checklist)). |
| Keep version history | Shortlisting is based on the submitted solutions, and we may have to hand over the final code. |
| Final package | Code, outputs and the methodology document (see [§14](#14-final-deliverables-package)). |

Hand-written abbreviation lists (e.g. `rd → road`, `pvt → private`) are **our own domain rules**, not external data. We'll still disclose them in the documentation.

---

## 5. How we are scored and why it matters

### 5.1 The formula
**F0.5** is calculated **for each S1 entity separately**, then **averaged over all S1 entities** (macro average):

```
F0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

- **Precision** is the share of our predicted matches that are correct. **Recall** is the share of true matches we found.
- β = 0.5 means **precision counts twice as much as recall**. A false merge (joining two different businesses) is punished more than a missed match.

### 5.2 Special cases (these matter a lot)

| True matches | Our prediction | Score for that S1 |
|---|---|---|
| none (singleton) | empty | **1.0** ✅ |
| none (singleton) | anything | **0.0** ❌ |
| some | empty | **0.0** |
| {A, B} | {A, B, C} | P=0.67, R=1.0 → **0.714** (example from the spec) |
| {A, B, C} | {A} | P=1.0, R=0.33 → **0.714** |
| {A, B, C} | {A, B, C, X} | P=0.75, R=1.0 → **0.789** |

### 5.3 What this means for how we build the model
1. **Each S1 needs two decisions:** does it have *any* match, and if so, *which* ones. Correctly predicting "no match" is worth a full 1.0.
2. **Only predict when confident.** One wrong match on a singleton turns 1.0 into 0.
3. **Tune thresholds on the actual metric**, not on accuracy, AUC or pairwise F1. We have our own copy of the scorer in `src/metric.py`.
4. **"Exclusivity" trick:** S1 is deduplicated, so each S2/S3 record should belong to **at most one** S1 business. If an S2 record scores high against two S1s, keep only the better one. **Confirmed on Day 1 EDA: 0 of 7,638,365 ground-truth matches violate this — safe to enforce as a hard rule** (see [§0](#0-day-1-eda--what-we-actually-found)).

---

## 6. Our solution: overview

The standard, proven ER setup is: **normalise → block (generate candidates) → score pairs with ML → make per-entity decisions tuned to the metric**.

```
  S1, S2, S3 records
        │
        ▼
 ┌──────────────────┐   lowercase, strip accents, expand abbreviations,
 │ 1. Normalisation │   split legal suffixes, extract postal codes & numbers
 └──────────────────┘
        │
        ▼
 ┌──────────────────┐   for each S1: union of top-K candidates from
 │ 2. Blocking      │   TF-IDF char n-grams, multilingual embeddings (FAISS),
 │  (candidates)    │   exact keys (postal code, rare tokens)
 └──────────────────┘   → candidate_pairs.tsv   (goal: ≥98–99% recall)
        │
        ▼
 ┌──────────────────┐   ~40 features per pair: fuzzy string sims, TF-IDF &
 │ 3. Pair scoring  │   embedding cosine, number/postcode agreement, rank
 │  (LightGBM)      │   features → probability that the pair is a match
 └──────────────────┘   (+ optional fine-tuned cross-encoder on Day 3)
        │
        ▼
 ┌──────────────────┐   threshold τ, exclusivity (each S2/S3 → best S1 only),
 │ 4. Decision      │   relative filter vs best candidate; τ tuned on
 │  layer           │   out-of-fold macro F0.5
 └──────────────────┘   → matching_results.tsv
```

**Why this approach:**
- **Blocking** avoids comparing every record with every other, which is impossible at scale, and it's also something the organisers explicitly ask us to report.
- **Gradient-boosted trees on similarity features** are the strongest, fastest and most explainable baseline for ER. They train in minutes on CPU.
- **Multilingual embeddings** help with transliteration and with France.
- The **decision layer** is where we optimise the unusual precision-heavy metric directly.

---

## 7. Our solution: stage by stage

### Stage 0: EDA (exploratory data analysis). Script: `src/eda.py`
Questions that shape the design:
- How many records per source and country? *(This decides compute strategy.)*
- What share of S1s are **singletons**? How many matches does a typical S1 have?
- Do matches come mostly from S2 or from S3?
- **Is any S2/S3 ID matched to more than one S1?** *(This validates the exclusivity trick.)*
- **Do matched pairs always have the same country label?** *(If yes, we can block within each country, which is faster and more precise.)*
- A manual read of ~50 matches and ~50 near-miss non-matches per country, to learn the real noise patterns.

### Stage 1: Normalisation (country-agnostic)
- Unicode NFKD, **strip accents** (é → e, important for French), lowercase, remove punctuation, `&` → `and`.
- **Names:** keep a *full* form and a *core* form with the legal suffix removed (`pvt ltd`, `llc`, `inc`, `corp`, `co`, `sa`, `sas`, `sarl`, `gmbh`, …). Store the suffix separately as a feature.
- **Addresses:** expand abbreviations (`rd`, `st`, `ave`, `blvd`, `bd`, `av`, `marg`, `nagar`…). Extract **numbers** (house/shop numbers) and **postal codes**: US ZIP is 5 digits, Indian PIN is 6 digits, French code postal is 5 digits. Flag **landmark phrases** ("near", "opp", "behind").
- Keep both the raw and normalised versions.

### Stage 2: Blocking / candidate generation
For each S1, take the **union** of the top-K candidates from several cheap retrievers, **restricted to the same country as a hard filter** — Day 1 EDA confirmed 100% of train matches share a country, so this is now a rule, not just an optimisation.

**Revised after Day 1 EDA (see §0): staged by day, not all at once**, because a full embedding pass over ~24M strings realistically takes 2–4 hours on our GPU — too slow to gate the first submission on.

| Day | Retriever | Catches |
|---|---|---|
| **1** | TF-IDF on **character 3–5-grams of the name**, chunked sparse top-K (e.g. `sparse_dot_topn`, never a dense `cosine_similarity` — that matrix doesn't fit in memory at this scale) | typos, transliteration (partially), abbreviations |
| **1** | TF-IDF on **name + address**, same chunked approach | businesses with generic names but distinctive addresses |
| **1** | **Exact keys**: same postal code + same first name token; shared rare tokens | cheap, high-precision extra candidates |
| **2** | **Multilingual sentence embeddings** (e.g. `multilingual-e5-small`, MIT) + FAISS top-K, run as a background job, **duplicate strings deduped before embedding** | reordered words, meaning-level similarity, French text, cross-script transliteration (confirmed necessary — see the Devanagari/Latin example in §0) |

- **Goal:** keep **≥ 98–99% of true matches** (blocking recall) with a small candidate list. **K starts at ~15 per retriever** (lower than originally planned — the confirmed hard exclusivity constraint means we need less redundancy in the pool), re-tuned once measured recall is in.
- To keep features cheap at scale, a **second pass** re-ranks the union with a quick combined score and keeps the top ~15–20.
- **Output:** `candidate_pairs.tsv`.

### Stage 3: Pair features + LightGBM classifier
For every (S1, candidate) pair we compute about 40 features:

| Group | Examples |
|---|---|
| Fuzzy string similarity (`rapidfuzz`) | ratio, partial_ratio, token_sort_ratio, token_set_ratio, Jaro-Winkler, Levenshtein distance; computed on name, core name and address |
| Vector similarity | TF-IDF cosine (char and word), embedding cosine, Jaccard of tokens |
| Rare-token evidence | IDF-weighted overlap (sharing a rare word like "Zenith" beats sharing "Traders") |
| Name structure | acronym match (IBM ↔ International Business Machines), legal suffix agreement, one name contained in the other |
| Address evidence | postal code equal / different / missing; house numbers agree / conflict; missing-part flags |
| **Context / rank** (usually the strongest) | candidate's rank and score gap to the best candidate for this S1; reverse rank (how this S1 ranks among all S1s for this candidate); number of candidates; source S2 vs S3 |
| Country | only "same country?" as a yes/no flag, **never** the country name itself |

- **Model:** **LightGBM** (MIT), binary classification (match / not match).
- **Validation:** 5-fold **GroupKFold grouped by S1 ID**, so every pair for an S1 lands in the same fold and nothing leaks across folds. This gives **out-of-fold (OOF)** probabilities for every training pair.
- **Optional Day 3 improvement:** fine-tune a small multilingual **cross-encoder** (multilingual MiniLM, Apache-2.0, or `xlm-roberta-base`, MIT) on text pairs like `"name | address [SEP] name | address"`, and add its score as a LightGBM feature. For speed, run it only on uncertain pairs.

### Stage 4: Decision layer (metric-aware)
1. Keep candidates with probability **p > τ**.
2. **Exclusivity:** if one S2/S3 ID passes for several S1s, keep it only for the S1 with the highest p.
3. **Relative filter (optional):** keep a candidate only if p ≥ α × (best p for that S1). This removes weak extra matches.
4. **Grid-search τ and α** on OOF predictions using the exact macro F0.5 scorer. Report the overall score, singletons only, and with-matches only.
5. **Output:** `matching_results.tsv`, which must be a subset of the candidates.

---

## 8. Handling France (the unseen country)

We have **no French training labels**, so we have to build a model that transfers to new countries:
- **Simulate the problem:** train on **US only → evaluate on India**, and the reverse. The score drop estimates the "unseen-country penalty". We prefer features and models that stay robust in this test.
- Use **accent stripping**, **multilingual embeddings** and **generic** features (numbers, postal codes, rare-token overlap) rather than US/India-specific rules.
- Add French legal forms and address abbreviations to the normalisation lists: SA, SAS, SARL, EURL, SCI; rue, avenue/av, boulevard/bd, place/pl, chemin/ch.
- If French predictions look unreliable, use a **slightly stricter threshold** for unseen countries. Under F0.5, being cautious is the safer mistake.

---

## 9. Scalability and hardware

**Main development machine:** Ryzen 7 5800H (8 cores / 16 threads), 15.4 GB RAM, RTX 3050 Ti (4 GB VRAM), Python 3.11, 48 GB free disk.

**Key point:** we never compare every record with every other. **The real dataset is ~2.2M S1 × ~5M S2 × ~5.3M S3 in train, and ~1.7M / 4.9M / 5.1M in test** — about 5x bigger than the "1M" placeholder we planned around. Brute force (S1 × S2+S3) would be **~2×10¹³ pairs**, completely impossible. Blocking reduces the work to roughly *N × K* pairs, and at this scale it's not optional — it's the only way this runs at all on a 16 GB laptop.

We already hit this in practice on Day 1: a first version of `eda.py` that used plain Python loops over the 7.6M ground-truth pairs (`Series.get()` per pair) either hung or crashed (silent exit, then a segfault) — rewriting it with vectorized pandas (`explode`, `.map`, `.value_counts`) fixed it and dropped runtime to under 2 minutes for the full ground-truth analysis. **Every later stage (blocking, features, training) needs to follow the same rule: vectorize or chunk, never a per-row Python loop over millions of rows.**

Loading all 6 train+test files as plain UTF-8 strings with pandas took ~75 seconds total and peaked around 6-7 GB RAM — comfortably inside budget as long as we don't hold train and test in memory simultaneously, or re-read files we already have.

Rough estimates at the confirmed real scale (~5M records per noisy source) *(to be replaced with measured timings once blocking is built)*:

| Stage | Estimated time | How we keep it manageable |
|---|---|---|
| Load raw TSVs | ~75s total (measured, all 6 files) | already fine as-is |
| Normalisation | low tens of minutes | vectorised pandas, cached to parquet, one source at a time |
| TF-IDF blocking | tens of minutes to ~1-2h | one country at a time (100% of train matches stay within-country, so this is safe and cuts work 2-3x), chunks of 10–50k rows, sparse top-K |
| Embeddings | ~30-90 min on GPU (many hours on CPU — avoid) | fp16, short max length, computed once per ~10M strings and saved to disk |
| FAISS search | minutes | per-country index |
| Pair features | ~1-3h on 16 threads (with K≈20/S1 that's ~4×10⁷ pairs across S1×(S2+S3)) | two-stage blocking (cheap score → top ~20 → full features), multiprocessing, float32, chunked writes to parquet |
| LightGBM | minutes; a few GB RAM | downsample easy negatives (we now know ~25-27% of S2/S3 are true non-matches, not a rare class) for training, predict in chunks |
| Cross-encoder (optional) | only on uncertain-band pairs | small model, fp16. Free Colab/Kaggle GPU if needed |

**Memory rules for 16 GB RAM:**
- Process one country at a time.
- Use float32.
- Never build a dense N×M matrix.
- Each stage saves its output to `artifacts/`, so a crash only reruns that stage.

---

## 10. Feasibility and risks

| Aspect | Assessment |
|---|---|
| Time (3 days) | ✅ Feasible: working baseline on Day 1, strong model on Day 2, polish and packaging on Day 3 |
| Compute | ✅ Everything except the optional cross-encoder runs on a laptop CPU. The GPU speeds up embeddings |
| Licences | ✅ LightGBM, rapidfuzz, scikit-learn, FAISS: MIT/BSD. e5 and XLM-R: MIT. MiniLM: Apache-2.0. *Re-check each model card* |
| Rule compliance | ✅ No external lookups anywhere in the pipeline |

| Risk | Mitigation |
|---|---|
| France behaves differently from US/India | cross-country simulation, multilingual features, conservative threshold |
| Blocking misses true matches (caps recall) | measure recall@K on the holdout, use several retrievers together |
| Overfitting the public leaderboard (the final ranking uses the **private** one) | trust OOF scores, don't chase small public LB changes |
| Rejected submission wastes one of the 5 daily uploads | run the validator before every upload |
| Running out of memory on large data | chunking, per-country processing, float32, cached stages |

---

## 11. 3-day timeline and task split

### Day 1 (25 Sep): baseline — done
- [x] Project scaffold, rules file (`CLAUDE.md`), metric with tests, EDA script
- [x] Download the dataset, run EDA, share findings with the team (see [§0](#0-day-1-eda--what-we-actually-found))
- [x] Normalisation + token/bigram blocking, country as a hard filter (see [§0b](#0b-day-1-implementation--scaling-the-pipeline-to-real-data-size)); recall measured on a 50k sample (56.2%) — full-scale run in progress
- [x] rapidfuzz features + LightGBM + threshold, exclusivity enforced as a hard rule → **OOF macro F0.5 = 0.6632 on the 50k sample**
- [ ] Full-scale (2.2M-row) train run completing; **first leaderboard submission still pending** — needs the test-set run + `decide.py` (see below)

### Day 2 (26 Sep): strong model — in progress
- [ ] Finish the full-scale train blocking run, retrain on full data
- [ ] Run blocking on the test set, build `decide.py` → `output/matching_results.tsv`, validate, **first submission**
- [ ] Multilingual embeddings + FAISS (deduped strings, run in background) added to blocking and features
- [ ] Context/rank features, τ/α tuning on OOF
- [ ] US→India transfer test (the France proxy)
- [ ] 2–4 submissions comparing variants

### Day 3 (27 Sep): polish and deliver
- [ ] Optional cross-encoder on uncertain pairs, ensemble
- [ ] Final threshold choice (based on OOF, not public LB)
- [ ] Clean code, README (done — [`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md)), `requirements.txt` (done), fill in `Documentation_template.md`
- [ ] Build the final zip **well before 23:59 IST**

### Suggested ownership (adjust to team size)

| Area | Owner |
|---|---|
| EDA + normalisation rules (incl. French lists) | _TBD_ |
| Blocking + recall measurement | _TBD_ |
| Features + LightGBM + decision layer | _TBD_ |
| Embeddings / cross-encoder | _TBD_ |
| Submissions log, validation, documentation, packaging | _TBD_ |

**Remember:** only **one login per person** on the challenge portal, and one person should own uploads so we don't lose track of the 5-per-day limit.

---

## 12. Repository layout and how to run

```
D:\ml-challenge\
├── CLAUDE.md                     # rules & context for Claude Code sessions
├── PROJECT_OVERVIEW.md           # this document
├── experiments.md                # log of every experiment (recall, OOF F0.5)
├── submissions.md                # log of every leaderboard upload
├── student_resource\             # organisers' data + utils (do not edit)
│   ├── dataset\train\ , dataset\test\
│   └── utils\validate_submission.py
├── artifacts\                    # cached intermediate outputs (parquet/npy/models)
├── output\                       # matching_results.tsv, candidate_pairs.tsv
└── code\business_entity_resolution\
    ├── requirements.txt          # ✅ pinned to installed versions
    ├── README.md                 # ✅ exact reproduction steps
    ├── tests\test_metric.py      # ✅ metric unit tests
    └── src\
        ├── metric.py             # ✅ macro F0.5 scorer (+ CLI)
        ├── eda.py                # ✅ EDA report
        ├── normalize.py          # ✅ Stage 1: text cleaning
        ├── blocking.py           # ✅ Stage 2: candidate generation (dict-based, current default)
        ├── blocking_sparse.py    # 🧪 Stage 2 speed rewrite (scipy sparse matmul) — NOT yet
        │                         #    validated against blocking.py's known-correct recall
        ├── features.py           # ✅ Stage 3a: pairwise similarity features (26 features)
        ├── train.py              # ✅ Stage 3b: LightGBM + GroupKFold + threshold sweep
        ├── decide.py             # planned: exclusivity + threshold → matching_results.tsv
        └── run_pipeline.py       # planned: data → blocking → matching → output
```

**Setup:**
```bash
python -m pip install -r code/business_entity_resolution/requirements.txt
```

**Run the metric tests:**
```bash
python code/business_entity_resolution/tests/test_metric.py
```

**Run EDA** (from `code/business_entity_resolution/`):
```bash
python src/eda.py --data ../../student_resource/dataset
```

**Score a prediction file against the ground truth:**
```bash
python src/metric.py --truth <ground_truth.tsv> --pred <matching_results.tsv>
```

**Run the full Stage 0-3 pipeline** (from `code/business_entity_resolution/`) — see that
directory's own [README.md](code/business_entity_resolution/README.md) for full detail, flags,
and the current known limitations (full-scale blocking is slow — check that before running it
on the full dataset):
```bash
python src/blocking.py --data ../../student_resource/dataset --artifacts ../../artifacts \
    --split train --out ../../artifacts/candidate_pairs_train.tsv --measure-recall
python src/features.py --data ../../student_resource/dataset --artifacts ../../artifacts \
    --candidates ../../artifacts/candidate_pairs_train.tsv --split train \
    --out ../../artifacts/features_train.parquet
python src/train.py --features ../../artifacts/features_train.parquet \
    --gt ../../student_resource/dataset/train/train_ground_truth.tsv \
    --out-oof ../../artifacts/oof_train.parquet
```

**Development tool:** we're building with **Claude Code**. `CLAUDE.md` holds the rules so every session follows them. Each pipeline stage is a small script that you can run and check on its own.

---

## 13. Submission workflow (checklist)

Before every upload:
1. [ ] OOF macro F0.5 is logged in `experiments.md`
2. [ ] Both TSVs were regenerated from the current code
3. [ ] `validate_submission.py` prints **PASS**
4. [ ] Row count equals the number of test S1 entities (France included)
5. [ ] Upload `matching_results.tsv` to the portal
6. [ ] Log it in `submissions.md`: number, time, code version, OOF score, public LB score, notes
7. [ ] Keep a copy of the exact files (e.g. `output/history/sub_03/`)

---

## 14. Final deliverables package

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/                # all source code, commented
│       ├── README.md           # how to reproduce end-to-end
│       └── requirements.txt    # pinned versions
└── Documentation_template.md   # filled-in methodology
```

**The documentation must cover:**
- methodology;
- candidate generation / blocking strategy, including recall and reduction ratio;
- model architecture and feature engineering;
- experiments and conclusions;
- anything else relevant, such as the France handling and the licences of the models used.

The **top 100 teams** will be asked for these details again, and their packages are reviewed before the final rankings.

---

## 15. Glossary

| Term | Meaning |
|---|---|
| **Entity Resolution (ER)** | Working out which records refer to the same real-world thing |
| **Singleton** | An S1 business with no matches in S2/S3 |
| **Blocking / candidate generation** | A cheap first pass that picks a short list of plausible matches per S1, so we don't compare all pairs |
| **Blocking recall** | Share of true matches that survive blocking. It's the maximum recall the whole system can reach |
| **Reduction ratio** | Share of all possible pairs that blocking removed |
| **False merge** | Predicting that two different businesses are the same (a false positive). Heavily penalised |
| **OOF (out-of-fold)** | Predictions on training data from a model that never saw that data. Used for honest tuning |
| **GroupKFold** | Cross-validation that keeps all rows of one group (here, one S1) in the same fold |
| **TF-IDF char n-grams** | Vectorises text using overlapping character pieces ("sha", "har", "arm"…), so it tolerates typos |
| **Embeddings** | Neural-network vectors that capture meaning, so similar texts get similar vectors |
| **FAISS** | Fast nearest-neighbour search library for vectors (MIT licence) |
| **LightGBM** | Fast gradient-boosted decision tree library (MIT licence) |
| **Cross-encoder** | A transformer that reads both records together and outputs a match score. Accurate but slow |
| **F0.5** | F-score that weighs precision twice as much as recall |

---

## 16. FAQ

**Can we use ChatGPT, Google or another API to check whether two businesses are the same?**
No. That counts as an external lookup and leads to disqualification. Only the provided data plus MIT/Apache models of ≤ 8B parameters that run inside our pipeline are allowed.

**Can we use a large language model?**
Only if it's MIT or Apache-2.0 licensed and ≤ 8B parameters. It's allowed but probably not worth the compute. LightGBM plus a small cross-encoder should give most of the benefit.

**Why not just predict the best candidate for every S1?**
Every singleton would then score 0. We have to predict "no match" whenever we aren't confident.

**Should we chase the public leaderboard?**
Only moderately. The final ranking uses the **private** leaderboard. Out-of-fold validation on the training data is our main guide.

**What if France scores badly?**
We can't see French labels, so we rely on the US↔India transfer test and multilingual/generic features, and we lean conservative on French predictions.
