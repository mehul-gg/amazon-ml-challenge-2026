# Amazon ML Challenge 2026 — Business Entity Resolution

Full plan: `C:\Users\sohil\.claude\plans\c-users-sohil-downloads-amazon-ml-chall-modular-galaxy.md`

## Task
For every Source 1 (S1) record, find all matching Source 2/3 records (0, 1 or many). Fields: entity_id, business_name, business_address, country.

## Hard rules (never violate)
- **No external data lookups**: no geocoding, registry APIs, web scraping or augmentation. Only the provided data plus pretrained models.
- Final model(s) must be **MIT or Apache-2.0** and **≤ 8B params**. Check the model card license before adding any model.
- Country is an open set: test contains **France** (unseen in train). Never hard-code/filter/one-hot {US, India}.
- All files are **TSV** — always `pd.read_csv(..., sep="\t", dtype=str, keep_default_na=False)`.
- Outputs in `output/`: `matching_results.tsv` (source1_entity_id, matched_entity_ids) and `candidate_pairs.tsv` (source1_entity_id, candidate_entity_ids). One row per test S1, no duplicate IDs, only S2/S3 IDs, matches ⊆ candidates.
- Run `utils/validate_submission.py` before every upload (5 uploads/day limit).

## Metric
Macro F0.5 per S1 entity (`code/business_entity_resolution/src/metric.py`). Singleton + empty prediction = 1.0; any false prediction on a singleton = 0. Favour precision.

## Layout
- `student_resource/` — organiser data + utils (read-only)
- `code/business_entity_resolution/src/` — pipeline scripts, each stage caches to `artifacts/`
- `experiments.md` — log every experiment (OOF score, blocking recall); `submissions.md` — log every upload + LB score

## Machine
Ryzen 7 5800H (16 threads), 15.4 GB RAM, RTX 3050 Ti 4 GB. Process per country, float32, chunked; never build dense N×M matrices.

## Confirmed dataset facts (Day 1 EDA, see experiments.md for full detail)
- Real scale: train S1 2.21M / S2 5.03M / S3 5.29M; test S1 1.73M / S2 4.89M / S3 5.08M. ~5x bigger than a "1M records" assumption — **never** write a per-row Python loop over pairs; always vectorize (pandas `explode`/`.map`/`merge`) or chunk. A naive Python loop over the 7.6M ground-truth pairs crashed/segfaulted; vectorized pandas did the same job in <2 min.
- Test country split: US 38% / India 47% / **France 15%** (~259k entities) — not a small edge case.
- Singleton rate is only 5.58% train-wide. Matches-per-S1 peaks at 2-5 (mode=3). Multi-match recall matters more than the singleton/non-singleton call.
- **Exclusivity confirmed exactly**: 0/7,638,365 ground-truth matched IDs belong to >1 S1. Safe to enforce as a hard constraint in the decision step.
- **Country agreement confirmed exactly**: 100% of train ground-truth matches share country. Safe (on train) to block within-country only. Not verifiable on France test (no labels) — sanity-check once predictions exist.
- ~3.3% of S2/S3 addresses are empty (never in S1) — name-only matching is routine, not rare.
- Real noise seen: transliteration (Devanagari vs Latin script, zero shared chars), typos, word reordering, literal `<NULL>` tokens inline, bracket noise around legal suffixes (`[[LLC]]`), DBA/trade names, phone numbers embedded in the name field.

## Revised blocking plan (post-EDA, see the plan file's "Revision after Day 1 EDA" section)
- **Country is now a hard filter in blocking**, not just a feature (100% agreement confirmed on train).
- **Exclusivity is a hard rule in the decision step**: each S2/S3 ID → its single best-scoring S1, no exceptions (0 violations confirmed on train).
- **Day 1 blocking = TF-IDF only**, chunked sparse top-N (`sparse_dot_topn` or chunked `rapidfuzz.process.cdist`) — never a dense `cosine_similarity` call, it won't fit in memory at ~5-10M docs. Multilingual embeddings + FAISS are a **Day 2** addition (run as a background job; dedupe identical normalized strings before embedding — ~24M strings is too slow otherwise on the 3050 Ti).
- Start candidate K at ~15/retriever (lower than a naive plan would use), re-tune after measuring recall on real data.
- LightGBM training set must be **negative-downsampled** — keep all positives + a curated multiple of hard negatives, not every candidate pair (blocking at this scale produces tens of millions of pairs).
