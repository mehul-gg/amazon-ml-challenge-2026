# Experiments log

## Plan revision decision (25 Sep, Day 1, post-EDA)
Reviewed the original plan against the EDA numbers below and revised it — full rationale in the plan file's "Revision after Day 1 EDA" section, mirrored in `CLAUDE.md` and `PROJECT_OVERVIEW.md` §0/§7/timeline. Summary: architecture unchanged; blocking reordered so Day 1 uses chunked sparse TF-IDF only (country as a hard filter) and embeddings+FAISS move to Day 2 as a background job (GPU embedding pass over ~24M strings is too slow to gate today's submission on); exclusivity is now enforced as a hard rule in the decision step (confirmed 0 violations); candidate K starts lower (~15); LightGBM training needs negative downsampling given the pair volume at this scale.

## Dataset scale (measured 25 Sep, Day 1)
- Train: S1 2,206,821 rows | S2 5,034,617 | S3 5,285,604 | ground truth 2,206,822 rows. Train dir ~1.3 GB.
- Test: S1 1,732,545 | S2 4,887,274 | S3 5,082,317. Test dir ~1.2 GB.
- Train S1 country split: US 1,323,633 (60.0%) / India 883,188 (40.0%).
- This is ~5x bigger than the "1M records" placeholder used in the original plan — blocking, chunking and per-country processing are not optional, they're required for this to run on the dev machine (15.4 GB RAM).
- EDA run took several minutes on full data (background job). Ran into two bugs at this scale, both fixed in `src/eda.py`: (1) naive Python loops over millions of ground-truth pairs caused a silent kill/segfault — rewrote with vectorized pandas (`explode`, `.map`, `.value_counts`) instead of per-row `.get()` calls; (2) re-reading train files twice (once for stats, once for GT analysis) pushed memory too far — now reads each file exactly once and reuses it; (3) Windows console is cp1252 by default and crashed printing non-ASCII sample text — forced UTF-8 stdout.

### Ground-truth findings (train, confirmed 25 Sep)
- **Singleton rate: 5.58%** (123,247 / 2,206,821 S1 entities have zero matches). Much lower than we should have assumed going in — most S1 businesses (94.4%) have at least one real match. Consistent across country: US 5.58%, India 5.59%.
- **Matches per S1 is NOT mostly 0/1** — it's concentrated at 2-5, roughly bell-shaped: {0: 123,247 | 1: 119,157 | 2: 375,212 | 3: 530,841 (mode) | 4: 484,115 | 5: 321,957 | 6: 164,868 | 7: 63,968 | 8: 18,680 | 9: 4,205 | 10: 534 | 11: 37}. Design implication: the "predict empty for singletons" case matters (5.6% of macro-F0.5 rows), but the bulk of the score comes from getting 2-5-way match sets right — recall on the FULL set matters more than we initially weighted it.
- **Exclusivity assumption CONFIRMED, perfectly**: 0 / 7,638,365 matched S2/S3 IDs are matched to more than one S1. This validates the plan's exclusivity step (assign each S2/S3 ID to its single best-scoring S1 only) — safe to enforce as a hard constraint, not just a soft filter.
- 7,638,365 total ground-truth pairs: 3,693,619 from S2, 3,944,746 from S3 (close to even).
- 73.36% of S2 records and 74.63% of S3 records have *some* S1 match; the rest (~25-27%) are true non-matches (noise/unrelated records) that must be correctly excluded — this is where precision is won or lost.
- **Country agreement: 100.00%** of matched pairs share the same country label. Blocking can safely restrict candidate search to the same country — this cuts the candidate search space by ~2-3x for free and is a strong precision signal. (Caveat: this is measured on train, which only has US/India; assuming it also holds between France test records and their true source — reasonable given the data design, but keep an eye on it since we can't verify directly on test.)
- business_address is empty for ~3.3% of S2/S3 records (never empty in S1) — normalization must handle missing addresses gracefully, matching on name alone in that case.

### Noise patterns observed in real sample matches (see `artifacts/eda_full_output_day1.txt` for the full 15-example dump)
- **Transliteration is real and severe**, not just theoretical: `Hotel Enterprises Limited` ↔ `होटल एंटरप्राइजेज लिमिटेड` (Devanagari) — same business, same address, no shared characters at all. Character n-gram / edit-distance features are useless here; we need the multilingual embedding retriever in blocking or these are simply unrecoverable.
- **Word-level corruption**: `Maure Williams` → `Maure Wilblims`, `Crystal Staffing` → `Crystal Sttfrifng`, `Warwick` → `Warwik` — classic typo/OCR-style noise, well suited to fuzzy ratio + Levenshtein.
- **Word reordering**: `Orellana Investments LLC` → `LLC Orellana Invsmbens`; `Crystal Staffing Solutions LLC` → `LLC Crystal Sttfrifng Solutions` — token_sort_ratio / token_set_ratio needed, plain `ratio()` will underscore these.
- **Junk/placeholder tokens in address**: literal `<NULL>` string appears in-line in an address (`939389287`). Normalization should strip known placeholder tokens (`<NULL>`, `N/A`, `NONE`, `-`) rather than treating them as content.
- **Bracket/marker noise around legal suffix**: `Obsidian, [[LLC]]`, `[Corp] Dick Regional Armada` — suffix detection regex needs to tolerate surrounding brackets/punctuation.
- **DBA (trade name) records**: `Korbrixx D.B.A. Obsidian, LLC` matches `Obsidian, LLC` — the DBA prefix needs to be either stripped or treated as a separate high-value feature ("does one name contain the other after suffix stripping").
- **Phone numbers embedded directly in the name field**: `Chordia + Pagnters - 7306204978` — names aren't clean; strip trailing numeric/phone-like tokens before name comparison, but don't discard them as a feature (a shared phone number substring would be a very strong signal if it recurs).
- **Missing address is common on the noisy sources**: several S2/S3 examples above have a blank address and only the name to go on (consistent with the 3.3% empty-address stat) — name-only matching path is not a rare edge case, it's routine.
- Confirms the plan's Stage 1/2 design (accent/script-aware embeddings, abbreviation expansion, suffix isolation, token-set fuzzy matching) is the right shape; adds concrete cleanup rules to implement: strip placeholder tokens, tolerate bracket noise around suffixes, strip/feature-ize embedded phone numbers.

## Blocking (src/blocking.py) — final Day 1 design and results (25 Sep, Day 1)

**Final approach**: country-scoped (hard filter) inverted-index token blocking, built as plain
Python dicts (`{key: [candidate_ids]}`), not a pandas DataFrame join. Three key types per
record: adjacent-word bigrams (`B:tok1_tok2`, most selective — naturally much rarer than either
constituent word), single word tokens (`W:tok`, fallback), postal code (`P:code`, exact). Each S1
uses only its own top-3 rarest keys (by candidate-side document frequency) plus postal code if
present — never every token — so every record contributes a small, bounded, maximally-selective
key set regardless of corpus size. Candidates are scored by an IDF-weighted sum (`1/df` per shared
key, postal code weighted at a flat 5.0) and the top-20 per S1 are kept.

**Debugging arc (each of these was a real bug confirmed on real data, not just tuning)**:
1. A flat `max_df=2000` cutoff dropped **every** key for names built from moderately-common
   words (`shree traders`, `federal united group`, `bay holdings group`) — even exact string
   matches got zero candidates. Root cause: at 5-10M-record scale, even fairly distinctive words
   ("twisted"=8,783 occurrences, "packaging"=6,402) blow past any low absolute cutoff; 62.5% of
   all word-key postings mass sat in just 867 of ~1M unique keys. Fixed by switching from "drop
   globally common keys" to "each S1 keeps its own rarest available keys" — recall 27%→39%.
2. Loosening `max_df` to fix that broke runtime: total join work scales with `sum(df)` over
   every key actually used, and moderately-common keys are still expensive in aggregate even
   under per-record selection. A `pandas.merge()`-based join of the full candidate index against
   even one small (~2,031-row) S1 batch produced a 77.8M-row intermediate and crashed on memory.
3. Added S1-side batching (bounds peak memory) and word-bigram keys (naturally far rarer than
   unigrams, so preferred automatically by the existing rarest-key selection) — recall →56.5%,
   zero-candidate exact-match failures →0.
4. Even with bigrams, the `pandas.merge()` architecture itself measured **~40+ hours extrapolated
   to the full 2.2M-row train set** — completely infeasible, because a DataFrame join
   materializes the full matched-row set for every batch regardless of how selective the
   individual keys are. **Fix: replaced the join with direct Python dict lookups** — build
   `{key: [candidate_ids]}` once per country, look up each S1's 2-4 selected keys directly,
   accumulate IDF-weighted scores in a plain dict. This is the same logical computation with
   none of the DataFrame materialization overhead.
5. Profiling the dict-index build itself found `groupby(...).apply(list)` spending ~56 of ~94s
   on just India due to pandas' per-group Python-call overhead across 2.3M groups — replaced
   with numpy `sort_values` + `np.unique(..., return_index=True)` + one bulk `.tolist()` call
   (not one per group) to slice group boundaries out of a single Python list.

**Measured results (train, India+US, 50,000-S1 sample — consistent with a 5,000-S1 sample too,
so numbers are stable, not sampling noise)**:
- Recall: 56.24% (97,217 / 172,874 ground-truth pairs found)
- Zero-candidate S1s (had a true match, got no candidates at all): 0 / 47,139
- Avg candidates per S1: 19.7 (near the K=20 cap)
- Runtime: India 254.6s / 20,004 rows, US 297.4s / 29,996 rows → **full train extrapolates to
  ~4.4 hours** (down from ~40+ hours with the pandas-merge version — ~9x improvement)

**This is a Day 1, single-retriever number, not the final target.** The plan's design is a union
of retrievers; Day 2's embedding+FAISS retriever is specifically meant to catch what token
blocking structurally cannot (transliteration — confirmed necessary: `होटल एंटरप्राइजेज लिमिटेड`
shares zero characters with its Latin-script match — and heavy typos). 56% recall from the cheap
first pass alone is a reasonable Day 1 checkpoint.

**Full-train run**: first launch (~15:56) crashed with `MemoryError` ~2 min in — caused by running
`features.py`/`train.py` concurrently with it (15:58-16:03) for Stage 3 development, exactly the
resource-contention risk flagged earlier but then triggered anyway to make parallel progress.
That trade paid off (got the validated 0.6632 OOF result on the 50k sample) but cost the run.
**Relaunched cleanly at 16:07 with no concurrent heavy jobs** — but crashed again at 18:19, this
time with `pyarrow.lib.ArrowMemoryError: malloc of size 1073741824 failed` while reading a
~272MB source1 parquet. Nothing on our side was running concurrently at that point (confirmed no
other Python processes) — this was external memory pressure elsewhere on the machine, not a bug
in our code. Memory was back to 5.3GB free 11 minutes later. **Relaunched a third time at
18:30**, ~4.4h ETA (~22:50 IST), writing to `artifacts/candidate_pairs_train_full.tsv`. The
50k-sample snapshot used for Stage 3 development is saved at
`artifacts/candidate_pairs_train_sample50k.tsv`.
Lesson: on a 15.4GB shared machine, an unattended multi-hour job can be taken out by memory
pressure we don't control, not just by our own bugs — worth checking `ps` / free RAM on any
unexpected crash before assuming the code is at fault.

**Attempt 3 (18:30) crashed again, immediately** — same error, same place. At that point 5.37GB
was free (not obviously insufficient), with Brave browser using ~4GB across many processes —
this smells like a Windows virtual-memory commit-limit issue (total reserved virtual memory
across all processes vs. physical RAM + pagefile) rather than a simple "not enough free RAM"
situation, which free-RAM alone doesn't show.

**Real fix applied regardless of root cause**: `load_normalized_trimmed()` in blocking.py was
calling `pd.read_parquet(cache_path)` with no column selection, loading **all 12 columns**
(including the full raw `business_name`/`business_address` text — the bulk of each file's size)
only to immediately select down to the 4 columns blocking actually needs and discard the rest.
Fixed via parquet column pushdown (`normalize_and_cache(..., columns=BLOCKING_COLS)`), so unused
columns are never read into memory at all. This is a real, general inefficiency worth having
fixed regardless of whether it was the proximate cause of these crashes — every future full-scale
read now uses meaningfully less peak memory.
**Attempt 4 launched 18:32** with the fix in place — crashed again 11s later, same allocation-failure
pattern but now inside `build_keys`'s bigram explode step, on a tiny 46.2MB allocation.

**Real root cause found**: checked Windows virtual memory directly (`Get-CimInstance
Win32_OperatingSystem`), not just physical RAM. `FreeVirtualMemory` was ~4.16GB — *lower* than
free physical RAM (5.19GB) — because the pagefile is small (~5.3GB, auto-managed) and total
system commit was near its ceiling, driven by the Brave browser's many processes reserving large
amounts of virtual address space. This is a Windows virtual-memory-commit-limit problem, not a
bug in the pipeline: every prior crash (40MB, 496MB, 1GB, 46MB, all different code paths) was
the same underlying cause at different, essentially random trigger points. **Free RAM alone was
not a reliable signal for whether an allocation would succeed on this machine** — worth checking
`FreeVirtualMemory` specifically, not just `FreePhysicalMemory`, on any further unexplained crash.

User closed/reduced Brave; `FreeVirtualMemory` rose to 9.43GB. **Attempt 5 launched 20:12.**

**Attempt 5 progress**: India completed successfully — **17,522,668 candidate pairs**. Wall-clock
showed 43,397.6s (~12h) but `Get-Process`'s cumulative CPU-time check showed only ~3h54m of real
CPU work across the whole run to that point — confirms the wall-clock inflation is genuinely from
the laptop sleeping (Windows suspends processes cleanly; time.time() elapsed still counts the
suspended duration), not a real slowdown. Requested `keep_awake` (session_idle) partway through
to make later readings trustworter — but that tool explicitly does NOT prevent lid-close or
manual sleep, only idle timeout, so wall-clock readings for the rest of this run are still not
fully trustworthy without the user manually keeping the lid open / disabling lid-close sleep.

**Attempt 5 then crashed during the US phase** with a plain `MemoryError` (not an allocation-size
message this time) inside `_score_s1_batch`'s per-entity score accumulation loop — after ~15h
wall-clock (~3h54m+ CPU time by that point). Root cause: with `top_r_tokens=3` word/bigram keys
plus a possible postal key, and each key capped at `max_df=50,000` postings, one entity's
`scores` accumulator dict could in the worst case (near-zero overlap between the four keys'
postings) reach ~4×50,000 = 200,000 entries — rare enough not to appear in 5k/50k-row samples,
real at 2.2M rows.

**Compounding problem this crash exposed**: the run had **no checkpointing** — a crash during the
US phase meant India's already-completed 17.5M pairs would have to be recomputed from scratch on
retry, at the cost of redoing ~2h of real compute (plus whatever wall-clock/sleep inflation).

**Two fixes applied**:
1. A cap (`MAX_SCORES_PER_ENTITY = 100_000`) on the per-entity accumulator: once an entity's own
   `scores` dict reaches this size it already has far more candidates than the top-k it needs, so
   further keys are skipped for that entity rather than let the dict grow unbounded.
2. **Per-country checkpointing** in `compute_candidates()`: each country's result is saved to
   `artifacts/blocking_checkpoints_{split}/_country_{country}.parquet` immediately after
   computing, and loaded from there instead of recomputed on a later run. Only applies to full
   (non-`--sample-s1`) runs, to avoid a stale sampled checkpoint being mistaken for a full one.

**Considered but explicitly declined**: parallelizing the blocking join across CPU cores via
`multiprocessing` to cut real compute time. Windows' `multiprocessing` uses `spawn`, not `fork` —
each worker would need its own full copy of the (potentially multi-GB) postings dict rather than
sharing one via copy-on-write, which risks reintroducing the exact memory crashes just fixed for
an unproven speed win. Not worth the risk under time pressure; worth reconsidering later with a
proper shared-memory design if blocking needs to be rerun many more times.

**Attempt 6 launched 11:50** (no checkpoint existed yet from attempt 5, so India recomputes once
more; from now on a crash won't lose a completed country's work).

## Full-train blocking: SUCCESS (26 Sep, attempt 6)

Both countries completed and checkpointed:
- India: 17,522,668 pairs (20,213.2s wall-clock this attempt)
- US: 25,878,257 pairs (8,873.1s wall-clock)
- **Total: 43,400,925 candidate pairs**, written to `artifacts/candidate_pairs_train_full.tsv`
  (2,206,821 rows — full S1 coverage) and `artifacts/candidate_scores_train.parquet` (with score
  + rank, for feature engineering).

`measure_recall` then crashed with `MemoryError` — but only in the diagnostic step, *after* both
output files were already safely written. Root cause: its pandas two-column merge (43.4M
candidate pairs x 7.6M ground-truth pairs) needed more memory than was available for the
internal join factorization. Fixed the same way as normalize.py's earlier bugs — avoid heavy
string-object structures at this row count. First fix attempt (string-concat + Python set) also
crashed; the working fix encodes both id columns to compact int64 codes (shared factorization
across both frames) and compares with `numpy.isin` — dense integer arrays instead of strings/
tuples/sets, dramatically lighter at this scale.

**Real full-scale recall: 56.10%** (4,285,280 / 7,638,365) — matches the 50k-sample estimate
(56.24%) almost exactly, confirming the sample-based calibration was representative all along.
Only 55 / 2,083,574 S1s with a true match got zero candidates (0.0026%) — negligible, consistent
with the sample's "0 zero-candidate failures" finding at a scale where a handful of edge cases
are expected.

**CPU-time monitoring during this run** (via `Get-Process`, not wall-clock) showed real,
observable variance in utilization: some windows near 100% CPU (steady progress), one ~1h
stretch at only ~6% (likely sleep or memory-pressure thrashing) — confirms `time.time()`-based
elapsed logging in these scripts still isn't a reliable duration signal on this machine; only
`Get-Process`'s cumulative CPU time is.

**Next**: run `features.py` on the full 43.4M-pair set, retrain `train.py` (now saves models via
`--model-dir`, see the `decide.py` addition below) to get a real full-scale OOF score, then move
to test-set blocking.

## Full train_oof memory debugging arc, then strategy pivot (26 Sep)

Training LightGBM on all 43.4M rows hit a long chain of full-scale-only memory issues, each
fixed in turn (see the commit history for exact details): mixed-dtype feature columns forcing
float64 upcasts inside both pandas and LightGBM's internal conversion, `df.iloc[]` on the full
28-column frame dragging id columns along unnecessarily, a fold's ~35M-row training slice and
the full 43.4M-row X_all needing to coexist in memory. The dtype fix (float32/int32/int8
throughout) and building X_all as one column-by-column-constructed array were real, confirmed
improvements — each got the crash further before the next bottleneck appeared, and each was
verified byte-for-byte / value-identical against the known-good 50k-sample result before being
trusted at scale.

The X_all-vs-fold-slice memory conflict was fixed by backing X_all with a memory-mapped file
(consistent with this project's earlier diagnosis that Windows' virtual-memory *commit limit*,
not physical RAM, is the tighter constraint on this machine) — but this then hit a **native
access violation inside LightGBM's C library** (`LGBM_DatasetCreateFromMat`), not a catchable
Python `MemoryError`. This is a different class of problem: LightGBM's low-level array handling
doesn't appear to interact safely with memmap-backed numpy arrays on Windows, and debugging a
native crash inside a compiled C extension is a much larger, less certain time investment than
the pure-Python memory fixes so far.

**Decision: stop pushing full 43.4M-row training through this machine, train on a large
subsample instead.** Full-scale blocking (43.4M pairs, 56.1% recall) and full-scale feature
computation (894MB parquet, correct/verified) both already succeeded — that work isn't wasted.
`sample_features.py` (new) samples ~400-500k S1 entities' worth of rows directly from the
already-computed `features_train_full.parquet` via pyarrow predicate pushdown (never loads the
full file), giving ~8-10M pairs — comfortably within memory territory already proven reliable
(the 50k sample's ~983k pairs trained without issue), while being a substantially larger and
more representative training set than the original 50k-S1 development sample.

## First complete, valid submission produced (26-27 Sep, overnight)

Picked back up the pivot plan from the previous entry and ran it through to completion:

1. **`sample_features.py`**: sampled 500,000 S1 entities (~9.8M pairs) from the full train
   features — took ~12s (predicate pushdown, never touches the full 43.4M-row file).
2. **Trained on the 500k sample**: OOF macro F0.5 = **0.6636** at τ=0.50 (consistent with the
   50k sample's 0.6631, now on 10x the data). 5 fold models saved to `artifacts/models_full/`.
3. **Full test-set blocking**: ran cleanly end to end this time (one transient crash on the
   France→India transition, same known `explode()` memory-pressure pattern, fixed by a plain
   retry — France's checkpoint was reused, no recomputation). Final numbers:
   - France: 5,067,199 pairs (34m)
   - India: 16,090,796 pairs (2h40m)
   - US: 12,839,502 pairs (41m — faster than estimated)
   - **Total: 33,997,497 candidate pairs, full coverage of all 1,732,544 test S1 entities**
   - Total wall-clock: ~3h23m — close to the ~3.5-4h estimate, no sleep-inflation this run.
4. **Test features**: 33,997,497 rows computed in ~10.5 min, no crash (all the batching/dtype
   fixes from the train run held up at test scale too).
5. **`decide.py`**: applied the 500k-sample model ensemble, enforced exclusivity, threshold
   τ=0.50 (from training) → `output/matching_results.tsv` (1,732,544 rows: 1,271,476 with a
   match, 461,068 predicted singletons) + `output/candidate_pairs.tsv` (copied through).
6. **`utils/validate_submission.py --check-ids`: PASS.** Every ID in both files genuinely
   exists in the test set; no format violations.

**Honest read on the singleton rate**: 461,068/1,732,544 ≈ 26.6% predicted singletons, far
above train's true ~5.6% singleton rate. This is expected given blocking recall (~56%) — many
entities with real matches simply have no candidate clearing threshold, so the model defaults
them to "no match." Not a bug, just the recall ceiling showing up directly in the submission;
the private-leaderboard score will very likely reflect that ceiling until Day 2 embeddings
improve recall.

**Status: a complete, valid, submission-ready file exists at `output/matching_results.tsv` /
`output/candidate_pairs.tsv`.** Uploading to the actual portal is the user's own action.

## First real leaderboard score: public LB F0.5 = 0.658 (27 Sep morning)

Uploaded the file above. **Public LB F0.5 = 0.658**, vs. OOF 0.6636 — only ~0.6% apart,
confirming the GroupKFold + exact-metric threshold sweep is well-calibrated, not overfit to
the training sample. Logged in `submissions.md`.

## Exact-name-match supplementary blocking pass (27 Sep)

Investigated real recall-miss cases (S1 entities whose true match is missing from candidates)
by inspecting 20 actual examples rather than theorizing. Found a genuinely mixed bag:
roughly a third were **fixable blocking-logic gaps**, not cases needing semantic embeddings —
word reordering breaking bigram overlap (`orthopedic safe health` vs `orthopedic health safe`,
zero shared bigrams despite identical words) and, more strikingly, **exact string matches still
missing** (`krishna power` -> `krishna power`, `united agro` -> `united agro` — identical
strings, dropped because the phrase is common enough to exceed even the rarest-token
selection's effective reach). Full multilingual embeddings were separately sized at ~12.6h of
CPU compute for the whole dataset — infeasible on the last day — so this cheaper, evidence-based
fix was pursued first.

**`exact_match_candidates.py`** (new): for each country, merge S1 and S2/S3 records on exact
`name_core` — vectorized per-country merge, not a Python loop over groups (would have repeated
blocking.py's original mistake). First uncapped run on train produced **75.2M pairs** — nearly
double the entire existing candidate set — because a handful of very generic/short names
(median group size 4, but max 1,359) dominate the count. Added `max_group_size=50` (mirrors
blocking.py's `max_df`, just for whole-name groups instead of tokens): cut train to **6.1M pairs**
(India 1.75M, US 4.35M), test to **5.0M pairs** (France 1.26M, India 1.64M, US 2.07M).

**`merge_candidates.py`** (new): unions blocking.py's token/bigram candidates with the
exact-match candidates, working entirely from flat parquet sources
(`candidate_scores_{split}.parquet`, never the comma-joined TSV — re-exploding that at full
scale crashed with MemoryError more than once already this project).

**Measured recall gain on train: 56.10% -> 57.75% (+1.65pp)**, from combined pairs
43.4M -> 44.46M (most exact-match pairs already existed in the token/bigram set).

**Retrained on a fresh 500k-S1 sample of the merged candidates: OOF macro F0.5 = 0.6762 at
τ=0.55** (up from 0.6636 at τ=0.50) — a genuine, real improvement matching the recall gain, not
noise. Models saved to `artifacts/models_merged/`.

Regenerated test predictions (`decide.py` with the new model + merged test candidates):
1,732,544 rows, 451,981 predicted singletons (down from 461,068), 1,280,563 with a match (up
from 1,271,476). **`utils/validate_submission.py --check-ids`: PASS.** New
`output/matching_results.tsv` is ready — a second, improved submission, not yet uploaded.

## decide.py added + train.py now persists models (26 Sep)

`train.py`'s `train_oof` only returned OOF predictions before — useless for inference on test
data, which has no fold assignment. Added `save_models()`: each fold's booster
(`fold_{i}.txt`), the feature-column list, and the tuned threshold go to `--model-dir/meta.json`.
`decide.py` (new): loads that ensemble, averages all folds' predictions (standard bagging — no
leakage risk since no fold ever saw test data), enforces exclusivity, applies the threshold, and
writes `matching_results.tsv` + copies `candidate_pairs.tsv` through unchanged (already in the
exact required format).

**Defaults landed on**: `--max-df 50000` (calibrated middle ground — lower drops legitimate keys,
higher reintroduces expensive joins), `--top-r-tokens 3`, `--s1-chunk-size 50000` (memory-only
concern now, not a join-size safety net), `k=20` candidates per S1.

## Stage 3 (features.py + train.py) — first end-to-end result (25 Sep, Day 1)

Built while the full-train blocking run (above) proceeds in the background. Developed and
validated against `artifacts/candidate_pairs_train_sample50k.tsv` (the 50k-S1 sample snapshot).

**Two more memory bugs caught before they hit the full run** (both same root cause as earlier:
scoping to the full dataset instead of the sample actually in scope):
1. `features.py`'s `load_filtered()` originally loaded the full ~670MB source2/3 parquet before
   `.isin()`-filtering — risky with as little as ~2.4-3.9GB free RAM while the concurrent
   full-train blocking job runs. Fixed with pyarrow predicate pushdown
   (`pd.read_parquet(..., filters=[("entity_id","in",ids)])`) so non-matching rows are never
   materialized as Python objects at all.
2. `attach_labels()` exploded the *full* 7.6M-row ground truth before merging against a 50k-S1
   sample's ~983k pairs — segfaulted (exit 139) under the same concurrent memory pressure. Fixed
   by filtering ground truth to the sample's S1 scope before exploding (same fix pattern as
   `blocking.py`'s `measure_recall`).

**26 pairwise features** (`f_` prefix): rapidfuzz ratio/partial/token_sort/token_set on core name
and address, name token Jaccard, legal-suffix agreement, postal/house-number agreement, landmark
flags, containment, acronym match, phone-in-name flags, name lengths. Verified on a 100-S1 smoke
test before running full: positives had mean `f_name_ratio` 90.5 vs negatives 67.0 — clearly
discriminative.

**Model**: LightGBM, 5-fold GroupKFold by `source1_entity_id` (no S1's pairs split across
train/val). Trained on the 50k-sample's 982,643 pairs (97,217 positives, 9.89%) in ~70s total.

**Top features by gain**: `f_addr_token_set` dominates by a wide margin (2.77M, ~7.6x the next
feature), then `f_addr_ratio`, `f_house_conflict`, `f_name_token_sort`. **Address similarity is
the single strongest signal** — stronger than any name feature. `f_acronym_match` and
`f_name_has_phone_a` had zero importance on this sample (rare conditions, small sample — revisit
once trained on the full dataset).

**Decision layer**: exclusivity enforced (each candidate → its single best-scoring S1 only, per
the confirmed hard rule), then threshold τ swept in the exact macro F0.5 metric.

**Result: OOF macro F0.5 = 0.6632 at τ=0.50** (singletons 0.8693, with-matches 0.6507). This is
capped by this sample's ~56% blocking recall — a real, honest number, but expect it to rise
substantially once blocking recall improves (Day 2 embeddings) and once trained on the full
2.2M-row train set instead of a 50k sample. Threshold behaved sensibly: singleton accuracy rises
monotonically with τ (as expected — higher bar to predict any match), while with-matches score
peaks around τ=0.40-0.55, consistent with the precision/recall trade-off F0.5 is designed around.

Artifacts: `artifacts/features_train_sample50k.parquet`, `artifacts/oof_train_sample50k.parquet`.

## Bug caught while building `src/normalize.py` (25 Sep, Day 1)
Two-part bug that would have silently corrupted every non-Latin-script name (Devanagari, and presumably other scripts) before either the classic 3-5 confirmed on Day 1) or the Day 2 embedding retriever ever saw the text — caught with a smoke test before it touched real data:
1. `strip_accents` used `unicodedata.combining(c)` to detect "accent marks to strip" after NFKD decomposition. That's too broad — it strips *any* Unicode combining mark, including Devanagari vowel signs (matras), which are not accents but structural parts of the letters. Fix: only strip combining marks in the U+0300-U+036F block (where Latin accents decompose to).
2. Separately, the punctuation-stripping regex was `[^\w\s&]` ("keep only word characters"). Python's `\w` does **not** include Unicode mark categories (Mn/Mc) — so this regex was *also* independently stripping Devanagari vowel signs (e.g. the "ो" in होटल is category `Mc`), regardless of fix #1. Fix: strip an explicit ASCII punctuation set instead of "anything not \w", so any script's letters/marks pass through untouched.
- Verified fix with a smoke test: `होटल एंटरप्राइजेज लिमिटेड` now survives `normalize_name()` unchanged (previously became `ह टल ए टरप र इज ज ल म ट ड` — unrecoverable garbage).
- **Lesson for the rest of the pipeline**: any future text-cleaning code must be smoke-tested against the real non-Latin samples in this file before running on full data — `\w`/accent-stripping bugs are easy to write and easy to miss if you only test on English samples.

| Date | Change | Blocking recall | Avg candidates/S1 | OOF F0.5 (all / singletons / with matches) | Notes |
|---|---|---|---|---|---|
