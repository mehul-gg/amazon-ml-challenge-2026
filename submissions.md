# Leaderboard submissions (max 5/day)

| # | Date/time (IST) | Git/commit or file | OOF F0.5 | Public LB F0.5 | Notes |
|---|---|---|---|---|---|
| 1 | 27 Sep, morning | commit 2bf926b — output/matching_results.tsv (500k-sample model, τ=0.50, token/bigram blocking only) | 0.6636 | **0.658** | First real submission. Public LB is only ~0.6% below OOF — validation methodology (GroupKFold, exact-metric threshold sweep) is well-calibrated, not overfit to the training sample. Blocking recall (~56%, single retriever) is the likely ceiling on both numbers. |
