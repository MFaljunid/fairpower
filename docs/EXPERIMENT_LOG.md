# fairpower: Experiment Log

Chronological record of every experiment, its configuration, raw output, and the decision it led to.
Terminal outputs are copied **verbatim**. Results are never overwritten (one folder per run).

---

## F0: Motivation and plan (2026-10-15)

Target: an international conference paper on the reliability of user-side fairness evaluation.
Evidence that motivates it (own work):
- leveldown audit: on MovieLens, the gender gap changed sign between validation and test items.
- LevelUp gate (ML-1M, iALS): the setting selected on one validation item per user raised target
  validation NDCG from 0.0265 to 0.0293 but changed test NDCG by -0.0001 (winner's curse).

Research questions:
- RQ1: minimum detectable gap (MDE) by dataset, grouping and number of held-out items per user.
- RQ2: how many gaps reported in the literature fall below the MDE of their own setting.
- RQ3: reliability of selecting fairness-intervention settings on validation (does the validation
  winner stay the winner on test?), with one vs several validation items per user.
- RQ4: remedies (several held-out items, temporal cross-validation, conservative selection).

Novelty check (2026-10-15): Rampisela (thesis, arXiv 2604.25032) studies properties of fairness
measures, not sample size or selection reliability; Ihemelandu and Ekstrand (2021) discuss statistical
inference for accuracy, not group gaps; Dong et al. (arXiv 2406.05247) give significance tests for
fairness metrics with missing labels in industry A/B tests, not offline group gaps.

---

## F1: Code check on synthetic data (2026-10-15)

Unit tests: 8 passed. With m = 1, temporal_multi gives the same split as leveldown's temporal_loo and
evaluate_multi gives the same per-user NDCG as leveldown's evaluate (tested on val and test).
MDE matches the closed form; gap_test has no false positive on null data and detects a real gap;
power curves increase with n; planted-gap power is about 5% at delta 0 and high at a large delta.

run_power.py on synthetic ML-100K-format data (500 users, m = 1, 3, 5): runs end to end; MDE
decreases with m (activity Q4 vs Q1: 65% -> 35% -> 28% of overall NDCG); planted-gap power at the MDE
is 0.71 to 0.87 against the nominal 0.80 (100 replicates, 125 to 146 users per group), which supports
the normal approximation on the zero-inflated metric.
