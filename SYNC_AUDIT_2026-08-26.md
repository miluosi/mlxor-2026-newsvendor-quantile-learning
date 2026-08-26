# Synchronization Audit: 2026-08-26

Source project: `/Users/seinzhou/Desktop/wsc_newexp`

Bundle: `/Users/seinzhou/Desktop/gendfl_rseto_ipa_reproducibility_bundle`

## Unchanged training core

The following current experiment components already matched the source project
byte for byte before this synchronization:

- shared conditional spline backbone and exact conditional quantile;
- likelihood-only Spline-ETO / GenDFL training;
- random-quantile Spline-QFR training;
- batched screen/replay RSETO-IPA training and gradient-variance diagnostic;
- Robbins-Monro projected SGD;
- Exp5 and Van Havre synthetic data generators;
- `R`, `M`, and `lambda` sensitivity drivers;
- ERM, LightGBM, ERM-NN, and conditional-mixture oracle baselines;
- real-world shared-spline experiment drivers.

No gradient, objective, data-generation, optimizer, or evaluation formula was
changed while updating the bundle.

## Updated reporting layer

- `syn_sensitivity_report.py`
- `bcmo_difference_report.py`
- `results_syn/read_results_syn_analysis/read_results_syn.ipynb`
- generated LaTeX, CSV, PNG, and PDF artifacts under `results_syn/`

The update adds combined Exp5 Metric 1/2 profiles for `lambda` and `M`, refreshes
the Van Havre and IPA-variance figures, and formats every selected RSETO reference
with complete `R`, `M`, and `lambda` values. The current Exp5 Metric 1 reference is
`R=64, M=128, lambda=0.5`.

## Exclusions retained

The bundle still excludes `.DS_Store`, Python caches, spreadsheet inspection
dumps, duplicate `results_syn/read_syn_summary/` previews, checkpoints, and raw
server run trees.

## Verification

- 50 Python files passed syntax parsing.
- All packaged Python files matched their current source files; the empty
  `model/__init__.py` package marker is bundle-only.
- 49 unit tests passed; one CUDA agreement test was skipped because CUDA was not
  available locally.
- The canonical notebook executed all 31 cells with zero errors using only the
  bundled result tree.
