# File Manifest

## Current shared-spline model layer

| File | Role |
|---|---|
| `model/shared_spline_flow.py` | Conditional one-dimensional rational-quadratic spline backbone and exact quantile |
| `model/gendfl_spline.py` | Shared base class and likelihood-only Spline-ETO / GenDFL training |
| `model/spline_qfr.py` | Random-quantile integrated pinball training |
| `model/rseto_ipa_spline.py` | Parallel RSETO-IPA screening, replay, training, and gradient-variance diagnostic |
| `model/projected_sa.py` | Robbins-Monro step schedule, box projection, and projected-SGD step |
| `model/generative_newsvendor_base.py` | Common generative-newsvendor interface |

## Synthetic data and experiments

| File | Role |
|---|---|
| `synthetic_fixed_dgp.py` | Fixed-parameter Exp5 conditional Gaussian mixture |
| `benchmark_literature_gaussian_rare_event_syn.py` | Van Havre-inspired separated rare Gaussian DGP |
| `benchmark_izbicki_2026_bimodal_newsvendor.py` | All-active conditional bimodal DGP and standalone experiment |
| `spline_sensitivity_common.py` | Shared split, scaling, costs, training, evaluation, output, and variance logic |
| `gendfl_simulation_num.py` | `R` sensitivity grid |
| `gendfl_m.py` | `M` sensitivity grid |
| `gendfl_lambda.py` | `lambda` sensitivity grid |
| `benchmark_shared_spline_flow_syn.py` | Shared-backbone synthetic benchmark and diagnostics |
| `benchmark_shared_spline_flow_toy_exp.py` | Exp1/Exp5 toy comparison entry point |
| `spline_low_data_shape_search.py` | Low-data and target-shape search experiment |
| `benchmark_rseto_ipa_acceleration.py` | Numerical equivalence and timing benchmark for IPA acceleration |

## Baselines and oracle

| File | Role |
|---|---|
| `synthetic_fixed_dgp_traditional_models.py` | ERM, LightGBM, ERM-NN, and Bayes conditional-mixture oracle |
| `GLR_lr2.py` | Legacy data/cost protocol reference retained for alignment metadata |
| `Walmart.csv` | Legacy row-count reference used to set synthetic sample size |

## Real-world experiments

| File | Role |
|---|---|
| `real_world_d3group_test.py` | d3group download, loading, feature preparation, and shared constants |
| `real_world_d3group_gendfl_common.py` | Shared initialization, loaders, exact quantiles, Metric 1/2, and output logic |
| `real_world_d3group_gendfl_benchmark.py` | Spline-ETO and Spline-QFR real-world driver |
| `real_world_d3group_gendfl_sqeto_ipa.py` | RSETO-IPA real-world driver |
| `real_world_d3group_paper_models_test.py` | Paper-model comparison driver |

## Reporting

| File | Role |
|---|---|
| `consolidate_syn_sensitivity_results.py` | Convert raw server trees to validated seed-level CSV tables |
| `export_synthetic_hyperparameter_results.py` | Legacy NPY-to-table exporter |
| `syn_sensitivity_report.py` | Van Havre/Exp5 summaries, LaTeX, combined Metric 1/2 parameter profiles, confidence plots, and boxplots |
| `bcmo_difference_report.py` | Paired BCMO differences, LaTeX, complete RSETO `(R, M, lambda)` reference labels, and no-BCMO-column boxplots |
| `results_syn/read_results_syn_analysis/read_results_syn.ipynb` | Canonical executable analysis notebook |

## Legacy GenDFL interfaces

The bundle also retains the earlier affine conditional-flow GenDFL, quantile-flow,
regularized-gradient interface, VAE, RealNVP, MeanFlow, DDPM, and DDIM comparison
files. These files are included for historical experiments but are not part of the
current shared-spline sensitivity pipeline.

## Excluded files

- `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, and `.DS_Store`
- `*.inspect.ndjson` spreadsheet verification dumps
- duplicated `results_syn/read_syn_summary/` previews
- model checkpoints and raw server run trees that were not present locally
