# Verification Record

Verification was repeated inside this copied bundle on 2026-08-26 after syncing
the current reporting code and canonical result artifacts.

## Source integrity

- 50 Python files are present.
- Every copied Python file has the same SHA-256 digest as its source in
  `wsc_newexp`.
- `model/__init__.py` is the only added Python file; it is an empty package marker.
- All 50 Python files pass `ast.parse` syntax validation.

## Unit and interface tests

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_*.py' -v
```

Result:

```text
Ran 49 tests in 1.087s
OK (skipped=1)
```

The skipped test is the CPU/GPU agreement test because CUDA was unavailable on
the local verification machine. The test remains included for execution on the
RTX 4090 server.

The passing tests cover:

- exact spline quantile monotonicity and inverse consistency;
- GenDFL likelihood-only training;
- shared initial backbone parameters;
- QFR integrated pinball loss;
- vectorized IPA versus the explicit replication loop;
- screen/replay versus the full autograd graph;
- projected stochastic approximation schedules;
- RSETO-IPA gradient-variance diagnostics;
- Metric 2 cost-pair handling;
- Exp5 and Van Havre data alignment;
- legacy GenDFL conditional-flow IPA/GLR interfaces.

## Experiment entry points

The following scripts successfully parse `--help` from the bundle:

```text
gendfl_lambda.py
gendfl_m.py
gendfl_simulation_num.py
benchmark_shared_spline_flow_syn.py
synthetic_fixed_dgp_traditional_models.py
real_world_d3group_gendfl_benchmark.py
real_world_d3group_gendfl_sqeto_ipa.py
```

## Data-generation smoke test

For dimension 4 and fold 0, all three current DGP interfaces produced independent
train/validation/test arrays with shapes:

```text
train       (11583, 5)
validation  (1287, 5)
test        (5791, 5)
```

The reported protocol identifiers were:

```text
fixed_parameter_conditional_mixture_v2
literature_van_havre_2015_sim4_conditional_v1
izbicki_2026_bimodal_full_all_active_projection_v1
```

## Result notebook

`results_syn/read_results_syn_analysis/read_results_syn.ipynb` was executed from
start to finish inside the bundle with a 900-second cell timeout.

```text
cells: 31
code cells: 17
execution errors: 0
```

The notebook regenerated the sensitivity figures, LaTeX tables, IPA variance
analysis, Van Havre comparison, and paired BCMO-difference outputs using only the
bundled `results_syn` directory.

## 2026-08-26 synchronization audit

- The 50 packaged Python files pass `ast.parse` validation.
- Every packaged Python file still matches its current `wsc_newexp` source by
  SHA-256; `model/__init__.py` remains the sole bundle-only package marker.
- Training core files had no differences before synchronization: the shared spline
  flow, GenDFL/Spline-ETO, Spline-QFR, RSETO-IPA, projected SGD, Exp5 and Van Havre
  generators, sensitivity drivers, and ERM/LightGBM/ERM-NN baseline script were
  already current.
- Updated files were `syn_sensitivity_report.py`, `bcmo_difference_report.py`, the
  canonical report notebook, and their generated result tables and figures.
- New Exp5 combined Metric 1/2 profile figures are included for `lambda` and `M`.
- RSETO reference labels now display complete `R`, `M`, and `lambda` values. In the
  current paired-BCMO output, Exp5 Metric 1 correctly selects
  `R=64, M=128, lambda=0.5`.
