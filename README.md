# GenDFL / Shared Spline / RSETO-IPA Reproduction Bundle

This directory is a self-contained snapshot of the GenDFL-related code, current
shared-spline experiments, RSETO-IPA implementation, synthetic data generators,
real-world experiment drivers, consolidated numerical results, plotting notebooks,
and generated figures from `wsc_newexp`, synchronized on 2026-08-26.

The Python files are copied without changing their mathematical or training logic.
The original import layout is preserved: run commands from this directory.

## 1. Current experimental pipeline

The paper-facing pipeline uses one common one-dimensional conditional rational
quadratic spline flow for all three learned methods:

1. **Spline-ETO / GenDFL** (`model/gendfl_spline.py`)
   trains the conditional distribution using negative log-likelihood only.
2. **Spline-QFR** (`model/spline_qfr.py`)
   samples quantile levels and minimizes integrated pinball loss.
3. **RSETO-IPA** (`model/rseto_ipa_spline.py`)
   combines distribution fidelity and the simulated newsvendor task gradient.

All three use `SharedConditionalSplineFlow` from
`model/shared_spline_flow.py`. At inference time, each method calls the exact
inverse spline quantile for every context and target service level. RSETO-IPA's
Monte Carlo order statistic is used during training only.

For shortage cost `c_u > 0` and signed overage coefficient `c_o < 0`, the target
service level is

```text
alpha = c_u / (c_u + abs(c_o)).
```

Metric 1 evaluates the cost pair assigned to the random-seed fold. Metric 2
recomputes the exact context-wise spline quantile for each of the ten test cost
pairs, evaluates the corresponding newsvendor loss, and averages those costs.

## 2. RSETO-IPA calculation

The regularized training objective is implemented as

```text
lambda * negative_log_likelihood
    + (1 - lambda) * smoothed_newsvendor_loss.
```

For a mini-batch of `B` contexts, RSETO-IPA generates a tensor of base noise with
shape `[B, R, M, 1]`:

- `R`: independent IPA replications (`simulation_number`);
- `M`: conditional demand draws per replication (`samples_per_replication`);
- each `[B, r, :, 0]` block supplies one empirical alpha-order statistic;
- gradients are averaged over the batch and the `R` replications.

The accelerated implementation preserves this estimator in two stages:

1. `screen_selected_base_noise` finds the selected order-statistic paths without
   retaining the full `B * R * M` autograd graph and chunks work according to
   `max_simulation_values`.
2. `replay_selected_quantiles` replays only the selected latent paths with
   gradients enabled.

Projected stochastic approximation is implemented in `model/projected_sa.py`:

```text
gamma_k = gamma_0 / (k + 1)^a,
theta_{k+1} = projection(theta_k - gamma_k * gradient_k).
```

The post-training variance diagnostic in
`RSETOIPASplineNewsvendor.estimate_batch_ipa_gradient_variance` repeats the same
batched estimator independently and records the variance of the averaged IPA
gradient. It is the statistic used to study variance reduction as `R` increases.

## 3. Synthetic data generators

The shared entry point is `build_sensitivity_data` in
`spline_sensitivity_common.py`. Select a generator with
`--data_synthetic {exp5,van-havre,izbicki-bimodal}`.

### Exp5

`synthetic_fixed_dgp.py` defines a fixed-parameter five-component conditional
Gaussian mixture. Mixture weights, component intercepts, regression vectors, and
feature means are generated once per experimental seed. Training and test samples
use independent sample seeds while sharing the same DGP parameters.

`Walmart.csv` contains 6,435 rows and is used only to recover the legacy sample
scale. The default pool contains 12,870 observations, split into 11,583 training
and 1,287 validation observations; the independent test set contains 5,791 rows.

### Van Havre Simulation 4

`LiteratureSeparatedRareGaussianDGP` in
`benchmark_literature_gaussian_rare_event_syn.py` implements the conditional,
positive-demand adaptation used by the experiments. The generator metadata and
observed rare-component fractions are stored with every run. The bundled
`data_support/van_havre_s1.r` is the retained reference script.

### Izbicki bimodal

`IzbickiBimodalFullDGP` in
`benchmark_izbicki_2026_bimodal_newsvendor.py` implements the all-active
conditional bimodal construction: every requested context dimension contributes
to the conditional target distribution.

## 4. Default synthetic protocol

The sensitivity drivers use:

```text
dimensions:       4, 9, 14, 19, 24
folds:            0,...,9
random seeds:     82, 15, 4, 95, 36, 32, 29, 18, 14, 87
epochs:           50
batch size:       64
optimizer:        projected SGD
gamma_0:          1e-3
step exponent:    0.6
parameter box:    [-10, 10]
default M:        128
default R:        16
default lambda:   0.5
```

Sensitivity grids:

```text
R:       1, 4, 16, 64, 256
M:       8, 32, 128, 512, 2048
lambda:  0.1, 0.3, 0.5, 0.7, 0.9
```

Synthetic drivers default to no early stopping; `--use-early-stopping` enables
validation-news-vendor early stopping with patience 20. Real-world drivers use
100 maximum epochs, batch size 64, early stopping enabled, and patience 50.

## 5. Installation

Python 3.11 or later is recommended. On a CUDA server, install the PyTorch build
matching the server CUDA runtime first, then install the remaining requirements.

```bash
cd /Users/seinzhou/Desktop/gendfl_rseto_ipa_reproducibility_bundle
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`gurobipy` is needed only for the exact ERM baseline and requires a valid Gurobi
license. `lightgbm` is needed only for the LightGBM baseline.

## 6. Quick verification

Compile the packaged code:

```bash
python -m compileall -q .
```

Run the focused shared-spline test suite:

```bash
python -m unittest discover -s tests -p 'test_shared_spline_methods.py' -v
```

Run one small lambda experiment:

```bash
python gendfl_lambda.py \
  --single-run \
  --data_synthetic exp5 \
  --dim 4 --fold 0 \
  --epochs 2 \
  --lambda-test-list 0.5 \
  --output-root smoke_outputs/exp5_dim4_seed82 \
  --force
```

## 7. Full synthetic experiments

Use one RTX 4090 per process. The scripts print the active hyperparameter and
timing every ten epochs by default.

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u gendfl_simulation_num.py \
  --data_synthetic exp5 --epochs 50 \
  --output-parent exp_simulation \
  > logs/exp5_R_rerun.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u gendfl_m.py \
  --data_synthetic exp5 --epochs 50 \
  --output-parent exp_simulation \
  > logs/exp5_M_rerun.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u gendfl_lambda.py \
  --data_synthetic exp5 --epochs 50 \
  --output-parent exp_simulation \
  > logs/exp5_lambda_rerun.log 2>&1 &
```

Replace `exp5` with `van-havre` or `izbicki-bimodal`. Use a different
`--output-parent` for each DGP to avoid mixing result trees.

Run ERM, LightGBM, ERM-NN, and the Bayes conditional-mixture oracle:

```bash
python synthetic_fixed_dgp_traditional_models.py \
  --data-synthetic exp5 \
  --epochs 50 --batch-size 64 \
  --output-dir analysis_outputs_ete/fixed_dgp_exp5_traditional_50epochs_projected_sgd
```

For Van Havre, change `--data-synthetic` to `van-havre` and choose a separate
output directory.

## 8. Real-world experiments

The real-world pipeline uses the four d3group datasets (`m5`, `SID`, `yaz`, and
`bakery`). Missing d3group files are downloaded automatically by default.

Train and evaluate Spline-ETO and Spline-QFR:

```bash
CUDA_VISIBLE_DEVICES=0 python real_world_d3group_gendfl_benchmark.py \
  --dataset all --feature-combi calendar \
  --epochs 100 --early-stopping 50 --batch-size 64
```

Train and evaluate RSETO-IPA:

```bash
CUDA_VISIBLE_DEVICES=0 python real_world_d3group_gendfl_sqeto_ipa.py \
  --dataset all --feature-combi calendar \
  --simulation-number 16 --mnum 128 --lambda 0.5 \
  --epochs 100 --early-stopping 50 --batch-size 64
```

Both drivers use `analysis_outputs/d3_real_world_gendfl_initializations` for
shared initial checkpoints, so comparisons start from the same backbone state.

## 9. Results and plotting

`results_syn/` is the canonical, portable result snapshot. It excludes temporary
inspection dumps and duplicated preview directories.

- `syn_lambda_ipa.xlsx`, `syn_m_ipa.xlsx`, `syn_simulation_num_ipa.xlsx`:
  seed-level Exp5 RSETO-IPA sensitivity results.
- `syn_gendfl_spline.xlsx`, `syn_spline_qfr.xlsx`: seed-level spline baselines.
- `consolidated_csv/`: Exp5 and Van Havre seed-level results, ETE baselines, and
  IPA gradient-variance tables.
- `read_results_syn_analysis/read_results_syn.ipynb`: canonical report notebook.
- `read_results_syn_analysis/figures/`: generated LaTeX tables, confidence plots,
  Exp5 and Van Havre parameter profiles, variance analysis, and comprehensive
  boxplots.
- `difference_output_bcmo/`: paired model-minus-BCMO tables and plots. Each
  selected RSETO-IPA reference is labelled with its complete `(R, M, lambda)`
  setting; unspecified values are filled from the defaults `(16, 128, 0.5)`.

The 2026-08-26 synchronization changed only reporting code and generated report
artifacts. The shared-spline backbone, three training objectives, synthetic data
generators, sensitivity drivers, and baseline training code were already identical
to the current project and were not modified by the synchronization.

Rebuild every report from the bundled result files:

```bash
jupyter nbconvert \
  --to notebook --execute --inplace \
  results_syn/read_results_syn_analysis/read_results_syn.ipynb \
  --ExecutePreprocessor.timeout=900
```

The notebook reads only from `results_syn`; it does not require the original raw
experiment folders. `consolidate_syn_sensitivity_results.py` is included for a
fresh server run, but it expects raw `exp_simulation/`, `vanharve_simulation/`,
and `analysis_outputs_ete/` trees to exist.

## 10. Directory map

```text
gendfl_rseto_ipa_reproducibility_bundle/
|-- model/                    # current spline and legacy generative models
|-- tests/                    # focused unit and interface tests
|-- notebooks/                # DGP visual checks and legacy readers
|-- results_syn/              # canonical numerical results and figures
|-- logs/                     # retained server sensitivity logs
|-- data_support/             # Van Havre reference script
|-- gendfl_lambda.py          # lambda grid
|-- gendfl_m.py               # M grid
|-- gendfl_simulation_num.py  # R grid
|-- spline_sensitivity_common.py
|-- synthetic_fixed_dgp.py
|-- synthetic_fixed_dgp_traditional_models.py
|-- real_world_d3group_gendfl_*.py
|-- syn_sensitivity_report.py
|-- bcmo_difference_report.py
|-- Walmart.csv
|-- requirements.txt
|-- VERIFICATION.md
|-- FUNCTION_INDEX.md
`-- FILE_MANIFEST.md
```

## 11. Current versus legacy code

Use the shared-spline files for the current experiments. The following files are
retained only to make earlier comparisons and interfaces reproducible:

- `model/newsvendor_gendfl_conditional_flow.py`
- `model/newsvendor_quantile_flow.py`
- `model/gendfl_1d_interface.py`
- `run_generative_newsvendor_toy_exp.py`
- `benchmark_gendfl_quantile_flow_syn.py`
- VAE, RealNVP, MeanFlow, DDPM, and DDIM model files under `model/`

They are not used by `gendfl_lambda.py`, `gendfl_m.py`, or
`gendfl_simulation_num.py`.
