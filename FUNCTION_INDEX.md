# GenDFL Function Index

This index identifies the functions that form the current end-to-end experimental
path. Names beginning with `_` are internal helpers but are listed when they are
important to the RSETO-IPA computation.

## Shared spline backbone

### `model/shared_spline_flow.py`

- `SplineFlowConfig`: architecture configuration.
- `softplus_mlp`: positive-output context network used by spline parameterization.
- `ConditionalRQSLayer1D.encode_condition`: encode each context once.
- `ConditionalRQSLayer1D._spline_parameters`: construct normalized spline widths,
  heights, and derivatives.
- `ConditionalRQSLayer1D.forward_with_encoded`: forward or inverse scalar RQS map.
- `SharedConditionalSplineFlow.base_to_data_from_encoded`: differentiable inverse
  generation map from base noise to demand.
- `SharedConditionalSplineFlow.data_to_base_from_encoded`: map demand to base noise.
- `SharedConditionalSplineFlow.log_prob_from_encoded`: exact conditional log density.
- `SharedConditionalSplineFlow.sample_from_base_noise`: conditional sampling with
  externally supplied noise, including `[B,R,M,1]` RSETO tensors.
- `SharedConditionalSplineFlow.quantile`: exact conditional inverse-CDF quantile.
- `SharedConditionalSplineFlow.sample`: explicit conditional random sampling.

## GenDFL / Spline-ETO

### `model/gendfl_spline.py`

- `SplineConditionalNewsvendorBase.generative_loss`: conditional NLL.
- `SplineConditionalNewsvendorBase.quantile`: exact spline quantile interface.
- `SplineConditionalNewsvendorBase.critical_quantile_decision`: newsvendor decision
  at the configured service level.
- `SplineConditionalNewsvendorBase.exact_newsvendor_loss`: unsmoothed validation loss.
- `SplineConditionalNewsvendorBase.evaluate_exact_newsvendor`: validation evaluator.
- `SplineConditionalNewsvendorBase.train_spline_nll`: likelihood-only training loop.
- `SplineConditionalNewsvendorBase.train_gendfl_spline`: public GenDFL training alias.
- `GenDFLSplineNewsvendor`: current likelihood-only model class.

## Spline-QFR

### `model/spline_qfr.py`

- `pinball_loss`: quantile regression loss.
- `SplineQFRNewsvendor.qfr_objective`: random-quantile integrated pinball objective.
- `SplineQFRNewsvendor.train_spline_qfr`: projected-SGD/Adam training with exact
  unsmoothed newsvendor validation and optional early stopping.

## RSETO-IPA

### `model/rseto_ipa_spline.py`

- `increasing_sample_size`: optional iteration-dependent `M_k` schedule.
- `smooth_newsvendor_loss`: differentiable training task loss.
- `exact_spline_newsvendor_objective`: exact-quantile task objective used for checks.
- `screen_selected_base_noise`: chunked no-grad generation of `B * R * M` samples
  and selection of the empirical alpha-order-statistic noise paths.
- `replay_selected_quantiles`: differentiable replay of selected paths only.
- `_gradient_statistics`: parameter-gradient norms and variance summaries.
- `_gradients_are_finite`: finite-gradient guard.
- `RSETOIPASplineNewsvendor.rseto_ipa_objective`: full-graph reference objective.
- `RSETOIPASplineNewsvendor.rseto_ipa_replay_objective`: accelerated NLL/IPA joint
  objective using selected paths.
- `RSETOIPASplineNewsvendor.exact_task_objective`: exact spline task comparator.
- `RSETOIPASplineNewsvendor.estimate_batch_ipa_gradient_variance`: post-training
  variance of the same batch-averaged IPA estimator used in training.
- `RSETOIPASplineNewsvendor.train_rseto_ipa_spline`: complete projected-SGD
  RSETO-IPA training loop.

## Projected stochastic approximation

### `model/projected_sa.py`

- `robbins_monro_step_size`: decaying step size.
- `training_tensors`: materialize training tensors for deterministic sampling.
- `gradient_norm`: global gradient norm.
- `project_parameter_box`: in-place box projection and optional boundary-hit rate.
- `projected_sgd_step`: parameter update followed by projection.

## Synthetic data

### `synthetic_fixed_dgp.py`

- `ToyMixtureParameters`: immutable Exp5 DGP parameters.
- `make_toy_mixture_parameters`: generate fixed mixture parameters once per seed.
- `makettoy_multi_exp`: sample independent observations from fixed parameters.

### `spline_sensitivity_common.py`

- `make_cost_protocol`: reconstruct ten seeds, Metric 1 costs, and Metric 2 costs.
- `build_fixed_dgp_data`: Exp5 train/validation/test generation and alignment hashes.
- `build_van_havre_data`: Van Havre conditional rare-mixture data.
- `build_izbicki_bimodal_data`: all-active conditional bimodal data.
- `build_sensitivity_data`: route `--data_synthetic` to the selected DGP.
- `make_loader`: deterministic PyTorch loaders.
- `model_kwargs`: one shared spline architecture configuration.
- `predict_exact_quantile`: context-wise exact spline inference.
- `evaluate_metric1_metric2`: Metric 1 plus averaged cost-pair Metric 2.
- `_train_model`: dispatch GenDFL, QFR, or RSETO-IPA training.
- `_load_or_train_baseline`: reuse shared baseline checkpoints.
- `_estimate_final_ipa_gradient_variance`: final batch IPA variance calculation.
- `run_sensitivity`: complete data, initialization, training, evaluation, and saving
  workflow for one hyperparameter grid.
- `add_common_arguments`: shared experiment arguments.

## Real-world workflow

### `real_world_d3group_gendfl_common.py`

- `service_level`, `newsvendor_cost`, `metric2_cost`: common evaluation definitions.
- `load_or_create_shared_initialization`: one physical initial checkpoint shared by
  Spline-ETO, Spline-QFR, and RSETO-IPA.
- `make_loaders`, `model_kwargs`: common data and architecture setup.
- `predict_exact_quantiles`: exact context-wise quantiles for all test cost pairs.
- `common_training_signature`: auditable configuration and data hashes.
- `train_or_load`: checkpoint-aware training.
- `evaluate_one_model`: per-group Metric 1/2 evaluation.
- `run_one_dataset`, `run_real_world`: complete four-dataset experiment workflow.

## Legacy conditional-flow interface

### `model/newsvendor_gendfl_conditional_flow.py`

- `ConditionalFlow`: earlier affine scalar conditional flow.
- `pretrain_flow`: source-compatible likelihood training.
- `GenDFLConditionalFlowNewsvendor.train_conditional_flow`: pure conditional-flow
  training through the common generative interface.

### `model/gendfl_1d_interface.py`

- `make_gendfl_1d_model`: construct the one-target GenDFL model.
- `train_gendfl_1d`: public NLL, IPA-regularized, or GLR-regularized training entry.
- `predict_gendfl_1d_quantile`: one-dimensional conditional quantile inference.

### `model/newsvendor_quantile_flow.py`

- `AffineQuantileFlowNewsvendor.quantile`: affine-flow exact quantile.
- `integrated_pinball_loss`: random-quantile objective.
- `train_quantile_flow`: legacy quantile-flow training loop.

### `model/regularized_gradient_trainer.py`

- `RegularizedGenerativeTrainer.fit`: common loop/vmap IPA and GLR regularized
  training used by the legacy VAE, RealNVP, MeanFlow, DDPM, DDIM, and flow models.
