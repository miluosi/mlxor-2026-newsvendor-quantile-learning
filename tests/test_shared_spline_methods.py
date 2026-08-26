import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch.utils.data import DataLoader, TensorDataset

from gendfl_lambda import build_parser as build_lambda_parser
from gendfl_m import build_parser as build_m_parser
from gendfl_simulation_num import build_parser as build_simulation_parser
from model.gendfl_spline import GenDFLSplineNewsvendor
from model.projected_sa import projected_sgd_step, robbins_monro_step_size
from model.rseto_ipa_spline import (
    RSETOIPASplineNewsvendor,
    increasing_sample_size,
    screen_selected_base_noise,
)
from model.shared_spline_flow import SharedConditionalSplineFlow, SplineFlowConfig
from model.spline_qfr import SplineQFRNewsvendor
from real_world_d3group_gendfl_benchmark import build_parser as build_real_benchmark_parser
from real_world_d3group_gendfl_common import (
    load_or_create_shared_initialization,
    set_seed,
)
from real_world_d3group_gendfl_sqeto_ipa import build_parser as build_rseto_parser


def model_kwargs():
    return {
        "targetdim": 1,
        "labeldim": 3,
        "latent": 1,
        "data_len": 16,
        "epoch": 2,
        "target_quantile": 0.7,
        "cost_under": 7.0,
        "cost_over": 3.0,
        "num_transforms": 2,
        "num_bins": 8,
        "hidden_dim": 16,
        "hidden_layers": 2,
    }


class SharedSplineBackboneTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.config = SplineFlowConfig(
            context_dim=3,
            num_transforms=2,
            num_bins=8,
            hidden_dim=16,
            hidden_layers=2,
        )
        self.flow = SharedConditionalSplineFlow(self.config)
        self.context = torch.randn(4, 3)

    def test_inverse_logdet_monotonicity_and_log_prob(self):
        z = torch.linspace(-3.5, 3.5, 41).reshape(1, 41, 1).expand(4, -1, -1)
        target, forward_logdet = self.flow.base_to_data(z, self.context)
        reconstructed, inverse_logdet = self.flow.data_to_base(target, self.context)

        self.assertLess(float((reconstructed - z).abs().max().detach()), 2e-5)
        self.assertLess(float((forward_logdet + inverse_logdet).abs().max().detach()), 2e-5)
        self.assertTrue(torch.all(target[:, 1:] > target[:, :-1]))
        self.assertTrue(torch.isfinite(self.flow.log_prob(target, self.context)).all())

    def test_exact_quantiles_are_monotone_and_sampling_is_explicit(self):
        tau = torch.linspace(0.01, 0.99, 31).reshape(1, 31, 1).expand(4, -1, -1)
        quantiles = self.flow.quantile(self.context, tau)
        self.assertTrue(torch.all(quantiles[:, 1:] > quantiles[:, :-1]))

        base_noise = torch.randn(4, 3, 7, 1)
        first = self.flow.sample_from_base_noise(self.context, base_noise)
        second = self.flow.sample_from_base_noise(self.context, base_noise)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        self.assertTrue(first.requires_grad)

    def test_condition_network_runs_once_per_context_not_per_simulation(self):
        observed_shapes = []
        hooks = [
            layer.parameter_net.register_forward_hook(
                lambda _module, inputs, _output: observed_shapes.append(inputs[0].shape)
            )
            for layer in self.flow.layers
        ]
        try:
            self.flow.sample_from_base_noise(
                self.context,
                torch.randn(4, 3, 7, 1),
            )
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(observed_shapes, [self.context.shape] * len(self.flow.layers))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cpu_gpu_agreement(self):
        gpu_flow = SharedConditionalSplineFlow(self.config).cuda()
        gpu_flow.load_state_dict(copy.deepcopy(self.flow.state_dict()))
        z = torch.randn(4, 11, 1)
        cpu_result = self.flow.sample_from_base_noise(self.context, z)
        gpu_result = gpu_flow.sample_from_base_noise(self.context.cuda(), z.cuda()).cpu()
        torch.testing.assert_close(cpu_result, gpu_result, rtol=2e-4, atol=2e-5)


class SharedSplineMethodTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        kwargs = model_kwargs()
        base = GenDFLSplineNewsvendor(**kwargs)
        initial_state = copy.deepcopy(base.state_dict())
        self.models = [
            base,
            SplineQFRNewsvendor(**kwargs),
            RSETOIPASplineNewsvendor(**kwargs),
        ]
        for model in self.models[1:]:
            model.load_state_dict(copy.deepcopy(initial_state), strict=True)
        self.context = torch.randn(6, 3)
        self.demand = (0.5 * self.context[:, :1] + 0.1 * torch.randn(6, 1))

    def test_equal_independent_initialization_and_architecture(self):
        counts = [sum(parameter.numel() for parameter in model.parameters()) for model in self.models]
        self.assertEqual(len(set(counts)), 1)
        self.assertEqual(
            [tuple(model.state_dict().keys()) for model in self.models].count(
                tuple(self.models[0].state_dict().keys())
            ),
            3,
        )
        for left, right in zip(self.models[0].parameters(), self.models[1].parameters()):
            self.assertIsNot(left, right)
            torch.testing.assert_close(left, right)

        z = torch.randn(6, 9, 1)
        outputs = [model.sample_from_base_noise(self.context, z) for model in self.models]
        torch.testing.assert_close(outputs[0], outputs[1])
        torch.testing.assert_close(outputs[0], outputs[2])
        for model in self.models:
            forbidden = (torch.nn.modules.batchnorm._BatchNorm, torch.nn.Dropout)
            self.assertFalse(any(isinstance(module, forbidden) for module in model.modules()))

    def test_real_world_models_use_one_physical_initial_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            args = SimpleNamespace(
                shared_initialization_dir=Path(temporary_directory),
                feature_combi="calendar",
                num_transforms=2,
                num_bins=8,
                hidden_dim=16,
                hidden_layers=2,
                tail_bound=4.0,
            )
            paths = []
            hashes = []
            for model_class in (
                GenDFLSplineNewsvendor,
                SplineQFRNewsvendor,
                RSETOIPASplineNewsvendor,
            ):
                set_seed(47)
                model = model_class(**model_kwargs())
                path, state_hash = load_or_create_shared_initialization(
                    model,
                    args,
                    dataset="bakery",
                    feature_combo=["calendar"],
                    group=("store", "item"),
                    group_index=0,
                    cost_under=9.0,
                    cost_over=1.0,
                    seed=47,
                )
                paths.append(path)
                hashes.append(state_hash)
            self.assertEqual(len(set(paths)), 1)
            self.assertEqual(len(set(hashes)), 1)
            self.assertTrue(paths[0].exists())

    def test_real_world_early_stopping_default_is_fifty(self):
        for build_parser in (build_real_benchmark_parser, build_rseto_parser):
            args = build_parser().parse_args([])
            self.assertEqual(args.epochs, 100)
            self.assertTrue(args.use_early_stopping)
            self.assertEqual(args.early_stopping, 50)
            self.assertEqual(args.batch_size, 64)
            self.assertEqual(args.learning_rate, 1e-3)
            self.assertEqual(args.step_size_exponent, 0.6)
            self.assertEqual(args.parameter_box_lower, -10.0)
            self.assertEqual(args.parameter_box_upper, 10.0)

        rseto_args = build_rseto_parser().parse_args([])
        self.assertEqual(rseto_args.samples_per_replication, 128)
        self.assertEqual(rseto_args.max_simulation_values, 1048576)
        self.assertEqual(rseto_args.diagnostic_interval, 100)
        self.assertEqual(rseto_args.finite_check_interval, 100)
        self.assertTrue(rseto_args.train_data_on_device)
        benchmark_args = build_real_benchmark_parser().parse_args([])
        self.assertFalse(hasattr(benchmark_args, "gendfl_scenarios"))
        self.assertFalse(hasattr(benchmark_args, "gendfl_decision_weight"))

    def test_sensitivity_training_control_defaults_and_aliases(self):
        for build_parser in (
            build_lambda_parser,
            build_m_parser,
            build_simulation_parser,
        ):
            parser = build_parser()
            defaults = parser.parse_args([])
            self.assertEqual(defaults.epochs, 50)
            self.assertFalse(defaults.use_early_stopping)
            self.assertEqual(defaults.early_stopping, 20)
            self.assertEqual(defaults.samples_per_replication, 128)
            self.assertEqual(defaults.batch_size, 64)
            self.assertEqual(defaults.progress_interval, 10)
            self.assertEqual(defaults.data_synthetic, "exp5")
            self.assertEqual(defaults.gradient_variance_repeats, 8)
            self.assertEqual(defaults.gradient_variance_batch_size, 16)

            configured = parser.parse_args(
                [
                    "--epochs",
                    "35",
                    "--ifearflystop",
                    "--early-stopping-patience",
                    "7",
                    "--mnum",
                    "256",
                    "--batch-size",
                    "32",
                    "--progress-interval",
                    "5",
                    "--data_synthetic",
                    "van-havre",
                ]
            )
            self.assertEqual(configured.epochs, 35)
            self.assertTrue(configured.use_early_stopping)
            self.assertEqual(configured.early_stopping, 7)
            self.assertEqual(configured.samples_per_replication, 256)
            self.assertEqual(configured.batch_size, 32)
            self.assertEqual(configured.progress_interval, 5)
            self.assertEqual(configured.data_synthetic, "van-havre")

            izbicki = parser.parse_args(
                ["--data_synthetic", "izbicki-bimodal"]
            )
            self.assertEqual(izbicki.data_synthetic, "izbicki-bimodal")

    def test_real_world_training_control_aliases(self):
        benchmark = build_real_benchmark_parser().parse_args(
            [
                "--epochs",
                "35",
                "--no-use-early-stopping",
                "--early-stopping-patience",
                "7",
                "--batch-size",
                "32",
            ]
        )
        self.assertEqual(benchmark.epochs, 35)
        self.assertFalse(benchmark.use_early_stopping)
        self.assertEqual(benchmark.early_stopping, 7)
        self.assertEqual(benchmark.batch_size, 32)
        self.assertFalse(hasattr(benchmark, "gendfl_scenarios"))

        sqeto = build_rseto_parser().parse_args(
            [
                "--epochs",
                "35",
                "--no-use-early-stopping",
                "--early-stopping-patience",
                "7",
                "--batch-size",
                "32",
                "--mnum",
                "256",
            ]
        )
        self.assertEqual(sqeto.epochs, 35)
        self.assertFalse(sqeto.use_early_stopping)
        self.assertEqual(sqeto.early_stopping, 7)
        self.assertEqual(sqeto.batch_size, 32)
        self.assertEqual(sqeto.samples_per_replication, 256)

    def test_all_objectives_are_finite_and_differentiable(self):
        gendfl, qfr, ipa = self.models
        gendfl_loss = gendfl.generative_loss(self.demand, self.context)
        qfr_loss, qfr_details = qfr.qfr_objective(self.context, self.demand, num_tau=7)
        base_noise = torch.randn(6, 4, 9, 1)
        ipa_loss, ipa_details = ipa.rseto_ipa_objective(
            self.context,
            self.demand,
            replications=4,
            samples_per_replication=9,
            smoothing_mu=0.1,
            fidelity_weight=0.5,
            base_noise=base_noise,
        )

        self.assertTrue(torch.all(qfr_details["tau"] >= qfr.tau_eps))
        self.assertTrue(torch.all(qfr_details["tau"] <= 1.0 - qfr.tau_eps))
        qfr_residual = self.demand[:, None, :] - qfr_details["quantiles"]
        qfr_manual = torch.maximum(
            qfr_details["tau"] * qfr_residual,
            (qfr_details["tau"] - 1.0) * qfr_residual,
        ).mean()
        torch.testing.assert_close(qfr_loss, qfr_manual)
        self.assertEqual(ipa_details["base_noise"].shape, (6, 4, 9, 1))
        self.assertEqual(ipa_details["generated"].shape, (6, 4, 9, 1))
        self.assertEqual(ipa_details["selected_quantile"].shape, (6, 4))
        manual_order = torch.kthvalue(
            ipa_details["generated"].squeeze(-1),
            k=ipa_details["order_index"],
            dim=-1,
        ).values
        torch.testing.assert_close(manual_order, ipa_details["selected_quantile"])

        for model, loss in zip(self.models, (gendfl_loss, qfr_loss, ipa_loss)):
            model.zero_grad(set_to_none=True)
            loss.backward()
            gradients = [parameter.grad for parameter in model.parameters()]
            self.assertTrue(any(gradient is not None for gradient in gradients))
            self.assertTrue(
                all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
            )

    def test_gendfl_training_uses_only_conditional_nll(self):
        model = self.models[0]
        loader = DataLoader(
            TensorDataset(self.context, self.demand),
            batch_size=3,
            shuffle=False,
        )
        with mock.patch.object(
            model.backbone,
            "sample",
            side_effect=AssertionError("GenDFL must not sample training scenarios."),
        ), mock.patch.object(
            model,
            "newsvendor_loss",
            side_effect=AssertionError("GenDFL must not optimize newsvendor loss."),
        ):
            history = model.train_gendfl_spline(
                loader,
                loader,
                num_epochs=1,
                early_stopping=1,
                stop_early=False,
                restore_best=False,
            )

        self.assertEqual(history["training_objective"], "conditional_nll_only")
        self.assertIn("train_nll", history)
        self.assertNotIn("train_decision", history)

    def test_vectorized_ipa_matches_replication_loop(self):
        ipa = self.models[2]
        base_noise = torch.randn(6, 3, 8, 1)
        vectorized, vectorized_details = ipa.rseto_ipa_objective(
            self.context,
            self.demand,
            replications=3,
            samples_per_replication=8,
            smoothing_mu=0.07,
            fidelity_weight=0.4,
            base_noise=base_noise,
        )
        loop_totals = []
        loop_quantiles = []
        for replication in range(3):
            total, details = ipa.rseto_ipa_objective(
                self.context,
                self.demand,
                replications=1,
                samples_per_replication=8,
                smoothing_mu=0.07,
                fidelity_weight=0.4,
                base_noise=base_noise[:, replication : replication + 1],
            )

            loop_totals.append(total)
            loop_quantiles.append(details["selected_quantile"])
        loop_total = torch.stack(loop_totals).mean()
        loop_quantile = torch.cat(loop_quantiles, dim=1)
        torch.testing.assert_close(vectorized, loop_total, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            vectorized_details["selected_quantile"],
            loop_quantile,
            rtol=1e-5,
            atol=1e-6,
        )
        vectorized_gradients = torch.autograd.grad(
            vectorized,
            tuple(ipa.parameters()),
            retain_graph=True,
        )
        loop_gradients = torch.autograd.grad(loop_total, tuple(ipa.parameters()))
        for vectorized_gradient, loop_gradient in zip(
            vectorized_gradients,
            loop_gradients,
        ):
            torch.testing.assert_close(
                vectorized_gradient,
                loop_gradient,
                rtol=2e-5,
                atol=2e-6,
            )

    def test_batch_ipa_gradient_variance_is_finite_and_deterministic(self):
        model = self.models[2]
        first = model.estimate_batch_ipa_gradient_variance(
            self.context[:3],
            self.demand[:3],
            replications=2,
            samples_per_replication=7,
            smoothing_mu=0.1,
            diagnostic_repeats=4,
            max_simulation_values=64,
            seed=991,
        )
        second = model.estimate_batch_ipa_gradient_variance(
            self.context[:3],
            self.demand[:3],
            replications=2,
            samples_per_replication=7,
            smoothing_mu=0.1,
            diagnostic_repeats=4,
            max_simulation_values=64,
            seed=991,
        )
        self.assertGreaterEqual(first["ipa_gradient_variance_trace"], 0.0)
        self.assertGreater(first["ipa_gradient_mean_norm"], 0.0)
        self.assertEqual(first["gradient_variance_replications"], 2)
        self.assertEqual(first["gradient_variance_samples_per_replication"], 7)
        self.assertAlmostEqual(
            first["ipa_gradient_variance_trace"],
            second["ipa_gradient_variance_trace"],
            places=10,
        )

    def test_screen_replay_matches_full_graph_gradient_and_projected_step(self):
        reference = self.models[2]
        accelerated = RSETOIPASplineNewsvendor(**model_kwargs())
        accelerated.load_state_dict(copy.deepcopy(reference.state_dict()), strict=True)
        base_noise = torch.randn(6, 3, 9, 1)

        reference_loss, reference_details = reference.rseto_ipa_objective(
            self.context,
            self.demand,
            replications=3,
            samples_per_replication=9,
            smoothing_mu=0.07,
            fidelity_weight=0.4,
            base_noise=base_noise,
        )
        reference_parameters = tuple(reference.parameters())
        reference_gradients = torch.autograd.grad(
            reference_loss,
            reference_parameters,
        )

        selected_noise, screened_quantile, screening = screen_selected_base_noise(
            accelerated.backbone,
            self.context,
            replications=3,
            samples_per_replication=9,
            target_quantile=accelerated.target_quantile,
            max_simulation_values=18,
            base_noise=base_noise,
            collect_diagnostics=True,
            finite_check=True,
        )
        accelerated_loss, accelerated_details = accelerated.rseto_ipa_replay_objective(
            self.context,
            self.demand,
            selected_noise=selected_noise,
            smoothing_mu=0.07,
            fidelity_weight=0.4,
        )
        accelerated_parameters = tuple(accelerated.parameters())
        accelerated_gradients = torch.autograd.grad(
            accelerated_loss,
            accelerated_parameters,
        )

        self.assertTrue(bool(screening["finite"]))
        self.assertGreater(screening["chunk_count"], 1)
        self.assertTrue(
            torch.equal(
                screening["selected_index"],
                reference_details["selected_index"],
            )
        )
        torch.testing.assert_close(
            screened_quantile,
            reference_details["selected_quantile"],
        )
        torch.testing.assert_close(
            accelerated_details["selected_quantile"],
            reference_details["selected_quantile"],
        )
        torch.testing.assert_close(accelerated_loss, reference_loss)
        for reference_gradient, accelerated_gradient in zip(
            reference_gradients,
            accelerated_gradients,
        ):
            torch.testing.assert_close(
                accelerated_gradient,
                reference_gradient,
                rtol=2e-5,
                atol=2e-6,
            )

        for parameter, gradient in zip(reference_parameters, reference_gradients):
            parameter.grad = gradient
        for parameter, gradient in zip(accelerated_parameters, accelerated_gradients):
            parameter.grad = gradient
        projected_sgd_step(reference_parameters, 1e-3, -5.0, 5.0)
        projected_sgd_step(accelerated_parameters, 1e-3, -5.0, 5.0)
        for reference_parameter, accelerated_parameter in zip(
            reference_parameters,
            accelerated_parameters,
        ):
            torch.testing.assert_close(
                accelerated_parameter,
                reference_parameter,
                rtol=2e-5,
                atol=2e-6,
            )

    def test_screen_replay_lambda_endpoints_skip_unused_objective(self):
        model = self.models[2]
        base_noise = torch.randn(6, 2, 7, 1)
        selected_noise, _, _ = screen_selected_base_noise(
            model.backbone,
            self.context,
            replications=2,
            samples_per_replication=7,
            target_quantile=model.target_quantile,
            max_simulation_values=64,
            base_noise=base_noise,
        )

        ipa_only, ipa_details = model.rseto_ipa_replay_objective(
            self.context,
            self.demand,
            selected_noise=selected_noise,
            smoothing_mu=0.1,
            fidelity_weight=0.0,
        )
        self.assertEqual(float(ipa_details["fidelity_loss"]), 0.0)
        self.assertTrue(ipa_only.requires_grad)

        fidelity_only, fidelity_details = model.rseto_ipa_replay_objective(
            self.context,
            self.demand,
            selected_noise=None,
            smoothing_mu=0.1,
            fidelity_weight=1.0,
        )
        self.assertEqual(float(fidelity_details["ipa_task_loss"]), 0.0)
        self.assertIsNone(fidelity_details["selected_quantile"])
        self.assertTrue(fidelity_only.requires_grad)

    def test_theorem_schedules_and_kthvalue_gradient_route(self):
        step_sizes = [robbins_monro_step_size(k, 0.1, 0.6) for k in range(20)]
        sample_sizes = [increasing_sample_size(k, 8, 1.0, 0.5) for k in range(20)]
        self.assertTrue(all(left > right for left, right in zip(step_sizes, step_sizes[1:])))
        self.assertEqual(sample_sizes[0], 8)
        self.assertGreater(sample_sizes[-1], sample_sizes[0])

        values = torch.tensor(
            [[[0.4, -1.0, 2.0, 0.2]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        kth = torch.kthvalue(values, k=2, dim=-1)
        kth.values.sum().backward()
        expected = torch.zeros_like(values)
        expected.scatter_(-1, kth.indices.unsqueeze(-1), 1.0)
        torch.testing.assert_close(values.grad, expected)

    def test_default_projected_training_protocol_across_all_methods(self):
        loader = DataLoader(
            TensorDataset(self.context, self.demand),
            batch_size=3,
            shuffle=False,
        )
        common = {
            "num_epochs": 3,
            "learning_rate": 1e-3,
            "step_size_exponent": 0.6,
            "training_seed": 101,
            "parameter_box_lower": -5.0,
            "parameter_box_upper": 5.0,
            "stop_early": False,
            "restore_best": False,
            "early_stopping": 1,
            "warmup_epochs": 0,
        }
        histories = [
            self.models[0].train_gendfl_spline(
                loader,
                loader,
                **common,
            ),
            self.models[1].train_spline_qfr(
                loader,
                loader,
                num_tau=5,
                validation_num_tau=7,
                **common,
            ),
            self.models[2].train_rseto_ipa_spline(
                loader,
                loader,
                replications=2,
                samples_per_replication=7,
                m_growth=1.0,
                m_growth_exponent=0.5,
                max_simulation_values=256,
                **common,
            ),
        ]
        for history in histories:
            self.assertEqual(history["optimizer"], "projected_sgd")
            self.assertEqual(history["epochs_ran"], 3)
            self.assertEqual(history["steps_per_epoch"], 2)
            self.assertEqual(history["steps_ran"], 6)
            self.assertEqual(history["step_size_exponent"], 0.6)
        for index in (1, 2):
            self.assertEqual(
                histories[0]["step_size_first"],
                histories[index]["step_size_first"],
            )
            self.assertEqual(
                histories[0]["step_size_last"],
                histories[index]["step_size_last"],
            )
        for model in self.models:
            for parameter in model.parameters():
                self.assertTrue(torch.all(parameter >= -5.0))
                self.assertTrue(torch.all(parameter <= 5.0))

    def test_chunked_rseto_training_preserves_protocol_and_finiteness(self):
        loader = DataLoader(
            TensorDataset(self.context, self.demand),
            batch_size=3,
            shuffle=False,
        )
        first = self.models[2]
        second = RSETOIPASplineNewsvendor(**model_kwargs())
        second.load_state_dict(copy.deepcopy(first.state_dict()), strict=True)
        common = {
            "num_epochs": 1,
            "learning_rate": 1e-3,
            "step_size_exponent": 0.6,
            "stop_early": False,
            "restore_best": False,
            "early_stopping": 1,
            "replications": 3,
            "samples_per_replication": 7,
            "m_growth": 1.0,
            "m_growth_exponent": 0.5,
            "training_seed": 211,
        }
        first_history = first.train_rseto_ipa_spline(
            loader,
            loader,
            max_simulation_values=4096,
            **common,
        )
        second_history = second.train_rseto_ipa_spline(
            loader,
            loader,
            max_simulation_values=24,
            **common,
        )
        self.assertEqual(first_history["steps_ran"], second_history["steps_ran"])
        self.assertEqual(first_history["m_last"], second_history["m_last"])
        self.assertEqual(first_history["step_size_last"], second_history["step_size_last"])
        for model in (first, second):
            for parameter in model.parameters():
                self.assertTrue(torch.isfinite(parameter).all())

    def test_epoch_progress_callback_does_not_change_rseto_training(self):
        loader = DataLoader(
            TensorDataset(self.context, self.demand),
            batch_size=3,
            shuffle=False,
        )
        without_callback = self.models[2]
        with_callback = RSETOIPASplineNewsvendor(**model_kwargs())
        with_callback.load_state_dict(
            copy.deepcopy(without_callback.state_dict()),
            strict=True,
        )
        common = {
            "num_epochs": 2,
            "learning_rate": 1e-3,
            "step_size_exponent": 0.6,
            "stop_early": False,
            "restore_best": False,
            "early_stopping": 2,
            "replications": 2,
            "samples_per_replication": 7,
            "m_growth": 1.0,
            "m_growth_exponent": 0.5,
            "max_simulation_values": 64,
            "training_seed": 211,
        }
        callback_reports = []
        reference_history = without_callback.train_rseto_ipa_spline(
            loader,
            loader,
            **common,
        )
        callback_history = with_callback.train_rseto_ipa_spline(
            loader,
            loader,
            epoch_callback=lambda **report: callback_reports.append(report),
            **common,
        )

        self.assertEqual(repr(reference_history), repr(callback_history))
        self.assertEqual(len(callback_reports), common["num_epochs"])
        for name, reference_tensor in without_callback.state_dict().items():
            torch.testing.assert_close(
                reference_tensor,
                with_callback.state_dict()[name],
                rtol=0.0,
                atol=0.0,
            )

    def test_exact_oracle_has_finite_gradient(self):
        ipa = self.models[2]
        loss, quantile = ipa.exact_task_objective(
            self.context,
            self.demand,
            smoothing_mu=0.1,
        )
        gradients = torch.autograd.grad(loss, tuple(ipa.parameters()))
        self.assertEqual(quantile.shape, self.demand.shape)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_exact_validation_matches_manual_unsmoothed_cost(self):
        loader = DataLoader(
            TensorDataset(self.context, self.demand),
            batch_size=4,
            shuffle=False,
        )
        for model in self.models:
            with torch.no_grad():
                exact_quantile = model.critical_quantile_decision(self.context)
                manual = (
                    model.cu * torch.relu(self.demand - exact_quantile)
                    + model.co * torch.relu(exact_quantile - self.demand)
                ).mean()
            metrics = model.evaluate_exact_newsvendor(loader)
            self.assertAlmostEqual(metrics["newsvendor_loss"], float(manual), places=6)
            self.assertEqual(metrics["count"], self.demand.numel())

    def test_all_trainers_early_stop_on_exact_newsvendor_loss(self):
        loader = DataLoader(
            TensorDataset(self.context, self.demand),
            batch_size=3,
            shuffle=False,
        )
        common = {
            "num_epochs": 10,
            "early_stopping": 2,
            "warmup_epochs": 1,
            "min_delta_relative": 1e6,
        }
        histories = [
            self.models[0].train_gendfl_spline(
                loader,
                loader,
                **common,
            ),
            self.models[1].train_spline_qfr(
                loader,
                loader,
                num_tau=5,
                validation_num_tau=7,
                **common,
            ),
            self.models[2].train_rseto_ipa_spline(
                loader,
                loader,
                replications=2,
                samples_per_replication=7,
                **common,
            ),
        ]
        for history in histories:
            self.assertEqual(history["epochs_ran"], 4)
            self.assertEqual(history["best_epoch"], 1)
            self.assertTrue(torch.isnan(torch.tensor(history["val_newsvendor"][0])))
            self.assertAlmostEqual(
                history["best_val_newsvendor"],
                history["val_newsvendor"][1],
                places=7,
            )

    def test_tiny_training_loops(self):
        loader = DataLoader(
            TensorDataset(self.context, self.demand),
            batch_size=3,
            shuffle=False,
        )
        gendfl_history = self.models[0].train_gendfl_spline(
            loader,
            loader,
            num_epochs=1,
            early_stopping=1,
        )
        qfr_history = self.models[1].train_spline_qfr(
            loader,
            loader,
            num_epochs=1,
            early_stopping=1,
            num_tau=5,
            validation_num_tau=7,
        )
        ipa_history = self.models[2].train_rseto_ipa_spline(
            loader,
            loader,
            num_epochs=1,
            early_stopping=1,
            replications=2,
            samples_per_replication=7,
        )
        self.assertEqual(gendfl_history["epochs_ran"], 1)
        self.assertEqual(qfr_history["epochs_ran"], 1)
        self.assertEqual(ipa_history["epochs_ran"], 1)


if __name__ == "__main__":
    unittest.main()
