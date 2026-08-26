import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

from synthetic_fixed_dgp import make_toy_mixture_parameters, makettoy_multi_exp
from synthetic_fixed_dgp_traditional_models import (
    BayesConditionalMixtureOracle,
    build_parser,
    build_fixed_dgp_fold,
    build_van_havre_fold,
    evaluate_model,
)


class FixedDGPGeneratorTests(unittest.TestCase):
    def test_component_selects_matching_weight_and_intercept_per_observation(self):
        parameters = make_toy_mixture_parameters(
            num_features=4,
            random_state=19,
            num_exps=5,
            noise_scale=0.0,
        )
        data, returned_weights = makettoy_multi_exp(
            num_samples=500,
            num_features=4,
            random_state=19,
            num_exps=5,
            sample_random_state=10019,
            parameters=parameters,
        )
        context = data[:, :4]
        demand = data[:, 4]
        labels = data[:, 5].astype(int)
        expected = (
            np.einsum("ij,ij->i", context, parameters.weights[labels])
            + parameters.intercepts[labels]
        )
        np.testing.assert_allclose(demand, expected, rtol=0.0, atol=1e-10)
        np.testing.assert_array_equal(returned_weights, parameters.weights)
        self.assertGreater(np.unique(labels).size, 1)

    def test_train_and_test_are_independent_samples_from_one_dgp(self):
        data = build_fixed_dgp_fold(
            train_samples=400,
            test_samples=150,
            validation_size=0.1,
            dim=4,
            num_exps=5,
            random_state=82,
            split_seed=42,
        )
        self.assertEqual(data.X_train.shape[1], 4)
        self.assertEqual(data.X_validation.shape[1], 4)
        self.assertEqual(data.X_test.shape[1], 4)
        self.assertEqual(data.metadata["train_test_exact_context_overlap"], 0)
        self.assertTrue(data.metadata["same_weights"])
        self.assertTrue(data.metadata["same_intercepts"])
        self.assertTrue(data.metadata["same_component_probabilities"])
        self.assertNotEqual(
            data.metadata["train_sample_random_state"],
            data.metadata["test_sample_random_state"],
        )

    def test_default_full_data_are_identical_to_gendfl_data(self):
        from spline_sensitivity_common import build_fixed_dgp_data

        gendfl = build_fixed_dgp_data(
            dim=4,
            fold=0,
            walmart_path="Walmart.csv",
            glr_path="GLR_lr2.py",
        )
        traditional = build_fixed_dgp_fold(
            train_samples=12870,
            test_samples=None,
            validation_size=0.1,
            dim=4,
            num_exps=5,
            random_state=82,
            split_seed=42,
        )
        np.testing.assert_array_equal(
            traditional.X_train, gendfl.train_scaled[:, :-1]
        )
        np.testing.assert_array_equal(
            traditional.y_train, gendfl.train_scaled[:, -1]
        )
        np.testing.assert_array_equal(
            traditional.X_validation, gendfl.validation_scaled[:, :-1]
        )
        np.testing.assert_array_equal(
            traditional.X_test, gendfl.test_scaled[:, :-1]
        )
        np.testing.assert_array_equal(traditional.y_test, gendfl.test_raw[:, -1])

    def test_van_havre_builder_uses_independent_samples_and_alpha_0995(self):
        from spline_sensitivity_common import build_van_havre_data

        with tempfile.TemporaryDirectory() as temporary_directory:
            walmart_path = Path(temporary_directory) / "Walmart.csv"
            pd.DataFrame({"row": np.arange(400)}).to_csv(walmart_path, index=False)
            data = build_van_havre_data(
                dim=4,
                fold=0,
                walmart_path=walmart_path,
                glr_path="GLR_lr2.py",
            )

        self.assertEqual(data.train_raw.shape, (720, 5))
        self.assertEqual(data.validation_raw.shape, (80, 5))
        self.assertEqual(data.test_raw.shape, (360, 5))
        self.assertEqual(
            data.alignment["data_protocol"],
            "literature_van_havre_2015_sim4_conditional_v1",
        )
        self.assertAlmostEqual(data.alignment["target_quantile"], 0.995)
        self.assertEqual(data.alignment["single_cost"], [199.0, -1.0])
        self.assertEqual(
            data.alignment["mixture_parameters"]["raw_component_weights"],
            [0.6, 0.39, 0.01],
        )
        self.assertTrue(
            data.alignment["distribution_alignment"]["independent_sample_seeds"]
        )

    def test_traditional_van_havre_fold_matches_gendfl_arrays(self):
        from spline_sensitivity_common import build_van_havre_data

        with tempfile.TemporaryDirectory() as temporary_directory:
            walmart_path = Path(temporary_directory) / "Walmart.csv"
            pd.DataFrame({"row": np.arange(400)}).to_csv(walmart_path, index=False)
            gendfl = build_van_havre_data(
                dim=4,
                fold=0,
                walmart_path=walmart_path,
                glr_path="GLR_lr2.py",
            )
            traditional = build_van_havre_fold(
                dim=4,
                fold=0,
                reference_data=walmart_path,
                glr_path="GLR_lr2.py",
            )

        np.testing.assert_array_equal(
            traditional.X_train, gendfl.train_scaled[:, :-1]
        )
        np.testing.assert_array_equal(
            traditional.y_train, gendfl.train_scaled[:, -1]
        )
        np.testing.assert_array_equal(
            traditional.X_validation, gendfl.validation_scaled[:, :-1]
        )
        np.testing.assert_array_equal(
            traditional.y_validation, gendfl.validation_scaled[:, -1]
        )
        np.testing.assert_array_equal(
            traditional.X_test, gendfl.test_scaled[:, :-1]
        )
        np.testing.assert_array_equal(
            traditional.y_test, gendfl.test_raw[:, -1]
        )
        self.assertEqual(traditional.metadata["train_test_exact_context_overlap"], 0)

    def test_izbicki_builder_supports_multiple_dimensions_and_alpha_095(self):
        from spline_sensitivity_common import build_izbicki_bimodal_data

        with tempfile.TemporaryDirectory() as temporary_directory:
            walmart_path = Path(temporary_directory) / "Walmart.csv"
            pd.DataFrame({"row": np.arange(400)}).to_csv(walmart_path, index=False)
            data4 = build_izbicki_bimodal_data(
                dim=4,
                fold=0,
                walmart_path=walmart_path,
                glr_path="GLR_lr2.py",
            )
            data9 = build_izbicki_bimodal_data(
                dim=9,
                fold=0,
                walmart_path=walmart_path,
                glr_path="GLR_lr2.py",
            )

        self.assertEqual(data4.train_raw.shape, (720, 5))
        self.assertEqual(data4.validation_raw.shape, (80, 5))
        self.assertEqual(data4.test_raw.shape, (360, 5))
        self.assertEqual(data9.train_raw.shape, (720, 10))
        self.assertEqual(
            data4.alignment["data_protocol"],
            "izbicki_2026_bimodal_full_all_active_projection_v1",
        )
        self.assertEqual(data4.alignment["single_cost"], [19.0, -1.0])
        self.assertAlmostEqual(data4.alignment["target_quantile"], 0.95)
        self.assertEqual(data4.alignment["active_context_dimensions"], 4)
        self.assertEqual(data9.alignment["active_context_dimensions"], 9)
        self.assertEqual(data9.alignment["nuisance_context_dimensions"], 0)
        self.assertTrue(
            data9.alignment["mixture_parameters"]["all_context_dimensions_active"]
        )
        self.assertFalse(
            np.allclose(data4.train_raw[:, -1], data9.train_raw[:, -1])
        )
        self.assertFalse(np.allclose(data4.test_raw[:, -1], data9.test_raw[:, -1]))


class TraditionalMetricTests(unittest.TestCase):
    def test_defaults_match_the_requested_gendfl_protocol(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.data_synthetic, "exp5")
        self.assertEqual(args.num_exps_list, [5])
        self.assertEqual(args.dims, [4, 9, 14, 19, 24])
        self.assertEqual(args.folds, 10)
        self.assertEqual(args.epochs, 50)
        self.assertEqual(args.batch_size, 64)
        self.assertEqual(args.hidden_dim, 64)
        self.assertEqual(args.hidden_layers, 2)
        self.assertEqual(args.learning_rate, 1e-3)
        self.assertEqual(args.step_size_exponent, 0.6)
        self.assertEqual(args.parameter_box_lower, -10.0)
        self.assertEqual(args.parameter_box_upper, 10.0)
        self.assertEqual(args.lgb_n_estimators, 50)
        self.assertEqual(args.n_jobs, 1)
        self.assertIn("oracle_gmm", args.models)

        van_havre = build_parser().parse_args(
            ["--data-synthetic", "van-havre"]
        )
        self.assertEqual(van_havre.data_synthetic, "van-havre")

    def test_oracle_gmm_inverts_true_fixed_dgp_cdf_and_varies_by_cost(self):
        data = build_fixed_dgp_fold(
            train_samples=300,
            test_samples=60,
            validation_size=0.1,
            dim=3,
            num_exps=5,
            random_state=15,
            split_seed=42,
        )
        oracle = BayesConditionalMixtureOracle(
            data=data,
            cost_pair=np.array([4.0, -4.0]),
        )
        costs = np.array([[1.0, -9.0], [5.0, -5.0], [9.0, -1.0]])
        predictions = oracle.predict_for_costs(data.X_test_raw, costs)
        weights, means, sigmas = oracle.conditional_parameters(data.X_test_raw)
        target_quantiles = costs[:, 0] / (costs[:, 0] + np.abs(costs[:, 1]))
        recovered_cdf = np.sum(
            weights[None, :, :]
            * ndtr(
                (predictions[:, :, None] - means[None, :, :])
                / sigmas[None, :, :]
            ),
            axis=2,
        )
        np.testing.assert_allclose(
            recovered_cdf,
            np.broadcast_to(target_quantiles[:, None], recovered_cdf.shape),
            rtol=0.0,
            atol=1e-12,
        )
        self.assertFalse(np.allclose(predictions[0], predictions[2]))

        evaluation = evaluate_model(
            oracle,
            data,
            metric1_cost=np.array([4.0, -4.0]),
            metric2_costs=costs,
        )
        self.assertEqual(evaluation["metric2_predictions"].shape, (3, 60))
        self.assertFalse(
            np.allclose(
                evaluation["metric2_predictions"][0],
                evaluation["metric2_predictions"][2],
            )
        )

    def test_metric2_reuses_one_prediction_for_all_cost_pairs(self):
        class CountingModel:
            def __init__(self):
                self.calls = 0

            def predict(self, context):
                self.calls += 1
                return np.zeros(context.shape[0])

        data = build_fixed_dgp_fold(
            train_samples=200,
            test_samples=40,
            validation_size=0.1,
            dim=3,
            num_exps=5,
            random_state=15,
            split_seed=42,
        )
        model = CountingModel()
        evaluation = evaluate_model(
            model,
            data,
            metric1_cost=np.array([4.0, -4.0]),
            metric2_costs=np.array([[1.0, -9.0], [5.0, -5.0], [9.0, -1.0]]),
        )
        self.assertEqual(model.calls, 1)
        self.assertEqual(evaluation["metric2_predictions"].shape, (3, 40))
        np.testing.assert_array_equal(
            evaluation["metric2_predictions"][0],
            evaluation["metric2_predictions"][2],
        )


if __name__ == "__main__":
    unittest.main()
