import copy
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model.newsvendor_gendfl_conditional_flow import GenDFLConditionalFlowNewsvendor
from model.newsvendor_quantile_flow import (
    AffineQuantileFlowNewsvendor,
    pinball_loss,
)


def make_model(model_class):
    return model_class(
        targetdim=1,
        labeldim=2,
        latent=1,
        data_len=4,
        epoch=2,
        target_quantile=0.7,
        samplingnumber=7,
        cost_under=7.0,
        cost_over=3.0,
    )


class QuantileFlowTest(unittest.TestCase):
    def setUp(self):
        self.condition = torch.tensor(
            [[-0.4, 0.2], [0.1, 0.7], [0.6, -0.2], [-0.5, -0.1]],
            dtype=torch.float32,
        )
        self.target = torch.tensor([[0.5], [-0.3], [0.8], [-0.6]], dtype=torch.float32)

    def test_backbone_and_parameter_count_match_gendfl_exactly(self):
        torch.manual_seed(71)
        gendfl = make_model(GenDFLConditionalFlowNewsvendor)
        quantile_flow = make_model(AffineQuantileFlowNewsvendor)

        self.assertEqual(
            [type(layer) for layer in gendfl.flow.net],
            [type(layer) for layer in quantile_flow.flow.net],
        )
        self.assertEqual(
            [tuple(parameter.shape) for parameter in gendfl.flow.net.parameters()],
            [tuple(parameter.shape) for parameter in quantile_flow.flow.net.parameters()],
        )
        self.assertEqual(gendfl.state_dict().keys(), quantile_flow.state_dict().keys())
        self.assertEqual(
            sum(parameter.numel() for parameter in gendfl.parameters()),
            sum(parameter.numel() for parameter in quantile_flow.parameters()),
        )
        quantile_flow.load_state_dict(copy.deepcopy(gendfl.state_dict()), strict=True)
        for left, right in zip(gendfl.parameters(), quantile_flow.parameters()):
            torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)

    def test_quantile_formula_and_monotonicity(self):
        torch.manual_seed(72)
        model = make_model(AffineQuantileFlowNewsvendor)
        tau = torch.tensor([0.1, 0.5, 0.9])
        quantiles = model.quantile(tau, self.condition)

        output = model.flow.net(self.condition)
        location = output[:, 0:1]
        scale = model.sigma_min + F.softplus(output[:, 1:2])
        normal = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))
        expected = location[:, None, :] + scale[:, None, :] * normal.icdf(tau)[None, :, None]
        torch.testing.assert_close(quantiles, expected)
        self.assertTrue(torch.all(quantiles[:, 1:, :] > quantiles[:, :-1, :]))
        torch.testing.assert_close(
            model.critical_quantile_decision(self.condition),
            model.quantile(0.7, self.condition)[:, 0, :],
        )

    def test_integrated_pinball_matches_manual_loss(self):
        torch.manual_seed(73)
        model = make_model(AffineQuantileFlowNewsvendor)
        tau = torch.tensor([0.25, 0.75])
        result = model.integrated_pinball_loss(
            self.target,
            self.condition,
            tau=tau,
        )
        manual = pinball_loss(
            self.target[:, None, :],
            model.quantile(tau, self.condition),
            tau[None, :, None],
        ).mean()
        torch.testing.assert_close(result["loss"], manual)

    def test_training_updates_quantile_flow_and_returns_finite_inference(self):
        torch.manual_seed(74)
        model = make_model(AffineQuantileFlowNewsvendor)
        loader = DataLoader(
            TensorDataset(self.condition, self.target),
            batch_size=2,
            shuffle=False,
        )
        parameters_before = [parameter.detach().clone() for parameter in model.parameters()]
        history = model.train_quantile_flow(
            loader,
            loader,
            num_epochs=2,
            learning_rate=1e-3,
            early_stopping=2,
            num_quantile_levels=4,
            validation_quantile_levels=5,
        )

        self.assertTrue(np.isfinite(history["best_val_qfr_loss"]))
        self.assertGreaterEqual(history["epochs_ran"], 1)
        self.assertTrue(
            any(
                not torch.equal(before, after.detach())
                for before, after in zip(parameters_before, model.parameters())
            )
        )
        self.assertTrue(torch.isfinite(model.critical_quantile_decision(self.condition)).all())


if __name__ == "__main__":
    unittest.main()
