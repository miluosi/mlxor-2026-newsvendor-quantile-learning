import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model.newsvendor_ddim import ConditionalDDIMNewsvendor
from model.newsvendor_ddpm import ConditionalDDPMNewsvendor
from model.newsvendor_gendfl_conditional_flow import GenDFLConditionalFlowNewsvendor
from model.newsvendor_mean_flow import ConditionalMeanFlowNewsvendor
from model.newsvendor_realnvp import ConditionalRealNVPNewsvendor
from model.newsvendor_vae import ConditionalVAENewsvendor


def build_models(data_len=4):
    common = {
        "targetdim": 1,
        "labeldim": 2,
        "data_len": data_len,
        "epoch": 1,
        "quantiles": 0.6,
        "target_quantile": 0.6,
        "samplingnumber": 3,
        "hidden_dim": 4,
        "cost_under": 3.0,
        "cost_over": 2.0,
    }
    gendfl_common = dict(common)
    gendfl_common.pop("hidden_dim")
    return [
        ConditionalVAENewsvendor(latent=2, **common),
        ConditionalRealNVPNewsvendor(**common),
        ConditionalMeanFlowNewsvendor(**common),
        ConditionalDDPMNewsvendor(T=2, **common),
        ConditionalDDIMNewsvendor(T=2, tau=1, **common),
        GenDFLConditionalFlowNewsvendor(**gendfl_common),
    ]


class RegularizedGradientTrainerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.condition = torch.tensor(
            [[-0.4, 0.2], [0.1, 0.7]],
            dtype=torch.float32,
        )
        self.target = torch.tensor([[0.5], [-0.3]], dtype=torch.float32)

    def test_all_models_expose_both_regularized_backends(self):
        expected_methods = (
            "train_regularized_ipa_vmap",
            "train_regularized_ipa_loop",
            "train_regularized_glr_vmap",
            "train_regularized_glr_loop",
        )
        for model in build_models():
            for method_name in expected_methods:
                self.assertTrue(hasattr(model, method_name), f"{type(model).__name__}.{method_name}")

    def test_gendfl_scalar_flow_interface_matches_source(self):
        model = GenDFLConditionalFlowNewsvendor(
            targetdim=1,
            labeldim=2,
            latent=1,
            data_len=2,
            target_quantile=0.6,
            samplingnumber=5,
        )
        samples = model.sample(5, self.condition)
        z, log_det = model.transform_to_noise(self.target, self.condition)
        loss = model.generative_loss(self.target, self.condition)
        gradients = torch.autograd.grad(loss, tuple(model.parameters()))

        self.assertEqual(tuple(samples.shape), (2, 5))
        self.assertFalse(samples.requires_grad)
        self.assertEqual(tuple(z.shape), (2, 1))
        self.assertEqual(tuple(log_det.shape), (2,))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_gendfl_pure_conditional_flow_training(self):
        model = GenDFLConditionalFlowNewsvendor(
            targetdim=1,
            labeldim=2,
            latent=1,
            data_len=4,
            epoch=2,
            target_quantile=0.6,
            samplingnumber=3,
        )
        conditions = torch.tensor(
            [[-0.4, 0.2], [0.1, 0.7], [0.6, -0.2], [-0.5, -0.1]],
            dtype=torch.float32,
        )
        targets = torch.tensor([[0.5], [-0.3], [0.8], [-0.6]], dtype=torch.float32)
        loader = DataLoader(TensorDataset(conditions, targets), batch_size=2, shuffle=False)
        parameters_before = [parameter.detach().clone() for parameter in model.parameters()]

        history = model.pretrain_flow(
            loader,
            loader,
            num_epochs=2,
            learning_rate=1e-3,
            early_stopping=2,
        )

        self.assertEqual(history["epochs_ran"], 2)
        self.assertEqual(set(history).intersection({"regularizer_loss", "newsvendor_loss"}), set())
        self.assertTrue(np.isfinite(history["best_val_nll"]))
        self.assertTrue(
            any(
                not torch.equal(before, after.detach())
                for before, after in zip(parameters_before, model.parameters())
            )
        )

    def test_ipa_vmap_matches_loop_for_every_generator(self):
        for model in build_models(data_len=2):
            latent_samples = torch.randn(3, 2, 5, model.latent)
            vectorized = model.batched_ipa_regularizer(
                self.condition,
                self.target,
                k=3,
                num_samples=5,
                use_vmap=True,
                vmap_chunk_size=2,
                latent_samples=latent_samples,
            )
            vectorized_gradients = torch.autograd.grad(
                vectorized["loss"],
                tuple(model.parameters()),
                allow_unused=True,
            )
            loop = model.batched_ipa_regularizer(
                self.condition,
                self.target,
                k=3,
                num_samples=5,
                use_vmap=False,
                latent_samples=latent_samples,
            )
            loop_gradients = torch.autograd.grad(
                loop["loss"],
                tuple(model.parameters()),
                allow_unused=True,
            )

            torch.testing.assert_close(vectorized["loss"], loop["loss"], rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(
                vectorized["order_quantiles"],
                loop["order_quantiles"],
                rtol=1e-6,
                atol=1e-6,
            )
            self.assertEqual(vectorized["order_index"], 3)
            for vectorized_gradient, loop_gradient in zip(vectorized_gradients, loop_gradients):
                if vectorized_gradient is None or loop_gradient is None:
                    self.assertIsNone(vectorized_gradient)
                    self.assertIsNone(loop_gradient)
                else:
                    torch.testing.assert_close(
                        vectorized_gradient,
                        loop_gradient,
                        rtol=1e-5,
                        atol=1e-5,
                    )

    def test_glr_vmap_matches_loop_for_every_generator(self):
        q_values = torch.tensor([[0.3], [0.2]], dtype=torch.float32)
        for model in build_models(data_len=2):
            latent_samples = torch.randn(2, model.latent)
            latent_dimensions = torch.arange(2) % model.latent
            vectorized = model._glr_innerloop(
                self.condition,
                self.target,
                q_values,
                use_vmap=True,
                vmap_chunk_size=2,
                latent_samples=latent_samples,
                latent_dimensions=latent_dimensions,
            )
            loop = model._glr_innerloop(
                self.condition,
                self.target,
                q_values,
                use_vmap=False,
                latent_samples=latent_samples,
                latent_dimensions=latent_dimensions,
            )

            self.assertEqual(vectorized[0].keys(), loop[0].keys())
            for name in vectorized[0]:
                torch.testing.assert_close(
                    vectorized[0][name],
                    loop[0][name],
                    rtol=2e-5,
                    atol=2e-5,
                )
            torch.testing.assert_close(vectorized[1], loop[1], rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(vectorized[2], loop[2], rtol=2e-5, atol=2e-5)

    def test_all_models_train_with_ipa_and_glr_vmap_and_loop(self):
        combined_data = torch.tensor(
            [
                [-0.4, 0.2, 0.5],
                [0.1, 0.7, -0.3],
                [0.6, -0.2, 0.8],
                [-0.5, -0.1, -0.6],
            ],
            dtype=torch.float32,
        )
        indices = torch.arange(combined_data.shape[0])
        train_loader = DataLoader(
            TensorDataset(combined_data, indices),
            batch_size=2,
            shuffle=False,
        )
        val_loader = DataLoader(combined_data[:2], batch_size=2, shuffle=False)
        regularization_lambda = 0.25

        for method in ("ipa", "glr"):
            for use_vmap in (False, True):
                for model in build_models(data_len=len(combined_data)):
                    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]
                    trainer = model.make_regularized_trainer(
                        method=method,
                        regularization_lambda=regularization_lambda,
                        learning_rate=1e-3,
                        use_vmap=use_vmap,
                        k=2,
                        num_samples=3,
                        vmap_chunk_size=2,
                        glr_inner_steps=1,
                    )
                    history = trainer.fit(
                        train_loader,
                        val_loader,
                        num_epochs=1,
                        early_stopping=1,
                    )

                    self.assertEqual(history["method"], method)
                    self.assertEqual(history["use_vmap"], use_vmap)
                    self.assertEqual(history["epochs_ran"], 1)
                    self.assertTrue(np.isfinite(history["total_loss"][0]))
                    self.assertAlmostEqual(
                        history["total_loss"][0],
                        history["generative_loss"][0]
                        + regularization_lambda * history["regularizer_loss"][0],
                        places=5,
                    )
                    self.assertTrue(
                        any(
                            not torch.equal(before, after.detach())
                            for before, after in zip(parameters_before, model.parameters())
                        ),
                        f"{type(model).__name__} did not update for {method}/{use_vmap}",
                    )


if __name__ == "__main__":
    unittest.main()
