import unittest

import numpy as np
import torch

from model.gendfl_1d_interface import (
    predict_gendfl_1d_quantile,
    train_gendfl_1d,
)


class GenDFL1DInterfaceTest(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor(
            [[-0.4, 0.2], [0.1, 0.7], [0.6, -0.2], [-0.5, -0.1]],
            dtype=torch.float32,
        )
        self.y = torch.tensor([[0.5], [-0.3], [0.8], [-0.6]], dtype=torch.float32)

    def test_nll_ipa_and_glr_public_training_interface(self):
        for method in ("nll", "ipa", "glr"):
            model, history = train_gendfl_1d(
                self.x,
                self.y,
                self.x,
                self.y,
                method=method,
                cost_under=7.0,
                cost_over=3.0,
                regularization_lambda=0.25,
                epochs=1,
                batch_size=4,
                learning_rate=1e-3,
                early_stopping=1,
                sampling_number=5,
                ipa_replicates=2,
                glr_inner_steps=1,
                use_vmap=True,
                vmap_chunk_size=4,
                max_grad_norm=None,
                shuffle=True,
                random_seed=100 + len(method),
                device="cpu",
            )
            self.assertEqual(history["method"], method)
            self.assertEqual(history["epochs_ran"], 1)
            if method == "nll":
                self.assertTrue(np.isfinite(history["nll_loss"][0]))
            else:
                self.assertTrue(np.isfinite(history["generative_loss"][0]))
                self.assertGreater(history["combined_gradient_norm"][0], 0.0)

            prediction = predict_gendfl_1d_quantile(model, self.x, num_samples=7)
            self.assertEqual(tuple(prediction.shape), (4, 1))
            self.assertTrue(torch.isfinite(prediction).all())

    def test_one_dimensional_target_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            train_gendfl_1d(
                self.x,
                torch.cat([self.y, self.y], dim=1),
                method="ipa",
                epochs=1,
            )


if __name__ == "__main__":
    unittest.main()
