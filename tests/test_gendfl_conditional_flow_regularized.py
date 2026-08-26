import ast
import copy
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model.newsvendor_gendfl_conditional_flow import (
    ConditionalFlow,
    GenDFLConditionalFlowNewsvendor,
    pretrain_flow,
)


GEN_DFL_SOURCE = Path(
    "/Users/seinzhou/Desktop/gen_dfl-main/end2end_cflowdfl_undergrounding.py"
)


def load_source_implementation():
    source_tree = ast.parse(GEN_DFL_SOURCE.read_text())
    selected_nodes = [
        node
        for node in source_tree.body
        if (
            isinstance(node, ast.ClassDef)
            and node.name == "ConditionalFlow"
        )
        or (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "pretrain_flow"
        )
    ]
    source_module = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(source_module)
    namespace = {"np": np, "torch": torch, "nn": nn}
    exec(compile(source_module, str(GEN_DFL_SOURCE), "exec"), namespace)
    return namespace["ConditionalFlow"], namespace["pretrain_flow"]


def make_model(data_len=4):
    return GenDFLConditionalFlowNewsvendor(
        targetdim=1,
        labeldim=2,
        latent=1,
        data_len=data_len,
        epoch=1,
        target_quantile=0.7,
        samplingnumber=7,
        cost_under=7.0,
        cost_over=3.0,
        innerloop=1,
    )


def gradient_norm(gradients):
    return torch.sqrt(sum(gradient.detach().pow(2).sum() for gradient in gradients))


class GenDFLConditionalFlowStrictTest(unittest.TestCase):
    def setUp(self):
        self.condition = torch.tensor(
            [[-0.4, 0.2], [0.1, 0.7], [0.6, -0.2], [-0.5, -0.1]],
            dtype=torch.float32,
        )
        self.target = torch.tensor([[0.5], [-0.3], [0.8], [-0.6]], dtype=torch.float32)

    def test_forward_sample_and_pretrain_match_source_exactly(self):
        SourceConditionalFlow, source_pretrain_flow = load_source_implementation()
        torch.manual_seed(811)
        source_model = SourceConditionalFlow(c_dim=1, x_dim=2)
        migrated_model = ConditionalFlow(c_dim=1, x_dim=2)
        migrated_model.load_state_dict(copy.deepcopy(source_model.state_dict()))

        source_z, source_log_det = source_model(self.target, self.condition)
        migrated_z, migrated_log_det = migrated_model(self.target, self.condition)
        torch.testing.assert_close(migrated_z, source_z, rtol=0.0, atol=0.0)
        torch.testing.assert_close(migrated_log_det, source_log_det, rtol=0.0, atol=0.0)

        torch.manual_seed(812)
        source_samples = source_model.sample(5, self.condition)
        torch.manual_seed(812)
        migrated_samples = migrated_model.sample(5, self.condition)
        torch.testing.assert_close(migrated_samples, source_samples, rtol=0.0, atol=0.0)

        loader = DataLoader(
            TensorDataset(self.condition, self.target),
            batch_size=2,
            shuffle=False,
        )
        source_losses = source_pretrain_flow(source_model, loader, num_epochs=1, lr=1e-3)
        migrated_losses = pretrain_flow(migrated_model, loader, num_epochs=1, lr=1e-3)
        self.assertEqual(source_losses, migrated_losses)
        for source_parameter, migrated_parameter in zip(
            source_model.parameters(), migrated_model.parameters()
        ):
            torch.testing.assert_close(
                migrated_parameter,
                source_parameter,
                rtol=0.0,
                atol=0.0,
            )

    def test_differentiable_decode_is_the_source_inverse_sampling_map(self):
        torch.manual_seed(101)
        model = make_model()
        num_samples = 5
        latent = torch.randn(len(self.condition), num_samples, 1)
        repeated_condition = self.condition[:, None, :].expand(-1, num_samples, -1)
        decoded = model.decode(
            latent.reshape(-1, 1),
            repeated_condition.reshape(-1, self.condition.shape[1]),
        ).reshape(len(self.condition), num_samples, 1)

        output = model.flow.net(self.condition)
        mean = output[:, 0:1].unsqueeze(1)
        standard_deviation = torch.exp(0.5 * output[:, 1:2]).unsqueeze(1)
        source_inverse = mean + standard_deviation * latent
        torch.testing.assert_close(decoded, source_inverse, rtol=1e-6, atol=1e-7)
        self.assertTrue(decoded.requires_grad)

    def test_ipa_joint_gradient_is_nll_plus_weighted_ipa(self):
        torch.manual_seed(102)
        model = make_model()
        regularization_lambda = 0.25
        latent_samples = torch.randn(3, len(self.condition), 5, 1)
        generative_loss = model.generative_loss(self.target, self.condition)
        ipa_loss = model.batched_ipa_regularizer(
            self.condition,
            self.target,
            k=3,
            num_samples=5,
            use_vmap=True,
            latent_samples=latent_samples,
        )["loss"]
        parameters = tuple(model.generation_parameters())
        generative_gradients = torch.autograd.grad(
            generative_loss,
            parameters,
            retain_graph=True,
        )
        ipa_gradients = torch.autograd.grad(ipa_loss, parameters, retain_graph=True)
        combined_gradients = torch.autograd.grad(
            generative_loss + regularization_lambda * ipa_loss,
            parameters,
        )

        self.assertGreater(float(gradient_norm(generative_gradients)), 0.0)
        self.assertGreater(float(gradient_norm(ipa_gradients)), 0.0)
        for combined, generative, ipa in zip(
            combined_gradients,
            generative_gradients,
            ipa_gradients,
        ):
            torch.testing.assert_close(
                combined,
                generative + regularization_lambda * ipa,
                rtol=1e-6,
                atol=1e-7,
            )

    def test_glr_optimizer_gradient_is_nll_plus_weighted_glr(self):
        torch.manual_seed(100)
        model = make_model()
        regularization_lambda = 0.25
        trainer = model.make_regularized_trainer(
            method="glr",
            regularization_lambda=regularization_lambda,
            learning_rate=1e-3,
            use_vmap=True,
            num_samples=7,
            vmap_chunk_size=4,
            glr_inner_steps=1,
            max_grad_norm=None,
        )
        model._reset_glr_state(len(self.condition))
        generative_loss = model.generative_loss(self.target, self.condition)
        parameters_by_name = model.generation_named_parameters()
        generative_gradients = torch.autograd.grad(
            generative_loss,
            tuple(parameters_by_name.values()),
            retain_graph=True,
        )

        torch.manual_seed(1)
        glr_result = model.regularized_glr_gradient(
            self.condition,
            self.target,
            np.arange(len(self.condition)),
            use_vmap=True,
            vmap_chunk_size=4,
            inner_steps=1,
        )
        glr_gradients = tuple(glr_result["gradient"].values())
        self.assertGreater(float(gradient_norm(generative_gradients)), 0.0)
        self.assertGreater(float(gradient_norm(glr_gradients)), 0.0)

        zero_step_optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        trainer._set_combined_glr_gradient(
            generative_loss,
            glr_result,
            zero_step_optimizer,
        )
        for (name, parameter), generative_gradient in zip(
            parameters_by_name.items(),
            generative_gradients,
        ):
            expected = (
                generative_gradient
                + regularization_lambda * glr_result["gradient"][name]
            )
            torch.testing.assert_close(parameter.grad, expected, rtol=1e-6, atol=1e-7)

    def test_public_regularized_ipa_and_glr_training_update_conflow(self):
        combined = torch.cat([self.condition, self.target], dim=1)
        indices = torch.arange(len(combined))
        train_loader = DataLoader(
            TensorDataset(combined, indices),
            batch_size=len(combined),
            shuffle=False,
        )
        val_loader = DataLoader(combined, batch_size=len(combined), shuffle=False)

        for method in ("ipa", "glr"):
            torch.manual_seed(103 if method == "ipa" else 104)
            model = make_model()
            parameters_before = [parameter.detach().clone() for parameter in model.parameters()]
            train_method = getattr(model, f"train_regularized_{method}_vmap")
            common_arguments = {
                "traindata_loader": train_loader,
                "valdata_loader": val_loader,
                "num_epochs": 1,
                "early_stopping": 1,
                "regularization_lambda": 0.25,
                "learning_rate": 1e-3,
                "num_samples": 7,
                "vmap_chunk_size": 4,
                "max_grad_norm": None,
            }
            if method == "ipa":
                common_arguments["k"] = 3
            else:
                common_arguments["glr_inner_steps"] = 1
            history = train_method(**common_arguments)

            self.assertEqual(history["method"], method)
            self.assertEqual(history["regularization_lambda"], 0.25)
            self.assertTrue(np.isfinite(history["generative_loss"][0]))
            self.assertGreater(history["combined_gradient_norm"][0], 0.0)
            self.assertTrue(
                any(
                    not torch.equal(before, after.detach())
                    for before, after in zip(parameters_before, model.parameters())
                )
            )


if __name__ == "__main__":
    unittest.main()
