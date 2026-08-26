import argparse
import ast
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model.newsvendor_vae import ConditionalVAENewsvendor
from model.newsvendor_gendfl_conditional_flow import ConditionalFlow
from run_generative_newsvendor_toy_exp import run_one_setting


OUTPUT_DIR = Path("analysis_outputs/interface_and_gendfl_syn_validation")
REFERENCE_PATH = Path(
    "analysis_outputs/generative_newsvendor_toy_five_strategies_n200/toy_results_summary.csv"
)
GEN_DFL_SOURCE_PATH = Path(
    "/Users/seinzhou/Desktop/gen_dfl-main/end2end_cflowdfl_undergrounding.py"
)


def compare_migrated_flow_with_source():
    source_tree = ast.parse(GEN_DFL_SOURCE_PATH.read_text())
    source_class_node = next(
        node
        for node in source_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ConditionalFlow"
    )
    source_module = ast.Module(body=[source_class_node], type_ignores=[])
    ast.fix_missing_locations(source_module)
    namespace = {"torch": torch, "nn": nn}
    exec(compile(source_module, str(GEN_DFL_SOURCE_PATH), "exec"), namespace)
    SourceConditionalFlow = namespace["ConditionalFlow"]

    torch.manual_seed(811)
    source_model = SourceConditionalFlow(c_dim=1, x_dim=2)
    migrated_model = ConditionalFlow(c_dim=1, x_dim=2)
    migrated_model.load_state_dict(copy.deepcopy(source_model.state_dict()))
    condition = torch.tensor([[-0.4, 0.2], [0.1, 0.7]], dtype=torch.float32)
    target = torch.tensor([[0.5], [-0.3]], dtype=torch.float32)
    source_z, source_log_det = source_model(target, condition)
    migrated_z, migrated_log_det = migrated_model(target, condition)
    torch.manual_seed(812)
    source_samples = source_model.sample(5, condition)
    torch.manual_seed(812)
    migrated_samples = migrated_model.sample(5, condition)
    return {
        "forward_z_max_abs_difference": float((source_z - migrated_z).abs().max().detach()),
        "forward_log_det_max_abs_difference": float(
            (source_log_det - migrated_log_det).abs().max().detach()
        ),
        "sample_max_abs_difference": float((source_samples - migrated_samples).abs().max()),
    }


def make_vae(data_len):
    return ConditionalVAENewsvendor(
        targetdim=1,
        labeldim=2,
        latent=2,
        data_len=data_len,
        epoch=1,
        quantiles=0.7,
        target_quantile=0.7,
        lambda1=0.25,
        lambda_gradient=0.25,
        samplingnumber=5,
        cost_under=7.0,
        cost_over=3.0,
        hidden_dim=8,
        innerloop=1,
    )


def direct_vae_generative_loss(model, target, condition):
    mu, logvar = model.encoder(target, condition)
    std = torch.exp(0.5 * logvar)
    latent = mu + std * torch.randn_like(std)
    reconstruction = model.decode(latent, condition)
    reconstruction_loss = torch.nn.functional.mse_loss(reconstruction, target, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kld


def direct_batched_ipa_loss(model, target, condition, k, num_samples):
    batch_size = condition.shape[0]
    latent_samples = torch.randn(k, batch_size, num_samples, model.latent)
    condition_rep = condition[:, None, :].expand(batch_size, num_samples, condition.shape[1])
    condition_rep = condition_rep.reshape(batch_size * num_samples, condition.shape[1])
    replicate_quantiles = []
    order_index = int(np.ceil(model.target_quantile * num_samples))
    for replicate_idx in range(k):
        generated = model.decode(
            latent_samples[replicate_idx].reshape(batch_size * num_samples, model.latent),
            condition_rep,
        ).reshape(batch_size, num_samples)
        replicate_quantiles.append(
            torch.kthvalue(generated, order_index, dim=1).values.unsqueeze(1)
        )
    quantiles = torch.stack(replicate_quantiles, dim=0)
    difference = target.unsqueeze(0) - quantiles
    losses = torch.where(
        difference > 0,
        model.cu * difference,
        model.co * (-difference),
    )
    return losses.mean()


def maximum_parameter_difference(model_a, model_b):
    return max(
        float(torch.max(torch.abs(parameter_a.detach() - parameter_b.detach())))
        for parameter_a, parameter_b in zip(model_a.parameters(), model_b.parameters())
    )


def compare_vae_ipa_interface(combined_data, regularization_lambda=0.25):
    indices = torch.arange(combined_data.shape[0])
    loader = DataLoader(
        TensorDataset(combined_data, indices),
        batch_size=combined_data.shape[0],
        shuffle=False,
    )
    torch.manual_seed(91)
    interface_model = make_vae(len(combined_data))
    direct_model = make_vae(len(combined_data))
    direct_model.load_state_dict(copy.deepcopy(interface_model.state_dict()))

    update_seed = 1201
    trainer = interface_model.make_regularized_trainer(
        method="ipa",
        regularization_lambda=regularization_lambda,
        learning_rate=1e-3,
        use_vmap=True,
        k=3,
        num_samples=5,
        vmap_chunk_size=3,
        max_grad_norm=None,
    )
    torch.manual_seed(update_seed)
    interface_history = trainer.fit(loader, loader, num_epochs=1, early_stopping=1)

    optimizer = torch.optim.Adam(direct_model.parameters(), lr=1e-3)
    torch.manual_seed(update_seed)
    direct_batch = next(iter(loader))[0]
    condition, target = direct_model._split_batch(direct_batch, targetdim=1)
    generative_loss = direct_vae_generative_loss(direct_model, target, condition)
    ipa_loss = direct_batched_ipa_loss(direct_model, target, condition, k=3, num_samples=5)
    total_loss = generative_loss + regularization_lambda * ipa_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {
        "method": "ipa",
        "maximum_parameter_difference": maximum_parameter_difference(interface_model, direct_model),
        "interface_total_loss": interface_history["total_loss"][0],
        "direct_total_loss": float(total_loss.detach()),
        "absolute_loss_difference": abs(interface_history["total_loss"][0] - float(total_loss.detach())),
    }


def compare_vae_glr_interface(combined_data, regularization_lambda=0.25):
    indices = torch.arange(combined_data.shape[0])
    loader = DataLoader(
        TensorDataset(combined_data, indices),
        batch_size=combined_data.shape[0],
        shuffle=False,
    )
    torch.manual_seed(92)
    interface_model = make_vae(len(combined_data))
    direct_model = make_vae(len(combined_data))
    direct_model.load_state_dict(copy.deepcopy(interface_model.state_dict()))

    update_seed = 1202
    trainer = interface_model.make_regularized_trainer(
        method="glr",
        regularization_lambda=regularization_lambda,
        learning_rate=1e-3,
        use_vmap=True,
        num_samples=5,
        vmap_chunk_size=len(combined_data),
        glr_inner_steps=1,
        max_grad_norm=None,
    )
    torch.manual_seed(update_seed)
    trainer.fit(loader, loader, num_epochs=1, early_stopping=1)

    direct_model.data_len = len(combined_data)
    direct_model._reset_glr_state(len(combined_data))
    direct_model.k_step = 1.0
    optimizer = torch.optim.Adam(direct_model.parameters(), lr=1e-3)
    torch.manual_seed(update_seed)
    direct_batch, global_indices = next(iter(loader))
    condition, target = direct_model._split_batch(direct_batch, targetdim=1)
    generative_loss = direct_vae_generative_loss(direct_model, target, condition)
    glr_result = direct_model.regularized_glr_gradient(
        condition,
        target,
        global_indices.numpy(),
        use_vmap=True,
        vmap_chunk_size=len(combined_data),
        inner_steps=1,
    )
    named_parameters = direct_model.generation_named_parameters()
    generative_gradients = torch.autograd.grad(
        generative_loss,
        tuple(named_parameters.values()),
        allow_unused=True,
    )
    optimizer.zero_grad()
    for (name, parameter), gradient in zip(named_parameters.items(), generative_gradients):
        if gradient is None:
            gradient = torch.zeros_like(parameter)
        parameter.grad = (
            gradient.detach() + regularization_lambda * glr_result["gradient"][name]
        ).clone()
    optimizer.step()

    return {
        "method": "glr",
        "maximum_parameter_difference": maximum_parameter_difference(interface_model, direct_model),
        "interface_total_loss": np.nan,
        "direct_total_loss": np.nan,
        "absolute_loss_difference": np.nan,
    }


def run_conflow_synthetic_comparison():
    conflow_output = OUTPUT_DIR / "conflow_exp1_exp5_n200"
    conflow_output.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(
        output_dir=conflow_output,
        num_samples=200,
        dim=4,
        batch_size=32,
        epochs=100,
        early_stopping=10,
        samplingnumber=8,
        generate_size=128,
        hidden_dim=16,
        vae_latent=4,
        diffusion_steps=6,
        ddim_tau=2,
        lambda1=0.5,
        lambda_gradient=0.5,
        gen_lr=1e-3,
        ipa_lr=1e-3,
        glr_lr=5e-4,
        selection_metric="generative",
        cost_under=7.0,
        cost_over=3.0,
        random_state=42,
        test_size=0.25,
        val_size=0.2,
    )
    rows = [
        run_one_setting(num_exps, "generator_only", "conflow", args, conflow_output, torch.device("cpu"))
        for num_exps in (1, 5)
    ]
    conflow_results = pd.DataFrame(rows)
    conflow_results.to_csv(conflow_output / "conflow_results.csv", index=False)

    reference = pd.read_csv(REFERENCE_PATH)
    reference = reference[reference["strategy"] == "generator_only"].copy()
    comparisons = []
    for row in rows:
        num_exps = row["num_exps"]
        subset = reference[reference["num_exps"] == num_exps]
        realnvp = subset[subset["model"] == "realnvp"].iloc[0]
        best = subset.loc[subset["test_newsvendor_cost"].idxmin()]
        conflow_cost = row["test_newsvendor_cost"]
        comparisons.append(
            {
                "num_exps": num_exps,
                "conflow_cost": conflow_cost,
                "previous_realnvp_cost": realnvp["test_newsvendor_cost"],
                "delta_vs_realnvp": conflow_cost - realnvp["test_newsvendor_cost"],
                "delta_pct_vs_realnvp": 100.0
                * (conflow_cost - realnvp["test_newsvendor_cost"])
                / realnvp["test_newsvendor_cost"],
                "previous_best_model": best["model"],
                "previous_best_cost": best["test_newsvendor_cost"],
                "delta_pct_vs_previous_best": 100.0
                * (conflow_cost - best["test_newsvendor_cost"])
                / best["test_newsvendor_cost"],
            }
        )
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(conflow_output / "comparison_to_previous_n200.csv", index=False)
    return conflow_results, comparison


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_equivalence = compare_migrated_flow_with_source()
    combined_data = torch.tensor(
        [
            [-0.4, 0.2, 0.5],
            [0.1, 0.7, -0.3],
            [0.6, -0.2, 0.8],
            [-0.5, -0.1, -0.6],
        ],
        dtype=torch.float32,
    )
    interface_comparison = pd.DataFrame(
        [
            compare_vae_ipa_interface(combined_data),
            compare_vae_glr_interface(combined_data),
        ]
    )
    interface_comparison.to_csv(OUTPUT_DIR / "vae_interface_equivalence.csv", index=False)
    conflow_results, conflow_comparison = run_conflow_synthetic_comparison()

    summary = {
        "gendfl_source_equivalence": source_equivalence,
        "vae_interface_max_parameter_difference": float(
            interface_comparison["maximum_parameter_difference"].max()
        ),
        "conflow_max_abs_delta_pct_vs_realnvp": float(
            conflow_comparison["delta_pct_vs_realnvp"].abs().max()
        ),
    }
    with (OUTPUT_DIR / "summary.json").open("w") as output_file:
        json.dump(summary, output_file, indent=2)

    print("\nVAE interface equivalence")
    print(interface_comparison.to_string(index=False))
    print("\nStrict gen_dfl ConditionalFlow synthetic results")
    print(conflow_results[["num_exps", "test_newsvendor_cost", "elapsed_seconds"]].to_string(index=False))
    print("\nComparison with previous n=200 generator-only results")
    print(conflow_comparison.to_string(index=False))
    print("\nMigrated ConditionalFlow vs desktop source")
    print(pd.Series(source_equivalence).to_string())
    print("\nSaved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
