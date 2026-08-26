import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from model.newsvendor_ddim import ConditionalDDIMNewsvendor
from model.newsvendor_ddpm import ConditionalDDPMNewsvendor
from model.newsvendor_gendfl_conditional_flow import (
    GenDFLConditionalFlowNewsvendor,
    pretrain_flow as pretrain_gendfl_flow,
)
from model.newsvendor_mean_flow import ConditionalMeanFlowNewsvendor
from model.newsvendor_realnvp import ConditionalRealNVPNewsvendor
from model.newsvendor_vae import ConditionalVAENewsvendor


def makettoy_multi_exp(num_samples, num_features, random_state, num_exps=1):
    samples_per_exp = num_samples // num_exps
    remaining_samples = num_samples % num_exps
    all_data = []
    np.random.seed(random_state)
    meanx = []
    for i in range(num_features):
        np.random.seed(random_state + i)
        meanx.append(np.random.uniform(-50, 50))
    meanx = np.array(meanx)
    peakmean = np.random.uniform(0, 250, size=num_exps)
    label_list = []
    w_np = np.zeros((num_exps, num_features))
    for i in range(num_exps):
        peak_samples = samples_per_exp + (1 if i < remaining_samples else 0)
        label_list.extend([i] * peak_samples)
        np.random.seed(random_state + i)
        X = np.random.normal(loc=meanx, scale=1, size=(peak_samples, num_features))
        w = np.random.normal(loc=0, scale=1, size=num_features)
        w_np[i, :] = w
        y = X @ w + peakmean[i] + np.random.normal(0, 10, size=peak_samples)
        all_data.append(np.column_stack((X, y)))
    combined = np.vstack(all_data)
    combined_labels = np.array(label_list).reshape(-1, 1)
    combined = np.hstack((combined, combined_labels))
    np.random.seed(random_state)
    np.random.shuffle(combined)
    return combined, w_np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def newsvendor_cost_np(y_true, q_pred, cu, co):
    diff = y_true - q_pred
    return np.where(diff > 0, cu * diff, co * (-diff)).mean()


def make_model(name, dim, train_len, args, quantile, cu, co, seed, device):
    common = dict(
        targetdim=1,
        labeldim=dim,
        data_len=train_len,
        epoch=args.epochs,
        quantiles=quantile,
        lambda1=args.lambda1,
        lambda_gradient=args.lambda_gradient,
        samplingnumber=args.samplingnumber,
        target_quantile=quantile,
        cost_under=cu,
        cost_over=co,
        random_seed=seed,
        innerloop=1,
        hidden_dim=args.hidden_dim,
    )
    if name == "vae":
        model = ConditionalVAENewsvendor(latent=args.vae_latent, **common)
    elif name == "realnvp":
        model = ConditionalRealNVPNewsvendor(latent=1, **common)
    elif name == "meanflow":
        model = ConditionalMeanFlowNewsvendor(latent=1, **common)
    elif name == "ddpm":
        model = ConditionalDDPMNewsvendor(latent=1, T=args.diffusion_steps, **common)
    elif name == "ddim":
        model = ConditionalDDIMNewsvendor(latent=1, T=args.diffusion_steps, tau=args.ddim_tau, **common)
    elif name in {"conflow", "gendfl"}:
        common.pop("hidden_dim")
        model = GenDFLConditionalFlowNewsvendor(latent=1, **common)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model.to(device)


@torch.no_grad()
def evaluate_model(model, X_test_scaled, y_test_raw, scaler_y, cu, co, quantile, generate_size, batch_size=128):
    model.eval()
    device = next(model.parameters()).device
    preds = []
    for start in range(0, X_test_scaled.shape[0], batch_size):
        x_batch = torch.tensor(X_test_scaled[start:start + batch_size], dtype=torch.float32, device=device)
        n = x_batch.shape[0]
        condition = x_batch[:, None, :].expand(n, generate_size, x_batch.shape[1]).reshape(n * generate_size, x_batch.shape[1])
        z = torch.randn(n * generate_size, model.latent, device=device)
        generated_scaled = model.decode(z, condition).reshape(n, generate_size, 1)
        q_scaled = torch.quantile(generated_scaled, quantile, dim=1).detach().cpu().numpy().reshape(-1, 1)
        q_raw = scaler_y.inverse_transform(q_scaled).reshape(-1)
        preds.append(q_raw)
    q_pred = np.concatenate(preds, axis=0)
    return newsvendor_cost_np(y_test_raw, q_pred, cu, co), q_pred


def load_best_checkpoint(model, best_model_path, device):
    if not best_model_path:
        return False
    best_model_path = Path(best_model_path)
    if not best_model_path.exists():
        return False
    try:
        state_dict = torch.load(best_model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return True


def normalize_strategies(strategy_text):
    aliases = {
        "generator": "generator_only",
        "gen": "generator_only",
        "gen_only": "generator_only",
        "linear_combination": "ipa_linear_combination",
        "separate": "ipa_separate",
        "ipa_linear": "ipa_linear_combination",
        "ipa_separate_update": "ipa_separate",
        "glr_linear": "glr_linear_combination",
        "glr_twoupdate": "glr_separate",
        "glr_two_update": "glr_separate",
    }
    strategies = []
    for item in strategy_text.split(","):
        item = item.strip()
        if not item:
            continue
        strategies.append(aliases.get(item, item))
    valid = {
        "generator_only",
        "ipa_linear_combination",
        "ipa_separate",
        "glr_linear_combination",
        "glr_separate",
    }
    unknown = sorted(set(strategies) - valid)
    if unknown:
        raise ValueError(f"Unknown strategies: {unknown}. Valid strategies are: {sorted(valid)}")
    return strategies


def run_one_setting(num_exps, strategy, model_name, args, output_dir, device):
    set_seed(args.random_state + num_exps * 100)
    data, _ = makettoy_multi_exp(
        num_samples=args.num_samples,
        num_features=args.dim,
        random_state=args.random_state,
        num_exps=num_exps,
    )
    data = data[:, :-1].astype(np.float32)
    train_val, test = train_test_split(data, test_size=args.test_size, random_state=args.random_state)
    train, val = train_test_split(train_val, test_size=args.val_size, random_state=args.random_state)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_x.fit_transform(train[:, :-1]).astype(np.float32)
    X_val = scaler_x.transform(val[:, :-1]).astype(np.float32)
    X_test = scaler_x.transform(test[:, :-1]).astype(np.float32)
    y_train = scaler_y.fit_transform(train[:, -1:].astype(np.float32)).astype(np.float32)
    y_val = scaler_y.transform(val[:, -1:].astype(np.float32)).astype(np.float32)

    train_scaled = np.hstack([X_train, y_train]).astype(np.float32)
    val_scaled = np.hstack([X_val, y_val]).astype(np.float32)
    train_dataset = TensorDataset(torch.tensor(train_scaled), torch.arange(train_scaled.shape[0]))
    val_dataset = TensorDataset(torch.tensor(val_scaled), torch.arange(val_scaled.shape[0]))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    cu = args.cost_under
    co = args.cost_over
    quantile = cu / (cu + co)
    model = make_model(model_name, args.dim, train_scaled.shape[0], args, quantile, cu, co, args.random_state, device)
    run_tag = f"n{args.num_samples}_{model_name}_{strategy}_exp{num_exps}"

    start = time.time()
    if strategy == "generator_only":
        if model_name in {"conflow", "gendfl"}:
            best_model_path = output_dir / "checkpoints" / f"{run_tag}_final_model.pth"
            flow_train_loader = DataLoader(
                TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                batch_size=args.batch_size,
                shuffle=True,
            )
            nll_losses = pretrain_gendfl_flow(
                model.flow,
                flow_train_loader,
                num_epochs=args.epochs,
                lr=args.gen_lr,
                device=device,
            )
            history = {"epoch": list(range(args.epochs)), "train_nll": nll_losses}
            best_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            avg_innerloop_time = 0.0
        else:
            history, best_model_path, avg_innerloop_time = model.train_generator_only(
                args.epochs,
                1,
                train_loader,
                val_loader,
                args.early_stopping,
                save_name=run_tag,
                randomnumber=args.random_state,
                gen_lr=args.gen_lr,
                selection_metric=args.selection_metric,
            )
    elif strategy == "ipa_linear_combination":
        history, best_model_path, avg_innerloop_time = model.trainconvae_sgd_linear_combination(
            args.epochs,
            1,
            train_loader,
            val_loader,
            args.early_stopping,
            save_name=run_tag,
            randomnumber=args.random_state,
            gen_lr=args.gen_lr,
            ipa_lr=args.ipa_lr,
            selection_metric=args.selection_metric,
        )
    elif strategy == "ipa_separate":
        history, best_model_path, avg_innerloop_time = model.trainconvae_sgd_separate_update(
            args.epochs,
            1,
            train_loader,
            val_loader,
            args.early_stopping,
            save_name=run_tag,
            randomnumber=args.random_state,
            gen_lr=args.gen_lr,
            ipa_lr=args.ipa_lr,
            selection_metric=args.selection_metric,
        )
    elif strategy in {"glr_linear_combination", "glr_separate"}:
        history = model.train_step_sqo_vectorized_SGD_LR_globalsingle(
            train_loader,
            val_loader,
            args.early_stopping,
            args.batch_size,
            ifsave=False,
            save_tag=run_tag,
            ifonlyglr=False,
            iftwoupdate=(strategy == "glr_separate"),
            gen_lr=args.glr_lr,
            selection_metric=args.selection_metric,
        )
        best_model_path = history.get("best_model_path")
        avg_innerloop_time = history.get("avg_innerloop_time", 0.0)
    else:
        raise ValueError(strategy)
    elapsed = time.time() - start
    evaluated_best_checkpoint = load_best_checkpoint(model, best_model_path, device)

    test_cost, q_pred = evaluate_model(
        model,
        X_test,
        test[:, -1].astype(np.float32),
        scaler_y,
        cu,
        co,
        quantile,
        args.generate_size,
    )
    history_path = output_dir / f"history_exp{num_exps}_{model_name}_{strategy}.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    pred_path = output_dir / f"pred_exp{num_exps}_{model_name}_{strategy}.csv"
    pd.DataFrame({"y_true": test[:, -1], "q_pred": q_pred}).to_csv(pred_path, index=False)
    return {
        "num_exps": num_exps,
        "model": model_name,
        "strategy": strategy,
        "test_newsvendor_cost": test_cost,
        "best_epoch": history.get("best_loss_epoch", history.get("best_epoch")),
        "last_val_generative_loss": (
            history["val_generative_loss"][-1]
            if history.get("val_generative_loss")
            else history["val_nll"][-1] if history.get("val_nll") else None
        ),
        "last_val_newsvendor_loss": history["val_newsvendor_loss"][-1] if history.get("val_newsvendor_loss") else None,
        "avg_innerloop_time": avg_innerloop_time,
        "avg_gradient_time": history.get("avg_gradient_time", 0.0),
        "elapsed_seconds": elapsed,
        "best_model_path": best_model_path,
        "evaluated_best_checkpoint": evaluated_best_checkpoint,
        "history_path": str(history_path),
        "prediction_path": str(pred_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/generative_newsvendor_toy"))
    parser.add_argument("--models", type=str, default="vae,realnvp,meanflow,ddpm,ddim")
    parser.add_argument(
        "--strategies",
        type=str,
        default="generator_only,ipa_linear_combination,ipa_separate,glr_linear_combination,glr_separate",
    )
    parser.add_argument("--num-exps-list", type=str, default="1,5")
    parser.add_argument("--num-samples", type=int, default=80)
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stopping", type=int, default=10)
    parser.add_argument("--samplingnumber", type=int, default=8)
    parser.add_argument("--generate-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--vae-latent", type=int, default=4)
    parser.add_argument("--diffusion-steps", type=int, default=6)
    parser.add_argument("--ddim-tau", type=int, default=2)
    parser.add_argument("--lambda1", type=float, default=0.5)
    parser.add_argument("--lambda-gradient", type=float, default=0.5)
    parser.add_argument("--gen-lr", type=float, default=1e-3)
    parser.add_argument("--ipa-lr", type=float, default=1e-3)
    parser.add_argument("--glr-lr", type=float, default=5e-4)
    parser.add_argument("--selection-metric", choices=["generative", "newsvendor", "total"], default="newsvendor")
    parser.add_argument("--cost-under", type=float, default=7.0)
    parser.add_argument("--cost-over", type=float, default=3.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--val-size", type=float, default=0.2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    strategies = normalize_strategies(args.strategies)
    num_exps_list = [int(x.strip()) for x in args.num_exps_list.split(",") if x.strip()]

    rows = []
    for num_exps in num_exps_list:
        for strategy in strategies:
            for model_name in models:
                print(f"\n[run] exp{num_exps} model={model_name} strategy={strategy}")
                row = run_one_setting(num_exps, strategy, model_name, args, args.output_dir, device)
                rows.append(row)
                pd.DataFrame(rows).to_csv(args.output_dir / "toy_results_detail.csv", index=False)

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["num_exps", "model", "strategy"], as_index=False)
        .agg(
            test_newsvendor_cost=("test_newsvendor_cost", "mean"),
            elapsed_seconds=("elapsed_seconds", "mean"),
            avg_gradient_time=("avg_gradient_time", "mean"),
        )
        .sort_values(["num_exps", "strategy", "test_newsvendor_cost"])
    )
    detail.to_csv(args.output_dir / "toy_results_detail.csv", index=False)
    summary.to_csv(args.output_dir / "toy_results_summary.csv", index=False)
    with pd.ExcelWriter(args.output_dir / "toy_results.xlsx") as writer:
        detail.to_excel(writer, sheet_name="detail", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
    print(f"\nSaved results to {args.output_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
