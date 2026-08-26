"""Consolidate Exp5 and Van Havre RSETO-IPA sensitivity outputs.

The raw experiment directories intentionally remain untouched.  This script
extracts one row per dimension, random seed, and hyperparameter value into
machine-readable CSV tables under ``results_syn/consolidated_csv``.  Styled
Excel workbooks are built from these tables by the companion artifact builder.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


DIMS = [4, 9, 14, 19, 24]
SEEDS = [82, 15, 4, 95, 36, 32, 29, 18, 14, 87]
SWEEP_VALUES = {
    "lambda": [0.1, 0.3, 0.5, 0.7, 0.9],
    "m": [8, 32, 128, 512, 2048],
    "simulation_num": [1, 4, 16, 64, 256],
}
VAN_HAVRE_PATTERN = re.compile(
    r"spline_sensitivity_van_havre_sim4_dim(?P<dim>\d+)_seed(?P<seed>\d+)"
)
EXP5_PATTERN = re.compile(
    r"spline_sensitivity_iid_exp5_dim(?P<dim>\d+)_seed(?P<seed>\d+)"
)

VARIANCE_COLUMNS = [
    "ipa_gradient_variance_trace",
    "ipa_gradient_variance_mean_per_parameter",
    "ipa_gradient_std_norm",
    "ipa_gradient_mean_norm",
    "ipa_gradient_relative_variance",
    "ipa_gradient_loss_mean",
    "ipa_gradient_loss_variance",
    "gradient_variance_repeats",
    "gradient_variance_batch_size",
    "gradient_variance_replications",
    "gradient_variance_samples_per_replication",
    "gradient_variance_seed",
    "gradient_parameter_count",
    "weighted_ipa_gradient_variance_trace",
    "ipa_gradient_weight",
    "ipa_batched_gradient_variance",
    "weighted_ipa_batched_gradient_variance",
]


def parse_experiment_identity(path: Path, pattern: re.Pattern[str]) -> tuple[int, int]:
    match = pattern.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected experiment directory name: {path}")
    return int(match.group("dim")), int(match.group("seed"))


def load_detail(root: Path, pattern: re.Pattern[str], sweep: str) -> pd.DataFrame:
    frames = []
    for detail_path in sorted(root.glob(f"spline_sensitivity_*/{sweep}/detail.csv")):
        experiment_dir = detail_path.parents[1]
        dim, seed = parse_experiment_identity(experiment_dir, pattern)
        frame = pd.read_csv(detail_path)
        frame.insert(0, "random_state", seed)
        frame.insert(0, "dim", dim)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No {sweep} detail.csv files found under {root}")
    return pd.concat(frames, ignore_index=True)


def ordered(frame: pd.DataFrame, parameter: str | None = None) -> pd.DataFrame:
    result = frame.copy()
    result["dim"] = pd.Categorical(result["dim"], DIMS, ordered=True)
    result["random_state"] = pd.Categorical(
        result["random_state"], SEEDS, ordered=True
    )
    sort_columns = ["dim", "random_state"]
    if parameter is not None:
        sort_columns.append(parameter)
    result = result.sort_values(sort_columns).reset_index(drop=True)
    result["dim"] = result["dim"].astype(int)
    result["random_state"] = result["random_state"].astype(int)
    return result


def select_rseto(
    detail: pd.DataFrame,
    *,
    dataset: str,
    sweep: str,
) -> pd.DataFrame:
    parameter = sweep
    frame = detail[detail["method"].str.startswith("rseto_ipa_")].copy()
    frame.insert(0, "dataset", dataset)
    frame[parameter] = pd.to_numeric(frame.pop("sweep_value"))
    if sweep in {"m", "simulation_num"}:
        frame[parameter] = frame[parameter].astype(int)

    expected_rows = len(DIMS) * len(SEEDS) * len(SWEEP_VALUES[sweep])
    if len(frame) != expected_rows:
        raise RuntimeError(
            f"{dataset}/{sweep}: expected {expected_rows} rows, found {len(frame)}"
        )
    keys = ["dim", "random_state", parameter]
    if frame.duplicated(keys).any():
        raise RuntimeError(f"{dataset}/{sweep}: duplicate keys detected")
    if set(frame["dim"]) != set(DIMS) or set(frame["random_state"]) != set(SEEDS):
        raise RuntimeError(f"{dataset}/{sweep}: dimension or seed coverage mismatch")
    actual_values = sorted(frame[parameter].unique())
    if not np.allclose(actual_values, SWEEP_VALUES[sweep]):
        raise RuntimeError(
            f"{dataset}/{sweep}: parameter values {actual_values} do not match expected"
        )

    leading = [
        "dataset",
        "dim",
        "random_state",
        parameter,
        "method",
        "metric1",
        "metric2",
    ]
    diagnostics = [
        column
        for column in [
            "replications",
            "samples_per_replication",
            "final_samples_per_replication",
            "fidelity_weight",
            "smoothing_mu",
            "configured_epochs",
            "epochs_ran",
            "batch_size",
            "steps_ran",
            "optimizer",
            "initial_checkpoint_sha256",
            *VARIANCE_COLUMNS,
        ]
        if column in frame.columns
    ]
    return ordered(frame[leading + diagnostics], parameter)


def select_baseline(
    detail: pd.DataFrame,
    *,
    dataset: str,
    method: str,
) -> pd.DataFrame:
    frame = detail[detail["method"].eq(method)].copy()
    frame.insert(0, "dataset", dataset)
    if len(frame) != len(DIMS) * len(SEEDS):
        raise RuntimeError(f"{dataset}/{method}: incomplete baseline coverage")
    if frame.duplicated(["dim", "random_state"]).any():
        raise RuntimeError(f"{dataset}/{method}: duplicate keys detected")
    columns = [
        "dataset",
        "dim",
        "random_state",
        "method",
        "metric1",
        "metric2",
        "configured_epochs",
        "epochs_ran",
        "batch_size",
        "steps_ran",
        "optimizer",
        "initial_checkpoint_sha256",
    ]
    return ordered(frame[[column for column in columns if column in frame.columns]])


def verify_shared_van_havre_baselines(details: dict[str, pd.DataFrame]) -> None:
    for method in ["gendfl_spline", "spline_qfr"]:
        reference = select_baseline(
            details["lambda"], dataset="van_havre", method=method
        )
        for sweep in ["m", "simulation_num"]:
            candidate = select_baseline(
                details[sweep], dataset="van_havre", method=method
            )
            np.testing.assert_allclose(
                reference[["metric1", "metric2"]].to_numpy(),
                candidate[["metric1", "metric2"]].to_numpy(),
                rtol=0.0,
                atol=1e-12,
                err_msg=f"Van Havre {method} differs between lambda and {sweep}",
            )


def variance_detail(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "dataset",
        "dim",
        "random_state",
        "simulation_num",
        "metric1",
        "metric2",
        "ipa_gradient_variance_trace",
        "weighted_ipa_gradient_variance_trace",
        "ipa_batched_gradient_variance",
        "weighted_ipa_batched_gradient_variance",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Variance results are missing columns: {missing}")
    if frame[required].isna().any().any():
        raise RuntimeError("Variance detail contains missing values")
    np.testing.assert_allclose(
        frame["ipa_gradient_variance_trace"],
        frame["ipa_batched_gradient_variance"],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        frame["weighted_ipa_gradient_variance_trace"],
        frame["weighted_ipa_batched_gradient_variance"],
        rtol=0.0,
        atol=1e-12,
    )
    columns = required + [
        column
        for column in VARIANCE_COLUMNS
        if column in frame.columns and column not in required
    ]
    return frame[columns].copy()


def build_variance_summary(combined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    measures = [
        "metric1",
        "metric2",
        "ipa_gradient_variance_trace",
        "weighted_ipa_gradient_variance_trace",
    ]
    by_dim = (
        combined.groupby(["dataset", "dim", "simulation_num"], sort=False)[measures]
        .agg(["mean", "std", "count"])
    )
    by_dim.columns = [f"{measure}_{stat}" for measure, stat in by_dim.columns]
    by_dim = by_dim.reset_index()
    overall = (
        combined.groupby(["dataset", "simulation_num"], sort=False)[measures]
        .agg(["mean", "std", "count"])
    )
    overall.columns = [f"{measure}_{stat}" for measure, stat in overall.columns]
    return by_dim, overall.reset_index()


def build_van_havre_mean_summary(sweeps: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for sweep, frame in sweeps.items():
        summary = (
            frame.groupby(["dim", sweep], sort=False)[["metric1", "metric2"]]
            .agg(["mean", "std", "count"])
        )
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary = summary.reset_index().rename(columns={sweep: "parameter"})
        summary.insert(0, "sweep", sweep)
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def load_ete_results(
    project_root: Path,
    *,
    experiment_directory: str,
    expected_dataset: str,
) -> pd.DataFrame:
    path = (
        project_root
        / "analysis_outputs_ete"
        / experiment_directory
        / "fixed_dgp_results_detail.csv"
    )
    frame = pd.read_csv(path, keep_default_na=False)
    frame = frame[frame["error"].fillna("").eq("")].copy()
    if set(frame["data_synthetic"]) != {expected_dataset}:
        raise RuntimeError(
            f"{path} contains datasets {sorted(frame['data_synthetic'].unique())}, "
            f"expected only {expected_dataset}"
        )
    expected_models = {"oracle_gmm", "erm", "lightgbm", "end_to_end"}
    if set(frame["model"]) != expected_models:
        raise RuntimeError(
            f"Van Havre ETE models {sorted(frame['model'].unique())} do not match "
            f"expected {sorted(expected_models)}"
        )
    if len(frame) != len(DIMS) * len(SEEDS) * len(expected_models):
        raise RuntimeError(f"{expected_dataset} ETE result coverage is incomplete")
    if frame.duplicated(["dim", "random_state", "model"]).any():
        raise RuntimeError(f"{expected_dataset} ETE results contain duplicate keys")
    columns = [
        "data_synthetic",
        "dim",
        "random_state",
        "model",
        "metric1",
        "metric2",
        "configured_epochs",
        "epochs_ran",
        "batch_size",
        "steps_ran",
        "optimizer",
        "data_protocol",
        "error",
    ]
    return ordered(frame[[column for column in columns if column in frame.columns]])


def build_van_havre_wide_mean_tables(
    sweeps: dict[str, pd.DataFrame],
    gendfl: pd.DataFrame,
    qfr: pd.DataFrame,
    ete: pd.DataFrame,
) -> dict[tuple[str, str], pd.DataFrame]:
    model_labels = {
        "oracle_gmm": "BCMO",
        "erm": "ERM",
        "lightgbm": "LightGBM",
        "end_to_end": "ERM-NN",
    }
    ete = ete.copy()
    ete["model"] = ete["model"].map(model_labels)
    model_order = ["BCMO", "ERM", "LightGBM", "ERM-NN"]
    tables = {}
    for sweep, frame in sweeps.items():
        for metric in ["metric1", "metric2"]:
            sweep_mean = (
                frame.pivot_table(
                    index="dim", columns=sweep, values=metric, aggfunc="mean"
                )
                .reindex(index=DIMS, columns=SWEEP_VALUES[sweep])
            )
            if sweep == "lambda":
                sweep_mean.columns = [f"Lambda {float(value):.1f}" for value in sweep_mean.columns]
            elif sweep == "m":
                sweep_mean.columns = [f"M {int(value)}" for value in sweep_mean.columns]
            else:
                sweep_mean.columns = [f"R {int(value)}" for value in sweep_mean.columns]
            baseline = pd.concat(
                [
                    gendfl.groupby("dim")[metric].mean().reindex(DIMS).rename("Spline-ETO"),
                    qfr.groupby("dim")[metric].mean().reindex(DIMS).rename("Spline-QFR"),
                ],
                axis=1,
            )
            ete_mean = (
                ete.pivot_table(index="dim", columns="model", values=metric, aggfunc="mean")
                .reindex(index=DIMS, columns=model_order)
            )
            table = pd.concat([sweep_mean, baseline, ete_mean], axis=1)
            table.index.name = "Dim"
            if table.isna().any().any():
                raise RuntimeError(f"Missing values in Van Havre {sweep}/{metric} mean table")
            tables[(sweep, metric)] = table.reset_index()
    return tables


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"[saved] {path} rows={len(frame)}")


def consolidate(project_root: Path) -> None:
    output = project_root / "results_syn" / "consolidated_csv"
    van_root = project_root / "vanharve_simulation"
    exp_root = project_root / "exp_simulation"

    van_details = {
        sweep: load_detail(van_root, VAN_HAVRE_PATTERN, sweep)
        for sweep in SWEEP_VALUES
    }
    verify_shared_van_havre_baselines(van_details)
    van_sweeps = {
        sweep: select_rseto(detail, dataset="van_havre", sweep=sweep)
        for sweep, detail in van_details.items()
    }
    van_gendfl = select_baseline(
        van_details["lambda"], dataset="van_havre", method="gendfl_spline"
    )
    van_qfr = select_baseline(
        van_details["lambda"], dataset="van_havre", method="spline_qfr"
    )
    van_ete = load_ete_results(
        project_root,
        experiment_directory="van_havre_traditional_50epochs_projected_sgd",
        expected_dataset="van-havre",
    )
    exp5_ete = load_ete_results(
        project_root,
        experiment_directory="fixed_dgp_exp5_traditional_50epochs_projected_sgd",
        expected_dataset="exp5",
    )

    for sweep, frame in van_sweeps.items():
        write_csv(frame, output / f"van_havre_{sweep}_ipa.csv")
    write_csv(van_gendfl, output / "van_havre_gendfl_spline.csv")
    write_csv(van_qfr, output / "van_havre_spline_qfr.csv")
    write_csv(van_ete, output / "van_havre_ete_models.csv")
    write_csv(exp5_ete, output / "exp5_ete_models.csv")
    write_csv(
        build_van_havre_mean_summary(van_sweeps),
        output / "van_havre_sensitivity_mean.csv",
    )
    for (sweep, metric), table in build_van_havre_wide_mean_tables(
        van_sweeps, van_gendfl, van_qfr, van_ete
    ).items():
        write_csv(table, output / f"van_havre_{sweep}_{metric}_mean.csv")

    exp_detail = load_detail(exp_root, EXP5_PATTERN, "simulation_num")
    exp_simulation = select_rseto(
        exp_detail, dataset="exp5", sweep="simulation_num"
    )
    exp_variance = variance_detail(exp_simulation)
    van_variance = variance_detail(van_sweeps["simulation_num"])
    combined_variance = pd.concat([exp_variance, van_variance], ignore_index=True)
    by_dim, overall = build_variance_summary(combined_variance)

    write_csv(exp_variance, output / "exp5_simulation_num_ipa_variance.csv")
    write_csv(
        van_variance, output / "van_havre_simulation_num_ipa_variance.csv"
    )
    write_csv(by_dim, output / "ipa_variance_simulation_by_dim_mean.csv")
    write_csv(overall, output / "ipa_variance_simulation_overall_mean.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    consolidate(arguments.project_root.resolve())
