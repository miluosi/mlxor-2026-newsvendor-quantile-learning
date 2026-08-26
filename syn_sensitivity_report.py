"""Tables and figures for consolidated Exp5 and Van Havre sensitivity results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DIMS = [4, 9, 14, 19, 24]
SEEDS = [82, 15, 4, 95, 36, 32, 29, 18, 14, 87]
SWEEP_INFO = {
    "lambda": {"symbol": r"$\lambda$", "values": [0.1, 0.3, 0.5, 0.7, 0.9]},
    "m": {"symbol": r"$M$", "values": [8, 32, 128, 512, 2048]},
    "simulation_num": {"symbol": r"$R$", "values": [1, 4, 16, 64, 256]},
}
DATASET_LABELS = {"exp5": "Synthetic Dataset 1", "van_havre": "Synthetic Dataset 2"}
MODEL_LABELS = {
    "gendfl_spline": "Spline-ETO",
    "spline_qfr": "Spline-QFR",
    "oracle_gmm": "BCMO",
    "erm": "ERM",
    "lightgbm": "LightGBM",
    "end_to_end": "ERM-NN",
}


def _validate_grid(frame: pd.DataFrame, parameter: str) -> None:
    expected = len(DIMS) * len(SEEDS) * len(SWEEP_INFO[parameter]["values"])
    if len(frame) != expected:
        raise RuntimeError(f"{parameter}: expected {expected} rows, found {len(frame)}")
    if frame.duplicated(["dim", "random_state", parameter]).any():
        raise RuntimeError(f"{parameter}: duplicate dimension/seed/parameter rows")
    if set(frame["dim"]) != set(DIMS) or set(frame["random_state"]) != set(SEEDS):
        raise RuntimeError(f"{parameter}: dimension or random-seed coverage mismatch")


def load_consolidated_results(project_root: Path | str = ".") -> dict[str, object]:
    project_root = Path(project_root).resolve()
    csv_root = project_root / "results_syn" / "consolidated_csv"
    van_sweeps = {
        sweep: pd.read_csv(csv_root / f"van_havre_{sweep}_ipa.csv")
        for sweep in SWEEP_INFO
    }
    for sweep, frame in van_sweeps.items():
        _validate_grid(frame, sweep)

    exp_variance = pd.read_csv(csv_root / "exp5_simulation_num_ipa_variance.csv")
    van_variance = pd.read_csv(
        csv_root / "van_havre_simulation_num_ipa_variance.csv"
    )
    _validate_grid(exp_variance, "simulation_num")
    _validate_grid(van_variance, "simulation_num")

    gendfl = pd.read_csv(csv_root / "van_havre_gendfl_spline.csv")
    qfr = pd.read_csv(csv_root / "van_havre_spline_qfr.csv")
    ete = pd.read_csv(csv_root / "van_havre_ete_models.csv")
    for name, frame in {"Spline-ETO": gendfl, "Spline-QFR": qfr}.items():
        if len(frame) != len(DIMS) * len(SEEDS):
            raise RuntimeError(f"{name}: incomplete Van Havre coverage")
    if len(ete) != len(DIMS) * len(SEEDS) * 4:
        raise RuntimeError("Van Havre ETE model coverage is incomplete")

    mean_tables = {
        (sweep, metric): pd.read_csv(
            csv_root / f"van_havre_{sweep}_{metric}_mean.csv"
        )
        for sweep in SWEEP_INFO
        for metric in ["metric1", "metric2"]
    }
    return {
        "project_root": project_root,
        "csv_root": csv_root,
        "van_sweeps": van_sweeps,
        "exp_variance": exp_variance,
        "van_variance": van_variance,
        "gendfl": gendfl,
        "qfr": qfr,
        "ete": ete,
        "mean_tables": mean_tables,
    }


def variance_summary_table(results: dict[str, object]) -> pd.DataFrame:
    combined = pd.concat(
        [results["exp_variance"], results["van_variance"]], ignore_index=True
    )
    measures = [
        "metric1",
        "metric2",
        "ipa_gradient_variance_trace",
        "weighted_ipa_gradient_variance_trace",
    ]
    summary = (
        combined.groupby(["dataset", "simulation_num"])[measures]
        .agg(["mean", "std", "count"])
    )
    summary.columns = [f"{measure}_{stat}" for measure, stat in summary.columns]
    return summary.reset_index()


def variance_reduction_table(results: dict[str, object]) -> pd.DataFrame:
    """Summarize variance reduction relative to one IPA replication."""
    summary = variance_summary_table(results)
    baseline = (
        summary[summary["simulation_num"].eq(1)]
        .set_index("dataset")["ipa_gradient_variance_trace_mean"]
    )
    summary["variance_relative_to_R1"] = summary.apply(
        lambda row: row["ipa_gradient_variance_trace_mean"] / baseline[row["dataset"]],
        axis=1,
    )
    summary["variance_reduction_vs_R1_pct"] = (
        100.0 * (1.0 - summary["variance_relative_to_R1"])
    )
    summary["R_times_variance"] = (
        summary["simulation_num"] * summary["ipa_gradient_variance_trace_mean"]
    )
    return summary[
        [
            "dataset",
            "simulation_num",
            "ipa_gradient_variance_trace_mean",
            "ipa_gradient_variance_trace_std",
            "ipa_gradient_variance_trace_count",
            "variance_relative_to_R1",
            "variance_reduction_vs_R1_pct",
            "R_times_variance",
            "metric1_mean",
            "metric2_mean",
        ]
    ]


def _mean_ci(frame: pd.DataFrame, group_columns: list[str], value: str) -> pd.DataFrame:
    summary = frame.groupby(group_columns)[value].agg(["mean", "std", "count"]).reset_index()
    summary["ci95"] = 1.96 * summary["std"] / np.sqrt(summary["count"])
    return summary


def _geometric_mean_ci(
    frame: pd.DataFrame,
    group_columns: list[str],
    value: str,
) -> pd.DataFrame:
    """Return geometric means and 95% CIs for a positive, heavy-tailed measure."""
    selected = frame[group_columns + [value]].copy()
    if (selected[value] <= 0).any():
        raise RuntimeError(f"{value} must be positive for log-scale confidence intervals")
    selected["log_value"] = np.log(selected[value])
    summary = (
        selected.groupby(group_columns)["log_value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    half_width = 1.96 * summary["std"] / np.sqrt(summary["count"])
    summary["center"] = np.exp(summary["mean"])
    summary["lower"] = np.exp(summary["mean"] - half_width)
    summary["upper"] = np.exp(summary["mean"] + half_width)
    return summary


def plot_ipa_variance_by_replications(
    results: dict[str, object],
    output_dir: Path | str | None = None,
) -> plt.Figure:
    combined = pd.concat(
        [results["exp_variance"], results["van_variance"]], ignore_index=True
    )
    colors = sns.color_palette("viridis", n_colors=len(DIMS))
    fig, axes = plt.subplots(2, len(DIMS), figsize=(25, 9), dpi=220, sharex=True)
    for row, dataset in enumerate(["exp5", "van_havre"]):
        for col, (dim, color) in enumerate(zip(DIMS, colors)):
            axis = axes[row, col]
            selected = combined[
                (combined["dataset"] == dataset) & (combined["dim"] == dim)
            ]
            summary = _geometric_mean_ci(
                selected, ["simulation_num"], "ipa_gradient_variance_trace"
            )
            x = summary["simulation_num"].to_numpy(dtype=float)
            center = summary["center"].to_numpy(dtype=float)
            lower = summary["lower"].to_numpy(dtype=float)
            upper = summary["upper"].to_numpy(dtype=float)
            axis.plot(x, center, color=color, marker="o", linewidth=2.4)
            axis.fill_between(
                x,
                lower,
                upper,
                color=color,
                alpha=0.22,
            )
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.set_xticks(SWEEP_INFO["simulation_num"]["values"])
            axis.set_xticklabels(SWEEP_INFO["simulation_num"]["values"])
            axis.set_title(f"{DATASET_LABELS[dataset]}: Dimension {dim}", fontsize=14, fontweight="bold")
            axis.set_xlabel("Replication count R", fontsize=12, fontweight="bold")
            if col == 0:
                axis.set_ylabel(
                    "Final-checkpoint IPA variance trace",
                    fontsize=12,
                    fontweight="bold",
                )
            axis.grid(True, which="both", linestyle="--", alpha=0.28)
    # fig.suptitle(
    #     "Batched IPA gradient variance versus replication count",
    #     fontsize=20,
    #     fontweight="bold",
    #     y=1.01,
    # )
    fig.tight_layout()
    if output_dir is not None:
        stem = Path(output_dir) / "ipa_variance_by_replication_and_dimension"
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    return fig


def plot_simulation_metric_sensitivity(
    results: dict[str, object],
    output_dir: Path | str | None = None,
) -> plt.Figure:
    combined = pd.concat(
        [results["exp_variance"], results["van_variance"]], ignore_index=True
    )
    colors = sns.color_palette("tab10", n_colors=len(DIMS))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=220, sharex=True)
    for row, dataset in enumerate(["exp5", "van_havre"]):
        for col, metric in enumerate(["metric1", "metric2"]):
            axis = axes[row, col]
            for dim, color in zip(DIMS, colors):
                selected = combined[
                    (combined["dataset"] == dataset) & (combined["dim"] == dim)
                ]
                summary = _mean_ci(selected, ["simulation_num"], metric)
                x = summary["simulation_num"].to_numpy(dtype=float)
                mean = summary["mean"].to_numpy(dtype=float)
                ci = summary["ci95"].to_numpy(dtype=float)
                axis.plot(x, mean, color=color, marker="o", linewidth=2.0, label=f"Dim {dim}")
                axis.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.12)
            axis.set_xscale("log", base=2)
            axis.set_xticks(SWEEP_INFO["simulation_num"]["values"])
            axis.set_xticklabels(SWEEP_INFO["simulation_num"]["values"])
            axis.set_title(
                f"{DATASET_LABELS[dataset]}: {metric.upper()}",
                fontsize=15,
                fontweight="bold",
            )
            axis.set_xlabel("Replication count R", fontsize=12, fontweight="bold")
            axis.set_ylabel("Average cost", fontsize=12, fontweight="bold")
            axis.grid(True, linestyle="--", alpha=0.28)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(
        "Metric sensitivity to IPA replication count",
        fontsize=20,
        fontweight="bold",
        y=0.995,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=len(DIMS),
        frameon=True,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    if output_dir is not None:
        stem = Path(output_dir) / "simulation_num_metric_sensitivity_exp5_van_havre"
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    return fig


def plot_van_havre_sweep_profiles(
    results: dict[str, object],
    sweep: str,
    output_dir: Path | str | None = None,
) -> plt.Figure:
    frame = results["van_sweeps"][sweep]
    values = SWEEP_INFO[sweep]["values"]
    symbol = SWEEP_INFO[sweep]["symbol"]
    colors = sns.color_palette("tab10", n_colors=len(values))
    seed_positions = np.arange(len(SEEDS))
    fig, axes = plt.subplots(2, len(DIMS), figsize=(28, 10), dpi=220, sharex=True)
    for row, metric in enumerate(["metric1", "metric2"]):
        for col, dim in enumerate(DIMS):
            axis = axes[row, col]
            dim_data = frame[frame["dim"] == dim]
            for value, color in zip(values, colors):
                series = (
                    dim_data[np.isclose(dim_data[sweep], value)]
                    .set_index("random_state")
                    .reindex(SEEDS)[metric]
                )
                if series.isna().any():
                    raise RuntimeError(f"Missing Van Havre {sweep}/{metric}/dim={dim}/value={value}")
                label = f"{symbol}={value:g}"
                axis.plot(
                    seed_positions,
                    series.to_numpy(dtype=float),
                    color=color,
                    linewidth=2.0,
                    marker="o",
                    markersize=4.5,
                    label=label,
                )
            axis.set_title(f"Dim {dim}", fontsize=15, fontweight="bold")
            axis.set_xticks(seed_positions)
            axis.set_xticklabels([])
            axis.set_xlabel("Random-seed run", fontsize=12, fontweight="bold")
            if col == 0:
                axis.set_ylabel(f"{metric.upper()} average cost", fontsize=12, fontweight="bold")
            axis.grid(True, alpha=0.28)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=len(values),
        frameon=True,
    )
    fig.suptitle(
        f"Synthetic Dataset 2 RSETO-IPA sensitivity to {symbol}",
        fontsize=21,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    if output_dir is not None:
        stem = Path(output_dir) / f"van_havre_{sweep}_metric_profiles"
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    return fig


def build_van_havre_comprehensive_data(results: dict[str, object]) -> tuple[pd.DataFrame, dict[str, str]]:
    rseto_frames = []
    for sweep, frame in results["van_sweeps"].items():
        symbol = {"lambda": "lambda", "m": "M", "simulation_num": "R"}[sweep]
        current = frame[["dim", "random_state", sweep, "metric1", "metric2"]].copy()
        current["method"] = current[sweep].map(lambda value: f"{symbol}={value:g}")
        rseto_frames.append(current[["dim", "random_state", "method", "metric1", "metric2"]])
    rseto = pd.concat(rseto_frames, ignore_index=True)
    reference = {metric: "R=16" for metric in ["metric1", "metric2"]}

    models = []
    for frame, label in [
        (results["gendfl"], "Spline-ETO"),
        (results["qfr"], "Spline-QFR"),
    ]:
        current = frame[["dim", "random_state", "metric1", "metric2"]].copy()
        current["method"] = label
        models.append(current)
    ete = results["ete"][["dim", "random_state", "model", "metric1", "metric2"]].copy()
    ete["method"] = ete["model"].map(MODEL_LABELS)
    models.append(ete[["dim", "random_state", "metric1", "metric2", "method"]])
    return pd.concat(models, ignore_index=True), reference


def build_van_havre_all_parameter_mean_table(
    results: dict[str, object],
    metric: str,
) -> pd.DataFrame:
    if metric not in {"metric1", "metric2"}:
        raise ValueError("metric must be metric1 or metric2")
    blocks = []
    for sweep in ["simulation_num", "m", "lambda"]:
        frame = results["van_sweeps"][sweep]
        values = SWEEP_INFO[sweep]["values"]
        block = frame.pivot_table(
            index="dim", columns=sweep, values=metric, aggfunc="mean"
        ).reindex(index=DIMS, columns=values)
        if sweep == "simulation_num":
            block.columns = [f"R {int(value)}" for value in values]
        elif sweep == "m":
            block.columns = [f"M {int(value)}" for value in values]
        else:
            block.columns = [f"Lambda {float(value):.1f}" for value in values]
        blocks.append(block)

    baselines = pd.concat(
        [
            results["gendfl"].groupby("dim")[metric].mean().reindex(DIMS).rename("Spline-ETO"),
            results["qfr"].groupby("dim")[metric].mean().reindex(DIMS).rename("Spline-QFR"),
        ],
        axis=1,
    )
    ete = results["ete"].copy()
    ete["model"] = ete["model"].map(MODEL_LABELS)
    model_order = ["BCMO", "ERM", "LightGBM", "ERM-NN"]
    ete_mean = ete.pivot_table(
        index="dim", columns="model", values=metric, aggfunc="mean"
    ).reindex(index=DIMS, columns=model_order)
    table = pd.concat([*blocks, baselines, ete_mean], axis=1)
    if table.isna().any().any():
        raise RuntimeError(f"Van Havre comprehensive {metric} table has missing values")
    table.index.name = "Dim"
    return table


def van_havre_all_parameter_latex_code(
    results: dict[str, object],
    metric: str,
) -> tuple[str, pd.DataFrame]:
    table = build_van_havre_all_parameter_mean_table(results, metric)
    panels = [
        ("Panel A: Replication count $R$", [f"R {value}" for value in SWEEP_INFO["simulation_num"]["values"]]),
        ("Panel B: Samples per replication $M$", [f"M {value}" for value in SWEEP_INFO["m"]["values"]]),
        ("Panel C: Gradient weight $\\lambda$", [f"Lambda {value:.1f}" for value in SWEEP_INFO["lambda"]["values"]]),
        ("Panel D: Comparison models", ["Spline-ETO", "Spline-QFR", "BCMO", "ERM", "LightGBM", "ERM-NN"]),
    ]
    labels = {
        **{f"R {value}": rf"$R={value}$" for value in SWEEP_INFO["simulation_num"]["values"]},
        **{f"M {value}": rf"$M={value}$" for value in SWEEP_INFO["m"]["values"]},
        **{f"Lambda {value:.1f}": rf"$\lambda={value:.1f}$" for value in SWEEP_INFO["lambda"]["values"]},
        **{name: name for name in ["Spline-ETO", "Spline-QFR", "BCMO", "ERM", "LightGBM", "ERM-NN"]},
    }
    row_minima = table.min(axis=1)
    row_end = " " + chr(92) * 2
    metric_label = "Metric 1" if metric == "metric1" else "Metric 2"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        rf"\caption{{Van Havre {metric_label}. Each entry is the mean over ten random seeds.}}",
        rf"\label{{tab:van_havre_all_parameters_{metric}}}",
    ]
    for panel_index, (title, columns) in enumerate(panels):
        if panel_index:
            lines.append(r"\par\medskip")
        lines.extend(
            [
                r"\begin{tabular}{rr" + "r" * len(columns) + "}",
                r"\toprule",
                rf"\multicolumn{{{len(columns) + 2}}}{{l}}{{\textbf{{{title}}}}}" + row_end,
                r"\midrule",
                "ID & Dim & " + " & ".join(labels[column] for column in columns) + row_end,
                r"\midrule",
            ]
        )
        for latex_id, dim in enumerate(DIMS):
            cells = [str(latex_id), str(dim)]
            for column in columns:
                value = float(table.loc[dim, column])
                formatted = f"{value:.2f}"
                if np.isclose(value, row_minima.loc[dim], rtol=1e-10, atol=1e-10):
                    formatted = rf"\textbf{{{formatted}}}"
                cells.append(formatted)
            lines.append(" & ".join(cells) + row_end)
        lines.extend([r"\bottomrule", r"\end{tabular}"])
    lines.append(r"\end{table}")
    return "\n".join(lines), table


def plot_van_havre_comprehensive_box(
    results: dict[str, object],
    output_dir: Path | str | None = None,
) -> plt.Figure:
    models, reference = build_van_havre_comprehensive_data(results)
    colors = sns.color_palette("Set2", n_colors=len(DIMS))
    hue_order = [f"Dim {dim}" for dim in DIMS]
    fig, axes = plt.subplots(2, 1, figsize=(17, 13), dpi=250)
    for axis, metric in zip(axes, ["metric1", "metric2"]):
        selected_reference = []
        for sweep, frame in results["van_sweeps"].items():
            symbol = {"lambda": "lambda", "m": "M", "simulation_num": "R"}[sweep]
            label = reference[metric]
            if not label.startswith(f"{symbol}="):
                continue
            value = float(label.split("=", 1)[1])
            selected_reference.append(
                frame[np.isclose(frame[sweep], value)][
                    ["dim", "random_state", "metric1", "metric2"]
                ]
            )
        if len(selected_reference) != 1:
            raise RuntimeError(f"Could not resolve Van Havre reference {reference[metric]}")
        reference_frame = selected_reference[0].copy()
        reference_label = "RSETO-IPA\n(reference)"
        reference_frame["method"] = reference_label
        plot_data = pd.concat([models, reference_frame], ignore_index=True)
        plot_data["dimension"] = plot_data["dim"].map(lambda value: f"Dim {int(value)}")
        order = [
            "Spline-ETO",
            "Spline-QFR",
            reference_label,
            "BCMO",
            "ERM",
            "LightGBM",
            "ERM-NN",
        ]
        sns.boxplot(
            data=plot_data,
            x="method",
            y=metric,
            hue="dimension",
            order=order,
            hue_order=hue_order,
            palette=colors,
            width=0.78,
            linewidth=0.8,
            fliersize=2.7,
            ax=axis,
        )
        axis.set_yscale("log")
        axis.set_title(metric.upper(), fontsize=16, fontweight="bold")
        axis.set_ylabel("Average cost", fontsize=14, fontweight="bold")
        axis.set_xlabel("Models", fontsize=15, fontweight="bold")
        axis.tick_params(axis="x", labelsize=14, rotation=18)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")
        axis.grid(True, axis="y", which="both", linestyle="--", alpha=0.28)
        if axis.get_legend() is not None:
            axis.get_legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Feature dimension",
        loc="center left",
        bbox_to_anchor=(0.985, 0.5),
        frameon=True,
    )
    fig.suptitle(
        "Van Havre complete model comparison",
        fontsize=20,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 0.975, 0.98], h_pad=3.0)
    if output_dir is not None:
        stem = Path(output_dir) / "van_havre_comprehensive_model_boxplot"
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    return fig
