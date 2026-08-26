"""Paired BCMO-difference tables and figures for Exp5 and Van Havre."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from syn_sensitivity_report import DIMS, SEEDS, load_consolidated_results


SWEEP_VALUES = {
    "simulation_num": [1, 4, 16, 64, 256],
    "m": [8, 32, 128, 512, 2048],
    "lambda": [0.1, 0.3, 0.5, 0.7, 0.9],
}
MODEL_ORDER = [
    *[f"R={value}" for value in SWEEP_VALUES["simulation_num"]],
    *[f"M={value}" for value in SWEEP_VALUES["m"]],
    *[f"lambda={value:.1f}" for value in SWEEP_VALUES["lambda"]],
    "Spline-ETO",
    "Spline-QFR",
    "BCMO",
    "ERM",
    "LightGBM",
    "ERM-NN",
]
DIFFERENCE_MODEL_ORDER = [method for method in MODEL_ORDER if method != "BCMO"]
COMPARISON_MODELS = ["Spline-ETO", "Spline-QFR", "ERM", "LightGBM", "ERM-NN"]
RSETO_METHODS = MODEL_ORDER[:15]
REFERENCE_RSETO_DEFAULTS = {"R": 16, "M": 128, "lambda": 0.5}


def complete_reference_parameters(method: str) -> dict[str, int | float]:
    """Expand one swept RSETO-IPA setting with the two fixed defaults."""
    parameters = REFERENCE_RSETO_DEFAULTS.copy()
    if method.startswith("R="):
        parameters["R"] = int(float(method.split("=", 1)[1]))
    elif method.startswith("M="):
        parameters["M"] = int(float(method.split("=", 1)[1]))
    elif method.startswith("lambda="):
        parameters["lambda"] = float(method.split("=", 1)[1])
    else:
        raise ValueError(f"Unknown RSETO-IPA reference setting: {method}")
    return parameters


def reference_parameter_text(method: str) -> str:
    parameters = complete_reference_parameters(method)
    return (
        f"R={parameters['R']}, M={parameters['M']}, "
        f"lambda={parameters['lambda']:g}"
    )


def reference_axis_label(method: str) -> str:
    parameters = complete_reference_parameters(method)
    return (
        "RSETO-IPA\n"
        rf"$R={parameters['R']},\ M={parameters['M']}$" "\n"
        rf"$\lambda={parameters['lambda']:g}$"
    )


def _standardize_sweep(frame: pd.DataFrame, sweep: str) -> pd.DataFrame:
    current = frame[["dim", "random_state", sweep, "metric1", "metric2"]].copy()
    if sweep == "simulation_num":
        current["method"] = current[sweep].map(lambda value: f"R={int(value)}")
    elif sweep == "m":
        current["method"] = current[sweep].map(lambda value: f"M={int(value)}")
    else:
        current["method"] = current[sweep].map(lambda value: f"lambda={float(value):.1f}")
    return current[["dim", "random_state", "method", "metric1", "metric2"]]


def _standardize_baseline(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    current = frame[["dim", "random_state", "metric1", "metric2"]].copy()
    current["method"] = method
    return current[["dim", "random_state", "method", "metric1", "metric2"]]


def _validate_seed_results(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    expected_rows = len(MODEL_ORDER) * len(DIMS) * len(SEEDS)
    if len(frame) != expected_rows:
        raise RuntimeError(f"{dataset}: expected {expected_rows} rows, found {len(frame)}")
    if frame.duplicated(["method", "dim", "random_state"]).any():
        raise RuntimeError(f"{dataset}: duplicate method/dimension/seed rows")
    if set(frame["method"]) != set(MODEL_ORDER):
        raise RuntimeError(f"{dataset}: model coverage mismatch")
    if set(frame["dim"]) != set(DIMS) or set(frame["random_state"]) != set(SEEDS):
        raise RuntimeError(f"{dataset}: dimension or seed coverage mismatch")
    if frame[["metric1", "metric2"]].isna().any().any():
        raise RuntimeError(f"{dataset}: missing metric values")
    result = frame.copy()
    result.insert(0, "dataset", dataset)
    return result.sort_values(["dim", "random_state", "method"]).reset_index(drop=True)


def load_exp5_seed_results(project_root: Path | str = ".") -> pd.DataFrame:
    project_root = Path(project_root).resolve()
    results_root = project_root / "results_syn"
    frames = [
        _standardize_sweep(pd.read_excel(results_root / "syn_simulation_num_ipa.xlsx"), "simulation_num"),
        _standardize_sweep(pd.read_excel(results_root / "syn_m_ipa.xlsx"), "m"),
        _standardize_sweep(pd.read_excel(results_root / "syn_lambda_ipa.xlsx"), "lambda"),
        _standardize_baseline(pd.read_excel(results_root / "syn_gendfl_spline.xlsx"), "Spline-ETO"),
        _standardize_baseline(pd.read_excel(results_root / "syn_spline_qfr.xlsx"), "Spline-QFR"),
    ]
    ete = pd.read_csv(results_root / "consolidated_csv" / "exp5_ete_models.csv")
    ete_labels = {
        "oracle_gmm": "BCMO",
        "erm": "ERM",
        "lightgbm": "LightGBM",
        "end_to_end": "ERM-NN",
    }
    ete = ete[["dim", "random_state", "model", "metric1", "metric2"]].copy()
    ete["method"] = ete.pop("model").map(ete_labels)
    if ete["method"].isna().any():
        raise RuntimeError("Exp5 ETE results contain an unknown method")
    frames.append(ete[["dim", "random_state", "method", "metric1", "metric2"]])
    return _validate_seed_results(pd.concat(frames, ignore_index=True), "exp5")


def load_van_havre_seed_results(project_root: Path | str = ".") -> pd.DataFrame:
    results = load_consolidated_results(project_root)
    frames = [
        _standardize_sweep(results["van_sweeps"]["simulation_num"], "simulation_num"),
        _standardize_sweep(results["van_sweeps"]["m"], "m"),
        _standardize_sweep(results["van_sweeps"]["lambda"], "lambda"),
        _standardize_baseline(results["gendfl"], "Spline-ETO"),
        _standardize_baseline(results["qfr"], "Spline-QFR"),
    ]
    ete = results["ete"][["dim", "random_state", "model", "metric1", "metric2"]].copy()
    ete["method"] = ete.pop("model").map(
        {
            "oracle_gmm": "BCMO",
            "erm": "ERM",
            "lightgbm": "LightGBM",
            "end_to_end": "ERM-NN",
        }
    )
    if ete["method"].isna().any():
        raise RuntimeError("Van Havre ETE results contain an unknown method")
    frames.append(ete[["dim", "random_state", "method", "metric1", "metric2"]])
    return _validate_seed_results(pd.concat(frames, ignore_index=True), "van_havre")


def subtract_paired_bcmo(frame: pd.DataFrame) -> pd.DataFrame:
    bcmo = frame[frame["method"].eq("BCMO")][
        ["dataset", "dim", "random_state", "metric1", "metric2"]
    ].rename(columns={"metric1": "bcmo_metric1", "metric2": "bcmo_metric2"})
    expected_oracle_rows = len(DIMS) * len(SEEDS)
    if len(bcmo) != expected_oracle_rows:
        raise RuntimeError(
            f"{frame['dataset'].iat[0]}: expected {expected_oracle_rows} BCMO rows, found {len(bcmo)}"
        )
    models = frame[~frame["method"].eq("BCMO")].copy()
    merged = models.merge(
        bcmo,
        on=["dataset", "dim", "random_state"],
        how="left",
        validate="many_to_one",
    )
    if merged[["bcmo_metric1", "bcmo_metric2"]].isna().any().any():
        raise RuntimeError("BCMO pairing produced missing oracle costs")
    merged["metric1_difference"] = merged["metric1"] - merged["bcmo_metric1"]
    merged["metric2_difference"] = merged["metric2"] - merged["bcmo_metric2"]
    expected_rows = len(DIFFERENCE_MODEL_ORDER) * len(DIMS) * len(SEEDS)
    if len(merged) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} paired differences, found {len(merged)}")
    return merged.sort_values(["dim", "random_state", "method"]).reset_index(drop=True)


def build_difference_mean_table(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    value_column = f"{metric}_difference"
    table = frame.pivot_table(
        index="dim", columns="method", values=value_column, aggfunc="mean"
    ).reindex(index=DIMS, columns=DIFFERENCE_MODEL_ORDER)
    if table.isna().any().any():
        raise RuntimeError(f"Missing values in {frame['dataset'].iat[0]} {metric} difference table")
    table.index.name = "Dim"
    return table


def difference_latex_code(
    table: pd.DataFrame,
    *,
    dataset_label: str,
    metric: str,
) -> str:
    panels = [
        ("Panel A: Replication count $R$", [f"R={value}" for value in SWEEP_VALUES["simulation_num"]]),
        ("Panel B: Samples per replication $M$", [f"M={value}" for value in SWEEP_VALUES["m"]]),
        ("Panel C: Gradient weight $\\lambda$", [f"lambda={value:.1f}" for value in SWEEP_VALUES["lambda"]]),
        ("Panel D: Comparison models", COMPARISON_MODELS),
    ]
    labels = {
        **{f"R={value}": rf"$R={value}$" for value in SWEEP_VALUES["simulation_num"]},
        **{f"M={value}": rf"$M={value}$" for value in SWEEP_VALUES["m"]},
        **{f"lambda={value:.1f}": rf"$\lambda={value:.1f}$" for value in SWEEP_VALUES["lambda"]},
        **{method: method for method in COMPARISON_MODELS},
    }
    row_minima = table.min(axis=1)
    row_end = " " + chr(92) * 2
    metric_label = "Metric 1" if metric == "metric1" else "Metric 2"
    dataset_slug = dataset_label.lower().replace(" ", "_")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        rf"\caption{{{dataset_label} {metric_label} cost difference relative to BCMO. Negative values outperform BCMO.}}",
        rf"\label{{tab:{dataset_slug}_bcmo_difference_{metric}}}",
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
    return "\n".join(lines)


def select_reference_methods(frame: pd.DataFrame) -> dict[str, str]:
    """Select the lowest-mean RSETO-IPA setting separately for each metric."""
    rseto = frame[frame["method"].isin(RSETO_METHODS)]
    if set(rseto["method"]) != set(RSETO_METHODS):
        raise RuntimeError("RSETO-IPA reference candidates are incomplete")
    references = {}
    for metric in ["metric1", "metric2"]:
        mean_differences = rseto.groupby("method")[f"{metric}_difference"].mean()
        references[metric] = str(mean_differences.idxmin())
    return references


def plot_bcmo_difference_boxplots(
    differences: dict[str, pd.DataFrame],
    output_dir: Path | str | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    dataset_labels = {"exp5": "Synthetic Dataset 1", "van_havre": "Synthetic Dataset 2"}
    references = {
        dataset: select_reference_methods(frame)
        for dataset, frame in differences.items()
    }
    reference_rows = [
        {"dataset": dataset, **methods}
        for dataset, methods in references.items()
    ]
    reference_table = pd.DataFrame(reference_rows)
    for metric in ["metric1", "metric2"]:
        reference_table[metric] = reference_table[metric].map(reference_parameter_text)
    hue_order = [f"Dim {dim}" for dim in DIMS]
    palette = sns.color_palette("Set2", n_colors=len(DIMS))
    fig, axes = plt.subplots(2, 2, figsize=(20, 13), dpi=300, sharex=False)
    
    for row, dataset in enumerate(["exp5", "van_havre"]):
        frame = differences[dataset]
        for col, metric in enumerate(["metric1", "metric2"]):
            axis = axes[row, col]
            reference_method = references[dataset][metric]
            reference_label = reference_axis_label(reference_method)
            selected = frame[
                frame["method"].isin(COMPARISON_MODELS + [reference_method])
            ].copy()
            selected.loc[selected["method"].eq(reference_method), "method"] = reference_label
            selected["dimension"] = selected["dim"].map(lambda value: f"Dim {int(value)}")
            order = [
                "Spline-ETO",
                "Spline-QFR",
                reference_label,
                "ERM",
                "LightGBM",
                "ERM-NN",
            ]
            sns.boxplot(
                data=selected,
                x="method",
                y=f"{metric}_difference",
                hue="dimension",
                order=order,
                hue_order=hue_order,
                palette=palette,
                width=0.78,
                linewidth=0.8,
                fliersize=2.5,
                ax=axis,
            )
            axis.axhline(0.0, color="black", linestyle="--", linewidth=1.4, alpha=0.8)
            axis.set_yscale("symlog", linthresh=10)
            axis.set_title(
                f"{dataset_labels[dataset]}: {metric.upper()}",
                fontsize=15,
                fontweight="bold",
            )
            axis.set_ylabel("Cost difference from BCMO", fontsize=12, fontweight="bold")
            axis.set_xlabel("Models", fontsize=12, fontweight="bold")
            axis.tick_params(axis="x", labelsize=15, rotation=0, pad=6)
            for label in axis.get_xticklabels():
                label.set_horizontalalignment("center")
                label.set_multialignment("center")
                label.set_linespacing(1.05)
            axis.grid(True, axis="y", which="both", linestyle="--", alpha=0.25)
            if axis.get_legend() is not None:
                axis.get_legend().remove()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Feature dimension",
        loc="center left",
        bbox_to_anchor=(0.985, 0.5),
        frameon=True,
        fontsize=12,
        title_fontsize=13,
    )
    # fig.suptitle(
    #     "Paired model cost differences relative to BCMO",
    #     fontsize=20,
    #     fontweight="bold",
    #     y=0.995,
    # )
    fig.tight_layout(rect=[0, 0.035, 0.975, 0.97], h_pad=2.5, w_pad=2.0)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = output_dir / "bcmo_difference_boxplot"
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    return fig, reference_table


def build_all_outputs(project_root: Path | str = ".") -> dict[str, object]:
    project_root = Path(project_root).resolve()
    output_dir = project_root / "results_syn" / "difference_output_bcmo"
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_results = {
        "exp5": load_exp5_seed_results(project_root),
        "van_havre": load_van_havre_seed_results(project_root),
    }
    differences = {
        dataset: subtract_paired_bcmo(frame)
        for dataset, frame in seed_results.items()
    }
    combined = pd.concat(differences.values(), ignore_index=True)
    combined.to_csv(output_dir / "bcmo_paired_seed_differences.csv", index=False)

    mean_tables = {}
    latex_codes = {}
    dataset_labels = {"exp5": "Synthetic Dataset 1", "van_havre": "Synthetic Dataset 2"}
    for dataset, frame in differences.items():
        for metric in ["metric1", "metric2"]:
            table = build_difference_mean_table(frame, metric)
            mean_tables[(dataset, metric)] = table
            table.reset_index().to_csv(
                output_dir / f"{dataset}_{metric}_bcmo_difference_mean.csv",
                index=False,
            )
            latex = difference_latex_code(
                table,
                dataset_label=dataset_labels[dataset],
                metric=metric,
            )
            latex_codes[(dataset, metric)] = latex
            (output_dir / f"{dataset}_{metric}_bcmo_difference.tex").write_text(
                latex + "\n", encoding="utf-8"
            )

    figure, references = plot_bcmo_difference_boxplots(differences, output_dir)
    references.to_csv(output_dir / "reference_rseto_parameters.csv", index=False)
    return {
        "output_dir": output_dir,
        "seed_results": seed_results,
        "differences": differences,
        "combined": combined,
        "mean_tables": mean_tables,
        "latex_codes": latex_codes,
        "reference_parameters": references,
        "figure": figure,
    }
