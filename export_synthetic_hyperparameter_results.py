"""Export seed-level synthetic sensitivity results and LaTeX mean tables.

Every sensitivity metric is read from its individual ``.npy`` file and checked
against the corresponding ``detail.csv`` entry before it is exported.
"""

from __future__ import annotations

import argparse
import json
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
SWEEP_PREFIX = {"lambda": "lambda", "m": "m", "simulation_num": "R"}
REFERENCE_ORDER = ["GenDFL", "Spline QFR", "ERM", "LightGBM", "Benchmark"]
METHOD_LABELS = {
    "gendfl_spline": "GenDFL",
    "spline_qfr": "Spline QFR",
    "erm": "ERM",
    "lightgbm": "LightGBM",
    "end_to_end": "Benchmark",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, default=Path("analysis_outputs"))
    parser.add_argument(
        "--ete-detail",
        type=Path,
        default=Path(
            "analysis_outputs_ete/"
            "fixed_dgp_exp5_traditional_50epochs_projected_sgd/"
            "fixed_dgp_results_detail.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def scalar_npy(path: Path) -> float:
    values = np.asarray(np.load(path, allow_pickle=False), dtype=float).reshape(-1)
    if values.size != 1:
        raise ValueError(f"Expected one scalar in {path}, found shape {values.shape}")
    value = float(values[0])
    if not np.isfinite(value):
        raise ValueError(f"Non-finite metric in {path}: {value}")
    return value


def expected_methods(sweep: str) -> list[str]:
    prefix = SWEEP_PREFIX[sweep]
    methods = ["gendfl_spline", "spline_qfr"]
    for value in SWEEP_VALUES[sweep]:
        value_text = f"{value:.1f}" if sweep == "lambda" else str(int(value))
        methods.append(f"rseto_ipa_{prefix}{value_text}")
    return methods


def load_sensitivity_npy(analysis_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dim in DIMS:
        for seed in SEEDS:
            experiment_dir = analysis_root / f"spline_sensitivity_iid_exp5_dim{dim}_seed{seed}"
            for sweep in SWEEP_VALUES:
                sweep_dir = experiment_dir / sweep
                detail_path = sweep_dir / "detail.csv"
                if not detail_path.exists():
                    raise FileNotFoundError(detail_path)
                detail = pd.read_csv(detail_path)
                if detail["method"].duplicated().any():
                    raise ValueError(f"Duplicate methods in {detail_path}")
                detail = detail.set_index("method", drop=False)
                methods = expected_methods(sweep)
                if set(detail.index) != set(methods):
                    raise ValueError(
                        f"Unexpected methods in {detail_path}: "
                        f"missing={sorted(set(methods) - set(detail.index))}, "
                        f"extra={sorted(set(detail.index) - set(methods))}"
                    )
                for method in methods:
                    metric1_path = sweep_dir / f"{method}_metric1.npy"
                    metric2_path = sweep_dir / f"{method}_metric2.npy"
                    metric1 = scalar_npy(metric1_path)
                    metric2 = scalar_npy(metric2_path)
                    detail_row = detail.loc[method]
                    if not np.isclose(metric1, float(detail_row["metric1"]), rtol=1e-9, atol=1e-9):
                        raise ValueError(f"metric1 mismatch for {metric1_path}")
                    if not np.isclose(metric2, float(detail_row["metric2"]), rtol=1e-9, atol=1e-9):
                        raise ValueError(f"metric2 mismatch for {metric2_path}")
                    is_rseto = method.startswith("rseto_ipa_")
                    parameter = float(detail_row["sweep_value"]) if is_rseto else np.nan
                    rows.append(
                        {
                            "dim": dim,
                            "random_state": seed,
                            "sweep": sweep,
                            "parameter": parameter,
                            "model": "RSETO-IPA" if is_rseto else METHOD_LABELS[method],
                            "method": method,
                            "metric1": metric1,
                            "metric2": metric2,
                            "metric1_npy": str(metric1_path),
                            "metric2_npy": str(metric2_path),
                            "detail_csv": str(detail_path),
                        }
                    )
    frame = pd.DataFrame(rows)
    expected = len(DIMS) * len(SEEDS) * len(SWEEP_VALUES) * 7
    if len(frame) != expected:
        raise ValueError(f"Loaded {len(frame)} sensitivity rows, expected {expected}")
    return frame.sort_values(
        ["sweep", "dim", "random_state", "model", "parameter"], na_position="first"
    ).reset_index(drop=True)


def deduplicate_spline_baseline(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    selected = frame[frame["model"] == model].copy()
    rows = []
    for (dim, seed), group in selected.groupby(["dim", "random_state"], sort=True):
        if set(group["sweep"]) != set(SWEEP_VALUES):
            raise ValueError(f"Missing {model} sweep copy for dim={dim}, seed={seed}")
        for metric in ["metric1", "metric2"]:
            values = group[metric].to_numpy(dtype=float)
            if not np.allclose(values, values[0], rtol=1e-9, atol=1e-9):
                raise ValueError(f"Inconsistent {model} {metric} across sweeps for dim={dim}, seed={seed}")
        first = group.sort_values("sweep").iloc[0]
        rows.append(
            {
                "dim": int(dim),
                "random_state": int(seed),
                "model": model,
                "metric1": float(first["metric1"]),
                "metric2": float(first["metric2"]),
                "verified_sweeps": ",".join(sorted(group["sweep"].tolist())),
            }
        )
    result = pd.DataFrame(rows)
    expected = len(DIMS) * len(SEEDS)
    if len(result) != expected:
        raise ValueError(f"Loaded {len(result)} unique {model} rows, expected {expected}")
    return result.sort_values(["dim", "random_state"]).reset_index(drop=True)


def load_ete_references(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "error" in frame and frame["error"].notna().any():
        raise RuntimeError(frame.loc[frame["error"].notna(), ["dim", "random_state", "model", "error"]])
    frame = frame[frame["dim"].isin(DIMS) & frame["random_state"].isin(SEEDS)].copy()
    frame["model"] = frame["model"].map(METHOD_LABELS)
    if frame["model"].isna().any():
        raise ValueError("Unknown ETE model label")
    frame = frame[["dim", "random_state", "model", "metric1", "metric2"]]
    expected = len(DIMS) * len(SEEDS) * 3
    if len(frame) != expected:
        raise ValueError(f"Loaded {len(frame)} ETE rows, expected {expected}")
    if frame.duplicated(["dim", "random_state", "model"]).any():
        raise ValueError("Duplicate ETE reference rows")
    return frame.sort_values(["dim", "random_state", "model"]).reset_index(drop=True)


def parameter_column_name(sweep: str, value: float) -> str:
    if sweep == "lambda":
        return f"Lambda {float(value):.1f}"
    if sweep == "m":
        return f"M {int(value)}"
    return f"R {int(value)}"


def build_mean_table(
    rseto: pd.DataFrame,
    references: pd.DataFrame,
    sweep: str,
    metric: str,
) -> pd.DataFrame:
    values = SWEEP_VALUES[sweep]
    selected = rseto[rseto["sweep"] == sweep]
    counts = selected.groupby(["dim", "parameter"])["random_state"].nunique()
    if not (counts == len(SEEDS)).all():
        raise ValueError(f"Incomplete seed coverage for {sweep} {metric}")
    sweep_mean = selected.pivot_table(index="dim", columns="parameter", values=metric, aggfunc="mean")
    sweep_mean = sweep_mean.reindex(index=DIMS, columns=values)
    sweep_mean.columns = [parameter_column_name(sweep, value) for value in values]
    reference_mean = references.pivot_table(index="dim", columns="model", values=metric, aggfunc="mean")
    reference_mean = reference_mean.reindex(index=DIMS, columns=REFERENCE_ORDER)
    return pd.concat([sweep_mean, reference_mean], axis=1).reset_index().rename(columns={"dim": "Dim"})


def latex_rows(table: pd.DataFrame) -> list[str]:
    lines = []
    for _, row in table.iterrows():
        numeric = pd.to_numeric(row.drop(labels="Dim"), errors="raise")
        minimum = float(numeric.min())
        cells = [str(int(row["Dim"]))]
        for column in table.columns[1:]:
            value = float(row[column])
            formatted = f"{value:.2f}"
            if np.isclose(value, minimum, rtol=1e-10, atol=1e-10):
                formatted = rf"\textbf{{{formatted}}}"
            cells.append(formatted)
        lines.append(" & ".join(cells) + r" \\")
    return lines


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def export_results(analysis_root: Path, ete_detail: Path, output_dir: Path) -> dict[str, object]:
    raw_dir = output_dir / "raw"
    mean_dir = output_dir / "mean_tables"
    latex_dir = output_dir / "latex"
    for directory in [raw_dir, mean_dir, latex_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    all_sensitivity = load_sensitivity_npy(analysis_root)
    rseto = all_sensitivity[all_sensitivity["model"] == "RSETO-IPA"].copy()
    spline_all = all_sensitivity[all_sensitivity["model"].isin(["GenDFL", "Spline QFR"])].copy()
    gendfl_all = spline_all[spline_all["model"] == "GenDFL"].copy()
    qfr_all = spline_all[spline_all["model"] == "Spline QFR"].copy()
    gendfl_unique = deduplicate_spline_baseline(all_sensitivity, "GenDFL")
    qfr_unique = deduplicate_spline_baseline(all_sensitivity, "Spline QFR")
    ete = load_ete_references(ete_detail)
    references = pd.concat([gendfl_unique, qfr_unique, ete], ignore_index=True, sort=False)
    references = references[["dim", "random_state", "model", "metric1", "metric2"]]

    datasets = {
        "all_hyperparameter_seed_results": all_sensitivity,
        "rseto_ipa_sweep_seed_results": rseto,
        "spline_baseline_all_sweeps": spline_all,
        "gendfl_spline_all_sweeps": gendfl_all,
        "gendfl_spline_seed_results": gendfl_unique,
        "spline_qfr_all_sweeps": qfr_all,
        "spline_qfr_seed_results": qfr_unique,
        "ete_reference_seed_results": ete,
        "reference_model_seed_results": references,
    }
    for name, frame in datasets.items():
        write_csv(frame, raw_dir / f"{name}.csv")

    coverage = (
        rseto.groupby(["sweep", "dim", "parameter"])["random_state"]
        .nunique()
        .rename("seed_count")
        .reset_index()
        .sort_values(["sweep", "dim", "parameter"])
    )
    write_csv(coverage, raw_dir / "coverage.csv")

    mean_tables: dict[str, pd.DataFrame] = {}
    latex_by_table: dict[str, list[str]] = {}
    combined_lines = [
        "% Mean results and LaTeX rows",
        "% Each entry is the mean over ten random seeds.",
        "% The smallest mean in each dimension is bolded; no standard deviations are appended.",
    ]
    for sweep in SWEEP_VALUES:
        for metric in ["metric1", "metric2"]:
            name = f"{sweep}_{metric}"
            table = build_mean_table(rseto, references, sweep, metric)
            rows = latex_rows(table)
            mean_tables[name] = table
            latex_by_table[name] = rows
            write_csv(table, mean_dir / f"{name}_mean.csv")
            (latex_dir / f"{name}_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
            combined_lines.extend(["", f"% {sweep} - {metric}", *rows])
    (latex_dir / "all_mean_latex_rows.tex").write_text(
        "\n".join(combined_lines) + "\n", encoding="utf-8"
    )

    markdown_lines = [
        "### Mean results and LaTeX rows",
        "",
        "Each entry is the mean over the ten random seeds. The smallest mean in each dimension is bolded, without appending standard deviations.",
    ]
    for name, rows in latex_by_table.items():
        markdown_lines.extend(["", f"#### {name}", "", "```latex", *rows, "```"])
    (latex_dir / "mean_results_and_latex_rows.md").write_text(
        "\n".join(markdown_lines) + "\n", encoding="utf-8"
    )

    detail_workbook = output_dir / "synthetic_hyperparameter_all_seed_results.xlsx"
    detail_sheet_names = {
        "all_hyperparameter_seed_results": "all_seed_results",
        "rseto_ipa_sweep_seed_results": "rseto_ipa_sweeps",
        "spline_baseline_all_sweeps": "spline_baseline_sweeps",
        "gendfl_spline_all_sweeps": "gendfl_all_sweeps",
        "gendfl_spline_seed_results": "gendfl_seed_results",
        "spline_qfr_all_sweeps": "qfr_all_sweeps",
        "spline_qfr_seed_results": "qfr_seed_results",
        "ete_reference_seed_results": "ete_seed_results",
        "reference_model_seed_results": "reference_seed_results",
    }
    with pd.ExcelWriter(detail_workbook) as writer:
        for name, frame in datasets.items():
            frame.to_excel(writer, sheet_name=detail_sheet_names[name], index=False)
        coverage.to_excel(writer, sheet_name="coverage", index=False)
        for name, table in mean_tables.items():
            table.to_excel(writer, sheet_name=f"mean_{name}"[:31], index=False)

    manifest = {
        "experiment": "iid exp5 synthetic spline sensitivity",
        "dimensions": DIMS,
        "random_states": SEEDS,
        "sweeps": SWEEP_VALUES,
        "validation": "Every sensitivity metric NPY was checked against detail.csv.",
        "row_counts": {name: len(frame) for name, frame in datasets.items()},
        "coverage_rows": len(coverage),
        "mean_tables": list(mean_tables),
        "detail_workbook": detail_workbook.name,
        "latex_files": [f"{name}_rows.tex" for name in mean_tables] + ["all_mean_latex_rows.tex"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    workbook_payload = {
        "datasets": {
            name: frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")
            for name, frame in datasets.items()
        },
        "coverage": coverage.to_dict(orient="records"),
        "mean_tables": {name: table.to_dict(orient="records") for name, table in mean_tables.items()},
        "latex_rows": latex_by_table,
        "manifest": manifest,
    }
    (output_dir / "workbook_payload.json").write_text(
        json.dumps(workbook_payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    return manifest


def main() -> None:
    args = parse_args()
    manifest = export_results(args.analysis_root, args.ete_detail, args.output_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
