#!/usr/bin/env python3
"""
Aggregate per-seed result CSVs into a LaTeX table.

Usage:
    python aggregate_results.py --dataset heart --sensitive sex_age

Optional:
    --methods 1hot generic intersectional multitask   # override default set
    --results_root results                             # root folder containing dataset folders
    --precision 3                                      # decimal places
    --output table.tex                                 # write LaTeX table to file instead of stdout

The script expects directory layout:
results/<dataset>/<method>/<sensitive>/results/seed_*.csv
Each seed_*.csv has columns test_group_0, test_group_1, ... and possibly other metrics (ignored).
It computes mean and std across seeds for every test_group_* column for each method.
Adds a Gain column: macro-average(test_group_* means) difference vs generic (method_mean - generic_mean).
"""
from __future__ import annotations
import argparse
import csv
import math
import sys
from pathlib import Path
from typing import List, Dict, Tuple

DEFAULT_METHODS = ["generic", "1hot", "intersectional", "multitask", "decoupled"]

# Mapping for group labels - now using attribute-index format
# For sex_age: sex[0]age[0] = "Young,Male", sex[0]age[1] = "Old,Male", etc.
GROUP_LABELS = {
    "sex_age": {
        (0, 0): "Young,Male",
        (0, 1): "Young,Female",
        (1, 0): "Old,Male",
        (1, 1): "Old,Female",
    },
    "sex": {
        (0,): "Male",
        (1,): "Female",
    },
    "age": {
        (0,): "Young",
        (1,): "Old",
    },
}

# Method name formatting
METHOD_NAMES = {
    "generic": "Generic",
    "1hot": "1hot",
    "intersectional": "Intersectional",
    "multitask": "Multitask",
    "decoupled": "Decoupled",
}


def read_seed_file(path: Path) -> Dict[str, float]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            return {}
        return {k: float(v) for k, v in rows[0].items() if v is not None and v != ""}


def collect_method_seeds(results_dir: Path) -> List[Dict[str, float]]:
    if not results_dir.exists():
        return []
    seed_files = sorted(results_dir.glob("seed_*.csv"))
    data = []
    for fp in seed_files:
        try:
            row = read_seed_file(fp)
            if row:
                data.append(row)
        except Exception:
            continue
    return data


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    m = sum(values) / len(values)
    if len(values) == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return m, math.sqrt(var)


def format_cell(mean: float, std: float, precision: int) -> str:
    if math.isnan(mean):
        return "--"
    return f"{mean*100:.{precision}f} $\\pm$ {std*100:.{precision}f}"


def build_single_metric_table(
    dataset: str,
    sensitive: str,
    root: Path,
    methods: List[str],
    precision: int,
    metric: str,
) -> str:
    """Build LaTeX table for a single metric (loss, acc, auroc)."""
    import re

    method_stats: Dict[str, List[Dict[str, float]]] = {}

    # Collect seed data per method
    for method in methods:
        results_path = root / dataset / method / sensitive / "results"
        seed_rows = collect_method_seeds(results_path)
        if not seed_rows:
            continue
        method_stats[method] = seed_rows

    if not method_stats:
        raise SystemExit("No data found for specified configuration.")

    # Determine group columns from the GENERIC method to ensure consistency
    generic_rows = method_stats.get("generic")
    if not generic_rows:
        raise SystemExit("Generic method results required.")

    # Keep only test_group_*_[metric] columns for the given metric (from generic method)
    group_cols = [
        c
        for c in generic_rows[0].keys()
        if c.startswith("test_group_") and c.endswith(f"_{metric}")
    ]

    # Extract and sort by group index
    def extract_group_index(col):
        # Handle both new format (test_group_sex[0]age[0]_metric) and old format (test_group_0_metric)
        if "[" in col:
            matches = re.findall(r"\[(\d+)\]", col)
            return tuple(int(m) for m in matches)
        else:
            # Old format: extract number after test_group_
            match = re.search(r"test_group_(\d+)_", col)
            if match:
                return (int(match.group(1)),)
        return ()

    group_cols = sorted(group_cols, key=extract_group_index)
    if not group_cols:
        raise SystemExit(f"No test_group_*_{metric} columns found in seed CSVs.")

    table_rows = []
    generic_group_metrics: Dict[str, Dict[str, float]] = (
        {}
    )  # group_col -> {metric: mean}
    overall_metrics: Dict[str, Tuple[float, float]] = (
        {}
    )  # method -> (mean, std) for test_[metric]

    # Determine if this is an inverted metric (lower is better)
    inverted_metric = metric in ["loss", "ece"]

    for method in methods:
        rows = method_stats.get(method)
        if not rows:
            continue
        group_cells: List[Tuple[float, float]] = []
        for gcol in group_cols:
            metric_col = gcol  # Column name already has the metric

            if metric_col in rows[0]:
                metric_vals = [r[metric_col] for r in rows if metric_col in r]
                m, s = mean_std(metric_vals)
                group_cells.append((m, s))

                # Store generic baseline metrics
                if method == "generic":
                    if gcol not in generic_group_metrics:
                        generic_group_metrics[gcol] = {}
                    generic_group_metrics[gcol][metric] = m
            else:
                group_cells.append((float("nan"), float("nan")))

        # Collect overall test_[metric] for this method
        overall_col = f"test_{metric}"
        if overall_col in rows[0]:
            overall_vals = [r[overall_col] for r in rows if overall_col in r]
            m, s = mean_std(overall_vals)
            overall_metrics[method] = (m, s)
        elif metric in ["loss", "ece"]:
            # For loss/ece, compute macro-average of group metrics
            group_metrics_list = []
            for gcol in group_cols:
                metric_col = gcol
                if metric_col in rows[0]:
                    metric_vals = [r[metric_col] for r in rows if metric_col in r]
                    if metric_vals:
                        m_group, _ = mean_std(metric_vals)
                        group_metrics_list.append(m_group)
            if group_metrics_list:
                macro_avg = sum(group_metrics_list) / len(group_metrics_list)
                # Estimate std as average of group stds
                group_stds = []
                for gcol in group_cols:
                    metric_col = gcol
                    if metric_col in rows[0]:
                        metric_vals = [r[metric_col] for r in rows if metric_col in r]
                        if len(metric_vals) > 1:
                            _, s_group = mean_std(metric_vals)
                            group_stds.append(s_group)
                avg_std = sum(group_stds) / len(group_stds) if group_stds else 0.0
                overall_metrics[method] = (macro_avg, avg_std)

        table_rows.append((method, group_cells))

    if not generic_group_metrics:
        raise SystemExit(f"Generic method results required for metric '{metric}'.")

    # LaTeX assembly
    lines: List[str] = []
    lines.append("\\begin{table*}")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{llrrrr}")
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Groups} & \\textbf{Generic} & \\multicolumn{4}{c}{\\textbf{Gains over Generic}} \\\\"
    )
    lines.append("\\cmidrule(r){3-6}")
    lines.append(
        "& & \\textbf{1Hot} & \\textbf{Intersectional} & \\textbf{Multitask} & \\textbf{Decoupled} \\\\"
    )
    lines.append("\\midrule")

    # Determine metric title for caption
    metric_title = metric.upper()
    if metric == "auroc":
        metric_title = "AUC"

    # Output one row per group
    for idx, gcol in enumerate(group_cols):
        # Parse group column name to get indices
        group_indices = extract_group_index(gcol)

        # Get group label from mapping
        group_labels = GROUP_LABELS.get(sensitive, {})
        g_label = group_labels.get(group_indices, None)

        # If no predefined label, build one from the column
        if not g_label:
            if "[" in gcol:
                parts = re.findall(r"(\w+)\[(\d+)\]", gcol)
                g_label = ",".join([f"{attr}[{idx}]" for attr, idx in parts])
            else:
                g_label = gcol.replace("test_group_", "").replace(f"_{metric}", "")

        row_cells = [f"\\texttt{{{g_label}}}"]

        # Generic method - show absolute value
        for method, group_cells in table_rows:
            if method == "generic":
                m, s = group_cells[idx]
                if math.isnan(m):
                    row_cells.append("--")
                else:
                    row_cells.append(format_cell(m, s, precision))
                break

        # Other methods - show gains
        for method, group_cells in table_rows:
            if method != "generic":
                m, s = group_cells[idx]
                if math.isnan(m):
                    row_cells.append("--")
                else:
                    # Show gain
                    gen_m = generic_group_metrics[gcol].get(metric, float("nan"))
                    if math.isnan(gen_m):
                        row_cells.append("--")
                    else:
                        if inverted_metric:
                            gain = gen_m - m  # For loss/ece: lower is better
                        else:
                            gain = m - gen_m  # For acc/auroc: higher is better
                        row_cells.append(f"{gain*100:+.{precision}f}")

        lines.append(" & ".join(row_cells) + " \\\\")

    # Best/Worst Gain row
    lines.append("\\midrule")
    row_cells = [f"\\textbf{{Best/Worst Gain}}"]

    # Generic column
    row_cells.append("--")

    # Gains for each non-generic method
    for method, group_cells in table_rows:
        if method == "generic":
            continue
        gains = []
        for idx, (m, s) in enumerate(group_cells):
            if not math.isnan(m):
                gcol = group_cols[idx]
                gen_m = generic_group_metrics[gcol].get(metric, float("nan"))
                if not math.isnan(gen_m):
                    if inverted_metric:
                        gain = gen_m - m
                    else:
                        gain = m - gen_m
                    gains.append(gain)

        if gains:
            best = max(gains)
            worst = min(gains)
            row_cells.append(f"{best*100:+.{precision}f}/{worst*100:+.{precision}f}")
        else:
            row_cells.append("--")

    lines.append(" & ".join(row_cells) + " \\\\")

    # Overall row
    row_cells = [f"\\textbf{{Overall}}"]

    # Generic overall test_[metric]
    if "generic" in overall_metrics:
        m, s = overall_metrics["generic"]
        row_cells.append(format_cell(m, s, precision))
    else:
        row_cells.append("--")

    # Gains for each non-generic method
    for method, _ in table_rows:
        if method == "generic":
            continue
        if method in overall_metrics:
            m, s = overall_metrics[method]
            if not math.isnan(m):
                gen_m, gen_s = overall_metrics.get(
                    "generic", (float("nan"), float("nan"))
                )
                if not math.isnan(gen_m):
                    if inverted_metric:
                        gain = gen_m - m
                    else:
                        gain = m - gen_m
                    row_cells.append(f"{gain*100:+.{precision}f}")
                else:
                    row_cells.append("--")
            else:
                row_cells.append("--")
        else:
            row_cells.append("--")

    lines.append(" & ".join(row_cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    # Add caption with metric name and dataset
    caption = f"{metric_title} metrics for the {dataset} test dataset, with different personalization strategies"
    lines.append(f"\\caption{{{caption}}}")

    lines.append("\\end{table*}")
    return "\n".join(lines)

    method_stats: Dict[str, List[Dict[str, float]]] = {}
    group_cols_set = set()

    # Collect seed data per method
    for method in methods:
        results_path = root / dataset / method / sensitive / "results"
        seed_rows = collect_method_seeds(results_path)
        if not seed_rows:
            continue
        for row in seed_rows:
            for col in row.keys():
                if col.startswith("test_group_") and col.endswith("_acc"):
                    group_cols_set.add(col)
        method_stats[method] = seed_rows

    if not method_stats:
        raise SystemExit("No data found for specified configuration.")

    # Keep only test_group_<int>_acc columns for accuracy
    group_cols = [
        c for c in group_cols_set if c.startswith("test_group_") and c.endswith("_acc")
    ]
    # Extract group index from format test_group_<idx>_acc
    group_cols = sorted(group_cols, key=lambda c: int(c.split("_")[2]))
    if not group_cols:
        raise SystemExit("No test_group_<id>_acc columns found in seed CSVs.")

    table_rows = (
        []
    )  # (method, [(mean,std), ...], (test_acc_mean, test_acc_std), (test_auroc_mean, test_auroc_std))
    generic_group_means: Dict[str, float] = {}  # col_name -> mean for generic
    generic_group_metrics: Dict[str, Dict[str, float]] = (
        {}
    )  # group_col -> {metric: mean}

    for method in methods:
        rows = method_stats.get(method)
        if not rows:
            continue
        group_cells: List[Tuple[float, float]] = []
        for gcol in group_cols:
            vals = [r[gcol] for r in rows if gcol in r]
            m, s = mean_std(vals)
            group_cells.append((m, s))
            # Store generic group means for later comparison
            if method == "generic":
                generic_group_means[gcol] = m
                # Also collect per-group metrics for generic (auc, loss, ece)
                # Extract group index from gcol (test_group_<idx>_acc)
                group_idx = gcol.split("_")[2]
                if gcol not in generic_group_metrics:
                    generic_group_metrics[gcol] = {}
                generic_group_metrics[gcol]["acc"] = m
                for metric in ["auroc", "loss", "ece"]:
                    metric_col = f"test_group_{group_idx}_{metric}"
                    if metric_col in rows[0]:
                        metric_vals = [r[metric_col] for r in rows if metric_col in r]
                        metric_m, _ = mean_std(metric_vals)
                        generic_group_metrics[gcol][metric] = metric_m
        # overall test_acc
        test_acc_vals = [r["test_acc"] for r in rows if "test_acc" in r]
        test_acc_mean, test_acc_std = mean_std(test_acc_vals)
        # overall test_auroc
        test_auroc_vals = [r["test_auroc"] for r in rows if "test_auroc" in r]
        test_auroc_mean, test_auroc_std = mean_std(test_auroc_vals)
        table_rows.append(
            (
                method,
                group_cells,
                (test_acc_mean, test_acc_std),
                (test_auroc_mean, test_auroc_std),
            )
        )

    if not generic_group_means:
        raise SystemExit("Generic method results required to compute per-group Gains.")

    # LaTeX assembly with new template: metrics as column groups, methods as subcolumns
    lines: List[str] = []
    lines.append("\\begin{table*}")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{l")
    lines.append("                *{5}{r}  % Loss")
    lines.append("                *{5}{r}  % ECE")
    lines.append("                *{5}{r}  % ACC")
    lines.append("                *{5}{r}} % AUC")
    lines.append("\\toprule")
    lines.append("& ")
    lines.append("\\multicolumn{5}{c}{\\textbf{Loss}} &")
    lines.append("\\multicolumn{5}{c}{\\textbf{ECE}} &")
    lines.append("\\multicolumn{5}{c}{\\textbf{ACC}} &")
    lines.append("\\multicolumn{5}{c}{\\textbf{AUC}}")
    lines.append("\\\\")
    lines.append("\\cmidrule(lr){2-6}")
    lines.append("\\cmidrule(lr){7-11}")
    lines.append("\\cmidrule(lr){12-16}")
    lines.append("\\cmidrule(lr){17-21}")

    method_abbrev = {
        "generic": "Gen.",
        "1hot": "1hot",
        "intersectional": "Int.",
        "multitask": "Multi",
        "decoupled": "Decoup.",
    }
    method_short_names = [method_abbrev.get(m, m) for m in methods]
    header_cells = ["\\textbf{Group}"] + method_short_names * 4  # 4 metric groups
    lines.append(" & ".join(header_cells) + " \\\\")
    lines.append("\\midrule")

    # Build dicts for metrics: group_col -> method -> {metric: (mean, std)}
    group_data: Dict[str, Dict[str, Dict[str, tuple]]] = {}
    for gcol in group_cols:
        group_data[gcol] = {}
        for m in methods:
            group_data[gcol][m] = {}

    # Also store test metrics per method and track gains per metric
    method_metrics: Dict[str, tuple] = (
        {}
    )  # method -> (test_acc_mean, test_acc_std, test_auroc_mean, test_auroc_std)
    method_metric_gains: Dict[str, Dict[str, List[float]]] = (
        {}
    )  # method -> metric -> list of gains per group
    for m in methods:
        method_metric_gains[m] = {
            metric: [] for metric in ["acc", "auc", "loss", "ece"]
        }

    for method, group_cells, (t_mean, t_std), (auroc_mean, auroc_std) in table_rows:
        method_metrics[method] = (t_mean, t_std, auroc_mean, auroc_std)
        rows = method_stats.get(method, [])

        for (m, s), gcol in zip(group_cells, group_cols):
            group_idx = gcol.split("_")[2]
            group_data[gcol][method]["acc"] = (m, s)

            # Collect other metrics (loss, ece, auroc)
            for metric, col_suffix in [
                ("loss", "loss"),
                ("ece", "ece"),
                ("auroc", "auroc"),
            ]:
                metric_col = f"test_group_{group_idx}_{col_suffix}"
                if metric_col in rows[0]:
                    metric_vals = [r[metric_col] for r in rows if metric_col in r]
                    metric_m, metric_s = mean_std(metric_vals)
                    group_data[gcol][method][metric] = (metric_m, metric_s)

            # Compute gains for each metric only for non-generic methods
            if method != "generic" and rows:
                # Accuracy gain
                if "acc" in generic_group_metrics.get(gcol, {}):
                    gen_m = generic_group_metrics[gcol]["acc"]
                    method_metric_gains[method]["acc"].append(m - gen_m)

                # AUROC gain
                auroc_col = f"test_group_{group_idx}_auroc"
                if auroc_col in rows[0]:
                    auc_vals = [r[auroc_col] for r in rows if auroc_col in r]
                    auc_m, _ = mean_std(auc_vals)
                    if "auroc" in generic_group_metrics.get(gcol, {}):
                        gen_auc = generic_group_metrics[gcol]["auroc"]
                        method_metric_gains[method]["auc"].append(auc_m - gen_auc)

                # Loss gain (negative is better, so we negate)
                loss_col = f"test_group_{group_idx}_loss"
                if loss_col in rows[0]:
                    loss_vals = [r[loss_col] for r in rows if loss_col in r]
                    loss_m, _ = mean_std(loss_vals)
                    if "loss" in generic_group_metrics.get(gcol, {}):
                        gen_loss = generic_group_metrics[gcol]["loss"]
                        method_metric_gains[method]["loss"].append(
                            gen_loss - loss_m
                        )  # negate

                # ECE gain (negative is better, so we negate)
                ece_col = f"test_group_{group_idx}_ece"
                if ece_col in rows[0]:
                    ece_vals = [r[ece_col] for r in rows if ece_col in r]
                    ece_m, _ = mean_std(ece_vals)
                    if "ece" in generic_group_metrics.get(gcol, {}):
                        gen_ece = generic_group_metrics[gcol]["ece"]
                        method_metric_gains[method]["ece"].append(
                            gen_ece - ece_m
                        )  # negate

    # Get group label mapping
    group_labels = GROUP_LABELS.get(sensitive, {})

    # Output one row per group: group label + 4 metric groups x 5 methods
    # Map display metrics to gain metric keys
    metric_key_map = {"loss": "loss", "ece": "ece", "acc": "acc", "auroc": "auc"}

    for idx, gcol in enumerate(group_cols):
        g_label = group_labels.get(idx, gcol.replace("test_", "").replace("_", " "))
        row_cells = [f"\\texttt{{{g_label}}}"]

        # For each metric (Loss, ECE, ACC, AUC)
        for metric in ["loss", "ece", "acc", "auroc"]:
            for method in methods:
                m_val, s_val = group_data[gcol][method].get(
                    metric, (float("nan"), float("nan"))
                )
                if math.isnan(m_val):
                    row_cells.append("--")
                elif method == "generic" or metric == "acc":
                    # Generic method or accuracy: show absolute value as percentage
                    row_cells.append(format_cell(m_val, s_val, precision))
                else:
                    # Show gain
                    gen_val, _ = group_data[gcol]["generic"].get(
                        metric, (float("nan"), float("nan"))
                    )
                    # For loss/ece: negative gain is better (lower is better), so negate
                    if metric in ["loss", "ece"]:
                        gain = gen_val - m_val
                    else:
                        gain = m_val - gen_val
                    if math.isnan(gain):
                        row_cells.append("--")
                    else:
                        row_cells.append(f"{gain*100:+.{precision}f}")

        lines.append(" & ".join(row_cells) + " \\\\")

    # Best/Worst Gain rows for each metric
    lines.append("\\midrule")
    for gain_metric in ["loss", "ece", "acc", "auroc"]:
        row_cells = [f"\\textbf{{Best/Worst Gain ({gain_metric.upper()})}}"]
        gain_key = metric_key_map[gain_metric]  # Map display name to dict key

        # For each metric group column
        for metric in ["loss", "ece", "acc", "auroc"]:
            for method in methods:
                if method == "generic":
                    row_cells.append("--")
                elif metric == gain_metric and method_metric_gains[method].get(
                    gain_key
                ):
                    # Show gain values only for the matching metric
                    gains = method_metric_gains[method][gain_key]
                    if gains:
                        best = max(gains)
                        worst = min(gains)
                        row_cells.append(
                            f"{best*100:+.{precision}f}/{worst*100:+.{precision}f}"
                        )
                    else:
                        row_cells.append("--")
                else:
                    # Other metrics: show placeholder
                    row_cells.append("--")

        lines.append(" & ".join(row_cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate seed CSVs into LaTeX table")
    p.add_argument(
        "--dataset", required=True, help="Dataset name (root folder under results)"
    )
    p.add_argument(
        "--sensitive", required=True, help="Sensitive attribute folder (e.g., sex_age)"
    )
    p.add_argument(
        "--methods",
        nargs="*",
        default=DEFAULT_METHODS,
        help="List of methods to include",
    )
    p.add_argument(
        "--metrics",
        nargs="*",
        default=["all"],
        help="Metrics to generate: 'all', 'loss', 'acc', 'auroc' (default: all)",
    )
    p.add_argument("--results_root", default="results", help="Root results directory")
    p.add_argument("--precision", type=int, default=2, help="Decimal precision")
    p.add_argument("--output", help="Optional output file for LaTeX table")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.results_root)

    # Determine which metrics to generate
    available_metrics = ["loss", "acc", "auroc"]
    if args.metrics == ["all"] or (len(args.metrics) == 1 and args.metrics[0] == "all"):
        metrics_to_generate = available_metrics
    else:
        # Filter to only requested metrics that are available
        metrics_to_generate = [m for m in args.metrics if m in available_metrics]
        if not metrics_to_generate:
            print(
                f"Error: No valid metrics specified. Available: {available_metrics}",
                file=sys.stderr,
            )
            sys.exit(1)

    metric_titles = {"loss": "Loss", "acc": "Accuracy", "auroc": "AUC"}

    # Determine output path
    if args.output:
        # User specified custom output file
        output_path = Path(args.output)
        output_single_file = True
    else:
        # Default: save to results/tables/ directory
        output_dir = Path("results") / "tables"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_single_file = False

    all_tables = []
    for metric in metrics_to_generate:
        try:
            table = build_single_metric_table(
                args.dataset, args.sensitive, root, args.methods, args.precision, metric
            )
            all_tables.append(f"\n% {metric_titles[metric]} Table\n{table}")

            # Save individual metric table if not using single output file
            if not output_single_file:
                metric_output_file = (
                    output_dir / f"{args.dataset}_{args.sensitive}_{metric}.tex"
                )
                metric_output_file.write_text(
                    f"% {metric_titles[metric]} Table\n{table}"
                )
                print(f"Saved: {metric_output_file}", file=sys.stderr)
        except SystemExit as e:
            print(
                f"Warning: Could not generate table for metric '{metric}': {e}",
                file=sys.stderr,
            )

    # If single output file specified, write all tables to it
    if output_single_file and all_tables:
        output = "\n\n".join(all_tables)
        Path(output_path).write_text(output)
        print(f"Saved: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
