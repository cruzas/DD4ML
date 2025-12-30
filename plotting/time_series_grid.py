#!/usr/bin/env python3
import hashlib
import json
import math
import os
import time
from functools import lru_cache

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import wandb

# ==========================================
# PRESENTATION STYLE SETTINGS
# ==========================================
mpl.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "figure.titlesize": 24,
        "legend.fontsize": 20,
        "legend.handlelength": 3.0,
        "lines.linewidth": 3,
    }
)


def latex_opt(opt: str) -> str:
    opt_lower = opt.lower()
    if "apts" in opt_lower:
        base = r"\mathrm{SAPTS}"
        if "ip" in opt_lower:
            return base + r"_{\mathrm{IP}}"
        if "_p" in opt_lower:
            return base + r"_{\mathrm{P}}"
        if "_d" in opt_lower:
            return base + r"_{\mathrm{D}}"
        return base
    parts = opt.split("_", 1)
    if len(parts) == 2:
        return rf"\mathrm{{{parts[0]}}}_{{\mathrm{{{parts[1]}}}}}"
    return rf"\mathrm{{{opt}}}"


def _metric_label(metric: str) -> str:
    name_map = {"acc": "accuracy", "accuracy": "accuracy", "loss": "loss"}
    base = name_map.get(metric, metric).replace("_", " ")
    if base == "accuracy":
        return r"Avg. Accuracy (\%)"
    if base == "loss":
        return r"Avg. Empirical Loss"
    return f"Avg. {base.capitalize()}"


def _safe_int(v):
    try:
        return int(float(v))
    except:
        return -1


def _loose_match(actual, target):
    if actual == target:
        return True
    try:
        if abs(float(actual) - float(target)) < 1e-6:
            return True
    except:
        pass
    return False


# ==========================================
# FETCHING & CACHING
# ==========================================
def _load_history_cached_by_id(api, project_path, run_id, cache_dir, dataset):
    path = os.path.join(cache_dir, dataset, f"{run_id}.pkl") if cache_dir else None
    if path and os.path.exists(path):
        try:
            return pd.read_pickle(path)
        except:
            pass
    try:
        run = api.run(f"{project_path}/{run_id}")
        p_keys = [
            "running_time",
            "grad_evals",
            "loss",
            "accuracy",
            "acc",
            "epoch",
            "iter",
        ]
        summary_keys = set(run.summary.keys())
        keys = ["running_time", "grad_evals", "loss"] + [
            k for k in p_keys if k in summary_keys
        ]
        df = pd.DataFrame([row for row in run.scan_history(keys=keys)])
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_pickle(path)
        return df
    except:
        return pd.DataFrame()


def _list_runs_cached(api, project_path, filters, cache_dir, dataset):
    can = json.dumps(
        {"project": project_path, "filters": filters}, sort_keys=True, default=str
    )
    kh = hashlib.md5(can.encode("utf-8")).hexdigest()
    path = (
        os.path.join(cache_dir, dataset, "_runs_index", f"{kh}.pkl")
        if cache_dir
        else None
    )
    if (
        path
        and os.path.exists(path)
        and (time.time() - os.path.getmtime(path) < 12 * 3600)
    ):
        try:
            return pd.read_pickle(path)["runs"]
        except:
            pass
    runs = api.runs(project_path, filters=filters)
    recs = [
        {"id": r.id, "name": r.name, "config": dict(getattr(r, "config", {}))}
        for r in runs
    ]
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.to_pickle({"_fetched_at": time.time(), "runs": recs}, path)
    return recs


def get_style_for_combo(optimizer_name, n_val):
    opt, n = str(optimizer_name).lower(), _safe_int(n_val)
    if "sgd" in opt:
        return "black", "--"
    if "_p" in opt:
        return {2: "#d62728", 4: "#ff7f0e", 8: "#8c564b"}.get(n, "#e377c2"), "-."
    if "ip" in opt:
        return {2: "#2ca02c", 4: "#bcbd22", 8: "#17becf"}.get(n, "#7f7f7f"), ":"
    return {2: "#1f77b4", 4: "#9467bd", 8: "#000080"}.get(n, "#1f77b4"), "-"


# ==========================================
# SUMMARY TABLE GENERATION (LaTeX)
# ==========================================
def generate_summary_table(
    data_cache, metrics, combo_key, get_history_fn, dataset_name, regime, file_path
):
    """Saves a consolidated LaTeX table evaluated at the max common training point."""
    bs_header = "Effective Batch Size" if regime == "weak" else "Global Batch Size"
    is_ts = "tinyshakespeare" in dataset_name.lower()
    progression_key = "iter" if is_ts else "epoch"
    progression_label = "iteration" if is_ts else "epoch"

    # Find the maximum valid point common to all runs in this regime
    all_max_points = []
    for size in data_cache:
        for r in data_cache[size]:
            h = get_history_fn(r["id"])
            if not h.empty and progression_key in h.columns:
                all_max_points.append(h[progression_key].max())

    eval_point = min(all_max_points) if all_max_points else 0

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        table_dataset_name = dataset_name
        if table_dataset_name.lower() == "mnist":
            table_dataset_name = "MNIST"
        elif table_dataset_name.lower() == "cifar10":
            table_dataset_name = "CIFAR-10"
        elif table_dataset_name.lower() == "tinyshakespeare":
            table_dataset_name = "Tiny Shakespeare"
        elif table_dataset_name.lower() == "poisson2d":
            table_dataset_name = "Poisson 2D"

        f.write(
            f"% --- Consolidated LaTeX Table for {table_dataset_name} ({regime} scaling) ---\n"
        )
        f.write("\\begin{table}[H]\n  \\centering\n")
        f.write(
            f"  \\caption{{Summary results for {table_dataset_name} ({regime} scaling) "
            f"evaluated at the max shared {progression_label} ({eval_point}).}}\n"
        )
        f.write("  \\resizebox{\\textwidth}{!}{\n")

        col_setup = (
            "l l l l "
            + "l " * len(metrics)
            + "S[separate-uncertainty, table-format=4.2(4)] l"
        )
        f.write(f"    \\begin{{tabular}}{{{col_setup}}}\n      \\hline\n")
        headers = (
            [bs_header, "Optimizer", "$N$", "Gradient Evaluations"]
            + [
                ("Accuracy (\%)" if "acc" in m.lower() else m.capitalize())
                for m in metrics
            ]
            + ["{Time (s)}", "Speedup"]
        )
        f.write("      " + " & ".join(headers) + " \\\\\n      \\hline\n")

        for size in sorted(data_cache.keys()):
            runs = data_cache[size]
            variant_baselines = {}
            for r in runs:
                raw_opt = r["config"].get("optimizer", "unk").lower()
                opt_key = "sgd" if "sgd" in raw_opt else raw_opt
                n = _safe_int(
                    r["config"].get(
                        "num_subdomains" if "sgd" in raw_opt else combo_key, 1
                    )
                )
                if n == 2:
                    h = get_history_fn(r["id"])
                    if not h.empty and progression_key in h.columns:
                        idx = (h[progression_key] - eval_point).abs().idxmin()
                        time_at_pt = (
                            h.loc[idx, "running_time"] - h["running_time"].min()
                        )
                        variant_baselines.setdefault(opt_key, []).append(time_at_pt)

            avg_variant_baselines = {
                k: np.mean(v) for k, v in variant_baselines.items() if v
            }

            unique_combos = sorted(
                {
                    (
                        r["config"].get("optimizer", "unk"),
                        _safe_int(
                            r["config"].get(
                                (
                                    "num_subdomains"
                                    if "sgd" in r["config"].get("optimizer", "").lower()
                                    else combo_key
                                ),
                                1,
                            )
                        ),
                    )
                    for r in runs
                },
                key=lambda x: (0 if "sgd" in x[0].lower() else 1, x[0], x[1]),
            )

            last_variant = None
            for opt, n in unique_combos:
                if "sgd" in opt.lower() and n < 2:
                    continue
                if last_variant is not None and opt != last_variant:
                    f.write("      \\hdashline\n")
                last_variant = opt

                relevant_runs = [
                    r
                    for r in runs
                    if r["config"].get("optimizer") == opt
                    and _safe_int(
                        r["config"].get(
                            "num_subdomains" if "sgd" in opt.lower() else combo_key, 1
                        )
                    )
                    == n
                ]
                opt_label = "SGD" if "sgd" in opt.lower() else f"${latex_opt(opt)}$"
                row_data = [str(size), opt_label, str(n)]

                ge_at_pt, times_at_pt, metric_vals = [], [], {m: [] for m in metrics}
                for r in relevant_runs:
                    h = get_history_fn(r["id"])
                    if not h.empty and progression_key in h.columns:
                        idx = (h[progression_key] - eval_point).abs().idxmin()
                        row = h.loc[idx]
                        if "grad_evals" in h.columns:
                            ge_at_pt.append(row["grad_evals"])
                        if "running_time" in h.columns:
                            times_at_pt.append(
                                row["running_time"] - h["running_time"].min()
                            )
                        for m in metrics:
                            if m in h.columns:
                                metric_vals[m].append(row[m])

                row_data.append(f"${np.mean(ge_at_pt):.0f}$" if ge_at_pt else "N/A")
                for m in metrics:
                    vals = metric_vals[m]
                    if vals:
                        prec = 2 if "acc" in m.lower() else 3
                        row_data.append(
                            f"${np.mean(vals):.{prec}f} \\pm {np.std(vals):.{prec}f}$"
                        )
                    else:
                        row_data.append("N/A")

                if times_at_pt:
                    avg_t = np.mean(times_at_pt)
                    row_data.append(f"{avg_t:.2f} \\pm {np.std(times_at_pt):.2f}")
                    v_key = "sgd" if "sgd" in opt.lower() else opt.lower()
                    speedup = (
                        avg_variant_baselines.get(v_key, avg_t) / avg_t
                        if avg_t > 0
                        else 1.0
                    )
                    row_data.append(f"${speedup:.2f}$")
                else:
                    row_data.extend(["N/A", "N/A"])
                f.write("      " + " & ".join(row_data) + " \\\\\n")
            f.write("      \\hline\n")
        f.write("    \\end{tabular}}\n\\end{table}\n")


# ==========================================
# PLOTTING
# ==========================================
def plot_grid_presentation(
    project_path,
    x_axis,
    base_key,
    batch_sizes,
    metrics,
    filters_base,
    sgd_filters=None,
    y_limits=None,
    save_path=None,
    table_path=None,
    y_log=False,
    combo_key="num_subdomains",
    cache_dir=None,
    regime="unknown",
):
    api = wandb.Api(timeout=300)
    dataset_name = filters_base.get("config.dataset_name", "dataset")

    @lru_cache(maxsize=None)
    def get_history(run_id):
        return _load_history_cached_by_id(
            api, project_path, run_id, cache_dir, dataset_name
        )

    def _res_n(cfg):
        opt = cfg.get("optimizer", "").lower()
        k = (
            "num_stages"
            if ("cifar" in dataset_name.lower() and "sgd" in opt)
            else combo_key
        )
        v = cfg.get(
            k, cfg.get("num_stages" if k == "num_subdomains" else "num_subdomains", 1)
        )
        return _safe_int(v) if _safe_int(v) > 0 else 1

    data_cache = {}
    for size in batch_sizes:
        base_f = {**filters_base, f"config.{base_key}": size}
        runs = _list_runs_cached(api, project_path, base_f, cache_dir, dataset_name)
        if size in (sgd_filters or {}):
            lr_f = sgd_filters[size]
            sgd_api_f = {"config.dataset_name": dataset_name, "config.optimizer": "sgd"}
            runs += [
                m
                for m in _list_runs_cached(
                    api, project_path, sgd_api_f, cache_dir, dataset_name
                )
                if (
                    _safe_int(m["config"].get("batch_size", -1)) == size
                    or _safe_int(m["config"].get("effective_batch_size", -1)) == size
                )
                and all(_loose_match(m["config"].get(k), v) for k, v in lr_f.items())
            ]
        data_cache[size] = runs

    if x_axis == "grad_evals" and table_path:
        generate_summary_table(
            data_cache,
            metrics,
            combo_key,
            get_history,
            dataset_name,
            regime,
            table_path,
        )

    nrows, ncols = len(metrics), len(batch_sizes)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 5 * nrows),
        squeeze=False,
        sharey="row",
    )

    metric_bounds = {m: [float("inf"), float("-inf")] for m in metrics}
    for size, runs in data_cache.items():
        for m in runs:
            hist = get_history(m["id"])
            if not hist.empty:
                for m_name in metrics:
                    if m_name in hist.columns:
                        y_vals = pd.to_numeric(hist[m_name], errors="coerce").dropna()
                        if y_log or m_name == "loss":
                            y_vals = y_vals[y_vals > 0]
                        if not y_vals.empty:
                            metric_bounds[m_name][0] = min(
                                metric_bounds[m_name][0], y_vals.min()
                            )
                            metric_bounds[m_name][1] = max(
                                metric_bounds[m_name][1], y_vals.max()
                            )

    all_combos = set()
    for j, size in enumerate(batch_sizes):
        runs_all = data_cache[size]
        combos = sorted(
            {
                (m["config"].get("optimizer", "unk"), _res_n(m["config"]))
                for m in runs_all
            },
            key=lambda x: (str(x[0]), x[1]),
        )
        all_combos.update(combos)
        for i, metric in enumerate(metrics):
            ax = axes[i, j]
            for combo in combos:
                opt_name, n_val = combo
                if "sgd" in str(opt_name).lower() and n_val > 1:
                    continue
                series_list = []
                relevant_runs = [
                    r
                    for r in runs_all
                    if (r["config"].get("optimizer", "unk"), _res_n(r["config"]))
                    == combo
                ]
                for m in relevant_runs:
                    hist = get_history(m["id"])
                    if (
                        not hist.empty
                        and x_axis in hist.columns
                        and metric in hist.columns
                    ):
                        df = hist[[x_axis, metric]].dropna()
                        if not df.empty:
                            if x_axis == "running_time":
                                df[x_axis] -= df[x_axis].min()
                            series_list.append(df)
                if series_list:
                    all_x = np.concatenate([s[x_axis].values for s in series_list])
                    grid = np.linspace(all_x.min(), all_x.max(), 200)
                    data = np.stack(
                        [np.interp(grid, s[x_axis], s[metric]) for s in series_list]
                    )
                    color, ls = get_style_for_combo(opt_name, n_val)
                    ax.plot(grid, data.mean(axis=0), color=color, linestyle=ls)
                    if len(series_list) > 1:
                        ax.fill_between(
                            grid,
                            data.mean(axis=0) - data.std(axis=0),
                            data.mean(axis=0) + data.std(axis=0),
                            alpha=0.15,
                            color=color,
                        )
            if i == 0:
                ax.set_title(rf"{base_key.replace('_', ' ').upper()} = {size}")
            if j == 0:
                ax.set_ylabel(_metric_label(metric))
            if i == nrows - 1:
                ax.set_xlabel(
                    {
                        "grad_evals": "Gradient Evaluations",
                        "running_time": "Wall-clock Time (s)",
                        "epoch": "Epochs",
                        "iter": "Iterations",
                    }.get(x_axis, x_axis)
                )
            ax.grid(True, alpha=0.3)
            ax.set_xlim(left=0)
            is_log = y_log or metric == "loss"
            if is_log:
                ax.set_yscale("log")
            low, high = metric_bounds[metric]
            if not np.isinf(low):
                if dataset_name == "poisson2d" and metric == "loss":
                    ax.set_ylim(max(low, 1e-12), 10**3)
                elif y_limits and metric in y_limits:
                    ax.set_ylim(*y_limits[metric])
                else:
                    safe_low = max(low, 1e-12) if is_log else low
                    pad = high * 0.1 if is_log else (high - low) * 0.05
                    ax.set_ylim(safe_low / (1.1 if is_log else 1), high + pad)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return all_combos


# ==========================================
# MAIN
# ==========================================
def main():
    proj = "cruzas-universit-della-svizzera-italiana/thesis_results"
    base_fig_dir = os.path.expanduser(
        "~/Documents/GitHub/PhD-Thesis-Samuel-Cruz/figures/thesis"
    )
    base_table_dir = (
        "/Users/cruzalegriasamueladolfo/Documents/GitHub/PhD-Thesis-Samuel-Cruz/tables"
    )
    cache_dir = os.path.join(os.path.dirname(base_fig_dir), ".cache")

    configs = {
        "mnist": {
            "filters": {"config.model_name": "simple_cnn"},
            "sgd_filters_strong": {
                1024: {"learning_rate": 0.10},
                2048: {"learning_rate": 0.10},
                4096: {"learning_rate": 0.10},
            },
            "sgd_filters_weak": {
                128: {"learning_rate": 0.01},
                256: {"learning_rate": 0.10},
                512: {"learning_rate": 0.10},
            },
            "metrics": ["loss", "accuracy"],
            "y_limits": {"accuracy": (0, 100)},
            "y_log": False,
        },
        "cifar10": {
            "filters": {"config.model_name": "big_resnet"},
            "sgd_filters_strong": {
                256: {"learning_rate": 0.01},
                512: {"learning_rate": 0.1},
                1024: {"learning_rate": 0.1},
            },
            "sgd_filters_weak": {
                256: {"learning_rate": 0.01},
                512: {"learning_rate": 0.1},
                1024: {"learning_rate": 0.1},
            },
            "metrics": ["loss", "accuracy"],
            "y_limits": {"accuracy": (0, 100)},
            "y_log": False,
        },
        "tinyshakespeare": {
            "filters": {"config.model_name": "minigpt"},
            "sgd_filters_strong": {
                1024: {"learning_rate": 0.01},
                2048: {"learning_rate": 0.01},
                4096: {"learning_rate": 0.01},
            },
            "sgd_filters_weak": {
                128: {"learning_rate": 0.10},
                256: {"learning_rate": 0.10},
                512: {"learning_rate": 0.10},
            },
            "metrics": ["loss"],
            "y_log": False,
        },
        "poisson2d": {
            "filters": {"config.model_name": "pinn_ffnn"},
            "sgd_filters_strong": {
                128: {"learning_rate": 0.001},
                256: {"learning_rate": 0.10},
                512: {"learning_rate": 0.001},
            },
            "sgd_filters_weak": {
                64: {"learning_rate": 0.01},
                128: {"learning_rate": 0.001},
                256: {"learning_rate": 0.10},
            },
            "metrics": ["loss"],
            "y_log": True,
        },
    }

    for dataset, cfg in configs.items():
        ck = "num_stages" if dataset == "cifar10" else "num_subdomains"
        dataset_combos = set()
        for reg in ["weak", "strong"]:
            bkey = "effective_batch_size" if reg == "weak" else "batch_size"
            smap = cfg[f"sgd_filters_{reg}"]

            table_filename = f"{dataset}_{reg}_scaling.tex"
            table_path = os.path.join(base_table_dir, table_filename)

            for xax_raw in ["grad_evals", "running_time", "epoch"]:
                xax = (
                    "iter"
                    if dataset == "tinyshakespeare" and xax_raw == "epoch"
                    else xax_raw
                )
                save_path = os.path.join(
                    base_fig_dir, f"{dataset}_{reg}_{xax}_grid.pdf"
                )

                combos = plot_grid_presentation(
                    proj,
                    xax,
                    bkey,
                    sorted(smap.keys()),
                    cfg["metrics"],
                    {"config.dataset_name": dataset, **cfg["filters"]},
                    smap,
                    cfg.get("y_limits"),
                    save_path,
                    table_path,
                    cfg["y_log"],
                    ck,
                    cache_dir,
                    reg,
                )
                if combos:
                    dataset_combos.update(combos)

        if dataset_combos:
            sorted_combos = sorted(
                dataset_combos,
                key=lambda x: (0 if "sgd" in str(x[0]).lower() else 1, str(x[0]), x[1]),
            )
            leg_h, leg_l = [], []
            for c in sorted_combos:
                if "sgd" in str(c[0]).lower() and c[1] > 1:
                    continue
                col, ls = get_style_for_combo(c[0], c[1])
                leg_h.append(Line2D([0], [0], color=col, lw=4, linestyle=ls))
                lbl = (
                    r"SGD Baseline"
                    if str(c[0]).lower() == "sgd"
                    else rf"${latex_opt(str(c[0]))}\;\mid\;N={c[1]}$"
                )
                leg_l.append(lbl)
            leg_fig = plt.figure(figsize=(18, 1.5 if len(leg_l) <= 4 else 2.5))
            leg_fig.legend(
                leg_h, leg_l, loc="center", ncol=min(len(leg_l), 4), frameon=False
            )
            leg_fig.savefig(
                os.path.join(base_fig_dir, f"{dataset}_legend.pdf"), bbox_inches="tight"
            )
            plt.close(leg_fig)


if __name__ == "__main__":
    main()
