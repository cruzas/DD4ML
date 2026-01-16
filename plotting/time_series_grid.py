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
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "figure.titlesize": 26,
        "legend.fontsize": 20,
        "legend.handlelength": 3.0,
        "lines.linewidth": 3,
    }
)


def format_opt_name(name):
    """Maps internal optimizer names to pretty LaTeX names."""
    mapping = {"lssr1_tr": "L-SSR1-TR", "cg": "CG", "adam": "Adam", "sgd": "SGD"}
    return mapping.get(str(name).lower(), str(name).upper())


def latex_opt(opt: str) -> str:
    """Standardizes optimizer names for LaTeX math mode."""
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
# LATEX GENERATION
# ==========================================
def save_latex_figure_code(
    tex_dir,
    dataset,
    regime,
    xax,
    metrics,
    batch_sizes,
    sgd_lrs,
    delta,
    model_pretty,
    label_suffix,
    glob_opt,
    loc_opt,
    glob_so,
    loc_so,
    present_variants,
):
    """Saves a .tex file with a 3x2 grid of plots and a dynamic, order-aware caption."""
    os.makedirs(tex_dir, exist_ok=True)
    filename = f"{dataset}_{regime}_{xax}_grid.tex"
    filepath = os.path.join(tex_dir, filename)

    normalized_variants = {v.lower().replace("sapts", "apts") for v in present_variants}

    is_glob_2nd = str(glob_so).lower() == "true"
    is_loc_2nd = str(loc_so).lower() == "true"
    order_suffix = " (2nd order)" if (is_glob_2nd and is_loc_2nd) else " (1st order)"

    lr_label = "EBS" if regime == "weak" else "GBS"
    lr_parts = [f"{lr} for {lr_label}={bs}" for bs, lr in sgd_lrs.items()]
    lr_str = ", ".join(lr_parts)

    xax_label = {
        "grad_evals": "gradient evaluations",
        "running_time": "wall-clock time",
    }.get(xax, xax)
    m_desc = (
        "Empirical loss (left) and test accuracy (right)"
        if len(metrics) > 1
        else "Avg. Empirical Loss"
    )
    dataset_pretty = {
        "mnist": "MNIST",
        "cifar10": "CIFAR-10",
        "tinyshakespeare": "Tiny Shakespeare",
        "poisson2d": "Poisson 2D",
    }.get(dataset, dataset.capitalize())

    sapts_labels = []
    if "apts_d" in normalized_variants:
        sapts_labels.append(rf"\SAPTSD\ {order_suffix}")
    if "apts_p" in normalized_variants:
        sapts_labels.append(rf"\SAPTSP\ {order_suffix}")
    if "apts_ip" in normalized_variants:
        sapts_labels.append(rf"\SAPTSIP\ {order_suffix}")
    opt_list_str = ", ".join(["SGD"] + sapts_labels)

    content = rf"""\begin{{figure}}[H]
    \centering
    \includegraphics[width=\linewidth]{{figures/{dataset}_{regime}_{xax}_grid.pdf}}%
    \vspace{{1ex}}
    \includegraphics[width=\linewidth]{{figures/{dataset}_legend.pdf}}%
    \caption{{{m_desc} vs number of {xax_label}. Dataset: {dataset_pretty} ({regime} scaling). Network type: {model_pretty}. Optimizers: {opt_list_str}. \SAPTS\ Configuration: Global optimizer: {format_opt_name(glob_opt)}, Local optimizer: {format_opt_name(loc_opt)}, $\Delta_0 = {delta}$. Learning rates for SGD: {lr_str}. From top to bottom, the effective batch size is increased by a factor of 2. An overlap of approximately 33\% was applied between consecutive mini-batches and micro-batches.}}
\label{{fig:{dataset}_{regime}_{label_suffix}}}
\end{{figure}}
"""
    with open(filepath, "w") as f:
        f.write(content)


def generate_summary_table(
    table_dir,
    dataset,
    regime,
    metrics,
    sgd_lrs,
    glob_opt,
    loc_opt,
    glob_so,
    loc_so,
    present_variants,
    delta,
    model_pretty,
    data_cache,
    combo_key,
    get_history_fn,
):
    """Saves a consolidated LaTeX table evaluated at the max common training point."""
    os.makedirs(table_dir, exist_ok=True)
    file_path = os.path.join(table_dir, f"{dataset}_{regime}_scaling.tex")

    bs_label = "EBS" if regime == "weak" else "GBS"
    is_ts = "tinyshakespeare" in dataset.lower()
    prog_key = "iter" if is_ts else "epoch"
    prog_label = "iteration" if is_ts else "epoch"

    all_max_points = []
    for size in data_cache:
        for r in data_cache[size]:
            h = get_history_fn(r["id"])
            if not h.empty and prog_key in h.columns:
                all_max_points.append(h[prog_key].max())
    eval_point = min(all_max_points) if all_max_points else 0

    normalized_variants = {v.lower().replace("sapts", "apts") for v in present_variants}
    is_glob_2nd = str(glob_so).lower() == "true"
    is_loc_2nd = str(loc_so).lower() == "true"
    order_suffix = " (2nd order)" if (is_glob_2nd and is_loc_2nd) else " (1st order)"

    sapts_labels = []
    if "apts_d" in normalized_variants:
        sapts_labels.append(rf"\SAPTSD\ {order_suffix}")
    if "apts_p" in normalized_variants:
        sapts_labels.append(rf"\SAPTSP\ {order_suffix}")
    if "apts_ip" in normalized_variants:
        sapts_labels.append(rf"\SAPTSIP\ {order_suffix}")
    opt_list_str = ", ".join(["SGD"] + sapts_labels)

    lr_parts = [f"{lr} for {bs_label}={bs}" for bs, lr in sgd_lrs.items()]
    lr_str = ", ".join(lr_parts)

    dataset_pretty = {
        "mnist": "MNIST",
        "cifar10": "CIFAR-10",
        "tinyshakespeare": "Tiny Shakespeare",
        "poisson2d": "Poisson 2D",
    }.get(dataset, dataset.capitalize())
    m_desc = (
        "Summary of performance metrics (Loss and Accuracy)"
        if len(metrics) > 1
        else "Summary of performance metrics (Loss)"
    )

    with open(file_path, "w") as f:
        f.write(f"\\begin{{table}}[H]\n  \\centering\n")
        f.write(
            f"  \\caption{{{m_desc} for {dataset_pretty} ({regime} scaling regime) evaluated at shared {prog_label} {eval_point}. Network type: {model_pretty}. Optimizers: {opt_list_str}. \\SAPTS\\ Configuration: Global: {format_opt_name(glob_opt)}, Local: {format_opt_name(loc_opt)}, $\\Delta_0 = {delta}$. Learning rates for SGD: {lr_str}. An overlap of approximately 33\\% was applied between consecutive mini-batches and micro-batches.}}\n"
        )
        f.write("  \\resizebox{\\textwidth}{!}{\n")

        col_setup = (
            "l l l l "
            + "l " * len(metrics)
            + "S[separate-uncertainty, table-format=4.2(4.2)] l"
        )
        f.write(f"    \\begin{{tabular}}{{{col_setup}}}\n      \\hline\n")
        headers = (
            [bs_label, "Optimizer", "$N$", "Grad. Evals"]
            + [
                ("Accuracy (\%)" if "acc" in m.lower() else m.capitalize())
                for m in metrics
            ]
            + ["{Time (s)}", "Efficiency"]
        )
        f.write("      " + " & ".join(headers) + " \\\\\n      \\hline\n")

        for size in sorted(data_cache.keys()):
            runs = data_cache[size]
            variant_baselines = {}
            for r in runs:
                raw_opt = r["config"].get("optimizer", "unk").lower()
                n = _safe_int(
                    r["config"].get(
                        "num_subdomains" if "sgd" in raw_opt else combo_key, 1
                    )
                )
                if n == 2:
                    h = get_history_fn(r["id"])
                    if not h.empty and prog_key in h.columns:
                        idx = (h[prog_key] - eval_point).abs().idxmin()
                        variant_baselines.setdefault(raw_opt, []).append(
                            h.loc[idx, "running_time"] - h["running_time"].min()
                        )
            avg_baselines = {k: np.mean(v) for k, v in variant_baselines.items() if v}

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

            last_opt = None
            for opt, n in unique_combos:
                if "sgd" in opt.lower() and n < 2:
                    continue
                if last_opt is not None and opt != last_opt:
                    f.write("      \\hdashline\n")
                last_opt = opt

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

                ge, times, m_vals = [], [], {m: [] for m in metrics}
                for r in relevant_runs:
                    h = get_history_fn(r["id"])
                    if not h.empty and prog_key in h.columns:
                        idx = (h[prog_key] - eval_point).abs().idxmin()
                        row = h.loc[idx]
                        ge.append(row.get("grad_evals", 0))
                        times.append(row["running_time"] - h["running_time"].min())
                        for m in metrics:
                            m_vals[m].append(row.get(m, 0))

                row_data.append(f"${np.mean(ge):.0f}$" if ge else "N/A")
                for m in metrics:
                    vals = m_vals[m]
                    if vals:
                        prec = 2 if "acc" in m.lower() else 3
                        row_data.append(
                            f"${np.mean(vals):.{prec}f} \\pm {np.std(vals):.{prec}f}$"
                        )
                    else:
                        row_data.append("N/A")

                if times:
                    avg_t = np.mean(times)
                    row_data.append(f"{avg_t:.2f} \\pm {np.std(times):.2f}")
                    eff = (
                        avg_baselines.get(opt.lower(), avg_t) / avg_t
                        if avg_t > 0
                        else 1.0
                    )
                    row_data.append(f"${eff:.2f}$")
                else:
                    row_data.extend(["N/A", "N/A"])
                f.write("      " + " & ".join(row_data) + " \\\\\n")
            f.write("      \\hline\n")
        f.write("    \\end{tabular}}\n\\end{table}\n")


# ==========================================
# PLOTTING & METADATA EXTRACTION
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
            k, cfg.get("num_subdomains" if k == "num_stages" else "num_stages", 1)
        )
        return _safe_int(v) if _safe_int(v) > 0 else 1

    data_cache = {}
    config_meta = {
        "delta": "N/A",
        "glob_opt": "N/A",
        "loc_opt": "N/A",
        "glob_so": "N/A",
        "loc_so": "N/A",
    }
    present_variants = set()

    for size in batch_sizes:
        base_f = {**filters_base, f"config.{base_key}": size}
        runs = _list_runs_cached(api, project_path, base_f, cache_dir, dataset_name)
        for r in runs:
            opt_raw = r["config"].get("optimizer", "").lower()
            if "sgd" not in opt_raw:
                present_variants.add(opt_raw)
                if config_meta["delta"] == "N/A":
                    c = r["config"]
                    config_meta.update(
                        {
                            "delta": c.get("delta", "N/A"),
                            "glob_opt": c.get(
                                "glob_opt", c.get("global_optimizer", "N/A")
                            ),
                            "loc_opt": c.get(
                                "loc_opt", c.get("local_optimizer", "N/A")
                            ),
                            "glob_so": c.get("glob_second_order", "False"),
                            "loc_so": c.get("loc_second_order", "False"),
                        }
                    )

        if size in (sgd_filters or {}):
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
                and all(
                    _loose_match(m["config"].get(k), v)
                    for k, v in sgd_filters[size].items()
                )
            ]
        data_cache[size] = runs

    nrows, ncols = len(batch_sizes), len(metrics)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(8 * ncols, 6 * nrows),
        squeeze=False,
        sharey="col",
    )

    for i, size in enumerate(batch_sizes):
        runs_all = data_cache[size]
        combos = sorted(
            {
                (r["config"].get("optimizer", "unk"), _res_n(r["config"]))
                for r in runs_all
            },
            key=lambda x: (str(x[0]), x[1]),
        )
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            for combo in combos:
                opt_name, n_val = combo
                if "sgd" in str(opt_name).lower() and n_val > 1:
                    continue
                series_list = []
                for m in [
                    r
                    for r in runs_all
                    if (r["config"].get("optimizer", "unk"), _res_n(r["config"]))
                    == combo
                ]:
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
                ax.set_title(_metric_label(metric), pad=15)
            if j == ncols - 1:
                ax2 = ax.twinx()
                # Updated label based on scaling regime
                batch_label = (
                    "GLOBAL BATCH SIZE"
                    if regime == "strong"
                    else "EFFECTIVE BATCH SIZE"
                )
                ax2.set_ylabel(
                    rf"{batch_label} = {size}",
                    rotation=270,
                    labelpad=30,
                    fontweight="bold",
                    fontsize=20,
                )
                ax2.set_yticks([])
            if j == 0:
                ax.set_ylabel(_metric_label(metric))
            if i == nrows - 1:
                ax.set_xlabel(x_axis.replace("_", " ").title())
            ax.grid(True, alpha=0.3, linestyle="--")
            if metric == "loss" or y_log:
                ax.set_yscale("log")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97], h_pad=1.5, w_pad=1.5)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return data_cache, config_meta, present_variants


# ==========================================
# MAIN
# ==========================================
def main():
    proj = "cruzas-universit-della-svizzera-italiana/thesis_results"
    base_fig_dir = os.path.expanduser(
        "~/Documents/GitHub/PhD-Thesis-Samuel-Cruz/figures/thesis"
    )
    tex_dir = os.path.expanduser(
        "~/Documents/GitHub/PhD-Thesis-Samuel-Cruz/figures_tex"
    )
    table_dir = os.path.expanduser("~/Documents/GitHub/PhD-Thesis-Samuel-Cruz/tables")
    cache_dir = os.path.join(os.path.dirname(base_fig_dir), ".cache")

    configs = {
        "mnist": {
            "filters": {"config.model_name": "simple_cnn"},
            "model_pretty": "CNN",
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
            "y_log": False,
        },
        "cifar10": {
            "filters": {"config.model_name": "big_resnet"},
            "model_pretty": "ResNet",
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
            "y_log": False,
        },
        "tinyshakespeare": {
            "filters": {"config.model_name": "minigpt"},
            "model_pretty": "MiniGPT",
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
            "model_pretty": "PINN-FFNN",
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

    api = wandb.Api(timeout=300)

    for dataset, cfg in configs.items():
        ck = "num_stages" if dataset == "cifar10" else "num_subdomains"
        dataset_combos = set()
        for reg in ["weak", "strong"]:
            bkey = "effective_batch_size" if reg == "weak" else "batch_size"
            smap = cfg[f"sgd_filters_{reg}"]
            lrs = {bs: v["learning_rate"] for bs, v in smap.items()}

            data_cache, meta, variants = None, None, None
            for xax in ["grad_evals", "running_time"]:
                save_path = os.path.join(
                    base_fig_dir, f"{dataset}_{reg}_{xax}_grid.pdf"
                )
                data_cache, meta, variants = plot_grid_presentation(
                    proj,
                    xax,
                    bkey,
                    sorted(smap.keys()),
                    cfg["metrics"],
                    {"config.dataset_name": dataset, **cfg["filters"]},
                    smap,
                    None,
                    save_path,
                    cfg["y_log"],
                    ck,
                    cache_dir,
                    regime=reg,
                )

                save_latex_figure_code(
                    tex_dir,
                    dataset,
                    reg,
                    xax,
                    cfg["metrics"],
                    sorted(smap.keys()),
                    lrs,
                    meta["delta"],
                    cfg["model_pretty"],
                    f"scaling_{xax}",
                    meta["glob_opt"],
                    meta["loc_opt"],
                    meta["glob_so"],
                    meta["loc_so"],
                    variants,
                )

                if data_cache:
                    for size in data_cache:
                        for m in data_cache[size]:
                            opt = m["config"].get("optimizer", "unk")
                            n = _safe_int(m["config"].get(ck, 1))
                            dataset_combos.add((opt, n))

            generate_summary_table(
                table_dir,
                dataset,
                reg,
                cfg["metrics"],
                lrs,
                meta["glob_opt"],
                meta["loc_opt"],
                meta["glob_so"],
                meta["loc_so"],
                variants,
                meta["delta"],
                cfg["model_pretty"],
                data_cache,
                ck,
                lambda rid: _load_history_cached_by_id(
                    api, proj, rid, cache_dir, dataset
                ),
            )

        # Legend Logic
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
                if str(c[0]).lower() == "sgd":
                    lbl = r"SGD $\mid$ $N=1$"
                else:
                    lbl = rf"${latex_opt(str(c[0]))}$ $\mid$ $N={c[1]}$"
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
