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
    eval_point,
    data_cache,
    get_history_fn,
):
    os.makedirs(tex_dir, exist_ok=True)
    filename = f"{dataset}_{regime}_{xax}_grid.tex"
    filepath = os.path.join(tex_dir, filename)

    is_ts = "tinyshakespeare" in dataset.lower()
    prog_label = "iteration" if is_ts else "epoch"

    normalized_variants = {v.lower().replace("sapts", "apts") for v in present_variants}
    is_glob_2nd = str(glob_so).lower() == "true"
    is_loc_2nd = str(loc_so).lower() == "true"
    order_suffix = " (2nd order)" if (is_glob_2nd and is_loc_2nd) else " (1st order)"

    lr_label = "EBS" if regime == "weak" else "GBS"
    lr_parts = [f"{lr} for {lr_label}={bs}" for bs, lr in sgd_lrs.items()]
    lr_str = ", ".join(lr_parts)

    xax_label_raw = {
        "grad_evals": "grad. evals.",
        "running_time": "wall-clock time",
    }.get(xax, xax)

    m_desc = (
        "Average empirical loss (left) and average test accuracy (right)"
        if len(metrics) > 1
        else "Average empirical loss"
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

    xax_label_for_fig = xax_label_raw.strip(".")
    if xax_label_for_fig == "grad. evals":
        xax_label_for_fig = "number of gradient evaluations"

    content = rf"""\begin{{figure}}[htbp]
    \centering
    \includegraphics[width=\linewidth]{{figures/{dataset}_{regime}_{xax}_grid.pdf}}%
    \vspace{{1ex}}
    \includegraphics[width=\linewidth]{{figures/{dataset}_legend.pdf}}%
    \caption{{{m_desc} vs {xax_label_for_fig}. Dataset: {dataset_pretty} ({regime} scaling regime). Network type: {model_pretty}. Optimizers: {opt_list_str}. \SAPTS configuration: global optimizer: {format_opt_name(glob_opt)}, local optimizer: {format_opt_name(loc_opt)}, $\Delta^{{(0)}} = {delta}$. Learning rates for SGD: {lr_str}. Curves are clipped at {eval_point:.0f} {prog_label}s. An overlap of approximately 33\% was applied between consecutive mini-batches and micro-batches.}}
\label{{fig:{dataset}_{regime}_{label_suffix}}}
\end{{figure}}
"""
    with open(filepath, "w") as f:
        f.write(content)


def get_siunitx_format(val_list, default_prec=2):
    if not val_list:
        return f"1.{default_prec}(1.{default_prec})"
    max_int_m = 1
    max_int_s = 1
    for item in val_list:
        m, s = item if isinstance(item, (list, tuple)) else (item, 0)
        max_int_m = max(max_int_m, len(str(int(abs(m)))))
        max_int_s = max(max_int_s, len(str(int(abs(s)))))
    return f"{max_int_m}.{default_prec}({max_int_s}.{default_prec})"


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
    eval_point,
):
    os.makedirs(table_dir, exist_ok=True)
    file_path = os.path.join(table_dir, f"{dataset}_{regime}_scaling.tex")

    is_strong = regime.lower() == "strong"
    bs_acronym = "GBS" if is_strong else "EBS"
    bs_full = "global batch size (GBS)" if is_strong else "effective batch size (EBS)"

    metric_col_name = "Speedup" if is_strong else "Efficiency"

    is_ts = "tinyshakespeare" in dataset.lower()
    prog_key = "iter" if is_ts else "epoch"
    prog_label = "iteration" if is_ts else "epoch"

    # --- PASS 1: FORMATTING PASS ---
    metric_stats = {m: [] for m in metrics}
    time_stats = []
    for size in sorted(data_cache.keys()):
        for r in data_cache[size]:
            h = get_history_fn(r["id"])
            if (
                not h.empty
                and prog_key in h.columns
                and h[prog_key].max() >= eval_point
            ):
                idx = (h[prog_key] - eval_point).abs().idxmin()
                row = h.loc[idx]
                for m in metrics:
                    metric_stats[m].append((row.get(m, 0), 0.1))
                time_stats.append((row["running_time"] - h["running_time"].min(), 1.0))

    m_formats = {
        m: get_siunitx_format(metric_stats[m], 2 if "acc" in m.lower() else 3)
        for m in metrics
    }
    t_format = get_siunitx_format(time_stats, 2)

    # --- PASS 2: CAPTION LOGIC (Matched to Figure Logic with User Fixes) ---
    dataset_pretty = {
        "mnist": "MNIST",
        "cifar10": "CIFAR-10",
        "tinyshakespeare": "Tiny Shakespeare",
        "poisson2d": "Poisson 2D",
    }.get(dataset, dataset.capitalize())

    # Build SAPTS variant list and order suffix
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

    # Build Learning Rate string to match figure format
    lr_parts = [f"{lr} for {bs_acronym}={bs}" for bs, lr in sgd_lrs.items()]
    lr_str = ", ".join(lr_parts)

    # Final Table Caption Construction (Redundancy removed)
    table_caption = (
        f"{regime.capitalize()} scaling table. Dataset: {dataset_pretty}. "
        f"Network type: {model_pretty}. Optimizers: {opt_list_str}. "
        f"SAPTS configuration: global optimizer: {format_opt_name(glob_opt)}, local optimizer: {format_opt_name(loc_opt)}, $\\Delta^{{(0)}} = {delta}$. "
        f"Learning rates for SGD: {lr_str}. "
        f"Values are evaluated at {eval_point:.0f} {prog_label}s. "
        f"An overlap of approximately 33\% was applied between consecutive mini-batches and micro-batches."
    )

    # --- PASS 3: GENERATE THE TABLE ---
    with open(file_path, "w") as f:
        f.write(f"\\begin{{table}}[htbp]\n  \\centering\n")
        f.write(f"  \\caption{{{table_caption}}}\n")
        f.write(f"  \\label{{tab:{dataset}_{regime}_scaling}}\n")
        f.write("  \\resizebox{\\textwidth}{!}{\n")

        si_opts = (
            "separate-uncertainty, input-open-uncertainty=(, input-close-uncertainty=)"
        )
        col_setup = "c c c c c "
        for m in metrics:
            col_setup += f"S[{si_opts}, table-format={m_formats[m]}] "
        col_setup += f"S[{si_opts}, table-format={t_format}] c"

        f.write(f"    \\begin{{tabular}}{{{col_setup.strip()}}}\n      \\toprule\n")

        headers = (
            [f"{{{bs_acronym}}}", "{Optimizer}", "{Runs}", "{$N$}", "{Grad. Evals.}"]
            + [
                ("{Accuracy (\\%)}" if "acc" in m.lower() else f"{{{m.capitalize()}}}")
                for m in metrics
            ]
            + ["{Time (s)}", f"{{{metric_col_name}}}"]
        )
        f.write("      " + " & ".join(headers) + " \\\\\n      \\midrule\n")

        for size in sorted(data_cache.keys()):
            runs = data_cache[size]

            # 1. FIND THE GLOBAL BASELINE (SGD N=1) FOR THIS BATCH SIZE
            global_baseline_time = None
            sgd_n1_runs = [
                r
                for r in runs
                if r["config"].get("optimizer", "").lower() == "sgd"
                and _safe_int(r["config"].get("num_subdomains", 1)) == 1
            ]

            baseline_times = []
            for r in sgd_n1_runs:
                h = get_history_fn(r["id"])
                if (
                    not h.empty
                    and prog_key in h.columns
                    and h[prog_key].max() >= eval_point
                ):
                    idx = (h[prog_key] - eval_point).abs().idxmin()
                    baseline_times.append(
                        h.loc[idx, "running_time"] - h["running_time"].min()
                    )

            if baseline_times:
                global_baseline_time = np.mean(baseline_times)

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

            is_first_row_of_size = True
            last_opt = None
            for opt, n in unique_combos:
                if last_opt is not None and opt != last_opt:
                    f.write(f"      \\cmidrule(lr){{2-{len(headers)}}}\n")
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

                valid_stats = []
                for r in relevant_runs:
                    h = get_history_fn(r["id"])
                    if (
                        not h.empty
                        and prog_key in h.columns
                        and h[prog_key].max() >= eval_point
                    ):
                        idx = (h[prog_key] - eval_point).abs().idxmin()
                        valid_stats.append(
                            h.iloc[idx].to_dict() | {"start_t": h["running_time"].min()}
                        )

                if not valid_stats:
                    continue

                time_vals = [s["running_time"] - s["start_t"] for s in valid_stats]
                avg_time = np.mean(time_vals)

                if global_baseline_time:
                    relative_val = global_baseline_time / avg_time
                else:
                    relative_val = 1.0

                opt_label = "SGD" if "sgd" in opt.lower() else f"${latex_opt(opt)}$"
                row = [
                    str(size) if is_first_row_of_size else "",
                    opt_label,
                    str(len(valid_stats)),
                    str(n),
                    f"{np.mean([s.get('grad_evals', 0) for s in valid_stats]):.0f}",
                ]
                is_first_row_of_size = False

                for m in metrics:
                    vals = [s.get(m, 0) for s in valid_stats]
                    p = 2 if "acc" in m.lower() else 3
                    row.append(f"{np.mean(vals):.{p}f}({np.std(vals):.{p}f})")

                row.append(f"{avg_time:.2f}({np.std(time_vals):.2f})")
                row.append(f"{relative_val:.2f}")

                f.write("      " + " & ".join(row) + " \\\\\n")

            f.write(
                "      \\midrule\n"
                if size != sorted(data_cache.keys())[-1]
                else "      \\bottomrule\n"
            )

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
    eval_point=None,
):
    api = wandb.Api(timeout=300)
    dataset_name = filters_base.get("config.dataset_name", "dataset").lower()
    prog_key = "iter" if "tinyshakespeare" in dataset_name else "epoch"

    @lru_cache(maxsize=None)
    def get_history(run_id):
        return _load_history_cached_by_id(
            api, project_path, run_id, cache_dir, dataset_name
        )

    def _res_n(cfg):
        opt = cfg.get("optimizer", "").lower()
        k = "num_stages" if ("cifar" in dataset_name and "sgd" in opt) else combo_key
        v = cfg.get(
            k, cfg.get("num_subdomains" if k == "num_stages" else "num_stages", 1)
        )
        return _safe_int(v) if _safe_int(v) > 0 else 1

    data_cache, config_meta, present_variants = (
        {},
        {
            "delta": "N/A",
            "glob_opt": "N/A",
            "loc_opt": "N/A",
            "glob_so": "N/A",
            "loc_so": "N/A",
        },
        set(),
    )

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

    nrows, ncols = len(batch_sizes), len(metrics)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(7.5 * ncols, 3.5 * nrows if ncols == 1 else 5.5 * nrows),
        squeeze=False,
        sharey="col",
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

    for i, size in enumerate(batch_sizes):
        runs_all = data_cache[size]
        combos = sorted(
            {
                (r["config"].get("optimizer", "unk"), _res_n(r["config"]))
                for r in runs_all
            },
            key=lambda x: (str(x[0]), x[1]),
        )

        sgd_limit = None
        datasets_to_limit = ["poisson2d", "tinyshakespeare", "cifar10"]
        if any(ds in dataset_name for ds in datasets_to_limit):
            sgd_runs = [
                r for r in runs_all if "sgd" in r["config"].get("optimizer", "").lower()
            ]
            limit_candidates = []
            for r in sgd_runs:
                hist = get_history(r["id"])
                if (
                    not hist.empty
                    and prog_key in hist.columns
                    and x_axis in hist.columns
                ):
                    if hist[prog_key].max() >= eval_point:
                        limit_candidates.append(
                            hist[hist[prog_key] <= eval_point][x_axis].max()
                        )
            if limit_candidates:
                sgd_limit = max(limit_candidates)

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
                        and prog_key in hist.columns
                    ):
                        if hist[prog_key].max() >= eval_point:
                            df = hist[hist[prog_key] <= eval_point][
                                [x_axis, metric]
                            ].dropna()
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
                ax2.set_ylabel(
                    rf"{'GLOBAL' if regime == 'strong' else 'EFFECTIVE'} BATCH SIZE = {size}",
                    rotation=270,
                    labelpad=30,
                    fontweight="bold",
                    fontsize=14,
                )
                ax2.set_yticks([])
            if i == nrows - 1:
                label_name = (
                    "Grad. Evals."
                    if x_axis == "grad_evals"
                    else x_axis.replace("_", " ").title()
                )
                ax.set_xlabel(label_name)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_xlim(left=0)
            if sgd_limit:
                ax.set_xlim(right=sgd_limit)
            if y_log or metric == "loss":
                ax.set_yscale("log")
            low, high = metric_bounds[metric]
            if not np.isinf(low):
                if "poisson2d" in dataset_name and metric == "loss":
                    ax.set_ylim(max(low, 1e-12), 10**3)
                elif y_limits and metric in y_limits:
                    ax.set_ylim(*y_limits[metric])
                else:
                    ax.set_ylim(
                        (max(low, 1e-12) if (y_log or metric == "loss") else low) / 1.1,
                        high * 1.1,
                    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout(rect=[0, 0.01, 1, 0.99], h_pad=1.0, w_pad=1.0)
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
            "eval_point": 5,
            "filters": {"config.model_name": "simple_cnn"},
            "model_pretty": "CNN",
            "sgd_filters_strong": {
                1024: {"learning_rate": 0.1},
                2048: {"learning_rate": 0.1},
                4096: {"learning_rate": 0.1},
            },
            "sgd_filters_weak": {
                128: {"learning_rate": 0.01},
                256: {"learning_rate": 0.1},
                512: {"learning_rate": 0.1},
            },
            "metrics": ["loss", "accuracy"],
            "y_log": False,
        },
        "cifar10": {
            "eval_point": 25,
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
            "eval_point": 750,
            "filters": {"config.model_name": "minigpt"},
            "model_pretty": "MiniGPT",
            "sgd_filters_strong": {
                1024: {"learning_rate": 0.01},
                2048: {"learning_rate": 0.01},
                4096: {"learning_rate": 0.01},
            },
            "sgd_filters_weak": {
                128: {"learning_rate": 0.1},
                256: {"learning_rate": 0.1},
                512: {"learning_rate": 0.1},
            },
            "metrics": ["loss"],
            "y_log": False,
        },
        "poisson2d": {
            "eval_point": 500,
            "filters": {"config.model_name": "pinn_ffnn"},
            "model_pretty": "PINN-FFNN",
            "sgd_filters_strong": {
                128: {"learning_rate": 0.001},
                256: {"learning_rate": 0.1},
                512: {"learning_rate": 0.001},
            },
            "sgd_filters_weak": {
                64: {"learning_rate": 0.01},
                128: {"learning_rate": 0.001},
                256: {"learning_rate": 0.1},
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
                    eval_point=cfg["eval_point"],
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
                    cfg["eval_point"],
                    data_cache,
                    lambda rid: _load_history_cached_by_id(
                        api, proj, rid, cache_dir, dataset
                    ),
                )

                if xax == "grad_evals":
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
                        cfg["eval_point"],
                    )
                    for size in data_cache:
                        for m in data_cache[size]:
                            dataset_combos.add(
                                (
                                    m["config"].get("optimizer", "unk"),
                                    _safe_int(m["config"].get(ck, 1)),
                                )
                            )

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
                leg_l.append(
                    r"SGD $\mid$ $N=1$"
                    if str(c[0]).lower() == "sgd"
                    else rf"${latex_opt(str(c[0]))}$ $\mid$ $N={c[1]}$"
                )
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
