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
        "font.size": 20,
        "axes.titlesize": 24,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "figure.titlesize": 26,
        "legend.fontsize": 24,
        "legend.handlelength": 3.0,
        "lines.linewidth": 3,
    }
)


# ==========================================
# UTILITIES & CACHING
# ==========================================
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
    name_map = {
        "acc": "accuracy",
        "accuracy": "accuracy",
        "loss": "loss",
        "train_perplexity": "train perplexity",
    }
    base = name_map.get(metric, metric).replace("_", " ")
    if base == "accuracy":
        return r"Test Accuracy (\%)"
    if base == "train perplexity":
        return r"Training Perplexity"
    if base == "loss":
        return r"Empirical Loss"
    return base.capitalize()


def _safe_int(v):
    """Safely converts value to int, handling strings/floats."""
    try:
        return int(float(v))
    except:
        return -1


def _loose_match(actual, target):
    """
    Robust comparison for config values (LR, params).
    """
    if actual == target:
        return True

    # Try string comparison
    if str(actual) == str(target):
        return True

    # Try float approximation
    try:
        f_act = float(actual)
        f_tgt = float(target)
        if abs(f_act - f_tgt) < 1e-6:
            return True
    except:
        pass

    return False


def _cache_path_for_run(cache_dir, dataset, run_id):
    return os.path.join(cache_dir, dataset, f"{run_id}.pkl")


def _load_history_cached_by_id(api, project_path, run_id, cache_dir, dataset):
    """
    Fetches run history safely using scan_history with dynamic key filtering.
    """
    path = None
    if cache_dir:
        path = _cache_path_for_run(cache_dir, dataset, run_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            try:
                return pd.read_pickle(path)
            except Exception as e:
                print(f"Warning: Corrupt cache for {run_id}, reloading. Error: {e}")

    try:
        run = api.run(f"{project_path}/{run_id}")

        print(
            f"  -> [Network] Streaming history for run: {run_id} ({dataset}) ...",
            end=" ",
            flush=True,
        )

        potential_keys = [
            "running_time",
            "grad_evals",
            "loss",
            "accuracy",
            "acc",
            "train_perplexity",
            "epoch",
            "iter",
        ]

        # Check what actually exists in this run's summary
        available_keys = set(run.summary.keys())

        # Always force essential keys
        minimal_keys = ["running_time", "grad_evals", "loss"]

        for k in potential_keys:
            if k in available_keys and k not in minimal_keys:
                minimal_keys.append(k)

        try:
            # Pass the filtered list of keys
            history_gen = run.scan_history(keys=minimal_keys)
            data = [row for row in history_gen]
            df = pd.DataFrame(data)
            print(f"Done ({len(df)} rows).")
        except Exception as scan_err:
            print(f"Error scanning history: {scan_err}")
            return pd.DataFrame()

        if cache_dir and path:
            try:
                df.to_pickle(path)
            except:
                pass
        return df

    except Exception as e:
        print(f"\nERROR fetching run {run_id}: {e}")
        return pd.DataFrame()


def _runs_index_path(cache_dir, dataset, keyhash):
    return os.path.join(cache_dir, dataset, "_runs_index", f"{keyhash}.pkl")


def _list_runs_cached(api, project_path, filters, cache_dir, dataset, max_age_hours=12):
    # Determine cache path
    canonical = json.dumps(
        {"project": project_path, "filters": filters}, sort_keys=True, default=str
    )
    keyhash = hashlib.md5(canonical.encode("utf-8")).hexdigest()

    if cache_dir:
        path = _runs_index_path(cache_dir, dataset, keyhash)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        now = time.time()

        if os.path.exists(path):
            try:
                payload = pd.read_pickle(path)
                # Return cached runs if fresh enough
                if now - payload["_fetched_at"] <= max_age_hours * 3600:
                    return payload["runs"]
            except:
                pass

    print(f"  -> [Network] Fetching run list for filter: {filters} ...")
    runs = api.runs(project_path, filters=filters)

    records = []
    for r in runs:
        records.append(
            {"id": r.id, "name": r.name, "config": dict(getattr(r, "config", {}))}
        )

    if cache_dir:
        try:
            pd.to_pickle({"_fetched_at": time.time(), "runs": records}, path)
        except:
            pass

    return records


# ==========================================
# PLOTTING LOGIC
# ==========================================
def get_style_for_combo(optimizer_name, n_val):
    opt = str(optimizer_name).lower()
    try:
        n = int(n_val)
    except:
        n = 1

    if "sgd" in opt:
        return "black", "--"

    if "_p" in opt:
        ls = "-."
        if n == 2:
            c = "#d62728"
        elif n == 4:
            c = "#ff7f0e"
        elif n == 8:
            c = "#8c564b"
        else:
            c = "#e377c2"
        return c, ls

    if "ip" in opt:
        ls = ":"
        if n == 2:
            c = "#2ca02c"
        elif n == 4:
            c = "#bcbd22"
        elif n == 8:
            c = "#17becf"
        else:
            c = "#7f7f7f"
        return c, ls

    ls = "-"
    if n == 2:
        c = "#1f77b4"
    elif n == 4:
        c = "#9467bd"
    elif n == 8:
        c = "#000080"
    else:
        c = "#1f77b4"
    return c, ls


def plot_presentation_pair(
    project_path: str,
    x_axis: str,
    base_key: str,
    target_size: int,
    metrics: list[str],
    filters_base: dict,
    sgd_filters: dict = None,
    extra_filters: dict = None,
    x_limits: tuple = None,
    y_limits_dict: dict = None,
    save_path: str = None,
    y_log: bool = False,
    combo_key: str = "num_subdomains",
    cache_dir: str = None,
    print_stats: bool = True,
    regime: str = "unknown",
):
    extra_filters = extra_filters or {}
    sgd_filters = sgd_filters or {}

    api = wandb.Api(timeout=300)
    dataset_name = filters_base.get("config.dataset_name", "dataset")

    def _fmt_opt(opt_raw):
        opt = str(opt_raw)
        opt_lower = opt.lower()
        if opt_lower == "sgd":
            return "SGD"
        if "apts" in opt_lower:
            base = r"\mathrm{SAPTS}"
            if "ip" in opt_lower:
                return base + r"_{\mathrm{IP}}"
            if "_p" in opt_lower:
                return base + r"_{\mathrm{P}}"
            if "_d" in opt_lower:
                return base + r"_{\mathrm{D}}"
            return base
        return latex_opt(opt)

    def _get_order_str(cfg, prefix):
        is_2nd = cfg.get(f"{prefix}_second_order", False)
        is_dog = cfg.get(f"{prefix}_dogleg", False)
        if not is_2nd:
            return "1st"
        if is_dog:
            return "2nd+Dog"
        return "2nd"

    # 1. Fetch Generic Runs
    base_f = {**filters_base, f"config.{base_key}": target_size, **extra_filters}
    if print_stats:
        print(f"Fetching runs from: {project_path} | Filter: {base_f}")

    runs_gen = _list_runs_cached(api, project_path, base_f, cache_dir, dataset_name)
    runs_gen = [
        m
        for m in runs_gen
        if _safe_int(m["config"].get(base_key)) == _safe_int(target_size)
        and m["config"].get("optimizer", "").lower() != "sgd"
    ]

    # 2. Fetch Runs (SGD)
    runs_sgd = []
    if target_size in sgd_filters:
        lr_constraints = sgd_filters[target_size]

        sgd_api_f = {"config.dataset_name": dataset_name, "config.optimizer": "sgd"}
        if "config.model_name" in filters_base:
            sgd_api_f["config.model_name"] = filters_base["config.model_name"]

        runs_sgd_raw = _list_runs_cached(
            api, project_path, sgd_api_f, cache_dir, dataset_name
        )

        for m in runs_sgd_raw:
            if _safe_int(m["config"].get("batch_size", -1)) != _safe_int(target_size):
                if _safe_int(m["config"].get("effective_batch_size", -1)) != _safe_int(
                    target_size
                ):
                    continue

            match = True
            for k, target_v in lr_constraints.items():
                actual_v = m["config"].get(k)
                if not _loose_match(actual_v, target_v):
                    match = False
                    break
            if match:
                runs_sgd.append(m)

    runs_all = runs_gen + runs_sgd

    if print_stats:
        print(f"[{dataset_name}] Size {target_size}: Found {len(runs_all)} runs.")

    if not runs_all:
        if print_stats:
            print("No runs found. Skipping.")
        return

    @lru_cache(maxsize=None)
    def get_history(run_id):
        return _load_history_cached_by_id(
            api, project_path, run_id, cache_dir, dataset_name
        )

    # 3. Grouping logic
    def _resolve_n_val(cfg):
        opt = cfg.get("optimizer", "").lower()
        if "cifar" in dataset_name.lower() and "sgd" in opt:
            val = cfg.get("num_subdomains", "unk")
            if _safe_int(val) > 0:
                return val

        val = cfg.get(combo_key, "unk")
        if val == "unk":
            if combo_key == "num_stages":
                val = cfg.get("num_subdomains", "unk")
            elif combo_key == "num_subdomains":
                val = cfg.get("num_stages", "unk")

        if _safe_int(val) <= 0:
            return 1
        return val

    combos = sorted(
        {
            (m["config"].get("optimizer", "unk"), _resolve_n_val(m["config"]))
            for m in runs_all
        },
        key=lambda x: (str(x[0]), _safe_int(x[1])),
    )

    runs_by = {}
    for m in runs_all:
        val = _resolve_n_val(m["config"])
        combo = (m["config"].get("optimizer", "unk"), val)
        runs_by.setdefault(combo, []).append(m)

    # 4. Plotting
    ncols = len(metrics)
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(16, 6), squeeze=False)

    xlabel_map = {
        "grad_evals": r"Gradient Evaluations",
        "running_time": "Wall-clock Time (s)",
        "epoch": "Epochs",
        "iter": "Iterations",
    }

    final_stats = []

    for i, metric in enumerate(metrics):
        ax = axes[0, i]
        plot_min_x = float("inf")

        for combo in combos:
            opt_name, n_val = combo
            is_sgd_multi = "sgd" in str(opt_name).lower() and _safe_int(n_val) > 1
            runs_in_combo = runs_by.get(combo, [])
            series_list = []

            # Lists for collecting stats
            combo_times = []
            combo_grads = []
            combo_final_loss = []
            combo_final_acc = []

            for m in runs_in_combo:
                hist = get_history(m["id"])
                if hist.empty:
                    continue

                if i == 0 and print_stats:
                    if (
                        "running_time" in hist.columns
                        and not hist["running_time"].isna().all()
                    ):
                        combo_times.append(hist["running_time"].max())
                    if (
                        "grad_evals" in hist.columns
                        and not hist["grad_evals"].isna().all()
                    ):
                        combo_grads.append(hist["grad_evals"].max())

                    if "loss" in hist.columns:
                        s_loss = pd.to_numeric(hist["loss"], errors="coerce").dropna()
                        if not s_loss.empty:
                            combo_final_loss.append(s_loss.iloc[-1])

                    acc_val = None
                    if "accuracy" in hist.columns:
                        s_acc = pd.to_numeric(
                            hist["accuracy"], errors="coerce"
                        ).dropna()
                        if not s_acc.empty:
                            acc_val = s_acc.iloc[-1]
                    elif "acc" in hist.columns:
                        s_acc = pd.to_numeric(hist["acc"], errors="coerce").dropna()
                        if not s_acc.empty:
                            acc_val = s_acc.iloc[-1]
                    if acc_val is not None:
                        combo_final_acc.append(acc_val)

                if x_axis in hist.columns and metric in hist.columns:
                    df = hist[[x_axis, metric]].dropna()
                    if not df.empty:
                        # --- NORMALIZATION FIX START ---
                        # If plotting running_time, shift so start is 0
                        if x_axis == "running_time":
                            start_time = df[x_axis].min()
                            df[x_axis] = df[x_axis] - start_time
                        # --- NORMALIZATION FIX END ---

                        min_x_here = df[x_axis][df[x_axis] > 0].min()
                        if not np.isnan(min_x_here):
                            plot_min_x = min(plot_min_x, min_x_here)
                        series_list.append(df)

            if i == 0 and print_stats and (combo_times or combo_grads):
                avg_time = np.mean(combo_times) if combo_times else 0.0
                std_time = np.std(combo_times) if combo_times else 0.0
                avg_grad = np.mean(combo_grads) if combo_grads else 0.0
                std_grad = np.std(combo_grads) if combo_grads else 0.0

                loss_str = "-"
                if combo_final_loss:
                    l_avg = np.mean(combo_final_loss)
                    l_std = np.std(combo_final_loss)
                    loss_str = f"{l_avg:.2f} ± {l_std:.2f}"

                acc_str = "-"
                if combo_final_acc:
                    a_avg = np.mean(combo_final_acc)
                    a_std = np.std(combo_final_acc)
                    acc_str = f"{a_avg:.2f} ± {a_std:.2f}"

                formatted_opt = _fmt_opt(combo[0])
                first_cfg = runs_in_combo[0]["config"]
                opt_lower = str(combo[0]).lower()

                lr_str = "-"
                overlap_str = "-"
                delta_str = "-"
                max_delta_str = "-"
                min_delta_str = "-"

                ov_val = first_cfg.get("overlap")
                if ov_val is not None:
                    try:
                        overlap_str = f"{float(ov_val):g}"
                    except:
                        overlap_str = str(ov_val)

                if "sgd" in opt_lower:
                    raw_lr = first_cfg.get("learning_rate")
                    if raw_lr is not None:
                        try:
                            lr_str = f"{float(raw_lr):g}"
                        except:
                            lr_str = str(raw_lr)

                if "apts" in opt_lower:
                    d = first_cfg.get("delta")
                    md = first_cfg.get("max_delta")
                    mind = first_cfg.get("min_delta")
                    if d is not None:
                        try:
                            delta_str = f"{float(d):g}"
                        except:
                            delta_str = str(d)
                    if md is not None:
                        try:
                            max_delta_str = f"{float(md):g}"
                        except:
                            max_delta_str = str(md)
                    if mind is not None:
                        try:
                            min_delta_str = f"{float(mind):g}"
                        except:
                            min_delta_str = str(mind)

                g_opt = first_cfg.get("glob_opt") or first_cfg.get("global_optimizer")
                l_opt = first_cfg.get("loc_opt") or first_cfg.get("local_optimizer")
                if g_opt is None:
                    g_opt = "-"
                if l_opt is None:
                    l_opt = "-"

                if "sgd" in opt_lower:
                    g_info = "-"
                    l_info = "-"
                else:
                    g_info = _get_order_str(first_cfg, "glob")
                    l_info = _get_order_str(first_cfg, "loc")

                final_stats.append(
                    {
                        "Optimizer": formatted_opt,
                        "N": combo[1],
                        "Runs": len(runs_in_combo),
                        "LR": lr_str,
                        "Overlap": overlap_str,
                        "Delta": delta_str,
                        "Min Delta": min_delta_str,
                        "Max Delta": max_delta_str,
                        "Glob Opt": g_opt,
                        "Loc Opt": l_opt,
                        "Global": g_info,
                        "Local": l_info,
                        "Time (s)": f"{avg_time:.2f} ± {std_time:.2f}",
                        "Grad Evals": f"{avg_grad:.2f} ± {std_grad:.2f}",
                        "Final Loss": loss_str,
                        "Final Acc (%)": acc_str,
                        "_raw_time": avg_time,
                        "_raw_opt": combo[0],
                        "_n_val": _safe_int(combo[1]),
                    }
                )

            if is_sgd_multi:
                continue
            if not series_list:
                continue

            all_x = np.concatenate([s[x_axis].values for s in series_list])
            grid = np.linspace(all_x.min(), all_x.max(), 200)
            data = np.stack(
                [np.interp(grid, s[x_axis], s[metric]) for s in series_list]
            )
            mean = data.mean(axis=0)
            std = data.std(axis=0)

            color, ls = get_style_for_combo(opt_name, n_val)
            alpha_fill = 0.15 if "sgd" in str(opt_name).lower() else 0.25

            ax.plot(grid, mean, color=color, linestyle=ls, label=str(combo))
            if len(series_list) > 1:
                ax.fill_between(
                    grid, mean - std, mean + std, alpha=alpha_fill, color=color
                )

        ax.set_title(_metric_label(metric))
        ax.set_xlabel(xlabel_map.get(x_axis, x_axis))
        ax.grid(True, alpha=0.3, which="both")

        # --- SYMLOG X-AXIS HANDLING ---
        # NOTE: Removed 'running_time' symlog override to keep it linear as requested.
        if x_axis == "grad_evals" and "shakespeare" in dataset_name.lower():
            # Linear from 0 to 10 evals, Log after that
            ax.set_xscale("symlog", linthresh=10.0)
        # -----------------------------------

        if y_log or metric == "loss" or metric == "train_perplexity":
            ax.set_yscale("log")

        # --- SMART ANCHORING ---
        if x_limits and x_axis in x_limits:
            ax.set_xlim(*x_limits[x_axis])
        elif ax.get_xscale() == "linear":
            ax.set_xlim(left=0)
        elif ax.get_xscale() == "symlog":
            # UPDATED: Anchor to plot_min_x instead of 0
            # This keeps symlog scaling but removes empty space on the left.
            if plot_min_x != float("inf"):
                ax.set_xlim(left=plot_min_x)
            else:
                ax.set_xlim(left=0)
        elif ax.get_xscale() == "log":
            if plot_min_x != float("inf") and plot_min_x > 0:
                ax.set_xlim(left=plot_min_x)
        # -----------------------

        if y_limits_dict and metric in y_limits_dict:
            ax.set_ylim(*y_limits_dict[metric])

    plt.tight_layout()

    # --- PROCESS AND PRINT STATS TABLE ---
    if print_stats and final_stats:
        print("\n" + "=" * 120)
        bs_label = (
            "Effective Batch Size" if "effective" in base_key else "Global Batch Size"
        )
        print(f"SUMMARY STATISTICS: {dataset_name} | {regime.upper()} SCALING")
        print(f"Constraint: {bs_label} = {target_size}")
        print("=" * 120)

        baselines = {}
        for row in final_stats:
            opt_raw = str(row["_raw_opt"]).lower()
            n_val = row["_n_val"]
            t_val = row["_raw_time"]
            if "sgd" in opt_raw and n_val == 1:
                baselines["sgd_base"] = t_val
            elif "sgd" not in opt_raw and n_val == 2:
                baselines[row["_raw_opt"]] = t_val

        for row in final_stats:
            opt_raw = str(row["_raw_opt"]).lower()
            t_val = row["_raw_time"]
            base_time = None
            if "sgd" in opt_raw:
                base_time = baselines.get("sgd_base")
            else:
                base_time = baselines.get(row["_raw_opt"])

            if base_time and t_val > 1e-9:
                sp = base_time / t_val
                row["Speedup"] = f"{sp:.2f}x"
            else:
                row["Speedup"] = "-"

        df_stats = pd.DataFrame(final_stats)
        df_stats["sort_helper"] = df_stats["Optimizer"].apply(
            lambda x: "AAA" if "SGD" in x and "\\" not in x else x
        )
        df_stats = df_stats.sort_values(by=["sort_helper", "N"])

        cols_to_drop = ["sort_helper", "_raw_time", "_raw_opt", "_n_val"]
        ds_lower = dataset_name.lower()
        if "shakespeare" in ds_lower or "poisson" in ds_lower:
            cols_to_drop.append("Final Acc (%)")

        df_stats = df_stats.drop(
            columns=[c for c in cols_to_drop if c in df_stats.columns]
        )

        print(df_stats.to_string(index=False))
        print("=" * 120 + "\n")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {save_path}")
        plt.close(fig)

        legend_handles = []
        labels_full = []

        def _legend_label(opt, nval):
            if str(opt).lower() == "sgd":
                return r"SGD Baseline"
            n_str = "?" if nval in ("unk", None) else str(nval)
            opt_tex = latex_opt(str(opt).upper())
            return rf"${opt_tex}\;\mid\;N={n_str}$"

        for c in combos:
            opt_name, n_val = c
            is_sgd_multi = "sgd" in str(opt_name).lower() and _safe_int(n_val) > 1
            if is_sgd_multi:
                continue
            color, ls = get_style_for_combo(opt_name, n_val)
            legend_handles.append(Line2D([0], [0], color=color, lw=4, linestyle=ls))
            labels_full.append(_legend_label(*c))

        n_items = len(labels_full)
        if n_items > 4:
            ncols_leg = math.ceil(n_items / 2)
            fig_height = 2.5
        else:
            ncols_leg = n_items
            fig_height = 1.5

        leg_fig = plt.figure(figsize=(18, fig_height))
        leg_fig.legend(
            legend_handles, labels_full, loc="center", ncol=ncols_leg, frameon=False
        )
        leg_path = save_path.replace(".pdf", "_legend.pdf")
        leg_fig.savefig(leg_path, bbox_inches="tight")
        plt.close(leg_fig)
    else:
        plt.show()


# ==========================================
# MAIN CONFIGURATION
# ==========================================
def main():
    entity = "cruzas-universit-della-svizzera-italiana"
    project = "thesis_results"
    full_project = f"{entity}/{project}"

    base_dir = os.path.expanduser(
        "~/Documents/GitHub/PhD-Thesis-Samuel-Cruz/figures/presentation"
    )
    cache_dir = os.path.join(os.path.dirname(base_dir), ".cache")

    # ==========================================
    # 1. SCENARIOS (Enable both Weak and Strong)
    # ==========================================
    scenarios_map = {
        "mnist": [
            ("weak", 128),
            ("strong", 1024),
        ],
        "cifar10": [
            ("strong", 256),
        ],
        "tinyshakespeare": [("weak", 128), ("strong", 1024)],
        "poisson2d": [("weak", 64), ("strong", 256)],
    }

    x_axes_to_plot = ["grad_evals", "running_time", "epoch"]

    # ==========================================
    # 2. CONFIGS
    # ==========================================
    configs = {
        "mnist": {
            "filters": {"config.model_name": "simple_cnn"},
            "sgd_filters_weak": {
                128: {"learning_rate": 0.01},
            },
            "sgd_filters_strong": {
                1024: {"learning_rate": 0.10},
            },
            "metrics": ["loss", "accuracy"],
            "x_limits": None,
            "y_limits": {"accuracy": (0, 100)},
            "y_log": False,
        },
        "cifar10": {
            "filters": {"config.model_name": "big_resnet"},
            "sgd_filters_strong": {
                256: {"learning_rate": 0.01},
            },
            "metrics": ["loss", "accuracy"],
            "x_limits": None,
            "y_limits": {"accuracy": (20, 85)},
            "y_log": False,
        },
        "tinyshakespeare": {
            "filters": {"config.model_name": "minigpt"},
            "sgd_filters_weak": {
                128: {"learning_rate": 0.10},
            },
            "sgd_filters_strong": {
                1024: {"learning_rate": 0.01},
            },
            "metrics": ["loss"],
            "x_limits": None,
            "y_limits": None,
            "y_log": False,
        },
        "poisson2d": {
            "filters": {"config.model_name": "pinn_ffnn"},
            "sgd_filters_weak": {
                64: {"learning_rate": 0.01},
            },
            "sgd_filters_strong": {
                256: {"learning_rate": 0.1},
            },
            "metrics": ["loss"],
            "x_limits": None,
            "y_limits": None,
            "y_log": True,
        },
    }

    # Iterate over datasets
    for dataset, scenarios_list in scenarios_map.items():
        if dataset not in configs:
            print(f"Skipping {dataset} (No config found)")
            continue

        if dataset == "cifar10":
            combo_key = "num_stages"
        else:
            combo_key = "num_subdomains"

        print(f"--- Processing {dataset} (Grouping by {combo_key}) ---")

        for regime, size in scenarios_list:
            print(f"  > Scenario: {regime} / Size {size}")
            cfg = configs[dataset]

            base_key = "effective_batch_size" if regime == "weak" else "batch_size"

            if regime == "weak":
                current_sgd_filters = cfg.get("sgd_filters_weak", {})
            else:
                current_sgd_filters = cfg.get("sgd_filters_strong", {})

            fb = {"config.dataset_name": dataset, **cfg["filters"]}

            for i, x_axis_template in enumerate(x_axes_to_plot):
                x_axis = x_axis_template
                # Swap epoch for iter if dataset is tinyshakespeare
                if dataset == "tinyshakespeare" and x_axis == "epoch":
                    x_axis = "iter"

                fname = f"slide_{dataset}_{regime}_sz{size}_{x_axis}.pdf"
                save_path = os.path.join(base_dir, fname)

                do_print_stats = i == 0

                plot_presentation_pair(
                    project_path=full_project,
                    x_axis=x_axis,
                    base_key=base_key,
                    target_size=size,
                    metrics=cfg["metrics"],
                    filters_base=fb,
                    sgd_filters=current_sgd_filters,
                    x_limits=cfg["x_limits"],
                    y_limits_dict=cfg["y_limits"],
                    save_path=save_path,
                    y_log=cfg["y_log"],
                    combo_key=combo_key,
                    cache_dir=cache_dir,
                    print_stats=do_print_stats,
                    regime=regime,
                )


if __name__ == "__main__":
    main()
