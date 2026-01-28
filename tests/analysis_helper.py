#!/usr/bin/env python3
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cycler import cycler

try:
    import wandb
except ImportError:
    print("Error: wandb package not found. Install with: pip install wandb")
    sys.exit(1)

# =============================================================================
# MODERN COLOR SCHEME & BEAMER STYLING
# =============================================================================
COLOURS = {
    "modernBlue": "#0054A6",
    "modernLight": "#2980B9",
    "modernBlack": "#1E1E1E",
    "modernDark": "#34495E",
    "modernPink": "#FF2D55",
    "modernPurple": "#8E44AD",
    "modernTeal": "#00A896",
    # "background": "#FAFAFA",
    "background": "#FFFFFF",
}


def setup_plotting_style():
    """Sets up global matplotlib parameters for consistent, publication-quality plots."""
    custom_cycler = cycler(
        color=[
            COLOURS["modernBlue"],
            COLOURS["modernPink"],
            COLOURS["modernTeal"],
            COLOURS["modernPurple"],
            COLOURS["modernLight"],
            COLOURS["modernDark"],
        ]
    )

    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{sfmath}",
            "font.family": "serif",
            "font.size": 22,  # Increased from 18
            "axes.titlesize": 28,  # Increased from 22
            "axes.labelsize": 24,  # Increased from 20
            "xtick.labelsize": 20,  # Increased from 18
            "ytick.labelsize": 20,  # Increased from 18
            "figure.titlesize": 32,  # Increased from 26
            "legend.fontsize": 20,
            "legend.handlelength": 3.0,
            "lines.linewidth": 4,  # Thicker lines for visibility
            "lines.markersize": 12,
            "axes.prop_cycle": custom_cycler,
            "grid.alpha": 0.3,
            "figure.facecolor": COLOURS["background"],
        }
    )


def get_custom_heat_cmap(metric_type: str = "loss"):
    """Returns a diverging color map for heatmaps based on the metric type."""
    if metric_type == "loss":
        colors = [COLOURS["modernBlue"], "#FFFFFF", COLOURS["modernPink"]]
    else:
        colors = [COLOURS["modernPink"], "#FFFFFF", COLOURS["modernBlue"]]
    return mcolors.LinearSegmentedColormap.from_list("modern_theme", colors)


# =============================================================================
# WANDB DATA FETCHING & CACHING
# =============================================================================


def get_cache_dir(model_type: str = "generic") -> Path:
    """Get cache directory for storing wandb data for a specific model family."""
    cache_dir = Path.home() / ".cache" / f"dd4ml_hyperparam_analysis_{model_type}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_cache_key(project_path: str, filters: Optional[Dict[str, Any]]) -> str:
    """Generate a stable MD5 cache key for project and filters."""
    canonical = json.dumps(
        {"project": project_path, "filters": filters or {}}, sort_keys=True, default=str
    )
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()


def fetch_runs(
    project: str = "ohtests",
    entity: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    use_cache: bool = True,
    cache_max_age_hours: int = 24,
    model_type: str = "generic",
) -> List[Any]:
    """Fetch runs from WandB with localized caching logic."""
    api = wandb.Api()
    project_path = f"{entity}/{project}" if entity else project

    if use_cache:
        cache_dir = get_cache_dir(model_type)
        cache_key = _get_cache_key(project_path, filters)
        cache_file = cache_dir / f"runs_{cache_key}.pkl"

        if cache_file.exists():
            try:
                cache_data = pd.read_pickle(cache_file)
                if isinstance(cache_data, dict) and "_fetched_at" in cache_data:
                    age_hours = (time.time() - cache_data["_fetched_at"]) / 3600
                    if age_hours <= cache_max_age_hours:
                        print(f"Using cached runs (age: {age_hours:.1f}h)")
                        return cache_data["runs"]
            except Exception as e:
                print(f"Warning: Failed to load cache: {e}")

    print(f"Fetching runs from wandb project: {project_path}")
    try:
        runs = api.runs(project_path, filters=filters)
        runs_list = list(runs)

        if use_cache:
            try:
                cache_data = {"_fetched_at": time.time(), "runs": runs_list}
                pd.to_pickle(cache_data, cache_file)
                print(f"Cached {len(runs_list)} runs")
            except Exception as e:
                print(f"Warning: Failed to cache runs: {e}")

        return runs_list
    except Exception as e:
        print(f"Error fetching runs: {e}")
        print("\nTip: If project is not found, specify wandb entity with --entity")
        sys.exit(1)


def load_history_cached(
    run_id: str, run: Any, model_type: str = "generic"
) -> pd.DataFrame:
    """Load run history (metrics over time) with persistence."""
    cache_dir = get_cache_dir(model_type) / "histories"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{run_id}.pkl"

    if cache_file.exists():
        try:
            return pd.read_pickle(cache_file)
        except Exception:
            pass

    # Fetch from wandb
    history = run.history(samples=10000)

    # Cache it
    try:
        history.to_pickle(cache_file)
    except Exception:
        pass

    return history


# =============================================================================
# FORMATTING & VALIDATION UTILITIES
# =============================================================================


def format_optimizer_name(optimizer: str, num_subdomains: Optional[int] = None) -> str:
    """Format optimizer name for display using LaTeX notation."""
    formatter = {
        "sgd": r"SGD",
        "apts_d": r"$\mathrm{SAPTS}_D$",
        "apts_p": r"$\mathrm{SAPTS}_P$",
        "apts_ip": r"$\mathrm{SAPTS}_{IP}$",
        "apts": r"$\mathrm{SAPTS}$",
    }
    base_name = formatter.get(optimizer.lower(), optimizer)

    if num_subdomains is not None and optimizer.lower() in [
        "apts_d",
        "apts_p",
        "apts_ip",
    ]:
        base_name += f" ($N={num_subdomains}$)"

    return base_name


def validate_apts_parameters(df: pd.DataFrame) -> None:
    """Validate that APTS variants use standard overlap and batch increase factors."""
    print("\n" + "=" * 80)
    print("VALIDATING APTS PARAMETERS")
    print("=" * 80)

    apts_variants = ["apts_d", "apts_p", "apts_ip"]
    issues_found = False

    for optimizer in apts_variants:
        opt_df = df[df["optimizer"] == optimizer]
        if len(opt_df) == 0:
            continue

        for param, expected in [("overlap", 0.33), ("batch_inc_factor", 1.5)]:
            values = opt_df[param].dropna().unique()
            if len(values) > 0:
                if not all(abs(v - expected) < 0.01 for v in values):
                    print(
                        f"\n⚠ WARNING: {optimizer.upper()} has non-standard {param}: {values}"
                    )
                    print(f"  Expected: {expected}")
                    issues_found = True
                else:
                    print(f"✓ {optimizer.upper()}: {param} = {values[0]:.2f} (correct)")

    if not issues_found:
        print("\n✓ All APTS variants have correct parameters")


def export_results(df: pd.DataFrame, output_path: Path) -> None:
    """Export the results DataFrame to CSV while stripping bulky history data."""
    export_df = df.drop(columns=["history"], errors="ignore")
    export_df.to_csv(output_path, index=False)
    print(f"\nExported results to: {output_path}")


def finalize_plot(ax: plt.Axes, output_dir: Optional[Path], filename: str):
    """
    Standardize the finalization, saving, and closing of matplotlib figures.
    Removes the interactive 'show' step to allow for headless/automated execution.
    """
    plt.tight_layout()
    if output_dir:
        filepath = output_dir / filename
        plt.savefig(filepath, bbox_inches="tight")
        print(f"  Saved: {filepath}")

    # Crucial: Close the plot immediately to free memory and prevent
    # it from being shown if plt.show() is called later globally.
    plt.close()


def calculate_params(model_type: str, config: Dict) -> Optional[int]:
    """Centralized parameter calculation based on model architecture."""
    try:
        if model_type == "medium_ffnn":
            w, nl = config.get("width"), config.get("num_layers")
            if w and nl:
                return (784 * w) + ((nl - 1) * w * w) + (w * 10)

        elif model_type == "medium_cnn":
            f, nl, fw = (
                config.get("filters_per_layer"),
                config.get("num_conv_layers"),
                config.get("fc_width"),
            )
            pe = config.get("pool_every", 0)
            if f and nl and fw:
                conv = (1 * f * 9) + ((nl - 1) * f * f * 9)
                num_pools = nl // pe if pe > 0 else 0
                feat = 28 // (2**num_pools)
                return conv + (f * feat * feat * fw) + (fw * 10)

        elif model_type == "nanogpt":
            e, h, l = config.get("n_embd"), config.get("n_head"), config.get("n_layer")
            v, b = 65, 256  # Defaults for tinyshakespeare
            if e and h and l:
                emb = v * e + b * e
                blk = l * (10 * e * e + 4 * e)  # Simplified GPT block estimate
                head = 2 * e + e * v
                return emb + blk + head
    except:
        return None
    return None


def get_arch_keys(model_type: str) -> List[str]:
    """Returns config keys used to define the architecture for heatmaps."""
    mapping = {
        "medium_ffnn": ["width", "num_layers"],
        "medium_cnn": ["filters_per_layer", "num_conv_layers"],
        "nanogpt": ["n_embd", "n_layer"],
    }
    return mapping.get(model_type, [])


def get_arch_labels(model_type: str) -> List[str]:
    """Returns human-readable labels for the architecture axes."""
    mapping = {
        "medium_ffnn": ["Width", "Number of Layers"],
        "medium_cnn": ["Filters per Layer", "Number of Conv Layers"],
        "nanogpt": ["Embedding Dimension", "Number of Layers"],
    }
    return mapping.get(model_type, ["Param A", "Param B"])


def get_common_n_head(df: pd.DataFrame) -> Optional[int]:
    """Helper for GPT models to find the mode of n_head to keep heatmaps 2D."""
    if "n_head" in df.columns and not df["n_head"].dropna().empty:
        return df["n_head"].mode()[0]
    return None
