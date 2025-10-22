#!/usr/bin/env python3
"""
Script to consolidate heatmaps into single figures for CNN, GPT, and PINNs analysis scripts.

This applies the same pattern used in analyze_hyperparam_results_ffnn.py to combine
all heatmaps into a single figure with a shared colorbar.
"""

import re
from pathlib import Path


COMBINED_HEATMAP_TEMPLATE = '''    # Plot 3: Heatmap of {metric} by {x_axis} and {y_axis}
    # First, calculate global min/max across ALL data for consistent scaling
    print("\\n" + "=" * 80)
    print("CALCULATING GLOBAL HEATMAP SCALES")
    print("=" * 80)
    print(f"Using SGD with overlap={{sgd_overlap:.2f}}")

    all_loss_pivots_global = []
    apts_variants = ["apts_d", "apts_p", "apts_ip"]

    # For SGD, use specified overlap value
    sgd_df = df[df["optimizer"] == "sgd"]
    sgd_selected = sgd_df[
        (sgd_df["overlap"].notna()) & (abs(sgd_df["overlap"] - sgd_overlap) < 0.01)
    ]
    if len(sgd_selected) > 0:
        pivot = sgd_selected.pivot_table(
            values="final_loss", index="{y_index}", columns="{x_index}", aggfunc="mean"
        )
        if not pivot.empty:
            all_loss_pivots_global.append(pivot)

    # For APTS variants with overlap=0.33, batch_inc=1.5
    for optimizer in apts_variants:
        opt_df = df[
            (df["optimizer"] == optimizer)
            & (df["overlap"].notna())
            & (df["batch_inc_factor"].notna())
            & (abs(df["overlap"] - 0.33) < 0.01)
            & (abs(df["batch_inc_factor"] - 1.5) < 0.01)
        ]
        for num_subs in sorted(opt_df["num_subdomains"].dropna().unique()):
            sub_df = opt_df[opt_df["num_subdomains"] == num_subs]
            pivot = sub_df.pivot_table(
                values="final_loss",
                index="{y_index}",
                columns="{x_index}",
                aggfunc="mean",
            )
            if not pivot.empty:
                all_loss_pivots_global.append(pivot)

    if all_loss_pivots_global:
        global_loss_min = min(pivot.min().min() for pivot in all_loss_pivots_global)
        global_loss_max = max(pivot.max().max() for pivot in all_loss_pivots_global)
        print(
            f"\\nGlobal loss heatmap scale: {{global_loss_min:.4f}} to {{global_loss_max:.4f}}"
        )
    else:
        global_loss_min = None
        global_loss_max = None

    all_acc_pivots_global = []

    # For SGD, use specified overlap value
    if len(sgd_selected) > 0:
        pivot = sgd_selected.pivot_table(
            values="final_accuracy", index="{y_index}", columns="{x_index}", aggfunc="mean"
        )
        if not pivot.empty:
            all_acc_pivots_global.append(pivot)

    # For APTS variants with overlap=0.33, batch_inc=1.5
    for optimizer in apts_variants:
        opt_df = df[
            (df["optimizer"] == optimizer)
            & (df["overlap"].notna())
            & (df["batch_inc_factor"].notna())
            & (abs(df["overlap"] - 0.33) < 0.01)
            & (abs(df["batch_inc_factor"] - 1.5) < 0.01)
        ]
        for num_subs in sorted(opt_df["num_subdomains"].dropna().unique()):
            sub_df = opt_df[opt_df["num_subdomains"] == num_subs]
            pivot = sub_df.pivot_table(
                values="final_accuracy",
                index="{y_index}",
                columns="{x_index}",
                aggfunc="mean",
            )
            if not pivot.empty:
                all_acc_pivots_global.append(pivot)

    if all_acc_pivots_global:
        global_acc_min = min(pivot.min().min() for pivot in all_acc_pivots_global)
        global_acc_max = max(pivot.max().max() for pivot in all_acc_pivots_global)
        print(
            f"Global accuracy heatmap scale: {{global_acc_min:.4f}} to {{global_acc_max:.4f}}"
        )
    else:
        global_acc_min = None
        global_acc_max = None

    # Generate combined heatmaps (SGD with selected overlap + APTS with overlap=0.33, batch_inc=1.5)
    print("\\n--- Creating combined heatmaps ---")

    # Collect all heatmap data for LOSS
    if global_loss_min is not None and global_loss_max is not None:
        heatmap_data = []

        # Add SGD with selected overlap
        if len(sgd_selected) > 0:
            pivot = sgd_selected.pivot_table(
                values="final_loss",
                index="{y_index}",
                columns="{x_index}",
                aggfunc="mean",
            )
            if not pivot.empty:
                heatmap_data.append({{
                    "pivot": pivot,
                    "title": format_optimizer_name("sgd"),
                    "optimizer": "sgd",
                    "num_subs": None
                }})

        # Add APTS variants with overlap=0.33, batch_inc=1.5
        for optimizer in apts_variants:
            opt_df = df[
                (df["optimizer"] == optimizer)
                & (df["overlap"].notna())
                & (df["batch_inc_factor"].notna())
                & (abs(df["overlap"] - 0.33) < 0.01)
                & (abs(df["batch_inc_factor"] - 1.5) < 0.01)
            ]
            for num_subs in sorted(opt_df["num_subdomains"].dropna().unique()):
                sub_df = opt_df[opt_df["num_subdomains"] == num_subs]
                pivot = sub_df.pivot_table(
                    values="final_loss",
                    index="{y_index}",
                    columns="{x_index}",
                    aggfunc="mean",
                )
                if not pivot.empty:
                    heatmap_data.append({{
                        "pivot": pivot,
                        "title": format_optimizer_name(optimizer, int(num_subs)),
                        "optimizer": optimizer,
                        "num_subs": int(num_subs)
                    }})

        # Create combined figure with all loss heatmaps
        if heatmap_data:
            n_heatmaps = len(heatmap_data)
            n_cols = min(3, n_heatmaps)  # Max 3 columns
            n_rows = (n_heatmaps + n_cols - 1) // n_cols  # Ceiling division

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

            # Flatten axes array for easier indexing
            if n_heatmaps == 1:
                axes = np.array([axes])
            elif n_rows == 1:
                axes = axes.reshape(1, -1).flatten()
            else:
                axes = axes.flatten()

            # Plot each heatmap
            for idx, data in enumerate(heatmap_data):
                ax = axes[idx]
                row = idx // n_cols
                col = idx % n_cols

                sns.heatmap(
                    data["pivot"],
                    annot=True,
                    fmt=".4f",
                    cmap="RdYlGn_r",
                    ax=ax,
                    vmin=global_loss_min,
                    vmax=global_loss_max,
                    cbar=False,
                    xticklabels=True,  # Show all x tick labels
                    yticklabels=(col == 0),  # Show y tick labels only on leftmost column
                )
                ax.set_title(data["title"], fontsize=14, fontweight="bold")

                # Only show x-label on center plot of bottom row
                if row == n_rows - 1 and col == n_cols // 2:
                    ax.set_xlabel(r"{x_label}", fontsize=12)
                else:
                    ax.set_xlabel("")

                # Only show y-label on leftmost column
                if col == 0:
                    ax.set_ylabel(r"{y_label}", fontsize=12)
                else:
                    ax.set_ylabel("")

            # Hide unused subplots
            for idx in range(n_heatmaps, len(axes)):
                axes[idx].axis('off')

            # Add a single colorbar for all heatmaps
            from matplotlib import cm
            from matplotlib.colors import Normalize
            norm = Normalize(vmin=global_loss_min, vmax=global_loss_max)
            sm = cm.ScalarMappable(cmap="RdYlGn_r", norm=norm)
            sm.set_array([])

            # Adjust layout to make room for colorbar
            plt.tight_layout(rect=[0, 0, 0.95, 1])

            # Add colorbar to the right of all subplots
            cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label(r"Final avg. loss", fontsize=14)

            if output_dir:
                filepath = output_dir / "heatmap_loss_combined_{suffix}.pdf"
                plt.savefig(filepath, bbox_inches="tight")
                print(f"  Saved: {{filepath}}")
            else:
                plt.show()

            plt.close()

    # Collect all heatmap data for ACCURACY
    if global_acc_min is not None and global_acc_max is not None:
        heatmap_data = []

        # Add SGD with selected overlap
        if len(sgd_selected) > 0:
            pivot = sgd_selected.pivot_table(
                values="final_accuracy",
                index="{y_index}",
                columns="{x_index}",
                aggfunc="mean",
            )
            if not pivot.empty:
                heatmap_data.append({{
                    "pivot": pivot,
                    "title": format_optimizer_name("sgd"),
                    "optimizer": "sgd",
                    "num_subs": None
                }})

        # Add APTS variants with overlap=0.33, batch_inc=1.5
        for optimizer in apts_variants:
            opt_df = df[
                (df["optimizer"] == optimizer)
                & (df["overlap"].notna())
                & (df["batch_inc_factor"].notna())
                & (abs(df["overlap"] - 0.33) < 0.01)
                & (abs(df["batch_inc_factor"] - 1.5) < 0.01)
            ]
            for num_subs in sorted(opt_df["num_subdomains"].dropna().unique()):
                sub_df = opt_df[opt_df["num_subdomains"] == num_subs]
                pivot = sub_df.pivot_table(
                    values="final_accuracy",
                    index="{y_index}",
                    columns="{x_index}",
                    aggfunc="mean",
                )
                if not pivot.empty:
                    heatmap_data.append({{
                        "pivot": pivot,
                        "title": format_optimizer_name(optimizer, int(num_subs)),
                        "optimizer": optimizer,
                        "num_subs": int(num_subs)
                    }})

        # Create combined figure with all accuracy heatmaps
        if heatmap_data:
            n_heatmaps = len(heatmap_data)
            n_cols = min(3, n_heatmaps)  # Max 3 columns
            n_rows = (n_heatmaps + n_cols - 1) // n_cols  # Ceiling division

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

            # Flatten axes array for easier indexing
            if n_heatmaps == 1:
                axes = np.array([axes])
            elif n_rows == 1:
                axes = axes.reshape(1, -1).flatten()
            else:
                axes = axes.flatten()

            # Plot each heatmap
            for idx, data in enumerate(heatmap_data):
                ax = axes[idx]
                row = idx // n_cols
                col = idx % n_cols

                sns.heatmap(
                    data["pivot"],
                    annot=True,
                    fmt=".2f",
                    cmap="RdYlGn",
                    ax=ax,
                    vmin=global_acc_min,
                    vmax=global_acc_max,
                    cbar=False,
                    xticklabels=True,  # Show all x tick labels
                    yticklabels=(col == 0),  # Show y tick labels only on leftmost column
                )
                ax.set_title(data["title"], fontsize=14, fontweight="bold")

                # Only show x-label on center plot of bottom row
                if row == n_rows - 1 and col == n_cols // 2:
                    ax.set_xlabel(r"{x_label}", fontsize=12)
                else:
                    ax.set_xlabel("")

                # Only show y-label on leftmost column
                if col == 0:
                    ax.set_ylabel(r"{y_label}", fontsize=12)
                else:
                    ax.set_ylabel("")

            # Hide unused subplots
            for idx in range(n_heatmaps, len(axes)):
                axes[idx].axis('off')

            # Add a single colorbar for all heatmaps
            from matplotlib import cm
            from matplotlib.colors import Normalize
            norm = Normalize(vmin=global_acc_min, vmax=global_acc_max)
            sm = cm.ScalarMappable(cmap="RdYlGn", norm=norm)
            sm.set_array([])

            # Adjust layout to make room for colorbar
            plt.tight_layout(rect=[0, 0, 0.95, 1])

            # Add colorbar to the right of all subplots
            cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label(r"Final avg. accuracy (\\%)", fontsize=14)

            if output_dir:
                filepath = output_dir / "heatmap_accuracy_combined_{suffix}.pdf"
                plt.savefig(filepath, bbox_inches="tight")
                print(f"  Saved: {{filepath}}")
            else:
                plt.show()

            plt.close()
'''


def replace_heatmap_section(filepath: Path, config: dict):
    """Replace the heatmap generation section with combined version."""
    print(f"\\nProcessing {filepath.name}...")

    with open(filepath, 'r') as f:
        content = f.read()

    # Find the start of heatmap section
    heatmap_start_pattern = r'    # Plot 3: Heatmap of loss by .+ \(parameter-aware\)\n\n    # First, calculate global min/max'
    match = re.search(heatmap_start_pattern, content)

    if not match:
        print(f"  ✗ Could not find heatmap section start")
        return False

    start_pos = match.start()

    # Find the end (next function definition)
    next_func_pattern = r'\n\ndef create_sgd_parameter_comparison'
    end_match = re.search(next_func_pattern, content[start_pos:])

    if not end_match:
        print(f"  ✗ Could not find heatmap section end")
        return False

    end_pos = start_pos + end_match.start()

    # Generate the replacement code
    replacement = COMBINED_HEATMAP_TEMPLATE.format(**config)

    # Replace the section
    new_content = content[:start_pos] + replacement + content[end_pos:]

    # Write back
    with open(filepath, 'w') as f:
        f.write(new_content)

    print(f"  ✓ Replaced heatmap section")
    return True


def main():
    """Apply heatmap consolidation to all analysis files."""
    print("="*80)
    print("CONSOLIDATING HEATMAPS INTO SINGLE FIGURES")
    print("="*80)

    base_dir = Path(__file__).parent

    # Configuration for each file
    configs = [
        {
            "file": "analyze_hyperparam_results_cnn.py",
            "metric": "loss",
            "x_axis": "filters",
            "y_axis": "conv layers",
            "x_index": "filters",
            "y_index": "num_conv_layers",
            "x_label": "Filters per layer",
            "y_label": "Number of conv layers",
            "suffix": "cnn"
        },
        {
            "file": "analyze_hyperparam_results_gpt.py",
            "metric": "loss",
            "x_axis": "hidden size",
            "y_axis": "layers",
            "x_index": "hidden_size",
            "y_index": "num_layers",
            "x_label": "Hidden size",
            "y_label": "Number of layers",
            "suffix": "gpt"
        },
        {
            "file": "analyze_hyperparam_results_pinns.py",
            "metric": "loss",
            "x_axis": "width",
            "y_axis": "depth",
            "x_index": "width",
            "y_index": "num_layers",
            "x_label": "Width",
            "y_label": "Depth",
            "suffix": "pinns"
        },
    ]

    for config in configs:
        filepath = base_dir / config["file"]
        if filepath.exists():
            try:
                replace_heatmap_section(filepath, config)
            except Exception as e:
                print(f"  ✗ Error: {e}")
        else:
            print(f"  ✗ File not found: {config['file']}")

    print("\\n" + "="*80)
    print("CONSOLIDATION COMPLETE")
    print("="*80)
    print("\\nAll heatmaps have been consolidated into single figures with:")
    print("  - Shared colorbar on the right")
    print("  - X-axis label only on center-bottom plot")
    print("  - Y-axis labels only on leftmost column")
    print("  - Y-axis tick numbers only on leftmost column")


if __name__ == "__main__":
    main()
