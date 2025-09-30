#!/usr/bin/env python3
"""
Python equivalent of submit_jobs.sh for automating experiment configurations.

Usage:
    python submit_jobs.py --config mnist
    python submit_jobs.py --config allencahn1d --dry-run
"""

import argparse
import itertools
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


def parse_conf_file(conf_path: Path) -> Dict[str, Any]:
    """Parse a bash .conf file and extract variables."""
    config = {}

    with open(conf_path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            # Parse variable assignments
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                # Parse arrays (e.g., OPTIMIZERS=(sgd adam))
                if value.startswith("(") and value.endswith(")"):
                    value = value[1:-1].strip()
                    if value:
                        # Convert numeric strings to integers
                        items = value.split()
                        config[key] = [
                            int(item) if item.isdigit() else item for item in items
                        ]
                    else:
                        config[key] = []
                # Parse booleans
                elif value.lower() in ("true", "false"):
                    config[key] = value.lower() == "true"
                # Parse numbers
                elif value.isdigit():
                    config[key] = int(value)
                else:
                    config[key] = value

    return config


def set_optimizer_params(optimizer: str, config: Dict[str, Any]) -> None:
    """Set optimizer-specific parameters."""
    if optimizer == "apts_ip":
        config["USE_PMW"] = True
        config["NUM_SUBD"] = [1]
        config["NUM_REP"] = [1]


def set_model_params(model: str, config: Dict[str, Any]) -> None:
    """Set model-specific parameters."""
    if model == "nanogpt":
        config["EVAL_PARAMS"] = [
            "epochs=0",
            "max_iters=2000",
            "criterion=cross_entropy_transformers",
        ]
        config["BATCH_SIZES"] = [128]


def set_hardware_params(config: Dict[str, Any]) -> int:
    """Determine max GPUs based on environment."""
    if "/home/" in os.getcwd():
        return 1
    return 4


def set_apts_lssr1_tr_params(optimizer: str, config: Dict[str, Any]) -> None:
    """Set APTS/LSSR1_TR specific parameters."""
    if optimizer in ["apts_d", "apts_p", "apts_ip", "lssr1_tr", "tr"]:
        apts_params = [
            "batch_inc_factor=1.5",
            "overlap=0.33",
            "max_wolfe_iters=5",
            "max_zoom_iters=5",
            "mem_length=5",
        ]

        if optimizer != "lssr1_tr":
            apts_params.extend(
                [
                    "glob_opt=lssr1_tr",
                    "max_glob_iters=1",
                    "glob_second_order=true",
                    "loc_opt=lssr1_tr",
                    "max_loc_iters=2",
                    "loc_second_order=true",
                ]
            )

            if optimizer == "apts_d":
                apts_params.extend(["glob_pass=true", "foc=true"])
            elif optimizer == "apts_p":
                apts_params.append("glob_pass=true")
            elif optimizer == "apts_ip":
                apts_params.extend(
                    [
                        "batch_inc_factor=1.0",
                        "overlap=0.0",
                        "loc_opt=sgd",
                        "loc_second_order=false",
                        "glob_pass=true",
                    ]
                )
            elif optimizer == "tr":
                apts_params.append("glob_second_order=false")
        else:
            apts_params.append("glob_second_order=false")

        config["APTS_PARAMS"] = apts_params


def extract_apts_details(apts_params: List[str]) -> Dict[str, str]:
    """Extract APTS optimizer details from parameters."""
    details = {
        "glob_opt": "none",
        "loc_opt": "none",
        "glob_second_order": "false",
        "loc_second_order": "false",
    }

    for kv in apts_params:
        if "=" in kv:
            key, val = kv.split("=", 1)
            if key in details:
                details[key] = val

    return details


def calc_nodes(world_size: int, max_gpus: int) -> Tuple[int, int]:
    """Calculate optimal number of nodes and tasks per node."""
    for n in range(1, world_size + 1):
        tpn = world_size // n
        if world_size % n == 0 and tpn <= max_gpus:
            return n, tpn
    return world_size, 1


def update_config_yaml(yaml_path: Path, key: str, value: Any) -> None:
    """Update a value in a YAML config file."""
    with open(yaml_path, "r") as f:
        lines = f.readlines()

    # Find the key and update the value on the next line
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith(f"{key}:"):
            if i + 1 < len(lines) and "value:" in lines[i + 1]:
                # Update the value line
                indent = len(lines[i + 1]) - len(lines[i + 1].lstrip())
                lines[i + 1] = " " * indent + f"value: {value}\n"
                break
        i += 1

    with open(yaml_path, "w") as f:
        f.writelines(lines)


def run_experiment(config_file: Path, job_name: str, dry_run: bool = False) -> bool:
    """Run a single experiment directly."""
    if dry_run:
        print(f"  [DRY RUN] Would run: {job_name} with {config_file}")
        return True

    # Import and run directly
    try:
        # Save current sys.argv and working directory
        old_argv = sys.argv
        old_cwd = os.getcwd()

        # Set up arguments for run_config_file
        sys.argv = ["run_config_file.py", "--sweep_config", str(config_file)]

        # Change to the tests directory
        os.chdir(Path(__file__).parent)

        print(f"  → Running: {job_name}")

        # Import and call main_single_run
        from run_config_file import main_single_run

        main_single_run()

        print(f"  ✓ Completed: {job_name}")
        return True

    except Exception as e:
        print(f"  ✗ ERROR: Failed {job_name}: {e}", file=sys.stderr)
        return False
    finally:
        # Restore original state
        sys.argv = old_argv
        os.chdir(old_cwd)


def generate_and_submit_jobs(
    config: Dict[str, Any], script_dir: Path, dry_run: bool = False
) -> int:
    """Generate configurations and submit jobs."""
    max_gpus = set_hardware_params(config)
    submitted = 0
    skipped = 0

    # Set default values if not present
    paper_tr_updates = config.get("PAPER_TR_UPDATES", [True])
    glob_second_orders = config.get("GLOB_SECOND_ORDERS", [False])
    loc_second_orders = config.get("LOC_SECOND_ORDERS", [False])
    glob_doglegs = config.get("GLOB_DOGLEGS", [False])
    loc_doglegs = config.get("LOC_DOGLEGS", [False])
    apts_glob_opts = config.get("APTS_GLOB_OPTS", [])
    apts_loc_opts = config.get("APTS_LOC_OPTS", [])
    foc_opts = config.get("FOC_OPTS", [])

    optimizers = config.get("OPTIMIZERS", [])
    datasets = config.get("DATASETS", [])
    models = config.get("MODELS", [])

    # Main nested loop
    for optimizer in optimizers:
        for dataset in datasets:
            for model in models:
                for gso in glob_second_orders:
                    for lso in loc_second_orders:
                        for gdg in glob_doglegs:
                            # Skip invalid dogleg combinations
                            if gdg and not gso:
                                print("→ Skipping: global dogleg requires gso=true")
                                continue

                            # Determine local dogleg options
                            if optimizer.startswith("apts_"):
                                loc_dg_opts = loc_doglegs
                            else:
                                loc_dg_opts = [False]

                            for ldg in loc_dg_opts:
                                if ldg and not lso:
                                    print("→ Skipping: local dogleg requires lso=true")
                                    continue

                                for paper_tr_update in paper_tr_updates:
                                    # Set parameters
                                    set_optimizer_params(optimizer, config)
                                    set_model_params(model, config)
                                    set_apts_lssr1_tr_params(optimizer, config)

                                    apts_params = config.get("APTS_PARAMS", [])

                                    # Extract APTS details
                                    if (
                                        optimizer.startswith("apts_")
                                        or optimizer == "lssr1_tr"
                                    ):
                                        apts_details = extract_apts_details(apts_params)

                                        # Handle ASNTR batch increment factor
                                        if (
                                            optimizer == "asntr"
                                            or apts_details["glob_opt"] == "asntr"
                                            or apts_details["loc_opt"] == "asntr"
                                        ):
                                            apts_params = [
                                                (
                                                    p
                                                    if not p.startswith(
                                                        "batch_inc_factor="
                                                    )
                                                    else "batch_inc_factor=1.01"
                                                )
                                                for p in apts_params
                                            ]

                                        # Filter out certain parameters
                                        tmp_params = [
                                            p
                                            for p in apts_params
                                            if not any(
                                                p.startswith(k)
                                                for k in [
                                                    "glob_opt=",
                                                    "loc_opt=",
                                                    "glob_second_order=",
                                                    "loc_second_order=",
                                                    "foc=",
                                                    "glob_pass=",
                                                ]
                                            )
                                        ]
                                    else:
                                        apts_details = {
                                            "glob_opt": "none",
                                            "loc_opt": "none",
                                            "glob_second_order": str(gso).lower(),
                                            "loc_second_order": str(lso).lower(),
                                        }
                                        tmp_params = []

                                    # Handle APTS options
                                    glob_opt_list = (
                                        apts_glob_opts if apts_glob_opts else ["none"]
                                    )
                                    loc_opt_list = (
                                        apts_loc_opts if apts_loc_opts else ["none"]
                                    )

                                    for glob_opt in glob_opt_list:
                                        for loc_opt in loc_opt_list:
                                            # Determine FOC values
                                            if optimizer == "apts_d":
                                                foc_values = (
                                                    foc_opts if foc_opts else [False]
                                                )
                                            else:
                                                foc_values = [False]

                                            for foc in foc_values:
                                                # Rebuild APTS params
                                                current_apts_params = tmp_params.copy()

                                                if optimizer == "apts_d":
                                                    current_apts_params.extend(
                                                        [
                                                            "glob_pass=true",
                                                            f"foc={str(foc).lower()}",
                                                        ]
                                                    )

                                                if (
                                                    optimizer.startswith("apts_")
                                                    or optimizer == "lssr1_tr"
                                                ):
                                                    current_apts_params.extend(
                                                        [
                                                            f"glob_opt={glob_opt}",
                                                            f"loc_opt={loc_opt}",
                                                            f"glob_second_order={str(gso).lower()}",
                                                            f"loc_second_order={str(lso).lower()}",
                                                        ]
                                                    )
                                                    apts_details = extract_apts_details(
                                                        current_apts_params
                                                    )

                                                # Innermost loops for stages, subdomains, replicas, trials, batch sizes
                                                for num_stages in config.get(
                                                    "NUM_STAGES", [1]
                                                ):
                                                    for num_subd in config.get(
                                                        "NUM_SUBD", [1]
                                                    ):
                                                        for num_rep in config.get(
                                                            "NUM_REP", [1]
                                                        ):
                                                            for trial in range(
                                                                1,
                                                                config.get("TRIALS", 1)
                                                                + 1,
                                                            ):
                                                                for (
                                                                    batch_size
                                                                ) in config.get(
                                                                    "BATCH_SIZES", []
                                                                ):
                                                                    # Calculate batch sizes
                                                                    scaling_type = config.get(
                                                                        "SCALING_TYPE",
                                                                        "strong",
                                                                    )
                                                                    if (
                                                                        scaling_type
                                                                        == "weak"
                                                                    ):
                                                                        actual_bs = (
                                                                            batch_size
                                                                            * num_subd
                                                                        )
                                                                        eff_bs = (
                                                                            batch_size
                                                                        )
                                                                    else:
                                                                        actual_bs = (
                                                                            batch_size
                                                                        )
                                                                        eff_bs = (
                                                                            batch_size
                                                                            // num_subd
                                                                        )

                                                                    # Handle learning rates
                                                                    lr_values = config.get(
                                                                        "LEARNING_RATES",
                                                                        [],
                                                                    )
                                                                    if not lr_values:
                                                                        lr_values = [
                                                                            None
                                                                        ]

                                                                    for lr in lr_values:
                                                                        # Build job name
                                                                        eval_params = config.get(
                                                                            "EVAL_PARAMS",
                                                                            [
                                                                                "epochs=10",
                                                                                "max_iters=0",
                                                                                "criterion=cross_entropy",
                                                                            ],
                                                                        )
                                                                        epoch_count = next(
                                                                            (
                                                                                p.split(
                                                                                    "="
                                                                                )[1]
                                                                                for p in eval_params
                                                                                if p.startswith(
                                                                                    "epochs="
                                                                                )
                                                                            ),
                                                                            "10",
                                                                        )

                                                                        use_pmw = config.get(
                                                                            "USE_PMW",
                                                                            False,
                                                                        )

                                                                        # Get width for ffnn models
                                                                        width = config.get("WIDTH")

                                                                        job_name = f"{optimizer}_{dataset}_{model}_{actual_bs}_epochs_{epoch_count}_nsd_{num_subd}"

                                                                        # Add width to job name if ffnn model and width is specified
                                                                        if "ffnn" in model and width is not None:
                                                                            job_name += f"_w_{width}"

                                                                        if use_pmw:
                                                                            job_name += f"_nst_{num_stages}_nrpsd_{num_rep}"

                                                                        # Extract batch_inc_factor and overlap if present
                                                                        # Check both current_apts_params and original config APTS_PARAMS
                                                                        all_params = current_apts_params + config.get("APTS_PARAMS", [])
                                                                        batch_inc_factor = next(
                                                                            (
                                                                                p.split("=")[1]
                                                                                for p in all_params
                                                                                if p.startswith("batch_inc_factor=")
                                                                            ),
                                                                            None,
                                                                        )
                                                                        overlap = next(
                                                                            (
                                                                                p.split("=")[1]
                                                                                for p in all_params
                                                                                if p.startswith("overlap=")
                                                                            ),
                                                                            None,
                                                                        )

                                                                        if optimizer.startswith(
                                                                            "apts_"
                                                                        ):
                                                                            job_name += f"_gopt_{apts_details['glob_opt']}_lopt_{apts_details['loc_opt']}_gso_{apts_details['glob_second_order']}_lso_{apts_details['loc_second_order']}"
                                                                            if batch_inc_factor is not None:
                                                                                job_name += f"_bif_{batch_inc_factor}"
                                                                            if overlap is not None:
                                                                                job_name += f"_ovlp_{overlap}"

                                                                            # Add max_loc_iters to job name if present
                                                                            max_loc_iters = config.get("MAX_LOC_ITERS")
                                                                            if max_loc_iters is not None:
                                                                                job_name += f"_mli_{max_loc_iters}"

                                                                            if (
                                                                                optimizer
                                                                                == "apts_d"
                                                                            ):
                                                                                job_name += f"_foc_{str(foc).lower()}"
                                                                            if (
                                                                                apts_details[
                                                                                    "glob_opt"
                                                                                ]
                                                                                == "lssr1_tr"
                                                                                or apts_details[
                                                                                    "loc_opt"
                                                                                ]
                                                                                == "lssr1_tr"
                                                                            ):
                                                                                job_name += f"_ptru_{str(paper_tr_update).lower()}"
                                                                        elif (
                                                                            optimizer
                                                                            == "lssr1_tr"
                                                                        ):
                                                                            job_name += f"_gso_{apts_details['glob_second_order']}_ptru_{str(paper_tr_update).lower()}"
                                                                            if batch_inc_factor is not None:
                                                                                job_name += f"_bif_{batch_inc_factor}"
                                                                            if overlap is not None:
                                                                                job_name += f"_ovlp_{overlap}"
                                                                        else:
                                                                            job_name += f"_gso_{apts_details['glob_second_order']}"
                                                                            if batch_inc_factor is not None:
                                                                                job_name += f"_bif_{batch_inc_factor}"
                                                                            if overlap is not None:
                                                                                job_name += f"_ovlp_{overlap}"

                                                                        job_name += f"_gdg_{str(gdg).lower()}"
                                                                        if optimizer.startswith(
                                                                            "apts_"
                                                                        ):
                                                                            job_name += f"_ldg_{str(ldg).lower()}"
                                                                        job_name += f"_pmw_{str(use_pmw).lower()}"
                                                                        if (
                                                                            lr
                                                                            is not None
                                                                        ):
                                                                            job_name += f"_lr_{lr}"
                                                                        job_name += f"_trial_{trial}"

                                                                        # Calculate world size and nodes
                                                                        world_size = (
                                                                            num_stages
                                                                            * num_subd
                                                                            * num_rep
                                                                        )
                                                                        (
                                                                            nodes,
                                                                            ntasks_per_node,
                                                                        ) = calc_nodes(
                                                                            world_size,
                                                                            max_gpus,
                                                                        )

                                                                        # Create config file
                                                                        config_file = (
                                                                            script_dir
                                                                            / "config_files"
                                                                            / f"config_{job_name}.yaml"
                                                                        )

                                                                        if (
                                                                            config_file.exists()
                                                                        ):
                                                                            print(
                                                                                f"-> Skipping existing: {config_file.name}"
                                                                            )
                                                                            skipped += 1
                                                                            continue

                                                                        # Copy base config
                                                                        base_config = (
                                                                            script_dir
                                                                            / "config_files"
                                                                            / f"config_{optimizer}.yaml"
                                                                        )
                                                                        if (
                                                                            not base_config.exists()
                                                                        ):
                                                                            print(
                                                                                f"ERROR: Base config not found: {base_config}",
                                                                                file=sys.stderr,
                                                                            )
                                                                            continue

                                                                        shutil.copy(
                                                                            base_config,
                                                                            config_file,
                                                                        )

                                                                        # Update config values
                                                                        update_config_yaml(
                                                                            config_file,
                                                                            "optimizer",
                                                                            optimizer,
                                                                        )
                                                                        update_config_yaml(
                                                                            config_file,
                                                                            "batch_size",
                                                                            actual_bs,
                                                                        )
                                                                        update_config_yaml(
                                                                            config_file,
                                                                            "effective_batch_size",
                                                                            eff_bs,
                                                                        )
                                                                        update_config_yaml(
                                                                            config_file,
                                                                            "dataset_name",
                                                                            dataset,
                                                                        )
                                                                        update_config_yaml(
                                                                            config_file,
                                                                            "model_name",
                                                                            model,
                                                                        )
                                                                        update_config_yaml(
                                                                            config_file,
                                                                            "criterion",
                                                                            next(
                                                                                (
                                                                                    p.split(
                                                                                        "="
                                                                                    )[
                                                                                        1
                                                                                    ]
                                                                                    for p in eval_params
                                                                                    if p.startswith(
                                                                                        "criterion="
                                                                                    )
                                                                                ),
                                                                                "cross_entropy",
                                                                            ),
                                                                        )
                                                                        update_config_yaml(
                                                                            config_file,
                                                                            "epochs",
                                                                            epoch_count,
                                                                        )
                                                                        update_config_yaml(
                                                                            config_file,
                                                                            "max_iters",
                                                                            next(
                                                                                (
                                                                                    p.split(
                                                                                        "="
                                                                                    )[
                                                                                        1
                                                                                    ]
                                                                                    for p in eval_params
                                                                                    if p.startswith(
                                                                                        "max_iters="
                                                                                    )
                                                                                ),
                                                                                "0",
                                                                            ),
                                                                        )
                                                                        update_config_yaml(
                                                                            config_file,
                                                                            "num_subdomains",
                                                                            num_subd,
                                                                        )

                                                                        if (
                                                                            optimizer
                                                                            == "lssr1_tr"
                                                                            or (
                                                                                optimizer.startswith(
                                                                                    "apts_"
                                                                                )
                                                                                and (
                                                                                    apts_details[
                                                                                        "glob_opt"
                                                                                    ]
                                                                                    == "lssr1_tr"
                                                                                    or apts_details[
                                                                                        "loc_opt"
                                                                                    ]
                                                                                    == "lssr1_tr"
                                                                                )
                                                                            )
                                                                        ):
                                                                            update_config_yaml(
                                                                                config_file,
                                                                                "paper_tr_update",
                                                                                str(
                                                                                    paper_tr_update
                                                                                ).lower(),
                                                                            )

                                                                        for (
                                                                            kv
                                                                        ) in current_apts_params:
                                                                            if (
                                                                                "="
                                                                                in kv
                                                                            ):
                                                                                (
                                                                                    key,
                                                                                    val,
                                                                                ) = kv.split(
                                                                                    "=",
                                                                                    1,
                                                                                )
                                                                                update_config_yaml(
                                                                                    config_file,
                                                                                    key,
                                                                                    val,
                                                                                )

                                                                        update_config_yaml(
                                                                            config_file,
                                                                            "glob_dogleg",
                                                                            str(
                                                                                gdg
                                                                            ).lower(),
                                                                        )
                                                                        if optimizer.startswith(
                                                                            "apts_"
                                                                        ):
                                                                            update_config_yaml(
                                                                                config_file,
                                                                                "loc_dogleg",
                                                                                str(
                                                                                    ldg
                                                                                ).lower(),
                                                                            )

                                                                        if use_pmw:
                                                                            update_config_yaml(
                                                                                config_file,
                                                                                "num_stages",
                                                                                num_stages,
                                                                            )
                                                                            update_config_yaml(
                                                                                config_file,
                                                                                "num_replicas_per_subdomain",
                                                                                num_rep,
                                                                            )

                                                                        if config.get(
                                                                            "GRAD_ACC",
                                                                            False,
                                                                        ):
                                                                            update_config_yaml(
                                                                                config_file,
                                                                                "gradient_accumulation",
                                                                                "true",
                                                                            )
                                                                            update_config_yaml(
                                                                                config_file,
                                                                                "accumulation_steps",
                                                                                config.get(
                                                                                    "ACCUM_STEPS",
                                                                                    1,
                                                                                ),
                                                                            )

                                                                        if (
                                                                            lr
                                                                            is not None
                                                                        ):
                                                                            update_config_yaml(
                                                                                config_file,
                                                                                "learning_rate",
                                                                                lr,
                                                                            )

                                                                        # Update max_loc_iters if present
                                                                        max_loc_iters = config.get("MAX_LOC_ITERS")
                                                                        if max_loc_iters is not None:
                                                                            update_config_yaml(
                                                                                config_file,
                                                                                "max_loc_iters",
                                                                                max_loc_iters,
                                                                            )

                                                                        # Update width for ffnn models
                                                                        if "ffnn" in model and width is not None:
                                                                            update_config_yaml(
                                                                                config_file,
                                                                                "width",
                                                                                width,
                                                                            )

                                                                        # Run experiment directly
                                                                        if run_experiment(
                                                                            config_file,
                                                                            job_name,
                                                                            dry_run,
                                                                        ):
                                                                            submitted += (
                                                                                1
                                                                            )

    return submitted, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Python equivalent of submit_jobs.sh for automating experiment configurations."
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Configuration name (without .conf extension)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be submitted without actually submitting jobs",
    )

    args = parser.parse_args()

    # Determine script directory
    script_dir = Path(__file__).parent

    # Load configuration file
    conf_file = script_dir / "job_configs" / f"{args.config}.conf"
    if not conf_file.exists():
        print(f"Config file not found: {conf_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading configuration from: {conf_file}")
    config = parse_conf_file(conf_file)

    # Generate and submit jobs
    print("\nGenerating configurations and submitting jobs...")
    if args.dry_run:
        print("[DRY RUN MODE - No jobs will be submitted]\n")

    submitted, skipped = generate_and_submit_jobs(config, script_dir, args.dry_run)

    print(f"\n{'Would submit' if args.dry_run else 'Submitted'}: {submitted} jobs")
    print(f"Skipped (already exist): {skipped} configs")


if __name__ == "__main__":
    main()
