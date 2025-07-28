#!/usr/bin/env python3
"""
fix_gradient_evals.py

Multiply each run’s logged total_gradient_evals by its config.num_subdomains
and patch the corrected value back into W&B.
"""

import wandb


def main(entity: str, project: str):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    for run in runs:
        # retrieve config value
        nsd = run.config.get("num_subdomains", None)
        # retrieve original summary field
        orig = run.summary.get("grad_evals", None)
        if nsd is None or orig is None:
            continue  # nothing to correct

        corrected = orig * nsd
        run.summary.update({"grad_evals": corrected})

        # run.summary["grad_evals"] = corrected
        # run.summary.update()
        print(f"Patched run {run.id}: " f"{orig} × {nsd} → {corrected}")


if __name__ == "__main__":
    # customise your entity/project here
    ENTITY = "cruzas-universit-della-svizzera-italiana"
    PROJECT = "thesis_results"
    main(ENTITY, PROJECT)
