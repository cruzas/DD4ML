#!/usr/bin/env python3

from dd4ml.utils import compute_best_lr_per_batch_size, parse_cmd_args


def main():
    args = parse_cmd_args()
    verbose = True
    
    # Build project reference using command-line arguments.
    project = f"{args.entity}/{args.project}"
    compute_best_lr_per_batch_size(project, metric="loss", verbose=verbose)
    compute_best_lr_per_batch_size(project, metric="accuracy", verbose=verbose)
    

if __name__ == "__main__":
    main()
