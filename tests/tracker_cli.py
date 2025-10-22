#!/usr/bin/env python3
"""
Command-line interface for managing experiment tracker files.

Usage:
    python tracker_cli.py status cnn
    python tracker_cli.py list cnn --completed
    python tracker_cli.py list cnn --failed
    python tracker_cli.py clear cnn --failed-only
    python tracker_cli.py remove cnn exp_001
"""

import argparse
import json
from pathlib import Path
from experiment_tracker import ExperimentTracker


def get_tracker_path(test_type: str) -> Path:
    """Get the tracker file path for a given test type."""
    script_dir = Path(__file__).parent
    if test_type == "example":
        return script_dir / ".example_tracker.json"
    return script_dir / f".experiment_tracker_{test_type}.json"


def cmd_status(args):
    """Show tracker status summary."""
    tracker_path = get_tracker_path(args.test_type)
    if not tracker_path.exists():
        print(f"No tracker file found at: {tracker_path}")
        print("(This is normal if no experiments have been run yet)")
        return

    tracker = ExperimentTracker(tracker_path)

    print("=" * 80)
    print(f"TRACKER STATUS: {tracker_path.name}")
    print("=" * 80)
    print(f"\nTracker file: {tracker_path}")
    print(f"Completed experiments: {tracker.get_completed_count()}")
    print(f"Failed experiments: {tracker.get_failed_count()}")
    print()

    if tracker.get_completed_count() > 0:
        print("Recent completed experiments:")
        completed = tracker.list_completed()
        # Show last 5 completed
        for i, (exp_id, metadata) in enumerate(list(completed.items())[-5:]):
            print(f"  {i+1}. {exp_id}")
            print(f"     Completed: {metadata.get('completed_at', 'N/A')}")

    if tracker.get_failed_count() > 0:
        print(f"\nFailed experiments: {tracker.get_failed_count()}")
        print("Use 'list --failed' to see details")


def cmd_list(args):
    """List experiments."""
    tracker_path = get_tracker_path(args.test_type)
    if not tracker_path.exists():
        print(f"No tracker file found at: {tracker_path}")
        return

    tracker = ExperimentTracker(tracker_path)

    if args.failed:
        experiments = tracker.list_failed()
        title = "FAILED EXPERIMENTS"
    else:
        experiments = tracker.list_completed()
        title = "COMPLETED EXPERIMENTS"

    print("=" * 80)
    print(title)
    print("=" * 80)
    print(f"\nTotal: {len(experiments)}")
    print()

    if not experiments:
        print("No experiments found")
        return

    for exp_id, metadata in experiments.items():
        print(f"ID: {exp_id}")
        if args.failed:
            print(f"  Error: {metadata.get('error', 'N/A')}")
            print(f"  Failed at: {metadata.get('failed_at', 'N/A')}")
        else:
            print(f"  Optimizer: {metadata.get('optimizer', 'N/A')}")
            print(f"  Completed at: {metadata.get('completed_at', 'N/A')}")

        if args.verbose:
            print("  Metadata:")
            for key, value in metadata.items():
                if key not in ['completed_at', 'failed_at', 'error', 'optimizer']:
                    print(f"    {key}: {value}")
        print()


def cmd_clear(args):
    """Clear tracker data."""
    tracker_path = get_tracker_path(args.test_type)
    if not tracker_path.exists():
        print(f"No tracker file found at: {tracker_path}")
        return

    tracker = ExperimentTracker(tracker_path)

    if args.failed_only:
        count = tracker.get_failed_count()
        if not args.yes:
            response = input(f"Clear {count} failed experiments? (y/n): ")
            if response.lower() != 'y':
                print("Cancelled")
                return
        tracker.clear_failed()
        print(f"✓ Cleared {count} failed experiments")
    else:
        completed_count = tracker.get_completed_count()
        failed_count = tracker.get_failed_count()
        total = completed_count + failed_count

        if not args.yes:
            response = input(f"Clear ALL {total} experiments ({completed_count} completed, {failed_count} failed)? (y/n): ")
            if response.lower() != 'y':
                print("Cancelled")
                return
        tracker.clear()
        print(f"✓ Cleared all {total} experiments")


def cmd_remove(args):
    """Remove a specific experiment."""
    tracker_path = get_tracker_path(args.test_type)
    if not tracker_path.exists():
        print(f"No tracker file found at: {tracker_path}")
        return

    tracker = ExperimentTracker(tracker_path)

    if not tracker.is_completed(experiment_id=args.experiment_id):
        print(f"Experiment '{args.experiment_id}' not found in tracker")
        return

    if not args.yes:
        metadata = tracker.get_metadata(experiment_id=args.experiment_id)
        print(f"Experiment: {args.experiment_id}")
        print(f"Metadata: {metadata}")
        response = input("Remove this experiment? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled")
            return

    tracker.remove_experiment(args.experiment_id)
    print(f"✓ Removed experiment '{args.experiment_id}'")


def cmd_export(args):
    """Export tracker data to JSON."""
    tracker_path = get_tracker_path(args.test_type)
    if not tracker_path.exists():
        print(f"No tracker file found at: {tracker_path}")
        return

    with open(tracker_path, 'r') as f:
        data = json.load(f)

    output_path = Path(args.output) if args.output else Path(f"tracker_export_{args.test_type}.json")

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)

    print(f"✓ Exported tracker data to: {output_path}")
    print(f"  Completed: {len(data.get('experiments', {}))}")
    print(f"  Failed: {len(data.get('failed_experiments', {}))}")


def main():
    parser = argparse.ArgumentParser(
        description="Manage experiment tracker files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show status summary
  python tracker_cli.py status cnn

  # List completed experiments
  python tracker_cli.py list cnn

  # List failed experiments with details
  python tracker_cli.py list cnn --failed --verbose

  # Clear only failed experiments
  python tracker_cli.py clear cnn --failed-only

  # Clear all experiments (with confirmation)
  python tracker_cli.py clear cnn

  # Remove a specific experiment
  python tracker_cli.py remove cnn config_hyperparam_cnn_sgd_f32_cl4_trial1

  # Export tracker data
  python tracker_cli.py export cnn --output my_tracker.json
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Status command
    status_parser = subparsers.add_parser('status', help='Show tracker status')
    status_parser.add_argument('test_type', choices=['cnn', 'ffnn', 'gpt', 'example'],
                              help='Test type (cnn, ffnn, gpt, or example)')

    # List command
    list_parser = subparsers.add_parser('list', help='List experiments')
    list_parser.add_argument('test_type', choices=['cnn', 'ffnn', 'gpt', 'example'],
                            help='Test type')
    list_parser.add_argument('--failed', action='store_true',
                            help='List failed experiments instead of completed')
    list_parser.add_argument('--verbose', '-v', action='store_true',
                            help='Show detailed metadata')

    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear tracker data')
    clear_parser.add_argument('test_type', choices=['cnn', 'ffnn', 'gpt', 'example'],
                             help='Test type')
    clear_parser.add_argument('--failed-only', action='store_true',
                             help='Only clear failed experiments')
    clear_parser.add_argument('--yes', '-y', action='store_true',
                             help='Skip confirmation prompt')

    # Remove command
    remove_parser = subparsers.add_parser('remove', help='Remove a specific experiment')
    remove_parser.add_argument('test_type', choices=['cnn', 'ffnn', 'gpt', 'example'],
                              help='Test type')
    remove_parser.add_argument('experiment_id', help='Experiment ID to remove')
    remove_parser.add_argument('--yes', '-y', action='store_true',
                              help='Skip confirmation prompt')

    # Export command
    export_parser = subparsers.add_parser('export', help='Export tracker data')
    export_parser.add_argument('test_type', choices=['cnn', 'ffnn', 'gpt', 'example'],
                              help='Test type')
    export_parser.add_argument('--output', '-o', help='Output file path')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Dispatch to command handler
    command_handlers = {
        'status': cmd_status,
        'list': cmd_list,
        'clear': cmd_clear,
        'remove': cmd_remove,
        'export': cmd_export,
    }

    handler = command_handlers.get(args.command)
    if handler:
        handler(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
