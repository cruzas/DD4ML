#!/usr/bin/env python3
"""
Experiment Tracker

A robust system for tracking completed experiments to enable resuming
after crashes. Uses JSON-based persistence to store experiment metadata
and completion status.

Usage:
    from experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker()

    # Check if experiment was already completed
    if tracker.is_completed(experiment_id):
        print("Skipping already completed experiment")
        return

    # Run experiment...

    # Mark as completed
    tracker.mark_completed(experiment_id, metadata={
        "optimizer": "sgd",
        "filters": 32,
        "num_layers": 4
    })
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ExperimentTracker:
    """Tracks experiment completion status with JSON-based persistence."""

    def __init__(self, tracker_file: Optional[Path] = None):
        """
        Initialize the experiment tracker.

        Args:
            tracker_file: Path to the JSON file for storing tracking data.
                         Defaults to .experiment_tracker.json in the current directory.
        """
        if tracker_file is None:
            tracker_file = Path.cwd() / ".experiment_tracker.json"

        self.tracker_file = Path(tracker_file)
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        """Load tracking data from JSON file."""
        if self.tracker_file.exists():
            try:
                with open(self.tracker_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load tracker file: {e}")
                print("Starting with empty tracker")
                return {"experiments": {}, "version": "1.0"}
        return {"experiments": {}, "version": "1.0"}

    def _save(self):
        """Save tracking data to JSON file."""
        try:
            with open(self.tracker_file, 'w') as f:
                json.dump(self.data, f, indent=2, sort_keys=True)
        except IOError as e:
            print(f"Warning: Could not save tracker file: {e}")

    def _generate_id(self, config_path: Optional[Path] = None, **kwargs) -> str:
        """
        Generate a unique experiment ID from config path or parameters.

        Args:
            config_path: Path to the config file
            **kwargs: Additional parameters to include in the ID

        Returns:
            A unique experiment ID (hash of the inputs)
        """
        if config_path is not None:
            # Use config file name as base ID
            return config_path.stem
        else:
            # Generate ID from parameters
            param_str = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
            return hashlib.md5(param_str.encode()).hexdigest()[:16]

    def is_completed(self, experiment_id: Optional[str] = None,
                    config_path: Optional[Path] = None,
                    **kwargs) -> bool:
        """
        Check if an experiment has been completed.

        Args:
            experiment_id: Direct experiment ID
            config_path: Path to config file (used to generate ID if experiment_id not provided)
            **kwargs: Parameters used to generate ID if neither experiment_id nor config_path provided

        Returns:
            True if the experiment was completed successfully
        """
        if experiment_id is None:
            experiment_id = self._generate_id(config_path, **kwargs)

        return experiment_id in self.data["experiments"]

    def mark_completed(self, experiment_id: Optional[str] = None,
                      config_path: Optional[Path] = None,
                      metadata: Optional[Dict[str, Any]] = None,
                      **kwargs):
        """
        Mark an experiment as completed.

        Args:
            experiment_id: Direct experiment ID
            config_path: Path to config file (used to generate ID if experiment_id not provided)
            metadata: Additional metadata to store with the experiment
            **kwargs: Parameters used to generate ID and stored as metadata
        """
        if experiment_id is None:
            experiment_id = self._generate_id(config_path, **kwargs)

        # Combine explicit metadata with kwargs
        full_metadata = kwargs.copy()
        if metadata:
            full_metadata.update(metadata)

        # Store config path if provided
        if config_path is not None:
            full_metadata["config_path"] = str(config_path)

        # Add completion timestamp
        full_metadata["completed_at"] = datetime.now().isoformat()

        self.data["experiments"][experiment_id] = full_metadata
        self._save()

    def mark_failed(self, experiment_id: Optional[str] = None,
                   config_path: Optional[Path] = None,
                   error: Optional[str] = None,
                   **kwargs):
        """
        Mark an experiment as failed (does not mark as completed).

        This allows you to track failed experiments separately without
        preventing retry attempts.

        Args:
            experiment_id: Direct experiment ID
            config_path: Path to config file
            error: Error message or reason for failure
            **kwargs: Parameters used to generate ID
        """
        if experiment_id is None:
            experiment_id = self._generate_id(config_path, **kwargs)

        # Store in separate failed experiments dict
        if "failed_experiments" not in self.data:
            self.data["failed_experiments"] = {}

        self.data["failed_experiments"][experiment_id] = {
            "failed_at": datetime.now().isoformat(),
            "error": error,
            "config_path": str(config_path) if config_path else None,
            **kwargs
        }
        self._save()

    def get_metadata(self, experiment_id: Optional[str] = None,
                    config_path: Optional[Path] = None,
                    **kwargs) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a completed experiment.

        Args:
            experiment_id: Direct experiment ID
            config_path: Path to config file
            **kwargs: Parameters used to generate ID

        Returns:
            Metadata dict if experiment completed, None otherwise
        """
        if experiment_id is None:
            experiment_id = self._generate_id(config_path, **kwargs)

        return self.data["experiments"].get(experiment_id)

    def get_completed_count(self) -> int:
        """Get the number of completed experiments."""
        return len(self.data["experiments"])

    def get_failed_count(self) -> int:
        """Get the number of failed experiments."""
        return len(self.data.get("failed_experiments", {}))

    def list_completed(self) -> Dict[str, Dict[str, Any]]:
        """Get all completed experiments with their metadata."""
        return self.data["experiments"].copy()

    def list_failed(self) -> Dict[str, Dict[str, Any]]:
        """Get all failed experiments with their metadata."""
        return self.data.get("failed_experiments", {}).copy()

    def clear(self):
        """Clear all tracking data."""
        self.data = {"experiments": {}, "version": "1.0"}
        self._save()

    def clear_failed(self):
        """Clear failed experiments (keeping completed ones)."""
        if "failed_experiments" in self.data:
            del self.data["failed_experiments"]
        self._save()

    def remove_experiment(self, experiment_id: str):
        """Remove a specific experiment from tracking (allows re-running)."""
        if experiment_id in self.data["experiments"]:
            del self.data["experiments"][experiment_id]
            self._save()
