"""Lightweight exports only — heavy modules (runner, grid_builder) import submodules directly."""

from .config_schema import ExperimentConfig
from .registry import ExperimentRegistry

__all__ = ["ExperimentConfig", "ExperimentRegistry"]
