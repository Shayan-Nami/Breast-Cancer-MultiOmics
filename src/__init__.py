"""
Breast-Cancer-MultiOmics Source Package
Core utility functions, feature selection algorithms, statistical metrics,
and model training/tuning routines.
"""

from .utils import (
    load_object,
    save_object,
    ensure_directories,
    resolve_dir,
    reliefF,
    mrmr_selection,
    mrmr_feature_selection,
    bootstrap_ci,
    per_class_bootstrap_ci,
    tune_models_grid_search,
    plot_confusion_matrix,
    jaccard_similarity,
)

__all__ = [
    "load_object",
    "save_object",
    "ensure_directories",
    "resolve_dir",
    "reliefF",
    "mrmr_selection",
    "mrmr_feature_selection",
    "bootstrap_ci",
    "per_class_bootstrap_ci",
    "tune_models_grid_search",
    "plot_confusion_matrix",
    "jaccard_similarity",
]

__version__ = "1.0.0"
