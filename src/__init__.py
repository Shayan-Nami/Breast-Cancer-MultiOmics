"""
Breast-Cancer-MultiOmics Source Package.
Core IO utilities, feature selection, leak-free fold preprocessing,
model registry/tuning, statistical & survival analysis, and plotting helpers.
"""

from .utils import (
    MODEL_SPECS,
    build_model_specs,
    resolve_dir,
    ensure_directories,
    save_object,
    load_object,
    reliefF_multiclass,
    reliefF,
    mrmr_selection,
    mrmr_feature_selection,
    fit_transform_omics_fold,
    tune_models_grid_search,
    bootstrap_ci,
    bootstrap_class_report,
    per_class_bootstrap_ci,
    benjamini_hochberg,
    compute_multiclass_roc_pr,
    kaplan_meier_estimator,
    logrank_test_multigroup,
    jaccard_similarity,
    plot_confusion_matrix,
    plot_multiclass_roc,
    plot_kaplan_meier,
    plot_ppi_network,
)

__all__ = [
    "MODEL_SPECS",
    "build_model_specs",
    "resolve_dir",
    "ensure_directories",
    "save_object",
    "load_object",
    "reliefF_multiclass",
    "reliefF",
    "mrmr_selection",
    "mrmr_feature_selection",
    "fit_transform_omics_fold",
    "tune_models_grid_search",
    "bootstrap_ci",
    "bootstrap_class_report",
    "per_class_bootstrap_ci",
    "benjamini_hochberg",
    "compute_multiclass_roc_pr",
    "kaplan_meier_estimator",
    "logrank_test_multigroup",
    "jaccard_similarity",
    "plot_confusion_matrix",
    "plot_multiclass_roc",
    "plot_kaplan_meier",
    "plot_ppi_network",
]

__version__ = "2.0.0"
