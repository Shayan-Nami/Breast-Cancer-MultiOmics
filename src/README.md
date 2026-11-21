# `src/` module documentation

The `src/` package exposes the shared IO, feature selection, leak-free fold transformation, hyperparameter tuning, statistical/survival analysis, and plotting functions used identically by all 6 notebooks and `run_pipeline.py`.

## Module layout

```text
src/
├── __init__.py   # Public package exports (24 symbols, v2.0.0)
├── utils.py      # Core implementation
└── README.md     # API reference
```

## Public API

### 1. IO and directory management
- `resolve_dir(d='../outputs')`: Resolves relative output paths cleanly from either the repository root or `notebooks/`.
- `ensure_directories(base_dir='../outputs')`: Creates `outputs/`, `models/`, `results/`, and `logs/`.
- `save_object(obj, filename, output_dir='../outputs', prefer_format='auto')`: Saves DataFrames/Series as Parquet, NumPy arrays as compressed NPZ, and Python objects as Pickle. Creates only the target directory without spawning spurious nested subfolders.
- `load_object(filename, input_dir='../outputs')`: Auto-loads Parquet (unwrapping 1D Series), NPZ, or Pickle artifacts.

### 2. Multi-omics feature selection and leak-free fold transformation
- `reliefF_multiclass(X, y, n_neighbors=10, n_features_to_select=150)` (alias `reliefF`): Vectorized Kononenko (1994) multi-class ReliefF with Manhattan distance (~0.08s on `439 x 500`, numerically matching brute-force weights within `3.3e-15`).
- `mrmr_selection(X, y, k=50, method='mi_corr', random_state=42, n_features_to_select=None)` (alias `mrmr_feature_selection`): Supports Pearson redundancy (`'mi_corr'`) and vectorized `np.bincount` mutual information difference (`'mid'`).
- `fit_transform_omics_fold(...)`: Single leak-free fold transformer running `SimpleImputer(median)` -> `VarianceThreshold(20th pct)` -> `MinMaxScaler(clip=True)` -> `SelectKBest(f_classif)` -> optional `reliefF_multiclass` -> optional `mrmr_selection` strictly on training rows and applying learned transforms to validation rows.

### 3. Model registry and leak-free hyperparameter tuning
- `MODEL_SPECS` and `build_model_specs(random_state=42)`: Single source of truth defining estimators, parameter grids, class-weight support, and artifact filenames for `SVM (Linear)`, `Random Forest`, `KNN (Tuned)`, and `Logistic Regression (L2)`.
- `tune_models_grid_search(X_train, y_train, cv=3, random_state=42, resampling=None, class_weight=None, return_params=False, n_jobs=-1, model_names=None)`: Runs stratified `GridSearchCV` optimized for `balanced_accuracy`. Under `resampling='smote'`, wraps `SMOTE` inside `imblearn.pipeline.Pipeline` so synthetic minority samples never cross inner CV folds.

### 4. Statistical, survival, and visualization utilities
- `bootstrap_ci(...)` and `bootstrap_class_report(...)` (alias `per_class_bootstrap_ci`): Percentile 95% bootstrap CIs for overall metrics and per-subtype Recall, Precision, and F1.
- `benjamini_hochberg(pvals)`: BH-FDR adjustment preserving NaN positions.
- `compute_multiclass_roc_pr(y_true, y_proba, class_labels=None)`: One-vs-rest macro/weighted ROC-AUC, macro PR-AUC, and per-class curves.
- `kaplan_meier_estimator(times, events)` and `logrank_test_multigroup(times, events, groups)`: Exact product-limit KM curve and Mantel-Cox G-group log-rank test.
- `jaccard_similarity(set_a, set_b)`: Pairwise biomarker overlap index.
- `plot_confusion_matrix(...)`, `plot_multiclass_roc(...)`, `plot_kaplan_meier(...)`, and `plot_ppi_network(...)`: Publication-ready Matplotlib/NetworkX plots with explicit Figure/Axes management.
