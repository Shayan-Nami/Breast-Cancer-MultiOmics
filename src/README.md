# 📦 `src/` Module Documentation

The `src/` directory contains modular Python utilities and production-grade implementations of algorithms used across both the Jupyter notebooks and the standalone execution script (`run_pipeline.py`).

---

## 📂 Architecture Overview

```
src/
├── __init__.py      # Package entry point exposing core API functions
├── utils.py         # Primary module containing IO, feature selection, metrics, and tuning
└── README.md        # Technical documentation and API references
```

---

## 🛠️ API & Core Functions

### 1. I/O & Directory Management
- `resolve_dir(d='../outputs')`: Intelligently determines whether the calling script or notebook is executing from the project root or inside `notebooks/`, ensuring seamless cross-environment compatibility.
- `ensure_directories(base_dir='../outputs')`: Guarantees required subdirectories (`models/`, `results/`, `logs/`) exist prior to saving outputs.
- `save_object(obj, filename, output_dir='../outputs')`: Saves datasets with auto-detection of optimal formats:
  - `pandas.DataFrame` / `pandas.Series` $\to$ Apache Parquet (`.parquet`)
  - `numpy.ndarray` $\to$ Compressed NumPy Array (`.npz`)
  - Generic Python objects / string lists $\to$ Pickle (`.pkl`)
- `load_object(filename, input_dir='../outputs')`: Loads artifacts automatically detecting whether Parquet, NPZ, or Pickle is present.

### 2. Multi-Class Feature Selection Algorithms
- `reliefF(X, y, n_features_to_select=150, n_neighbors=10)`:
  - Multiclass-aware ReliefF implementation calculating feature relevance margins between nearest hits (same class) and nearest misses (differing classes).
  - Handles multi-modal and mixed continuous biomedical signals with distance weighting.
- `mrmr_feature_selection(X, y, n_features_to_select=50, variant='mi_corr')`:
  - **Minimum Redundancy Maximum Relevance**:
    - **Relevance**: Computed using mutual information with target labels ($I(X_i; Y)$ via `mutual_info_classif`).
    - **Redundancy**: Computed using mean absolute Pearson correlation across currently selected features ($|r(X_i, X_s)|$) or mutual information ($I(X_i; X_s)$ for the true MID variant).
    - Objective function: $\max_{X_i \in F \setminus S} \left[ I(X_i; Y) - \frac{1}{|S|} \sum_{X_s \in S} |r(X_i, X_s)| \right]$

### 3. Statistical Metrics & Validation
- `bootstrap_ci(y_true, y_pred, metric_func, n_bootstraps=1000, ci=95, random_state=42)`:
  - Generates empirical 95% Bootstrap Confidence Intervals for model performance metrics (Accuracy, Macro F1, etc.).
- `per_class_bootstrap_ci(y_true, y_pred, target_names, n_bootstraps=1000, ci=95, random_state=42)`:
  - Computes class-specific Precision, Recall, and F1-Score confidence intervals. Crucial for assessing statistical robustness on small subgroups like HER2 ($N=7$).
- `jaccard_similarity(set_a, set_b)`:
  - Computes biomarker set overlap stability across cross-validation folds: $J(A, B) = \frac{|A \cap B|}{|A \cup B|}$.

### 4. Hyperparameter Tuning & Visualization
- `tune_models_grid_search(X, y, cv=3, random_state=42)`:
  - Internal 3-fold stratified cross-validation for hyperparameter tuning on:
    - **Linear SVM**: Grid search over $C \in [0.01, 0.1, 1, 10]$.
    - **Random Forest**: Grid search over `n_estimators` $\in [100, 200, 300]$ and `max_depth` $\in [5, 10, \text{None}]$.
    - **KNN**: Grid search over `n_neighbors` $\in [3, 5, 7, 9]$ and `weights` $\in ['\text{uniform}', '\text{distance}']$.
- `plot_confusion_matrix(y_true, y_pred, target_names, title, save_path=None)`:
  - Generates standardized confusion matrices using clean Matplotlib `Axes` handling, preventing blank figure bugs in interactive environments.

---

## 💻 Quick Usage Example

```python
import numpy as np
from src import (
    load_object,
    save_object,
    mrmr_feature_selection,
    tune_models_grid_search,
    bootstrap_ci
)

# 1. Load preprocessed multi-omics data
X_train = load_object('X_train_rna_relief')
y_train = load_object('y_train')

# 2. Select top 50 features via mRMR
selected_indices, _, _ = mrmr_feature_selection(X_train, y_train, k=50)
X_train_sub = X_train[:, selected_indices]

# 3. Tune models in-fold
best_models = tune_models_grid_search(X_train_sub, y_train, cv=3)
rf_classifier = best_models['Random Forest']
print("Tuned Random Forest:", rf_classifier)
```
