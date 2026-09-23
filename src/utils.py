"""
src/utils.py
Common utility functions for Breast-Cancer-MultiOmics pipeline:
- IO operations (supporting Parquet, NPZ, Pickle, and JSON with auto-directory resolution)
- Multi-class ReliefF algorithm (with reliefF alias for notebook backward compatibility)
- Normalized mRMR feature selection (MI-Correlation and true MID variants)
- Statistical and validation metrics (Bootstrap CI, per-class Bootstrap CI, Jaccard similarity)
- Nested GridSearch hyperparameter tuning for SVM, RF, and KNN
- Plotting utilities with proper matplotlib axes handling
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, ConfusionMatrixDisplay
)

try:
    import pyarrow
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


def resolve_dir(d='../outputs'):
    """
    Intelligently resolves the outputs/inputs directory path regardless of whether
    the caller is executing from the project root or inside the notebooks/ directory.
    Prevents path leakage to parent/desktop folders for nested subdirectories.
    """
    if os.path.exists(d):
        return d

    norm_d = os.path.normpath(d)
    parts = norm_d.split(os.sep)

    # If caller is at root and path starts with '..' / 'outputs'
    if len(parts) >= 2 and parts[0] == '..' and parts[1] == 'outputs':
        rel_sub = os.sep.join(parts[2:]) if len(parts) > 2 else ''
        candidate = os.path.join('outputs', rel_sub) if rel_sub else 'outputs'
        if os.path.exists('outputs'):
            return candidate

    # If caller is in notebooks/ and path starts with 'outputs'
    if parts[0] == 'outputs':
        rel_sub = os.sep.join(parts[1:]) if len(parts) > 1 else ''
        candidate = os.path.join('..', 'outputs', rel_sub) if rel_sub else os.path.join('..', 'outputs')
        if os.path.exists(os.path.join('..', 'outputs')):
            return candidate

    return d


def ensure_directories(base_dir='../outputs'):
    """Ensures all output directories match the required project specification."""
    resolved_base = resolve_dir(base_dir)
    dirs = [
        resolved_base,
        os.path.join(resolved_base, 'models'),
        os.path.join(resolved_base, 'results'),
        os.path.join(resolved_base, 'logs'),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs


def save_object(obj, filename, output_dir='../outputs', prefer_format='auto'):
    """
    Saves an object using safe, efficient formats:
    - pandas Series -> Parquet with '__series_val__' column (plus PKL fallback)
    - pandas DataFrame -> Parquet (plus PKL fallback)
    - numpy ndarray -> NPZ compressed archive (plus PKL fallback)
    - other Python objects -> Pickle (HIGHEST_PROTOCOL)
    """
    resolved_dir = resolve_dir(output_dir)
    ensure_directories(resolved_dir)

    clean_name = filename
    for ext in ('.parquet', '.npz', '.pkl'):
        if clean_name.endswith(ext):
            clean_name = clean_name[:-len(ext)]
            break

    target_path_base = os.path.join(resolved_dir, clean_name)

    # 1. Pandas Series
    if isinstance(obj, pd.Series) and prefer_format in ('auto', 'parquet'):
        if HAS_PYARROW:
            parquet_path = target_path_base + '.parquet'
            df = obj.to_frame(name='__series_val__')
            df.to_parquet(parquet_path)
            print(f"💾 Saved: {parquet_path}")
            return parquet_path

    # 2. Pandas DataFrame
    if isinstance(obj, pd.DataFrame) and prefer_format in ('auto', 'parquet'):
        if HAS_PYARROW:
            parquet_path = target_path_base + '.parquet'
            obj.to_parquet(parquet_path)
            print(f"💾 Saved: {parquet_path}")
            return parquet_path

    # 3. Numpy ndarray
    if isinstance(obj, np.ndarray) and prefer_format in ('auto', 'npz'):
        npz_path = target_path_base + '.npz'
        np.savez_compressed(npz_path, data=obj)
        print(f"💾 Saved: {npz_path}")
        return npz_path

    # 4. Fallback to pickle for Python objects (dicts, lists, scikit-learn models)
    pkl_path = target_path_base + '.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"💾 Saved: {pkl_path}")
    return pkl_path


def load_object(filename, input_dir='../outputs'):
    """
    Loads an object checking Parquet, NPZ, and Pickle in order.
    Correctly unwraps 1D pd.Series when saved as Parquet to prevent
    the 'object too deep for desired array' error in np.bincount().
    """
    resolved_dir = resolve_dir(input_dir)

    clean_name = filename
    for ext in ('.parquet', '.npz', '.pkl'):
        if clean_name.endswith(ext):
            clean_name = clean_name[:-len(ext)]
            break

    target_path_base = os.path.join(resolved_dir, clean_name)

    # 1. Check Parquet
    parquet_path = target_path_base + '.parquet'
    if HAS_PYARROW and os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        if isinstance(df, pd.DataFrame):
            if '__series_val__' in df.columns and len(df.columns) == 1:
                return df['__series_val__']
            if len(df.columns) == 1 and any(tag in clean_name.lower() for tag in ['y_train', 'y_test', 'y_labels', 'final_y']):
                return df.iloc[:, 0]
        return df

    # 2. Check NPZ
    npz_path = target_path_base + '.npz'
    if os.path.exists(npz_path):
        with np.load(npz_path, allow_pickle=True) as data:
            if 'data' in data.files:
                return data['data']
            return {k: data[k] for k in data.files}

    # 3. Check PKL
    pkl_path = target_path_base + '.pkl'
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)

    # 4. Check literal file path
    literal_path = os.path.join(resolved_dir, filename)
    if os.path.exists(literal_path):
        if literal_path.endswith('.parquet'):
            df = pd.read_parquet(literal_path)
            if isinstance(df, pd.DataFrame) and '__series_val__' in df.columns:
                return df['__series_val__']
            return df
        if literal_path.endswith('.npz'):
            with np.load(literal_path, allow_pickle=True) as data:
                return data['data'] if 'data' in data.files else dict(data)
        with open(literal_path, 'rb') as f:
            return pickle.load(f)

    raise FileNotFoundError(f"❌ File not found in {resolved_dir} with extensions .parquet, .npz, or .pkl: {filename}")


def reliefF_multiclass(X, y, n_neighbors=10, n_features_to_select=150):
    """
    Kononenko (1994) Multi-Class ReliefF Algorithm.
    Correctly samples nearest hits from the same class and nearest misses from each opposing class,
    weighted by class prior probabilities: P(C) / (1 - P(C_i)).
    """
    n_samples, n_features = X.shape
    y_arr = np.asarray(y).ravel()
    classes = np.unique(y_arr)

    class_indices = {c: np.where(y_arr == c)[0] for c in classes}
    p_c = {c: len(class_indices[c]) / n_samples for c in classes}

    nbrs_per_class = {}
    for c in classes:
        k_c = min(n_neighbors + 1, len(class_indices[c]))
        nbrs_per_class[c] = NearestNeighbors(n_neighbors=k_c, metric='manhattan').fit(X[class_indices[c]])

    weights = np.zeros(n_features, dtype=np.float64)

    for i in range(n_samples):
        ri = X[i:i+1]
        ci = y_arr[i]

        # k Nearest Hits
        k_hit = min(n_neighbors + 1, len(class_indices[ci]))
        _, hit_local = nbrs_per_class[ci].kneighbors(ri, n_neighbors=k_hit)
        hit_ids = [class_indices[ci][idx] for idx in hit_local[0] if class_indices[ci][idx] != i][:n_neighbors]
        if hit_ids:
            diff_hits = np.sum(np.abs(X[i] - X[hit_ids]), axis=0)
            weights -= diff_hits / (n_samples * len(hit_ids))

        # k Nearest Misses from each opposing class
        denom = 1.0 - p_c[ci]
        denom = denom if denom > 0 else 1.0
        for c in classes:
            if c == ci or len(class_indices[c]) == 0:
                continue
            k_miss = min(n_neighbors, len(class_indices[c]))
            _, miss_local = nbrs_per_class[c].kneighbors(ri, n_neighbors=k_miss)
            miss_ids = [class_indices[c][idx] for idx in miss_local[0]][:n_neighbors]
            if miss_ids:
                diff_misses = np.sum(np.abs(X[i] - X[miss_ids]), axis=0)
                prob_weight = p_c[c] / denom
                weights += (prob_weight * diff_misses) / (n_samples * len(miss_ids))

    top_indices = np.argsort(weights)[::-1][:n_features_to_select]
    return top_indices, weights


# Notebook backward compatibility alias:
reliefF = reliefF_multiclass


def mrmr_selection(X, y, k=50, method='mi_corr', random_state=42, n_features_to_select=None):
    """
    Minimum Redundancy Maximum Relevance (mRMR) Feature Selection.
    Supports:
    - 'mi_corr': Relevance = Normalized Mutual Information, Redundancy = Pearson |Correlation|
    - 'mid': True Mutual Information Difference (Relevance = MI, Redundancy = Pairwise Feature-Feature MI)
    """
    if n_features_to_select is not None:
        k = n_features_to_select
    n_features = X.shape[1]
    k = max(1, min(k, n_features))
    y_arr = np.asarray(y).ravel()
    relevance = mutual_info_classif(X, y_arr, random_state=random_state)

    rel_min = np.min(relevance)
    rel_max = np.max(relevance)
    if rel_max > rel_min:
        rel_norm = (relevance - rel_min) / (rel_max - rel_min + 1e-8)
    else:
        rel_norm = np.zeros_like(relevance)

    selected = []
    candidates = list(range(n_features))

    first_feat = int(np.argmax(rel_norm))
    selected.append(first_feat)
    candidates.remove(first_feat)

    if method == 'mid':
        # Discretize continuous features into bins for pairwise mutual information
        n_bins = 10
        X_binned = np.zeros_like(X, dtype=np.int32)
        for j in range(n_features):
            col = X[:, j]
            c_min, c_max = col.min(), col.max()
            if c_max > c_min:
                bins = np.linspace(c_min, c_max, n_bins)
                X_binned[:, j] = np.digitize(col, bins)
            else:
                X_binned[:, j] = 0

        # Cache feature-feature MI with selected features
        mi_feat_cache = np.zeros((n_features, k), dtype=np.float64)

        for s_step in range(k - 1):
            last_selected = selected[-1]
            # Compute MI between candidates and last_selected feature
            for c in candidates:
                contingency = np.histogram2d(X_binned[:, c], X_binned[:, last_selected], bins=n_bins)[0]
                total = np.sum(contingency)
                if total > 0:
                    p_xy = contingency / total
                    p_x = np.sum(p_xy, axis=1, keepdims=True)
                    p_y = np.sum(p_xy, axis=0, keepdims=True)
                    p_prod = p_x * p_y
                    nz = (p_xy > 0) & (p_prod > 0)
                    mi_val = np.sum(p_xy[nz] * np.log(p_xy[nz] / p_prod[nz]))
                else:
                    mi_val = 0.0
                mi_feat_cache[c, s_step] = max(0.0, mi_val)

            cand_arr = np.array(candidates)
            cand_rel = rel_norm[cand_arr]
            cand_red = np.mean(mi_feat_cache[cand_arr, :s_step+1], axis=1)
            red_max = np.max(cand_red)
            cand_red_norm = (cand_red / red_max) if red_max > 0 else cand_red

            mrmr_scores = cand_rel - cand_red_norm
            best_cand_idx = int(np.argmax(mrmr_scores))
            best_feat = candidates[best_cand_idx]
            selected.append(best_feat)
            candidates.remove(best_feat)
    else:
        # 'mi_corr'
        corr_matrix = np.abs(np.corrcoef(X.T))
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        for _ in range(k - 1):
            cand_arr = np.array(candidates)
            cand_rel = rel_norm[cand_arr]
            cand_red = np.mean(corr_matrix[np.ix_(cand_arr, selected)], axis=1)
            mrmr_scores = cand_rel - cand_red

            best_cand_idx = int(np.argmax(mrmr_scores))
            best_feat = candidates[best_cand_idx]
            selected.append(best_feat)
            candidates.remove(best_feat)

    selected_arr = np.array(selected)
    return selected_arr, relevance[selected_arr], rel_norm[selected_arr]


# Function alias for flexibility:
mrmr_feature_selection = mrmr_selection


def bootstrap_ci(y_true, y_pred, metric_func=accuracy_score, n_bootstraps=1000, ci=95, random_state=42):
    """
    Calculates point estimate and non-parametric bootstrap confidence interval (percentile method).
    """
    rng = np.random.RandomState(random_state)
    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()
    n = len(y_true_arr)

    point_estimate = metric_func(y_true_arr, y_pred_arr)
    bootstrapped_scores = []

    for _ in range(n_bootstraps):
        indices = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true_arr[indices])) < 2:
            continue
        try:
            score = metric_func(y_true_arr[indices], y_pred_arr[indices])
            if not np.isnan(score):
                bootstrapped_scores.append(score)
        except Exception:
            continue

    if len(bootstrapped_scores) == 0:
        return point_estimate, point_estimate, point_estimate

    alpha = (100 - ci) / 2.0
    lower = np.percentile(bootstrapped_scores, alpha)
    upper = np.percentile(bootstrapped_scores, 100 - alpha)
    return point_estimate, lower, upper


def bootstrap_class_report(y_true, y_pred, target_names, ci=95, n_bootstraps=1000, random_state=42):
    """
    Calculates 95% bootstrap confidence intervals for overall metrics and per-class recall/precision,
    crucial for the minority Her2 class (N=7 samples in test).
    """
    records = []
    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()

    # Overall Metrics
    acc_pt, acc_lo, acc_hi = bootstrap_ci(y_true_arr, y_pred_arr, metric_func=accuracy_score, n_bootstraps=n_bootstraps, ci=ci, random_state=random_state)
    records.append({'Metric': 'Overall Accuracy', 'Subtype': 'All', 'Estimate': acc_pt, 'CI_Lower': acc_lo, 'CI_Upper': acc_hi})

    bal_pt, bal_lo, bal_hi = bootstrap_ci(y_true_arr, y_pred_arr, metric_func=balanced_accuracy_score, n_bootstraps=n_bootstraps, ci=ci, random_state=random_state)
    records.append({'Metric': 'Balanced Accuracy', 'Subtype': 'All', 'Estimate': bal_pt, 'CI_Lower': bal_lo, 'CI_Upper': bal_hi})

    f1_pt, f1_lo, f1_hi = bootstrap_ci(y_true_arr, y_pred_arr, metric_func=lambda yt, yp: f1_score(yt, yp, average='macro'), n_bootstraps=n_bootstraps, ci=ci, random_state=random_state)
    records.append({'Metric': 'Macro F1', 'Subtype': 'All', 'Estimate': f1_pt, 'CI_Lower': f1_lo, 'CI_Upper': f1_hi})

    # Per-class Recall & Precision
    for idx, c_name in enumerate(target_names):
        rec_fn = lambda yt, yp, c=idx: recall_score(yt, yp, labels=[c], average='macro', zero_division=0)
        prec_fn = lambda yt, yp, c=idx: precision_score(yt, yp, labels=[c], average='macro', zero_division=0)

        r_pt, r_lo, r_hi = bootstrap_ci(y_true_arr, y_pred_arr, metric_func=rec_fn, n_bootstraps=n_bootstraps, ci=ci, random_state=random_state)
        p_pt, p_lo, p_hi = bootstrap_ci(y_true_arr, y_pred_arr, metric_func=prec_fn, n_bootstraps=n_bootstraps, ci=ci, random_state=random_state)

        records.append({'Metric': 'Recall', 'Subtype': c_name, 'Estimate': r_pt, 'CI_Lower': r_lo, 'CI_Upper': r_hi})
        records.append({'Metric': 'Precision', 'Subtype': c_name, 'Estimate': p_pt, 'CI_Lower': p_lo, 'CI_Upper': p_hi})

    return pd.DataFrame(records)


# Function alias:
per_class_bootstrap_ci = bootstrap_class_report


def tune_models_grid_search(X_train, y_train, cv=3, random_state=42, class_weight=None):
    """
    Performs in-fold hyperparameter tuning using GridSearchCV (Requirement 14).
    Tuning is performed strictly on in-fold training data to prevent data leakage.
    Optionally supports cost-sensitive class_weight='balanced' (Requirement 10).
    """
    y_arr = np.asarray(y_train).ravel()

    param_grids = {
        'SVM (Linear)': (
            SVC(random_state=random_state, class_weight=class_weight),
            {'C': [0.1, 1.0, 10.0], 'kernel': ['linear']}
        ),
        'Random Forest': (
            RandomForestClassifier(random_state=random_state, class_weight=class_weight),
            {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}
        ),
        'KNN (k=5)': (
            KNeighborsClassifier(),
            {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
        )
    }

    best_models = {}
    cv_inner = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    for m_name, (base_clf, grid) in param_grids.items():
        gs = GridSearchCV(
            estimator=base_clf,
            param_grid=grid,
            cv=cv_inner,
            scoring='balanced_accuracy',
            n_jobs=1
        )
        gs.fit(X_train, y_arr)
        best_models[m_name] = gs.best_estimator_

    return best_models


def jaccard_similarity(set_a, set_b):
    """Calculates Jaccard index between two feature sets."""
    s_a = set(set_a)
    s_b = set(set_b)
    intersection = len(s_a.intersection(s_b))
    union = len(s_a.union(s_b))
    return intersection / union if union > 0 else 0.0


def plot_confusion_matrix(y_true, y_pred, target_names, title, save_path=None, cmap='Blues', close_fig=False):
    """
    Plots a confusion matrix cleanly on an explicitly created Figure and Axes,
    preventing the empty figure bug.
    """
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        np.asarray(y_true).ravel(),
        np.asarray(y_pred).ravel(),
        display_labels=target_names,
        cmap=cmap,
        colorbar=False,
        ax=ax
    )
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        if close_fig:
            plt.close(fig)
    return fig, ax
