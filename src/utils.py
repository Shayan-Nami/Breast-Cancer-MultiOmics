"""
src/utils.py
Core utilities for the Breast-Cancer-MultiOmics pipeline:
- IO helpers (Parquet, NPZ, Pickle with directory resolution)
- Vectorized multi-class ReliefF and fast mRMR feature selection
- Leak-free multi-omics fold preprocessing and feature selection
- Canonical model registry and leak-free GridSearchCV tuning (SMOTE / class_weight / none)
- Bootstrap CIs, Benjamini-Hochberg FDR, multi-class ROC/PR, Kaplan-Meier, and log-rank test
- Matplotlib plotting utilities for confusion matrices, ROC curves, survival curves, and PPI networks
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
from scipy import stats

from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    mutual_info_classif,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    average_precision_score,
    ConfusionMatrixDisplay,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

try:
    import pyarrow
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


MODEL_SPECS = {
    'SVM (Linear)': {
        'estimator_cls': SVC,
        'estimator_kwargs': {'kernel': 'linear', 'probability': True},
        'uses_random_state': True,
        'param_grid': {'C': [0.05, 0.1, 0.5, 1.0, 5.0]},
        'supports_class_weight': True,
        'model_file': 'svm_model',
        'cm_file': 'confusion_matrix_SVM.png',
    },
    'Random Forest': {
        'estimator_cls': RandomForestClassifier,
        'estimator_kwargs': {},
        'uses_random_state': True,
        'param_grid': {'n_estimators': [100, 200], 'max_depth': [5, 10, None]},
        'supports_class_weight': True,
        'model_file': 'random_forest_model',
        'cm_file': 'confusion_matrix_RF.png',
    },
    'KNN (Tuned)': {
        'estimator_cls': KNeighborsClassifier,
        'estimator_kwargs': {},
        'uses_random_state': False,
        'param_grid': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']},
        'supports_class_weight': False,
        'model_file': 'knn_model',
        'cm_file': 'confusion_matrix_KNN.png',
    },
    'Logistic Regression (L2)': {
        'estimator_cls': LogisticRegression,
        'estimator_kwargs': {'max_iter': 2000},
        'uses_random_state': True,
        'param_grid': {'C': [0.05, 0.1, 0.5, 1.0, 5.0]},
        'supports_class_weight': True,
        'model_file': 'logreg_model',
        'cm_file': 'confusion_matrix_LogReg.png',
    },
}


def build_model_specs(random_state=42):
    """Return fresh unfitted estimator instances, parameter grids, and metadata."""
    specs = {}
    for name, meta in MODEL_SPECS.items():
        kwargs = dict(meta['estimator_kwargs'])
        if meta['uses_random_state']:
            kwargs['random_state'] = random_state
        specs[name] = {
            'estimator': meta['estimator_cls'](**kwargs),
            'param_grid': {k: list(v) for k, v in meta['param_grid'].items()},
            'supports_class_weight': meta['supports_class_weight'],
            'model_file': meta['model_file'],
            'cm_file': meta['cm_file'],
        }
    return specs


def resolve_dir(d='../outputs'):
    """Resolve outputs/inputs directory whether called from project root or notebooks/."""
    if os.path.exists(d):
        return d

    norm_d = os.path.normpath(d)
    parts = norm_d.split(os.sep)

    if len(parts) >= 2 and parts[0] == '..' and parts[1] == 'outputs':
        rel_sub = os.sep.join(parts[2:]) if len(parts) > 2 else ''
        candidate = os.path.join('outputs', rel_sub) if rel_sub else 'outputs'
        if os.path.exists('outputs'):
            return candidate

    if parts[0] == 'outputs':
        rel_sub = os.sep.join(parts[1:]) if len(parts) > 1 else ''
        candidate = os.path.join('..', 'outputs', rel_sub) if rel_sub else os.path.join('..', 'outputs')
        if os.path.exists(os.path.join('..', 'outputs')):
            return candidate

    return d


def ensure_directories(base_dir='../outputs'):
    """Ensure required output subdirectories exist."""
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
    """Save DataFrames/Series as Parquet, ndarrays as compressed NPZ, or fallback to Pickle."""
    resolved_dir = resolve_dir(output_dir)
    os.makedirs(resolved_dir, exist_ok=True)

    clean_name = filename
    for ext in ('.parquet', '.npz', '.pkl'):
        if clean_name.endswith(ext):
            clean_name = clean_name[:-len(ext)]
            break

    target_path_base = os.path.join(resolved_dir, clean_name)

    if isinstance(obj, pd.Series) and prefer_format in ('auto', 'parquet'):
        if HAS_PYARROW:
            parquet_path = target_path_base + '.parquet'
            df = obj.to_frame(name='__series_val__')
            df.to_parquet(parquet_path)
            print(f"💾 Saved: {parquet_path}")
            return parquet_path

    if isinstance(obj, pd.DataFrame) and prefer_format in ('auto', 'parquet'):
        if HAS_PYARROW:
            parquet_path = target_path_base + '.parquet'
            obj.to_parquet(parquet_path)
            print(f"💾 Saved: {parquet_path}")
            return parquet_path

    if isinstance(obj, np.ndarray) and prefer_format in ('auto', 'npz'):
        npz_path = target_path_base + '.npz'
        np.savez_compressed(npz_path, data=obj)
        print(f"💾 Saved: {npz_path}")
        return npz_path

    pkl_path = target_path_base + '.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"💾 Saved: {pkl_path}")
    return pkl_path


def load_object(filename, input_dir='../outputs'):
    """Load an artifact checking Parquet, NPZ, Pickle, and literal path in order."""
    resolved_dir = resolve_dir(input_dir)

    clean_name = filename
    for ext in ('.parquet', '.npz', '.pkl'):
        if clean_name.endswith(ext):
            clean_name = clean_name[:-len(ext)]
            break

    target_path_base = os.path.join(resolved_dir, clean_name)

    parquet_path = target_path_base + '.parquet'
    if HAS_PYARROW and os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        if isinstance(df, pd.DataFrame):
            if '__series_val__' in df.columns and len(df.columns) == 1:
                return df['__series_val__']
            if len(df.columns) == 1 and any(
                tag in clean_name.lower() for tag in ['y_train', 'y_test', 'y_labels', 'final_y']
            ):
                return df.iloc[:, 0]
        return df

    npz_path = target_path_base + '.npz'
    if os.path.exists(npz_path):
        with np.load(npz_path, allow_pickle=True) as data:
            if 'data' in data.files:
                return data['data']
            return {k: data[k] for k in data.files}

    pkl_path = target_path_base + '.pkl'
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)

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

    raise FileNotFoundError(
        f"❌ File not found in {resolved_dir} with extensions .parquet, .npz, or .pkl: {filename}"
    )


def reliefF_multiclass(X, y, n_neighbors=10, n_features_to_select=150):
    """Kononenko (1994) multi-class ReliefF feature scorer with Manhattan distance."""
    X_arr = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X_arr.shape
    y_arr = np.asarray(y).ravel()
    classes = np.unique(y_arr)

    if n_samples == 0 or n_features == 0:
        return np.array([], dtype=np.int64), np.zeros(n_features, dtype=np.float64)

    class_indices = {c: np.where(y_arr == c)[0] for c in classes}
    p_c = {c: len(class_indices[c]) / n_samples for c in classes}

    denom_per_sample = np.empty(n_samples, dtype=np.float64)
    for c in classes:
        d = 1.0 - p_c[c]
        denom_per_sample[class_indices[c]] = d if d > 0.0 else 1.0

    weights = np.zeros(n_features, dtype=np.float64)

    for c in classes:
        c_idx = class_indices[c]
        n_c = len(c_idx)
        if n_c == 0:
            continue

        k_c = min(n_neighbors + 1, n_c)
        nn = NearestNeighbors(n_neighbors=k_c, metric='manhattan').fit(X_arr[c_idx])
        _, all_local = nn.kneighbors(X_arr, n_neighbors=k_c)

        n_hit = min(n_neighbors, n_c - 1)
        if n_hit > 0:
            hit_global = c_idx[all_local[c_idx]]
            not_self = hit_global != c_idx[:, None]
            keep = not_self & (np.cumsum(not_self, axis=1) <= n_hit)
            hit_selected = hit_global[keep].reshape(n_c, n_hit)

            diff_hits = np.zeros(n_features, dtype=np.float64)
            X_c = X_arr[c_idx]
            for j in range(n_hit):
                diff_hits += np.abs(X_c - X_arr[hit_selected[:, j]]).sum(axis=0)
            weights -= diff_hits / (n_samples * n_hit)

        opp_idx = np.where(y_arr != c)[0]
        k_miss = min(n_neighbors, n_c)
        if len(opp_idx) > 0 and k_miss > 0:
            miss_selected = c_idx[all_local[opp_idx, :k_miss]]
            w_opp = p_c[c] / denom_per_sample[opp_idx]
            diff_misses = np.zeros(n_features, dtype=np.float64)
            X_opp = X_arr[opp_idx]
            for j in range(k_miss):
                diff_misses += w_opp @ np.abs(X_opp - X_arr[miss_selected[:, j]])
            weights += diff_misses / (n_samples * k_miss)

    top_indices = np.argsort(weights)[::-1][:n_features_to_select]
    return top_indices, weights


reliefF = reliefF_multiclass


def mrmr_selection(X, y, k=50, method='mi_corr', random_state=42, n_features_to_select=None):
    """Minimum Redundancy Maximum Relevance (mRMR) selection ('mi_corr' or 'mid')."""
    if n_features_to_select is not None:
        k = n_features_to_select
    X_arr = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X_arr.shape
    k = max(1, min(int(k), n_features))
    y_arr = np.asarray(y).ravel()
    relevance = mutual_info_classif(X_arr, y_arr, random_state=random_state)

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

    if k > 1:
        if method == 'mid':
            n_bins = 10
            n_bins2 = n_bins * n_bins
            c_min = X_arr.min(axis=0)
            c_max = X_arr.max(axis=0)
            span = c_max - c_min
            X_binned = np.zeros((n_samples, n_features), dtype=np.int32)
            non_const = span > 0
            if np.any(non_const):
                scaled = (X_arr[:, non_const] - c_min[non_const]) / (span[non_const] + 1e-12) * n_bins
                X_binned[:, non_const] = np.clip(np.floor(scaled).astype(np.int32), 0, n_bins - 1)

            X_binned_T = np.ascontiguousarray(X_binned.T)
            mi_sum = np.zeros(n_features, dtype=np.float64)

            for s_step in range(k - 1):
                last_selected = selected[-1]
                cand_arr = np.asarray(candidates, dtype=np.intp)
                m = len(cand_arr)

                if n_samples > 0:
                    last_bins = X_binned_T[last_selected]
                    cand_bins = X_binned_T[cand_arr]
                    # Offset each candidate's joint bin table into a single flat bincount pass.
                    offsets = np.arange(m, dtype=np.intp)[:, None] * n_bins2
                    flat_idx = (offsets + cand_bins * n_bins + last_bins[None, :]).ravel()
                    counts = np.bincount(flat_idx, minlength=m * n_bins2).reshape(m, n_bins, n_bins)

                    p_xy = counts / float(n_samples)
                    p_x = p_xy.sum(axis=2, keepdims=True)
                    p_y = p_xy.sum(axis=1, keepdims=True)
                    p_prod = p_x * p_y
                    nz = (p_xy > 0) & (p_prod > 0)
                    mi_terms = np.zeros_like(p_xy)
                    mi_terms[nz] = p_xy[nz] * np.log(p_xy[nz] / p_prod[nz])
                    mi_vals = np.maximum(0.0, mi_terms.sum(axis=(1, 2)))
                    mi_sum[cand_arr] += mi_vals

                cand_rel = rel_norm[cand_arr]
                cand_red = mi_sum[cand_arr] / (s_step + 1)
                red_max = np.max(cand_red)
                cand_red_norm = (cand_red / red_max) if red_max > 0 else cand_red

                mrmr_scores = cand_rel - cand_red_norm
                best_cand_idx = int(np.argmax(mrmr_scores))
                best_feat = candidates[best_cand_idx]
                selected.append(best_feat)
                candidates.remove(best_feat)
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                corr_matrix = np.abs(np.corrcoef(X_arr.T))
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

            for _ in range(k - 1):
                cand_arr = np.asarray(candidates, dtype=np.intp)
                cand_rel = rel_norm[cand_arr]
                cand_red = np.mean(corr_matrix[np.ix_(cand_arr, selected)], axis=1)
                mrmr_scores = cand_rel - cand_red

                best_cand_idx = int(np.argmax(mrmr_scores))
                best_feat = candidates[best_cand_idx]
                selected.append(best_feat)
                candidates.remove(best_feat)

    selected_arr = np.array(selected)
    return selected_arr, relevance[selected_arr], rel_norm[selected_arr]


mrmr_feature_selection = mrmr_selection


def _fit_transform_single_modality(
    X_tr,
    X_va,
    y_arr,
    names,
    impute_strategy,
    var_percentile,
    clip_minmax,
    k_anova,
    use_relief,
    k_relief,
    relief_neighbors,
):
    """Fit preprocessing and univariate/ReliefF selectors strictly on training rows."""
    if names is None:
        if isinstance(X_tr, pd.DataFrame):
            kept_names = [str(c) for c in X_tr.columns]
        else:
            kept_names = [f'f{i}' for i in range(np.asarray(X_tr).shape[1])]
    else:
        kept_names = [str(n) for n in names]

    X_tr_m = np.asarray(X_tr, dtype=np.float64)
    X_va_m = np.asarray(X_va, dtype=np.float64)

    if impute_strategy is not None:
        imputer = SimpleImputer(strategy=impute_strategy, keep_empty_features=True)
        X_tr_m = imputer.fit_transform(X_tr_m)
        X_va_m = imputer.transform(X_va_m)

    if var_percentile is not None and var_percentile > 0 and X_tr_m.shape[1] > 1:
        variances = np.var(X_tr_m, axis=0)
        thresh = float(np.percentile(variances, var_percentile))
        if np.any(variances > thresh):
            vt = VarianceThreshold(threshold=thresh)
            X_tr_m = vt.fit_transform(X_tr_m)
            X_va_m = vt.transform(X_va_m)
            support = vt.get_support()
            kept_names = [n for n, keep in zip(kept_names, support) if keep]

    if clip_minmax is not None:
        scaler = MinMaxScaler(clip=clip_minmax)
        X_tr_m = scaler.fit_transform(X_tr_m)
        X_va_m = scaler.transform(X_va_m)

    if k_anova is not None and X_tr_m.shape[1] > 0:
        k_sel = min(int(k_anova), X_tr_m.shape[1])
        selector = SelectKBest(f_classif, k=k_sel)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            warnings.simplefilter('ignore', RuntimeWarning)
            X_tr_m = selector.fit_transform(X_tr_m, y_arr)
        X_va_m = selector.transform(X_va_m)
        support = selector.get_support()
        kept_names = [n for n, keep in zip(kept_names, support) if keep]

    if use_relief and k_relief is not None and X_tr_m.shape[1] > 0:
        k_rel = min(int(k_relief), X_tr_m.shape[1])
        rel_idx, _ = reliefF_multiclass(
            X_tr_m,
            y_arr,
            n_neighbors=relief_neighbors,
            n_features_to_select=k_rel,
        )
        X_tr_m = X_tr_m[:, rel_idx]
        X_va_m = X_va_m[:, rel_idx]
        kept_names = [kept_names[i] for i in rel_idx]

    return X_tr_m, X_va_m, kept_names


def fit_transform_omics_fold(
    X_rna_tr,
    X_rna_va,
    X_meth_tr,
    X_meth_va,
    y_tr,
    *,
    modalities='both',
    impute_strategy='median',
    var_percentile=20.0,
    clip_minmax=True,
    k_anova=500,
    use_relief=True,
    k_relief=150,
    relief_neighbors=10,
    use_mrmr=True,
    k_mrmr=50,
    mrmr_method='mid',
    random_state=42,
    rna_names=None,
    meth_names=None,
):
    """Leak-free fold transformer for RNA, Methylation, or combined multi-omics data."""
    if modalities not in ('both', 'rna', 'meth'):
        raise ValueError(f"Unsupported modalities={modalities!r}; expected 'both', 'rna', or 'meth'.")

    y_arr = np.asarray(y_tr).ravel()
    tr_blocks = []
    va_blocks = []
    prefixed_names = []
    raw_names = []
    modality_labels = []

    if modalities in ('both', 'rna'):
        rna_tr, rna_va, rna_kept = _fit_transform_single_modality(
            X_rna_tr,
            X_rna_va,
            y_arr,
            rna_names,
            impute_strategy,
            var_percentile,
            clip_minmax,
            k_anova,
            use_relief,
            k_relief,
            relief_neighbors,
        )
        tr_blocks.append(rna_tr)
        va_blocks.append(rna_va)
        prefixed_names.extend([f'RNA_{n}' for n in rna_kept])
        raw_names.extend(rna_kept)
        modality_labels.extend(['RNA'] * len(rna_kept))

    if modalities in ('both', 'meth'):
        meth_tr, meth_va, meth_kept = _fit_transform_single_modality(
            X_meth_tr,
            X_meth_va,
            y_arr,
            meth_names,
            impute_strategy,
            var_percentile,
            clip_minmax,
            k_anova,
            use_relief,
            k_relief,
            relief_neighbors,
        )
        tr_blocks.append(meth_tr)
        va_blocks.append(meth_va)
        prefixed_names.extend([f'Meth_{n}' for n in meth_kept])
        raw_names.extend(meth_kept)
        modality_labels.extend(['Methylation'] * len(meth_kept))

    comb_tr = np.hstack(tr_blocks) if len(tr_blocks) > 1 else tr_blocks[0]
    comb_va = np.hstack(va_blocks) if len(va_blocks) > 1 else va_blocks[0]

    mrmr_raw_mi = None
    mrmr_norm_mi = None
    if use_mrmr and comb_tr.shape[1] > 0:
        k_sel = min(int(k_mrmr), comb_tr.shape[1])
        sel_idx, mrmr_raw_mi, mrmr_norm_mi = mrmr_selection(
            comb_tr,
            y_arr,
            k=k_sel,
            method=mrmr_method,
            random_state=random_state,
        )
        comb_tr = comb_tr[:, sel_idx]
        comb_va = comb_va[:, sel_idx]
        prefixed_names = [prefixed_names[i] for i in sel_idx]
        raw_names = [raw_names[i] for i in sel_idx]
        modality_labels = [modality_labels[i] for i in sel_idx]

    n_rna = sum(1 for m in modality_labels if m == 'RNA')
    n_meth = sum(1 for m in modality_labels if m == 'Methylation')

    return {
        'X_tr': comb_tr,
        'X_va': comb_va,
        'feature_names': prefixed_names,
        'raw_names': raw_names,
        'modality_labels': modality_labels,
        'n_rna': n_rna,
        'n_meth': n_meth,
        'mrmr_raw_mi': mrmr_raw_mi,
        'mrmr_norm_mi': mrmr_norm_mi,
    }


def tune_models_grid_search(
    X_train,
    y_train,
    cv=3,
    random_state=42,
    resampling=None,
    class_weight=None,
    return_params=False,
    n_jobs=-1,
    model_names=None,
):
    """Leak-free stratified GridSearchCV over canonical models with in-fold SMOTE or class weighting."""
    y_arr = np.asarray(y_train).ravel()

    if resampling is None:
        resampling = 'class_weight' if class_weight == 'balanced' else 'smote'
    if resampling not in ('smote', 'class_weight', 'none'):
        raise ValueError(f"Unsupported resampling={resampling!r}; expected 'smote', 'class_weight', or 'none'.")

    specs = build_model_specs(random_state=random_state)
    if model_names is not None:
        missing = [m for m in model_names if m not in specs]
        if missing:
            raise KeyError(f"Unknown model name(s): {missing}")
        specs = {m: specs[m] for m in model_names}

    cv_inner = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    if resampling == 'smote':
        _, class_counts = np.unique(y_arr, return_counts=True)
        k_smote = max(1, min(5, int(np.min(class_counts) * (cv - 1) / cv) - 1))

    best_models = {}
    best_params = {}

    for m_name, spec in specs.items():
        estimator = spec['estimator']
        grid = spec['param_grid']

        if resampling == 'smote':
            pipe = ImbPipeline([
                ('smote', SMOTE(k_neighbors=k_smote, random_state=random_state)),
                ('clf', estimator),
            ])
            prefixed_grid = {f'clf__{k}': v for k, v in grid.items()}
            gs = GridSearchCV(
                estimator=pipe,
                param_grid=prefixed_grid,
                cv=cv_inner,
                scoring='balanced_accuracy',
                n_jobs=n_jobs,
            )
        else:
            if resampling == 'class_weight' and spec['supports_class_weight']:
                estimator.set_params(class_weight='balanced')
            gs = GridSearchCV(
                estimator=estimator,
                param_grid=grid,
                cv=cv_inner,
                scoring='balanced_accuracy',
                n_jobs=n_jobs,
            )

        gs.fit(X_train, y_arr)
        best_models[m_name] = gs.best_estimator_
        best_params[m_name] = gs.best_params_

    if return_params:
        return best_models, best_params
    return best_models


def bootstrap_ci(y_true, y_pred, metric_func=accuracy_score, n_bootstraps=1000, ci=95, random_state=42):
    """Calculate point estimate and percentile bootstrap confidence interval."""
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
    """Compute bootstrap CIs for overall metrics and per-class Recall, Precision, and F1."""
    records = []
    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()

    acc_pt, acc_lo, acc_hi = bootstrap_ci(
        y_true_arr, y_pred_arr, metric_func=accuracy_score,
        n_bootstraps=n_bootstraps, ci=ci, random_state=random_state,
    )
    records.append({'Metric': 'Overall Accuracy', 'Subtype': 'All', 'Estimate': acc_pt, 'CI_Lower': acc_lo, 'CI_Upper': acc_hi})

    bal_pt, bal_lo, bal_hi = bootstrap_ci(
        y_true_arr, y_pred_arr, metric_func=balanced_accuracy_score,
        n_bootstraps=n_bootstraps, ci=ci, random_state=random_state,
    )
    records.append({'Metric': 'Balanced Accuracy', 'Subtype': 'All', 'Estimate': bal_pt, 'CI_Lower': bal_lo, 'CI_Upper': bal_hi})

    f1_pt, f1_lo, f1_hi = bootstrap_ci(
        y_true_arr, y_pred_arr, metric_func=lambda yt, yp: f1_score(yt, yp, average='macro'),
        n_bootstraps=n_bootstraps, ci=ci, random_state=random_state,
    )
    records.append({'Metric': 'Macro F1', 'Subtype': 'All', 'Estimate': f1_pt, 'CI_Lower': f1_lo, 'CI_Upper': f1_hi})

    for idx, c_name in enumerate(target_names):
        rec_fn = lambda yt, yp, c=idx: recall_score(yt, yp, labels=[c], average='macro', zero_division=0)
        prec_fn = lambda yt, yp, c=idx: precision_score(yt, yp, labels=[c], average='macro', zero_division=0)
        f1_fn = lambda yt, yp, c=idx: f1_score(yt, yp, labels=[c], average='macro', zero_division=0)

        r_pt, r_lo, r_hi = bootstrap_ci(
            y_true_arr, y_pred_arr, metric_func=rec_fn,
            n_bootstraps=n_bootstraps, ci=ci, random_state=random_state,
        )
        p_pt, p_lo, p_hi = bootstrap_ci(
            y_true_arr, y_pred_arr, metric_func=prec_fn,
            n_bootstraps=n_bootstraps, ci=ci, random_state=random_state,
        )
        f_pt, f_lo, f_hi = bootstrap_ci(
            y_true_arr, y_pred_arr, metric_func=f1_fn,
            n_bootstraps=n_bootstraps, ci=ci, random_state=random_state,
        )

        records.append({'Metric': 'Recall', 'Subtype': c_name, 'Estimate': r_pt, 'CI_Lower': r_lo, 'CI_Upper': r_hi})
        records.append({'Metric': 'Precision', 'Subtype': c_name, 'Estimate': p_pt, 'CI_Lower': p_lo, 'CI_Upper': p_hi})
        records.append({'Metric': 'F1', 'Subtype': c_name, 'Estimate': f_pt, 'CI_Lower': f_lo, 'CI_Upper': f_hi})

    return pd.DataFrame(records)


per_class_bootstrap_ci = bootstrap_class_report


def benjamini_hochberg(pvals):
    """Apply Benjamini-Hochberg FDR adjustment while preserving NaN positions."""
    p = np.asarray(pvals, dtype=np.float64)
    out = np.full(p.shape, np.nan, dtype=np.float64)
    valid = ~np.isnan(p)
    if np.any(valid):
        out[valid] = stats.false_discovery_control(p[valid], method='bh')
    return out


def compute_multiclass_roc_pr(y_true, y_proba, class_labels=None):
    """Compute one-vs-rest macro/weighted ROC-AUC, macro PR-AUC, and per-class curves."""
    y_true_arr = np.asarray(y_true).ravel()
    proba = np.asarray(y_proba, dtype=np.float64)
    n_classes = proba.shape[1]
    classes = np.arange(n_classes) if class_labels is None else np.asarray(class_labels)

    y_bin = label_binarize(y_true_arr, classes=classes)
    if n_classes == 2 and y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    per_class = []
    valid_roc = []
    valid_pr = []

    for c_idx in range(n_classes):
        y_c = y_bin[:, c_idx]
        s_c = proba[:, c_idx]
        n_pos = int(y_c.sum())
        if 0 < n_pos < len(y_c):
            fpr, tpr, _ = roc_curve(y_c, s_c)
            c_roc = float(auc(fpr, tpr))
            c_pr = float(average_precision_score(y_c, s_c))
            valid_roc.append((c_roc, n_pos))
            valid_pr.append(c_pr)
        else:
            fpr = np.array([0.0, 1.0])
            tpr = np.array([0.0, 1.0])
            c_roc = float('nan')
            c_pr = float('nan')

        per_class.append({
            'class_idx': c_idx,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': c_roc,
            'pr_auc': c_pr,
        })

    roc_auc_macro = float(np.mean([v for v, _ in valid_roc])) if valid_roc else float('nan')
    total_support = sum(w for _, w in valid_roc)
    roc_auc_weighted = (
        float(sum(v * w for v, w in valid_roc) / total_support) if total_support > 0 else float('nan')
    )
    pr_auc_macro = float(np.mean(valid_pr)) if valid_pr else float('nan')

    return {
        'roc_auc_macro': roc_auc_macro,
        'roc_auc_weighted': roc_auc_weighted,
        'pr_auc_macro': pr_auc_macro,
        'per_class': per_class,
    }


def kaplan_meier_estimator(times, events):
    """Compute exact right-censored Kaplan-Meier step curve starting at (0.0, 1.0)."""
    t = np.asarray(times, dtype=np.float64).ravel()
    e = np.asarray(events, dtype=np.float64).ravel()
    valid = np.isfinite(t) & np.isfinite(e)
    t = t[valid]
    e = (e[valid] > 0).astype(np.int64)

    if len(t) == 0:
        return np.array([0.0]), np.array([1.0])

    event_times = np.unique(t[e == 1])
    t_curve = [0.0]
    s_curve = [1.0]
    surv = 1.0

    for ti in event_times:
        n_risk = int(np.sum(t >= ti))
        d_i = int(np.sum((t == ti) & (e == 1)))
        if n_risk > 0:
            surv *= 1.0 - (d_i / n_risk)
        t_curve.append(float(ti))
        s_curve.append(float(surv))

    t_max = float(np.max(t))
    if t_max > t_curve[-1]:
        t_curve.append(t_max)
        s_curve.append(s_curve[-1])

    return np.asarray(t_curve, dtype=np.float64), np.asarray(s_curve, dtype=np.float64)


def logrank_test_multigroup(times, events, groups):
    """Exact Mantel-Cox G-group log-rank test using hypergeometric variance-covariance."""
    t_raw = np.asarray(times, dtype=np.float64).ravel()
    e_raw = np.asarray(events, dtype=np.float64).ravel()
    g_raw = np.asarray(groups).ravel()

    valid = np.isfinite(t_raw) & np.isfinite(e_raw) & pd.notna(g_raw)
    t = t_raw[valid]
    e = (e_raw[valid] > 0).astype(np.int64)
    g = g_raw[valid]

    unique_groups = np.unique(g)
    G = len(unique_groups)
    group_counts = {grp: int(np.sum(g == grp)) for grp in unique_groups}
    group_events = {grp: int(np.sum((g == grp) & (e == 1))) for grp in unique_groups}

    if G < 2 or int(np.sum(e == 1)) == 0:
        return {
            'chi2': 0.0,
            'df': max(0, G - 1),
            'p_value': 1.0,
            'group_counts': group_counts,
            'group_events': group_events,
        }

    group_masks = [g == grp for grp in unique_groups]
    event_times = np.unique(t[e == 1])

    observed = np.zeros(G, dtype=np.float64)
    expected = np.zeros(G, dtype=np.float64)
    V = np.zeros((G, G), dtype=np.float64)

    for ti in event_times:
        at_risk = t >= ti
        is_event = (t == ti) & (e == 1)
        n_g = np.array([np.sum(at_risk & m) for m in group_masks], dtype=np.float64)
        d_g = np.array([np.sum(is_event & m) for m in group_masks], dtype=np.float64)
        n_j = float(np.sum(n_g))
        d_j = float(np.sum(d_g))

        if n_j <= 0.0:
            continue

        p_j = n_g / n_j
        observed += d_g
        expected += d_j * p_j

        if n_j > 1.0:
            factor = d_j * (n_j - d_j) / (n_j - 1.0)
            V += factor * (np.diag(p_j) - np.outer(p_j, p_j))

    U = observed - expected
    u_red = U[:-1]
    v_red = V[:-1, :-1]
    chi2_stat = float(max(0.0, u_red @ np.linalg.pinv(v_red) @ u_red))
    df = G - 1
    p_value = float(stats.chi2.sf(chi2_stat, df))

    return {
        'chi2': chi2_stat,
        'df': df,
        'p_value': p_value,
        'group_counts': group_counts,
        'group_events': group_events,
    }


def jaccard_similarity(set_a, set_b):
    """Calculate Jaccard index between two feature sets."""
    s_a = set(set_a)
    s_b = set(set_b)
    intersection = len(s_a.intersection(s_b))
    union = len(s_a.union(s_b))
    return intersection / union if union > 0 else 0.0


def _save_and_maybe_close(fig, save_path, close_fig):
    if save_path:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    if close_fig:
        plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, target_names, title, save_path=None, cmap='Blues', close_fig=False):
    """Plot a confusion matrix on an explicit Figure and Axes."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        np.asarray(y_true).ravel(),
        np.asarray(y_pred).ravel(),
        display_labels=target_names,
        cmap=cmap,
        colorbar=False,
        ax=ax,
    )
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    plt.tight_layout()
    _save_and_maybe_close(fig, save_path, close_fig)
    return fig, ax


def plot_multiclass_roc(y_true, y_proba, target_names, title, save_path=None, close_fig=False):
    """Plot one-vs-rest multi-class ROC curves with macro and weighted AUC summary."""
    metrics = compute_multiclass_roc_pr(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for entry in metrics['per_class']:
        idx = entry['class_idx']
        name = target_names[idx] if idx < len(target_names) else f'Class {idx}'
        auc_str = f"{entry['roc_auc']:.3f}" if not np.isnan(entry['roc_auc']) else 'N/A'
        ax.plot(entry['fpr'], entry['tpr'], linewidth=2.0, label=f"{name} (AUC = {auc_str})")

    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle='--', color='#64748b', linewidth=1.2, label='Chance (AUC = 0.500)')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)

    macro_str = f"{metrics['roc_auc_macro']:.3f}" if not np.isnan(metrics['roc_auc_macro']) else 'N/A'
    weighted_str = f"{metrics['roc_auc_weighted']:.3f}" if not np.isnan(metrics['roc_auc_weighted']) else 'N/A'
    ax.set_title(f"{title}\nMacro AUC = {macro_str} | Weighted AUC = {weighted_str}", fontsize=11, fontweight='bold', pad=10)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='lower right', fontsize=9, frameon=True)
    plt.tight_layout()
    _save_and_maybe_close(fig, save_path, close_fig)
    return fig, ax


def plot_kaplan_meier(times, events, groups, group_labels, title, save_path=None, close_fig=False):
    """Plot multi-group Kaplan-Meier survival curves annotated with Mantel-Cox log-rank p-value."""
    t_arr = np.asarray(times, dtype=np.float64).ravel()
    e_arr = np.asarray(events, dtype=np.float64).ravel()
    g_arr = np.asarray(groups).ravel()

    valid = np.isfinite(t_arr) & np.isfinite(e_arr) & pd.notna(g_arr)
    t_arr = t_arr[valid]
    e_arr = (e_arr[valid] > 0).astype(np.int64)
    g_arr = g_arr[valid]

    lr = logrank_test_multigroup(t_arr, e_arr, g_arr)
    unique_groups = np.unique(g_arr)

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    for idx, grp in enumerate(unique_groups):
        if isinstance(group_labels, dict):
            label = group_labels.get(grp, str(grp))
        elif isinstance(grp, (int, np.integer)) and 0 <= int(grp) < len(group_labels):
            label = group_labels[int(grp)]
        elif group_labels is not None and idx < len(group_labels):
            label = group_labels[idx]
        else:
            label = str(grp)

        mask = g_arr == grp
        t_curve, s_curve = kaplan_meier_estimator(t_arr[mask], e_arr[mask])
        n_g = lr['group_counts'].get(grp, int(np.sum(mask)))
        e_g = lr['group_events'].get(grp, int(np.sum(e_arr[mask])))
        ax.step(t_curve, s_curve, where='post', linewidth=2.0, label=f"{label} (n={n_g}, events={e_g})")

    p_str = f"{lr['p_value']:.2e}" if lr['p_value'] < 1e-3 else f"{lr['p_value']:.4f}"
    ax.text(
        0.03,
        0.05,
        f"Log-rank $\\chi^2$ = {lr['chi2']:.2f} (df={lr['df']})\np = {p_str}",
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment='bottom',
        bbox={'boxstyle': 'round,pad=0.35', 'facecolor': 'white', 'edgecolor': '#cbd5e1', 'alpha': 0.9},
    )

    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Survival Probability', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', fontsize=9, frameon=True)
    plt.tight_layout()
    _save_and_maybe_close(fig, save_path, close_fig)
    return fig, ax


def plot_ppi_network(edges_df, node_modalities, title, save_path=None, close_fig=False):
    """Plot a protein-protein interaction network colored by omics modality and sized by degree."""
    modality_colors = {
        'RNA': '#2563eb',
        'Methylation': '#d97706',
        'Shared': '#7c3aed',
        'Both': '#7c3aed',
    }
    default_color = '#64748b'
    mod_map = dict(node_modalities) if node_modalities is not None else {}

    fig, ax = plt.subplots(figsize=(8.0, 6.5))

    if edges_df is None or len(edges_df) == 0:
        ax.text(
            0.5,
            0.5,
            'No PPI interactions available to display.',
            ha='center',
            va='center',
            fontsize=11,
            color='#475569',
            transform=ax.transAxes,
        )
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        ax.set_axis_off()
        plt.tight_layout()
        _save_and_maybe_close(fig, save_path, close_fig)
        return fig, ax

    cols = list(edges_df.columns)
    src_col, dst_col = None, None
    for s_cand, d_cand in (
        ('source', 'target'),
        ('preferredName_A', 'preferredName_B'),
        ('protein1', 'protein2'),
        ('gene_a', 'gene_b'),
    ):
        if s_cand in cols and d_cand in cols:
            src_col, dst_col = s_cand, d_cand
            break
    if src_col is None:
        src_col, dst_col = cols[0], cols[1]

    weight_col = next((c for c in ('score', 'combined_score', 'weight') if c in cols), None)

    G = nx.Graph()
    for _, row in edges_df.iterrows():
        u = str(row[src_col])
        v = str(row[dst_col])
        if u == v:
            continue
        w = float(row[weight_col]) if weight_col is not None and pd.notna(row[weight_col]) else 1.0
        G.add_edge(u, v, weight=w)

    if G.number_of_nodes() == 0:
        ax.text(
            0.5,
            0.5,
            'No non-self PPI edges available to display.',
            ha='center',
            va='center',
            fontsize=11,
            color='#475569',
            transform=ax.transAxes,
        )
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        ax.set_axis_off()
        plt.tight_layout()
        _save_and_maybe_close(fig, save_path, close_fig)
        return fig, ax

    pos = nx.spring_layout(G, seed=42, k=1.2 / np.sqrt(max(G.number_of_nodes(), 1)))
    degrees = dict(G.degree())
    nodes = list(G.nodes())
    node_sizes = [320 + 140 * degrees[n] for n in nodes]
    node_colors = [modality_colors.get(mod_map.get(n, ''), default_color) for n in nodes]

    raw_weights = np.array([G[u][v].get('weight', 1.0) for u, v in G.edges()], dtype=np.float64)
    if len(raw_weights) > 0 and np.max(raw_weights) > np.min(raw_weights):
        w_norm = (raw_weights - np.min(raw_weights)) / (np.max(raw_weights) - np.min(raw_weights))
        edge_widths = 1.0 + 2.5 * w_norm
    else:
        edge_widths = np.full(len(raw_weights), 1.5)

    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, alpha=0.55, edge_color='#94a3b8')
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=nodes,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors='white',
        linewidths=1.2,
        ax=ax,
    )
    nx.draw_networkx_labels(G, pos, font_size=8.5, font_weight='bold', ax=ax)

    present_mods = {mod_map.get(n, 'Other') for n in nodes}
    legend_handles = []
    for mod_label in ('RNA', 'Methylation', 'Shared', 'Both'):
        if mod_label in present_mods:
            legend_handles.append(
                Line2D([0], [0], marker='o', color='w', label=mod_label,
                       markerfacecolor=modality_colors[mod_label], markersize=9)
            )
    if any(m not in modality_colors for m in present_mods):
        legend_handles.append(
            Line2D([0], [0], marker='o', color='w', label='Other',
                   markerfacecolor=default_color, markersize=9)
        )
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper left', fontsize=9, frameon=True, title='Modality')

    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.set_axis_off()
    plt.tight_layout()
    _save_and_maybe_close(fig, save_path, close_fig)
    return fig, ax
