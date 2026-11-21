"""
End-to-end Breast-Cancer-MultiOmics execution script synchronized with notebooks 1-6.

Pipeline specification:
- Stratified 80/20 split (split_seed=100), raw un-imputed train/test splits saved for leak-free CV
- Training-fit median imputation (SimpleImputer(strategy='median', keep_empty_features=True))
- Unscaled bottom-20% variance filtering per modality -> MinMaxScaler(clip=True)
- Leak-free 5-fold CV sensitivity curves for ANOVA, ReliefF, and mRMR-MID
- Full-chain 5-fold feature stability (Jaccard) and 7-configuration leak-free ablation
- True in-fold repeated nested CV (5x2) across 4 canonical classifiers x 2 imbalance strategies
- Single-pass held-out test evaluation (N=110) with bootstrap CIs, multi-class ROC/PR,
  permutation feature importance, cis RNA-Methylation correlation, tumor purity confounder audit,
  Kaplan-Meier overall survival stratification, and summary of archived biological database validations.
"""

import os
import sys
import time
import json
import warnings
from collections import Counter
import numpy as np
import pandas as pd
from scipy import stats
import networkx as nx
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.utils import (
    MODEL_SPECS,
    ensure_directories,
    save_object,
    reliefF_multiclass,
    mrmr_selection,
    fit_transform_omics_fold,
    tune_models_grid_search,
    bootstrap_ci,
    bootstrap_class_report,
    benjamini_hochberg,
    compute_multiclass_roc_pr,
    logrank_test_multigroup,
    jaccard_similarity,
    plot_confusion_matrix,
    plot_multiclass_roc,
    plot_kaplan_meier,
    plot_ppi_network,
)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class Logger:
    """Tee stdout to both terminal and pipeline log file."""

    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def run_pipeline(data_dir='data', output_dir='outputs', split_seed=100, cv_seed=42):
    t_start = time.perf_counter()
    ensure_directories(output_dir)
    results_dir = os.path.join(output_dir, 'results')
    models_dir = os.path.join(output_dir, 'models')

    log_path = os.path.join(output_dir, 'logs', 'pipeline_log.txt')
    logger = Logger(log_path)
    orig_stdout = sys.stdout
    sys.stdout = logger

    try:
        print('=' * 80)
        print('BREAST CANCER SUBTYPE PREDICTION VIA MULTI-OMICS INTEGRATION')
        print('=' * 80)

        print('\n[1/7] Reading and data harmonization')
        t0 = time.perf_counter()
        clinical_file = os.path.join(
            data_dir, 'Human_TCGA_BRCA_MS_Clinical_Clinical_01_28_2016_BI_Clinical_Firehose.tsi'
        )
        rna_file = os.path.join(
            data_dir, 'Human_TCGA_BRCA_UNC_RNAseq_HiSeq_RNA_01_28_2016_BI_Gene_Firehose.gz'
        )
        meth_file = os.path.join(
            data_dir, 'Human_TCGA_BRCA_JHU_USC_Methylation_Meth450_01_28_2016_BI_Gene_Firehose.gz'
        )

        for path in (clinical_file, rna_file, meth_file):
            if not os.path.exists(path):
                raise FileNotFoundError(f'Input file not found: {path}')

        clinical = pd.read_csv(clinical_file, sep='\t', index_col=0).T
        rna = pd.read_csv(rna_file, sep='\t', index_col=0).T
        meth = pd.read_csv(meth_file, sep='\t', index_col=0).T

        label_mapping = {'LumA': 0, 'LumB': 1, 'Her2': 2, 'Basal': 3}
        target_names = [k for k, _ in sorted(label_mapping.items(), key=lambda x: x[1])]

        clinical_clean = clinical.dropna(subset=['PAM50']).copy()
        clinical_clean = clinical_clean[clinical_clean['PAM50'].isin(label_mapping)].copy()
        clinical_clean['label'] = clinical_clean['PAM50'].map(label_mapping).astype(int)

        assert not clinical_clean.index.duplicated().any(), 'Duplicate IDs in clinical table'
        assert not rna.index.duplicated().any(), 'Duplicate IDs in RNA table'
        assert not meth.index.duplicated().any(), 'Duplicate IDs in Methylation table'

        common_patients = clinical_clean.index.intersection(rna.index).intersection(meth.index)
        X_rna_raw = rna.loc[common_patients].apply(pd.to_numeric, errors='coerce').astype(np.float32)
        X_meth_raw = meth.loc[common_patients].apply(pd.to_numeric, errors='coerce').astype(np.float32)
        y_raw = clinical_clean.loc[common_patients, 'label'].astype(int)

        rna_feature_names = list(X_rna_raw.columns)
        meth_feature_names = list(X_meth_raw.columns)

        clinical_metadata = clinical_clean.loc[common_patients].copy()
        for col in ('overall_survival', 'status', 'Tumor_purity', 'years_to_birth'):
            if col in clinical_metadata.columns:
                clinical_metadata[col] = pd.to_numeric(
                    clinical_metadata[col].astype(str).str.strip(), errors='coerce'
                )
        clinical_metadata['overall_survival_years'] = clinical_metadata['overall_survival'] / 365.25
        clinical_to_save = clinical_metadata.copy()
        for col in clinical_to_save.columns:
            if clinical_to_save[col].dtype == 'object' or str(clinical_to_save[col].dtype) == 'str':
                clinical_to_save[col] = clinical_to_save[col].astype('string')

        save_object(X_rna_raw, 'X_rna_raw', output_dir)
        save_object(X_meth_raw, 'X_meth_raw', output_dir)
        save_object(y_raw, 'y_labels', output_dir)
        save_object(list(common_patients), 'patient_ids', output_dir)
        save_object(rna_feature_names, 'rna_feature_names', output_dir)
        save_object(meth_feature_names, 'meth_feature_names', output_dir)
        save_object(clinical_to_save, 'clinical_metadata', output_dir)

        print(
            f'Synchronized N={len(common_patients)} | RNA={X_rna_raw.shape} | '
            f'Meth={X_meth_raw.shape} ({time.perf_counter() - t0:.2f}s)'
        )

        print('\n[2/7] Stratified split and median imputation')
        t0 = time.perf_counter()
        (
            X_train_rna_df,
            X_test_rna_df,
            X_train_meth_df,
            X_test_meth_df,
            y_train,
            y_test,
        ) = train_test_split(
            X_rna_raw,
            X_meth_raw,
            y_raw,
            test_size=0.2,
            stratify=y_raw,
            random_state=split_seed,
        )
        train_patient_ids = list(X_train_rna_df.index)
        test_patient_ids = list(X_test_rna_df.index)
        y_train_arr = np.asarray(y_train).ravel()
        y_test_arr = np.asarray(y_test).ravel()

        X_train_rna_raw = X_train_rna_df.to_numpy(dtype=np.float64)
        X_test_rna_raw = X_test_rna_df.to_numpy(dtype=np.float64)
        X_train_meth_raw = X_train_meth_df.to_numpy(dtype=np.float64)
        X_test_meth_raw = X_test_meth_df.to_numpy(dtype=np.float64)

        save_object(X_train_rna_raw, 'X_train_rna_raw', output_dir)
        save_object(X_test_rna_raw, 'X_test_rna_raw', output_dir)
        save_object(X_train_meth_raw, 'X_train_meth_raw', output_dir)
        save_object(X_test_meth_raw, 'X_test_meth_raw', output_dir)

        rna_imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        X_train_rna_imp = rna_imputer.fit_transform(X_train_rna_raw)
        X_test_rna_imp = rna_imputer.transform(X_test_rna_raw)
        assert X_train_rna_imp.shape == X_train_rna_raw.shape

        meth_imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        X_train_meth_imp = meth_imputer.fit_transform(X_train_meth_raw)
        X_test_meth_imp = meth_imputer.transform(X_test_meth_raw)
        assert X_train_meth_imp.shape == X_train_meth_raw.shape

        save_object(X_train_rna_imp, 'X_train_rna_imp', output_dir)
        save_object(X_test_rna_imp, 'X_test_rna_imp', output_dir)
        save_object(X_train_meth_imp, 'X_train_meth_imp', output_dir)
        save_object(X_test_meth_imp, 'X_test_meth_imp', output_dir)
        save_object(y_train, 'y_train', output_dir)
        save_object(y_test, 'y_test', output_dir)
        save_object(train_patient_ids, 'train_patient_ids', output_dir)
        save_object(test_patient_ids, 'test_patient_ids', output_dir)
        print(f'Train={len(y_train_arr)} | Test={len(y_test_arr)} ({time.perf_counter() - t0:.2f}s)')

        print('\n[3/7] Unscaled bottom-20% variance filtering and clipped MinMax scaling')
        t0 = time.perf_counter()
        var_rna = np.var(X_train_rna_imp, axis=0)
        thresh_rna = float(np.percentile(var_rna, 20))
        vt_rna = VarianceThreshold(threshold=thresh_rna)
        X_tr_rna_vraw = vt_rna.fit_transform(X_train_rna_imp)
        X_te_rna_vraw = vt_rna.transform(X_test_rna_imp)
        rna_var_indices = vt_rna.get_support(indices=True)
        rna_var_names = [rna_feature_names[i] for i in rna_var_indices]

        scaler_rna = MinMaxScaler(clip=True)
        X_train_rna_var = scaler_rna.fit_transform(X_tr_rna_vraw)
        X_test_rna_var = scaler_rna.transform(X_te_rna_vraw)

        var_meth = np.var(X_train_meth_imp, axis=0)
        thresh_meth = float(np.percentile(var_meth, 20))
        vt_meth = VarianceThreshold(threshold=thresh_meth)
        X_tr_meth_vraw = vt_meth.fit_transform(X_train_meth_imp)
        X_te_meth_vraw = vt_meth.transform(X_test_meth_imp)
        meth_var_indices = vt_meth.get_support(indices=True)
        meth_var_names = [meth_feature_names[i] for i in meth_var_indices]

        scaler_meth = MinMaxScaler(clip=True)
        X_train_meth_var = scaler_meth.fit_transform(X_tr_meth_vraw)
        X_test_meth_var = scaler_meth.transform(X_te_meth_vraw)

        assert X_test_rna_var.min() >= 0.0 and X_test_rna_var.max() <= 1.0
        assert X_test_meth_var.min() >= 0.0 and X_test_meth_var.max() <= 1.0

        save_object(X_train_rna_var, 'X_train_rna_var', output_dir)
        save_object(X_test_rna_var, 'X_test_rna_var', output_dir)
        save_object(rna_var_indices, 'feat_indices_rna_var', output_dir)
        save_object(rna_var_names, 'feat_names_rna_var', output_dir)
        save_object(X_train_meth_var, 'X_train_meth_var', output_dir)
        save_object(X_test_meth_var, 'X_test_meth_var', output_dir)
        save_object(meth_var_indices, 'feat_indices_meth_var', output_dir)
        save_object(meth_var_names, 'feat_names_meth_var', output_dir)
        print(
            f'RNA kept {X_train_rna_var.shape[1]} (p20={thresh_rna:.4f}) | '
            f'Meth kept {X_train_meth_var.shape[1]} (p20={thresh_meth:.5f}) ({time.perf_counter() - t0:.2f}s)'
        )

        print('\n[4/7] ANOVA F-test (k=500) and leak-free 5-fold sensitivity')
        t0 = time.perf_counter()
        K_ANOVA = 500
        cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=cv_seed)

        anova_rna = SelectKBest(f_classif, k=K_ANOVA)
        X_train_rna_anova = anova_rna.fit_transform(X_train_rna_var, y_train_arr)
        X_test_rna_anova = anova_rna.transform(X_test_rna_var)
        local_rna_a = anova_rna.get_support(indices=True)
        rna_anova_names = [rna_var_names[i] for i in local_rna_a]
        rna_anova_indices = rna_var_indices[local_rna_a]

        anova_meth = SelectKBest(f_classif, k=K_ANOVA)
        X_train_meth_anova = anova_meth.fit_transform(X_train_meth_var, y_train_arr)
        X_test_meth_anova = anova_meth.transform(X_test_meth_var)
        local_meth_a = anova_meth.get_support(indices=True)
        meth_anova_names = [meth_var_names[i] for i in local_meth_a]
        meth_anova_indices = meth_var_indices[local_meth_a]

        k_anova_grid = [100, 250, 500, 750, 1000]
        anova_accs = {k: [] for k in k_anova_grid}
        anova_baccs = {k: [] for k in k_anova_grid}
        for tr_idx, va_idx in cv5.split(X_train_rna_raw, y_train_arr):
            y_tr, y_va = y_train_arr[tr_idx], y_train_arr[va_idx]
            rf = fit_transform_omics_fold(
                X_train_rna_raw[tr_idx], X_train_rna_raw[va_idx], None, None, y_tr,
                modalities='rna', k_anova=None, use_relief=False, use_mrmr=False,
            )
            mf = fit_transform_omics_fold(
                None, None, X_train_meth_raw[tr_idx], X_train_meth_raw[va_idx], y_tr,
                modalities='meth', k_anova=None, use_relief=False, use_mrmr=False,
            )
            fr, _ = f_classif(rf['X_tr'], y_tr)
            fm, _ = f_classif(mf['X_tr'], y_tr)
            ord_r = np.argsort(np.nan_to_num(fr, nan=-np.inf))[::-1]
            ord_m = np.argsort(np.nan_to_num(fm, nan=-np.inf))[::-1]
            for kv in k_anova_grid:
                Xtr_k = np.hstack((rf['X_tr'][:, ord_r[:kv]], mf['X_tr'][:, ord_m[:kv]]))
                Xva_k = np.hstack((rf['X_va'][:, ord_r[:kv]], mf['X_va'][:, ord_m[:kv]]))
                preds = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Xtr_k, y_tr).predict(Xva_k)
                anova_accs[kv].append(accuracy_score(y_va, preds))
                anova_baccs[kv].append(balanced_accuracy_score(y_va, preds))

        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        ax.errorbar(
            k_anova_grid,
            [np.mean(anova_accs[k]) * 100 for k in k_anova_grid],
            yerr=[np.std(anova_accs[k]) * 100 for k in k_anova_grid],
            marker='o', linewidth=2, capsize=4, color='#2563eb', label='Accuracy (+/- 1 SD)',
        )
        ax.plot(
            k_anova_grid,
            [np.mean(anova_baccs[k]) * 100 for k in k_anova_grid],
            marker='s', linewidth=2, linestyle='--', color='#0891b2', label='Balanced accuracy',
        )
        ax.axvline(K_ANOVA, color='#dc2626', linestyle=':', linewidth=1.8, label=f'Selected k = {K_ANOVA}')
        ax.set_title('Leak-free ANOVA sensitivity (5-fold training CV)', fontweight='bold')
        ax.set_xlabel('Top features per modality (k)')
        ax.set_ylabel('Out-of-fold score (%)')
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(results_dir, 'cv_accuracy_vs_k_anova.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

        save_object(X_train_rna_anova, 'X_train_rna_anova', output_dir)
        save_object(X_test_rna_anova, 'X_test_rna_anova', output_dir)
        save_object(rna_anova_indices, 'feat_indices_rna_anova', output_dir)
        save_object(rna_anova_names, 'feat_names_rna_anova', output_dir)
        save_object(X_train_meth_anova, 'X_train_meth_anova', output_dir)
        save_object(X_test_meth_anova, 'X_test_meth_anova', output_dir)
        save_object(meth_anova_indices, 'feat_indices_meth_anova', output_dir)
        save_object(meth_anova_names, 'feat_names_meth_anova', output_dir)
        print(
            f'Leak-free ANOVA k=500 CV Acc={np.mean(anova_accs[K_ANOVA])*100:.2f}% '
            f'({time.perf_counter() - t0:.2f}s)'
        )

        print('\n[5/7] Multi-class ReliefF (k=150) and leak-free 5-fold sensitivity')
        t0 = time.perf_counter()
        K_RELIEF = 150
        top_rna_r, _ = reliefF_multiclass(X_train_rna_anova, y_train_arr, n_neighbors=10, n_features_to_select=K_RELIEF)
        X_train_rna_relief = X_train_rna_anova[:, top_rna_r]
        X_test_rna_relief = X_test_rna_anova[:, top_rna_r]
        rna_relief_names = [rna_anova_names[i] for i in top_rna_r]
        rna_relief_indices = rna_anova_indices[top_rna_r]

        top_meth_r, _ = reliefF_multiclass(X_train_meth_anova, y_train_arr, n_neighbors=10, n_features_to_select=K_RELIEF)
        X_train_meth_relief = X_train_meth_anova[:, top_meth_r]
        X_test_meth_relief = X_test_meth_anova[:, top_meth_r]
        meth_relief_names = [meth_anova_names[i] for i in top_meth_r]
        meth_relief_indices = meth_anova_indices[top_meth_r]

        k_relief_grid = [50, 100, 150, 200, 250]
        rel_accs = {k: [] for k in k_relief_grid}
        rel_baccs = {k: [] for k in k_relief_grid}
        for tr_idx, va_idx in cv5.split(X_train_rna_raw, y_train_arr):
            y_tr, y_va = y_train_arr[tr_idx], y_train_arr[va_idx]
            rf = fit_transform_omics_fold(
                X_train_rna_raw[tr_idx], X_train_rna_raw[va_idx], None, None, y_tr,
                modalities='rna', k_anova=K_ANOVA, use_relief=True, k_relief=max(k_relief_grid), use_mrmr=False,
            )
            mf = fit_transform_omics_fold(
                None, None, X_train_meth_raw[tr_idx], X_train_meth_raw[va_idx], y_tr,
                modalities='meth', k_anova=K_ANOVA, use_relief=True, k_relief=max(k_relief_grid), use_mrmr=False,
            )
            for kv in k_relief_grid:
                Xtr_k = np.hstack((rf['X_tr'][:, :kv], mf['X_tr'][:, :kv]))
                Xva_k = np.hstack((rf['X_va'][:, :kv], mf['X_va'][:, :kv]))
                preds = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Xtr_k, y_tr).predict(Xva_k)
                rel_accs[kv].append(accuracy_score(y_va, preds))
                rel_baccs[kv].append(balanced_accuracy_score(y_va, preds))

        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        ax.errorbar(
            k_relief_grid,
            [np.mean(rel_accs[k]) * 100 for k in k_relief_grid],
            yerr=[np.std(rel_accs[k]) * 100 for k in k_relief_grid],
            marker='s', linewidth=2, capsize=4, color='#16a34a', label='Accuracy (+/- 1 SD)',
        )
        ax.plot(
            k_relief_grid,
            [np.mean(rel_baccs[k]) * 100 for k in k_relief_grid],
            marker='o', linewidth=2, linestyle='--', color='#0891b2', label='Balanced accuracy',
        )
        ax.axvline(K_RELIEF, color='#dc2626', linestyle=':', linewidth=1.8, label=f'Selected k = {K_RELIEF}')
        ax.set_title('Leak-free ReliefF sensitivity (5-fold training CV)', fontweight='bold')
        ax.set_xlabel('Top features per modality (k)')
        ax.set_ylabel('Out-of-fold score (%)')
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(results_dir, 'cv_accuracy_vs_k_relieff.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

        save_object(X_train_rna_relief, 'X_train_rna_relief', output_dir)
        save_object(X_test_rna_relief, 'X_test_rna_relief', output_dir)
        save_object(rna_relief_indices, 'feat_indices_rna_relief', output_dir)
        save_object(rna_relief_names, 'feat_names_rna_relief', output_dir)
        save_object(X_train_meth_relief, 'X_train_meth_relief', output_dir)
        save_object(X_test_meth_relief, 'X_test_meth_relief', output_dir)
        save_object(meth_relief_indices, 'feat_indices_meth_relief', output_dir)
        save_object(meth_relief_names, 'feat_names_meth_relief', output_dir)
        print(
            f'Leak-free ReliefF k=150 CV Acc={np.mean(rel_accs[K_RELIEF])*100:.2f}% '
            f'({time.perf_counter() - t0:.2f}s)'
        )

        print('\n[6/7] mRMR-MID, nested CV, ablation, held-out test, and clinical validation')
        t0 = time.perf_counter()
        K_FINAL = 50
        X_train_combined = np.hstack((X_train_rna_relief, X_train_meth_relief))
        X_test_combined = np.hstack((X_test_rna_relief, X_test_meth_relief))
        comb_feat_names = [f'RNA_{g}' for g in rna_relief_names] + [f'Meth_{g}' for g in meth_relief_names]
        comb_feat_types = ['RNA'] * len(rna_relief_names) + ['Methylation'] * len(meth_relief_names)
        comb_raw_names = list(rna_relief_names) + list(meth_relief_names)

        final_idx, final_raw_mi, final_norm_mi = mrmr_selection(
            X_train_combined, y_train_arr, k=K_FINAL, method='mid', random_state=cv_seed
        )
        X_train_final = X_train_combined[:, final_idx]
        X_test_final = X_test_combined[:, final_idx]
        final_feature_names = [comb_feat_names[i] for i in final_idx]
        final_types = [comb_feat_types[i] for i in final_idx]
        final_symbols = [comb_raw_names[i] for i in final_idx]

        features_df = pd.DataFrame({
            'Rank': np.arange(1, K_FINAL + 1),
            'Feature_Name': final_feature_names,
            'Modality': final_types,
            'Gene_or_Probe': final_symbols,
            'Mutual_Info': final_raw_mi,
            'Normalized_MI': final_norm_mi,
        })
        features_df.to_csv(os.path.join(output_dir, 'final_50_features.csv'), index=False)
        n_final_rna = int((features_df['Modality'] == 'RNA').sum())
        n_final_meth = int((features_df['Modality'] == 'Methylation').sum())
        print(f'Final 50 mRMR-MID biomarkers: {n_final_rna} RNA + {n_final_meth} Methylation')

        k_mrmr_grid = [10, 20, 30, 40, 50, 60, 75, 100]
        mrmr_accs = {k: [] for k in k_mrmr_grid}
        mrmr_baccs = {k: [] for k in k_mrmr_grid}
        fold_top50_sets = []
        selection_counter = Counter()
        for tr_idx, va_idx in cv5.split(X_train_rna_raw, y_train_arr):
            y_tr, y_va = y_train_arr[tr_idx], y_train_arr[va_idx]
            fold_res = fit_transform_omics_fold(
                X_train_rna_raw[tr_idx], X_train_rna_raw[va_idx],
                X_train_meth_raw[tr_idx], X_train_meth_raw[va_idx],
                y_tr, modalities='both', k_anova=K_ANOVA, use_relief=True, k_relief=K_RELIEF,
                use_mrmr=True, k_mrmr=max(k_mrmr_grid), mrmr_method='mid', random_state=cv_seed,
                rna_names=rna_feature_names, meth_names=meth_feature_names,
            )
            top50 = fold_res['feature_names'][:K_FINAL]
            fold_top50_sets.append(set(top50))
            selection_counter.update(top50)
            for kv in k_mrmr_grid:
                pipe = ImbPipeline([
                    ('smote', SMOTE(random_state=cv_seed)),
                    ('clf', SVC(kernel='linear', C=1.0, random_state=cv_seed)),
                ])
                pipe.fit(fold_res['X_tr'][:, :kv], y_tr)
                preds = pipe.predict(fold_res['X_va'][:, :kv])
                mrmr_accs[kv].append(accuracy_score(y_va, preds))
                mrmr_baccs[kv].append(balanced_accuracy_score(y_va, preds))

        jaccard_pairs = [
            jaccard_similarity(fold_top50_sets[i], fold_top50_sets[j])
            for i in range(len(fold_top50_sets))
            for j in range(i + 1, len(fold_top50_sets))
        ]
        print(f'Full-chain 5-fold Jaccard stability: {np.mean(jaccard_pairs):.4f} +/- {np.std(jaccard_pairs):.4f}')

        final_set = set(final_feature_names)
        stability_df = pd.DataFrame([
            {
                'Feature_Name': feat,
                'Modality': 'RNA' if feat.startswith('RNA_') else 'Methylation',
                'Gene_or_Probe': feat.split('_', 1)[1],
                'Folds_Selected': cnt,
                'Selection_Frequency': cnt / 5.0,
                'In_Final_50': feat in final_set,
            }
            for feat, cnt in selection_counter.most_common()
        ]).sort_values(['Folds_Selected', 'In_Final_50', 'Feature_Name'], ascending=[False, False, True]).reset_index(drop=True)
        stability_df.to_csv(os.path.join(results_dir, 'feature_stability_frequency.csv'), index=False)

        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.errorbar(
            k_mrmr_grid,
            [np.mean(mrmr_accs[k]) * 100 for k in k_mrmr_grid],
            yerr=[np.std(mrmr_accs[k]) * 100 for k in k_mrmr_grid],
            marker='^', linewidth=2, capsize=4, color='#dc2626', label='Accuracy (+/- 1 SD)',
        )
        ax.plot(
            k_mrmr_grid,
            [np.mean(mrmr_baccs[k]) * 100 for k in k_mrmr_grid],
            marker='o', linewidth=2, linestyle='--', color='#2563eb', label='Balanced accuracy',
        )
        ax.axvline(K_FINAL, color='#111827', linestyle=':', linewidth=1.8, label=f'Selected k = {K_FINAL}')
        ax.set_title('Leak-free mRMR-MID sensitivity (5-fold full-chain CV)', fontweight='bold')
        ax.set_xlabel('Final selected features (k)')
        ax.set_ylabel('Out-of-fold score (%)')
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(results_dir, 'cv_accuracy_vs_k_mrmr.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

        rep_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=cv_seed)
        model_names = list(MODEL_SPECS.keys())
        strategy_map = [('SMOTE', 'smote'), ('Class_Weight_Balanced', 'class_weight')]
        nested_scores = {
            (m, sl): {'acc': [], 'bal_acc': [], 'f1_macro': [], 'roc_auc_macro': []}
            for m in model_names
            for sl, _ in strategy_map
        }

        for fold_num, (tr_idx, va_idx) in enumerate(rep_cv.split(X_train_rna_raw, y_train_arr), start=1):
            y_tr, y_va = y_train_arr[tr_idx], y_train_arr[va_idx]
            fold_res = fit_transform_omics_fold(
                X_train_rna_raw[tr_idx], X_train_rna_raw[va_idx],
                X_train_meth_raw[tr_idx], X_train_meth_raw[va_idx],
                y_tr, modalities='both', k_anova=K_ANOVA, use_relief=True, k_relief=K_RELIEF,
                use_mrmr=True, k_mrmr=K_FINAL, mrmr_method='mid', random_state=cv_seed,
            )
            for s_label, s_key in strategy_map:
                tuned = tune_models_grid_search(
                    fold_res['X_tr'], y_tr, cv=3, random_state=cv_seed, resampling=s_key, n_jobs=-1
                )
                for m_name in model_names:
                    clf = tuned[m_name]
                    preds = clf.predict(fold_res['X_va'])
                    proba = clf.predict_proba(fold_res['X_va'])
                    roc_info = compute_multiclass_roc_pr(y_va, proba)
                    b = nested_scores[(m_name, s_label)]
                    b['acc'].append(accuracy_score(y_va, preds))
                    b['bal_acc'].append(balanced_accuracy_score(y_va, preds))
                    b['f1_macro'].append(f1_score(y_va, preds, average='macro'))
                    b['roc_auc_macro'].append(roc_info['roc_auc_macro'])

        nested_rows = []
        for s_label, s_key in strategy_map:
            for m_name in model_names:
                b = nested_scores[(m_name, s_label)]
                nested_rows.append({
                    'Model': m_name,
                    'Strategy': s_label,
                    'Resampling_Key': s_key,
                    'CV_Accuracy_Mean': float(np.mean(b['acc'])),
                    'CV_Accuracy_Std': float(np.std(b['acc'])),
                    'CV_Balanced_Accuracy': float(np.mean(b['bal_acc'])),
                    'CV_Balanced_Accuracy_Std': float(np.std(b['bal_acc'])),
                    'CV_Macro_F1': float(np.mean(b['f1_macro'])),
                    'CV_Macro_ROC_AUC': float(np.mean(b['roc_auc_macro'])),
                })

        nested_cv_df = pd.DataFrame(nested_rows).sort_values(
            ['CV_Balanced_Accuracy', 'CV_Macro_F1'], ascending=False
        ).reset_index(drop=True)
        nested_cv_df.to_csv(os.path.join(results_dir, 'nested_cv_summary.csv'), index=False)
        print('\nNested CV summary (10 outer folds):')
        print(nested_cv_df.to_string(index=False))

        best_row = nested_cv_df.iloc[0]
        best_model_name = str(best_row['Model'])
        best_strategy_label = str(best_row['Strategy'])
        best_strategy_key = str(best_row['Resampling_Key'])
        print(f'Winning CV config: {best_model_name} + {best_strategy_label} (BalAcc={best_row["CV_Balanced_Accuracy"]*100:.2f}%)')

        ablation_configs = [
            ('RNA-Seq Only (Full Chain, mRMR-MID 50)', dict(modalities='rna', k_anova=500, use_relief=True, k_relief=150, use_mrmr=True, k_mrmr=50, mrmr_method='mid')),
            ('Methylation Only (Full Chain, mRMR-MID 50)', dict(modalities='meth', k_anova=500, use_relief=True, k_relief=150, use_mrmr=True, k_mrmr=50, mrmr_method='mid')),
            ('Multi-Omics: Full Pipeline (ANOVA -> ReliefF -> mRMR-MID 50)', dict(modalities='both', k_anova=500, use_relief=True, k_relief=150, use_mrmr=True, k_mrmr=50, mrmr_method='mid')),
            ('Multi-Omics: Full Pipeline (ANOVA -> ReliefF -> mRMR-mi_corr 50)', dict(modalities='both', k_anova=500, use_relief=True, k_relief=150, use_mrmr=True, k_mrmr=50, mrmr_method='mi_corr')),
            ('Multi-Omics: Without ReliefF (ANOVA -> mRMR-MID 50)', dict(modalities='both', k_anova=500, use_relief=False, use_mrmr=True, k_mrmr=50, mrmr_method='mid')),
            ('Multi-Omics: Without mRMR (ANOVA -> ReliefF 25+25)', dict(modalities='both', k_anova=500, use_relief=True, k_relief=25, use_mrmr=False)),
            ('Multi-Omics: ANOVA Only (Top 25+25)', dict(modalities='both', k_anova=25, use_relief=False, use_mrmr=False)),
        ]
        ablation_records = []
        for cfg_name, cfg_kwargs in ablation_configs:
            accs, baccs, f1s, n_rnas, n_meths, n_tots = [], [], [], [], [], []
            for tr_idx, va_idx in cv5.split(X_train_rna_raw, y_train_arr):
                y_tr, y_va = y_train_arr[tr_idx], y_train_arr[va_idx]
                fold_res = fit_transform_omics_fold(
                    X_train_rna_raw[tr_idx], X_train_rna_raw[va_idx],
                    X_train_meth_raw[tr_idx], X_train_meth_raw[va_idx],
                    y_tr, random_state=cv_seed, **cfg_kwargs,
                )
                pipe = ImbPipeline([
                    ('smote', SMOTE(random_state=cv_seed)),
                    ('clf', SVC(kernel='linear', C=1.0, random_state=cv_seed)),
                ])
                pipe.fit(fold_res['X_tr'], y_tr)
                preds = pipe.predict(fold_res['X_va'])
                accs.append(accuracy_score(y_va, preds))
                baccs.append(balanced_accuracy_score(y_va, preds))
                f1s.append(f1_score(y_va, preds, average='macro'))
                n_rnas.append(fold_res['n_rna'])
                n_meths.append(fold_res['n_meth'])
                n_tots.append(fold_res['X_tr'].shape[1])

            ablation_records.append({
                'Configuration': cfg_name,
                'Features_RNA': round(float(np.mean(n_rnas)), 1),
                'Features_Meth': round(float(np.mean(n_meths)), 1),
                'Features_Total': int(round(float(np.mean(n_tots)))),
                'CV_Accuracy': float(np.mean(accs)),
                'CV_Balanced_Accuracy': float(np.mean(baccs)),
                'CV_Balanced_Accuracy_Std': float(np.std(baccs)),
                'CV_Macro_F1': float(np.mean(f1s)),
            })

        ablation_df = pd.DataFrame(ablation_records)
        ablation_df.to_csv(os.path.join(results_dir, 'ablation_study.csv'), index=False)
        print('\nLeak-free ablation study:')
        print(ablation_df.to_string(index=False))

        plot_df = ablation_df.sort_values('CV_Balanced_Accuracy', ascending=True)
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        colors = ['#dc2626' if 'mRMR-MID 50)' in c and 'Multi-Omics: Full' in c else '#2563eb' for c in plot_df['Configuration']]
        ax.barh(
            plot_df['Configuration'],
            plot_df['CV_Balanced_Accuracy'] * 100,
            xerr=plot_df['CV_Balanced_Accuracy_Std'] * 100,
            color=colors, alpha=0.88, capsize=3,
        )
        ax.set_xlim(65, 102)
        ax.set_xlabel('Leak-free 5-fold CV balanced accuracy (%)')
        ax.set_title('Systematic leak-free ablation study (Fixed Linear SVM + SMOTE)', fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(results_dir, 'ablation_study.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

        final_models, final_best_params = tune_models_grid_search(
            X_train_final, y_train_arr, cv=3, random_state=cv_seed,
            resampling=best_strategy_key, return_params=True, n_jobs=-1,
        )

        eval_records = []
        per_class_frames = []
        test_predictions = {}
        test_probas = {}

        print(f'\nHeld-out test evaluation (N={len(y_test_arr)}, strategy={best_strategy_label}):')
        for m_name in model_names:
            clf = final_models[m_name]
            y_pred = clf.predict(X_test_final)
            y_proba = clf.predict_proba(X_test_final)
            test_predictions[m_name] = y_pred
            test_probas[m_name] = y_proba

            acc = accuracy_score(y_test_arr, y_pred)
            bal_acc = balanced_accuracy_score(y_test_arr, y_pred)
            f1_mac = f1_score(y_test_arr, y_pred, average='macro')
            f1_wt = f1_score(y_test_arr, y_pred, average='weighted')

            _, acc_lo, acc_hi = bootstrap_ci(y_test_arr, y_pred, metric_func=accuracy_score, n_bootstraps=1000, ci=95, random_state=cv_seed)
            _, f1_lo, f1_hi = bootstrap_ci(
                y_test_arr, y_pred, metric_func=lambda yt, yp: f1_score(yt, yp, average='macro'),
                n_bootstraps=1000, ci=95, random_state=cv_seed,
            )
            roc_pr = compute_multiclass_roc_pr(y_test_arr, y_proba)

            class_ci_df = bootstrap_class_report(y_test_arr, y_pred, target_names, ci=95, n_bootstraps=1000, random_state=cv_seed)
            class_ci_df['Model'] = m_name
            per_class_frames.append(class_ci_df)

            spec = MODEL_SPECS[m_name]
            save_object(clf, spec['model_file'], models_dir)
            plot_confusion_matrix(
                y_test_arr, y_pred, target_names,
                f'{m_name} held-out test (Acc={acc*100:.1f}%, BalAcc={bal_acc*100:.1f}%)',
                save_path=os.path.join(results_dir, spec['cm_file']),
                close_fig=True,
            )

            cv_match = nested_cv_df[
                (nested_cv_df['Model'] == m_name) & (nested_cv_df['Strategy'] == best_strategy_label)
            ].iloc[0]

            eval_records.append({
                'Model': m_name,
                'Strategy': best_strategy_label,
                'CV_Accuracy_Mean': float(cv_match['CV_Accuracy_Mean']),
                'CV_Accuracy_Std': float(cv_match['CV_Accuracy_Std']),
                'CV_Balanced_Accuracy': float(cv_match['CV_Balanced_Accuracy']),
                'CV_Macro_F1': float(cv_match['CV_Macro_F1']),
                'CV_Macro_ROC_AUC': float(cv_match['CV_Macro_ROC_AUC']),
                'Test_Accuracy': float(acc),
                'Test_Accuracy_95CI_Lower': float(acc_lo),
                'Test_Accuracy_95CI_Upper': float(acc_hi),
                'Test_Balanced_Accuracy': float(bal_acc),
                'Test_Macro_F1': float(f1_mac),
                'Test_Macro_F1_95CI_Lower': float(f1_lo),
                'Test_Macro_F1_95CI_Upper': float(f1_hi),
                'Test_Weighted_F1': float(f1_wt),
                'Test_ROC_AUC_Macro': float(roc_pr['roc_auc_macro']),
                'Test_ROC_AUC_Weighted': float(roc_pr['roc_auc_weighted']),
                'Test_PR_AUC_Macro': float(roc_pr['pr_auc_macro']),
                'Test_Samples': int(len(y_test_arr)),
                'Selected_Features': K_FINAL,
            })

            print(
                f'  {m_name:24s} | Acc={acc*100:.2f}% [{acc_lo*100:.2f}-{acc_hi*100:.2f}%] | '
                f'BalAcc={bal_acc*100:.2f}% | MacroF1={f1_mac*100:.2f}% | '
                f'ROC-AUC={roc_pr["roc_auc_macro"]:.4f} | PR-AUC={roc_pr["pr_auc_macro"]:.4f}'
            )

        eval_df = pd.DataFrame(eval_records)
        eval_df.to_csv(os.path.join(results_dir, 'evaluation_metrics.csv'), index=False)
        pd.concat(per_class_frames, ignore_index=True).to_csv(
            os.path.join(results_dir, 'per_class_metrics_bootstrap.csv'), index=False
        )

        save_object(X_train_final, 'Final_X_train', output_dir)
        save_object(X_test_final, 'Final_X_test', output_dir)
        save_object(y_train, 'Final_y_train', output_dir)
        save_object(y_test, 'Final_y_test', output_dir)
        save_object(final_feature_names, 'Final_Feature_Names', output_dir)

        plot_multiclass_roc(
            y_test_arr,
            test_probas[best_model_name],
            target_names,
            f'One-vs-Rest ROC curves: {best_model_name} ({best_strategy_label})',
            save_path=os.path.join(results_dir, 'roc_curves_best_model.png'),
            close_fig=True,
        )

        perm = permutation_importance(
            final_models[best_model_name],
            X_test_final,
            y_test_arr,
            scoring='balanced_accuracy',
            n_repeats=25,
            random_state=cv_seed,
            n_jobs=-1,
        )
        perm_df = pd.DataFrame({
            'Feature_Name': final_feature_names,
            'Modality': final_types,
            'Gene_or_Probe': final_symbols,
            'Importance_Mean': perm.importances_mean,
            'Importance_Std': perm.importances_std,
        }).sort_values('Importance_Mean', ascending=False).reset_index(drop=True)
        perm_df.to_csv(os.path.join(results_dir, 'permutation_feature_importance.csv'), index=False)

        top20_perm = perm_df.head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8.8, 6.2))
        ax.barh(
            top20_perm['Feature_Name'],
            top20_perm['Importance_Mean'],
            xerr=top20_perm['Importance_Std'],
            color=['#2563eb' if m == 'RNA' else '#d97706' for m in top20_perm['Modality']],
            alpha=0.88, capsize=3,
        )
        ax.axvline(0.0, color='#475569', linewidth=1.0)
        ax.set_xlabel('Mean decrease in held-out test balanced accuracy (25 repeats)')
        ax.set_title(f'Top 20 biomarkers by held-out permutation importance ({best_model_name})', fontweight='bold')
        ax.legend(handles=[Patch(facecolor='#2563eb', label='RNA-Seq'), Patch(facecolor='#d97706', label='Methylation')], loc='lower right')
        plt.tight_layout()
        fig.savefig(os.path.join(results_dir, 'permutation_feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

        rna_cols = set(X_rna_raw.columns)
        meth_cols = set(X_meth_raw.columns)
        cis_rows = []
        for _, row in features_df.iterrows():
            sym = row['Gene_or_Probe']
            if sym in rna_cols and sym in meth_cols:
                xr = X_rna_raw[sym].to_numpy(dtype=np.float64)
                xm = X_meth_raw[sym].to_numpy(dtype=np.float64)
                valid = np.isfinite(xr) & np.isfinite(xm)
                if int(valid.sum()) >= 10:
                    rho, pval = stats.spearmanr(xr[valid], xm[valid])
                    cis_rows.append({
                        'Feature_Name': row['Feature_Name'],
                        'Modality': row['Modality'],
                        'Gene_or_Probe': sym,
                        'N_Paired': int(valid.sum()),
                        'Spearman_Rho': float(rho),
                        'p_value': float(pval),
                    })
        cis_df = pd.DataFrame(cis_rows)
        cis_df['BH_FDR'] = benjamini_hochberg(cis_df['p_value'].to_numpy())
        cis_df['Significant_FDR05'] = cis_df['BH_FDR'] < 0.05
        cis_df['Direction'] = np.where(cis_df['Spearman_Rho'] < 0, 'Negative (silencing)', 'Positive')
        cis_df = cis_df.sort_values('Spearman_Rho').reset_index(drop=True)
        cis_df.to_csv(os.path.join(results_dir, 'cis_rna_meth_correlation.csv'), index=False)

        purity_vec = pd.to_numeric(clinical_metadata['Tumor_purity'], errors='coerce').to_numpy(dtype=np.float64)
        purity_rows = []
        for _, row in features_df.iterrows():
            sym = row['Gene_or_Probe']
            mod = row['Modality']
            vals = (X_rna_raw[sym] if mod == 'RNA' else X_meth_raw[sym]).to_numpy(dtype=np.float64)
            valid = np.isfinite(vals) & np.isfinite(purity_vec)
            rho, pval = stats.spearmanr(vals[valid], purity_vec[valid])
            purity_rows.append({
                'Feature_Name': row['Feature_Name'],
                'Modality': mod,
                'Gene_or_Probe': sym,
                'Mutual_Info': float(row['Mutual_Info']),
                'N_Valid': int(valid.sum()),
                'Purity_Spearman_Rho': float(rho),
                'Abs_Purity_Rho': float(abs(rho)),
                'p_value': float(pval),
            })
        purity_df = pd.DataFrame(purity_rows)
        purity_df['BH_FDR'] = benjamini_hochberg(purity_df['p_value'].to_numpy())
        purity_df['Strong_Purity_Flag'] = (purity_df['Abs_Purity_Rho'] >= 0.30) & (purity_df['BH_FDR'] < 0.05)
        purity_df = purity_df.sort_values('Abs_Purity_Rho', ascending=False).reset_index(drop=True)
        purity_df.to_csv(os.path.join(results_dir, 'tumor_purity_audit.csv'), index=False)

        os_years_all = pd.to_numeric(clinical_metadata['overall_survival'], errors='coerce') / 365.25
        os_event_all = pd.to_numeric(clinical_metadata['status'], errors='coerce')
        subtype_all = clinical_metadata['label'].to_numpy()
        lr_full = logrank_test_multigroup(os_years_all, os_event_all, subtype_all)
        fig_km, ax_km = plot_kaplan_meier(
            os_years_all, os_event_all, subtype_all, target_names,
            'Overall survival by PAM50 subtype (TCGA-BRCA cohort)',
            save_path=os.path.join(results_dir, 'kaplan_meier_survival.png'),
            close_fig=False,
        )
        ax_km.set_xlabel('Time (years)')
        fig_km.savefig(os.path.join(results_dir, 'kaplan_meier_survival.png'), dpi=300, bbox_inches='tight')
        plt.close(fig_km)

        print(
            f'Cis sig={int(cis_df["Significant_FDR05"].sum())}/{len(cis_df)} | '
            f'Purity median |rho|={purity_df["Abs_Purity_Rho"].median():.3f} | '
            f'KM log-rank p={lr_full["p_value"]:.4f} ({time.perf_counter() - t0:.2f}s)'
        )

        print('\n[7/7] Summarizing archived biological database validation artifacts')
        ensembl_path = os.path.join(results_dir, 'biomarker_ensembl_annotation.csv')
        edges_path = os.path.join(results_dir, 'ppi_network_edges_combined.csv')
        ot_path = os.path.join(results_dir, 'biomarker_opentargets_druggability.csv')
        hpa_path = os.path.join(results_dir, 'biomarker_hpa_protein.csv')
        ppi_comb_path = os.path.join(results_dir, 'ppi_network_summary_combined.json')

        if all(os.path.exists(p) for p in (ensembl_path, edges_path, ot_path, hpa_path, ppi_comb_path)):
            ensembl_df = pd.read_csv(ensembl_path)
            edges_df = pd.read_csv(edges_path)
            ot_df = pd.read_csv(ot_path)
            hpa_df = pd.read_csv(hpa_path)
            with open(ppi_comb_path, encoding='utf-8') as fh:
                ppi_comb = json.load(fh)

            G_ppi = nx.Graph()
            for _, r in edges_df.iterrows():
                G_ppi.add_edge(str(r['preferredName_A']), str(r['preferredName_B']), weight=float(r['score']))
            deg_dict = dict(G_ppi.degree())

            node_modalities = dict(zip(ensembl_df['Canonical_Symbol'], ensembl_df['Modality']))
            plot_ppi_network(
                edges_df, node_modalities,
                f'STRING v12 PPI network colored by omics modality (p = {ppi_comb["p_value"]:.2e})',
                save_path=os.path.join(results_dir, 'ppi_network_modality_colored.png'),
                close_fig=True,
            )

            def classify_tractability(row):
                if bool(row['SM_Approved_Drug']):
                    return 'Approved SM Drug'
                if bool(row['SM_Tractable']) or bool(row['AB_Tractable']):
                    return 'SM / AB Tractable'
                if bool(row['PROTAC_Tractable']):
                    return 'PROTAC Only'
                return 'No Current Tractability'

            ot_df['Tractability_Tier'] = ot_df.apply(classify_tractability, axis=1)
            scorecard = features_df[['Rank', 'Feature_Name', 'Modality', 'Gene_or_Probe', 'Mutual_Info']].rename(
                columns={'Rank': 'mRMR_Rank'}
            )
            scorecard = scorecard.merge(
                ensembl_df[['Feature_Name', 'Canonical_Symbol', 'Ensembl_ID', 'Is_PAM50_Gene']],
                on='Feature_Name', how='left',
            ).merge(
                perm_df[['Feature_Name', 'Importance_Mean']].rename(columns={'Importance_Mean': 'Perm_Importance'}),
                on='Feature_Name', how='left',
            ).merge(
                cis_df[['Feature_Name', 'Spearman_Rho', 'BH_FDR']].rename(
                    columns={'Spearman_Rho': 'Cis_RNA_Meth_Rho', 'BH_FDR': 'Cis_FDR'}
                ),
                on='Feature_Name', how='left',
            ).merge(
                purity_df[['Feature_Name', 'Abs_Purity_Rho']].rename(columns={'Abs_Purity_Rho': 'Purity_Abs_Rho'}),
                on='Feature_Name', how='left',
            )
            scorecard['STRING_Hub_Degree'] = scorecard['Canonical_Symbol'].map(deg_dict).fillna(0).astype(int)
            scorecard = scorecard.merge(
                ot_df[['Ensembl_ID', 'Breast_Cancer_Score', 'Tractability_Tier', 'Max_Drug_Stage']],
                on='Ensembl_ID', how='left',
            ).merge(
                hpa_df[['Ensembl_ID', 'Breast_Normal_IHC', 'FDA_Drug_Target']],
                on='Ensembl_ID', how='left',
            ).sort_values(
                ['Breast_Cancer_Score', 'STRING_Hub_Degree', 'Mutual_Info'],
                ascending=[False, False, False],
                na_position='last',
            ).reset_index(drop=True)
            scorecard.insert(0, 'Translational_Rank', np.arange(1, len(scorecard) + 1))
            scorecard.to_csv(os.path.join(results_dir, 'integrated_biomarker_scorecard.csv'), index=False)

            print(
                f'Biological database validation artifacts verified | '
                f'STRING nodes={int(ppi_comb["number_of_nodes"])}, edges={int(ppi_comb["number_of_edges"])}, '
                f'p={ppi_comb["p_value"]:.2e} | Scorecard rows={len(scorecard)}'
            )
        else:
            print('Biological database validation artifacts incomplete; skipped scorecard assembly.')

        print(f'\nPipeline completed in {time.perf_counter() - t_start:.2f}s.')
    finally:
        sys.stdout = orig_stdout
        logger.close()


if __name__ == '__main__':
    run_pipeline()
