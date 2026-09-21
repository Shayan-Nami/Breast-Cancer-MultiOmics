# 📊 Outputs Directory & File Specifications

This directory contains all **intermediate datasets**, **selected feature names**, **pre-trained machine learning models**, and **statistical validation artifacts** generated across the 6 phases of the **Breast-Cancer-MultiOmics** pipeline.

---

## 📁 Directory Architecture Overview

```text
outputs/
├── README.md                              # Comprehensive guide to all output artifacts
├── final_50_features.csv                  # Ranked table of final 50 biomarkers (31 RNA, 19 Methylation)
│
├── 📦 Pickle Files (.pkl) - Metadata & Feature Lists
│   ├── patient_ids.pkl                    # 549 synchronized TCGA-BRCA patient barcodes
│   ├── rna_feature_names.pkl              # 20,531 raw RNA-Seq gene symbols
│   ├── meth_feature_names.pkl             # 20,107 raw Methylation CpG gene names
│   ├── feat_names_rna_var.pkl             # 16,104 RNA genes passing Variance Threshold
│   ├── feat_names_meth_var.pkl            # 16,104 Methylation genes passing Variance Threshold
│   ├── feat_names_rna_anova.pkl           # Top 500 RNA genes selected by ANOVA F-test
│   ├── feat_names_meth_anova.pkl          # Top 500 Methylation genes selected by ANOVA F-test
│   ├── feat_names_rna_relief.pkl          # Top 150 RNA genes selected by ReliefF
│   ├── feat_names_meth_relief.pkl         # Top 150 Methylation genes selected by ReliefF
│   └── Final_Feature_Names.pkl            # Final 50 multi-omics biomarker names (mRMR)
│
├── 📦 Parquet Files (.parquet) - High-Throughput Tabular Data
│   ├── X_rna_raw.parquet                  # Full raw RNA-Seq matrix (549 samples × 20,531 genes)
│   ├── X_meth_raw.parquet                 # Full raw Methylation matrix (549 samples × 20,107 genes)
│   ├── y_labels.parquet                   # Subtype labels for all 549 patients (LumA, LumB, Her2, Basal)
│   ├── y_train.parquet                    # Train labels (N = 439 samples, 80%)
│   ├── y_test.parquet                     # Test labels (N = 110 samples, 20% isolated)
│   ├── Final_y_train.parquet              # Modeling-phase train labels
│   └── Final_y_test.parquet               # Modeling-phase test labels
│
├── 📦 NPZ Files (.npz) - Compressed High-Dimensional Matrices
│   ├── X_train_rna_imp.npz / X_test_...   # Median-imputed & log2-transformed continuous matrices
│   ├── X_train_meth_imp.npz / X_test_...  # Median-imputed methylation beta matrices
│   ├── X_train_*_var.npz / X_test_...     # Variance-filtered & MinMax normalized [0, 1]
│   ├── X_train_*_anova.npz / X_test_...   # Top 500 ANOVA-filtered matrices
│   ├── X_train_*_relief.npz / X_test_...  # Top 150 ReliefF-filtered matrices
│   ├── feat_indices_* (.npz)              # Numerical column indices selected at each stage
│   └── Final_X_train.npz / Final_X_test   # Final 50-feature multi-omics matrices (439 × 50 & 110 × 50)
│
├── 🤖 models/ - Trained & Hyperparameter-Tuned Classifiers
│   ├── random_forest_model.pkl            # Tuned RandomForestClassifier (GridSearchCV, CV BalAcc = 90.43%)
│   ├── svm_model.pkl                      # Tuned SVC (Linear kernel, C=0.1, CV BalAcc = 91.50%)
│   └── knn_model.pkl                      # Tuned KNeighborsClassifier (k=5, CV BalAcc = 85.17%)
│
├── 📈 results/ - Empirical Tables, Plots & Biological Validation
│   ├── evaluation_metrics.csv             # Final test set performance comparison
│   ├── ablation_study.csv                 # 6-configuration controlled ablation study
│   ├── per_class_metrics_bootstrap.csv    # 95% Bootstrap Confidence Intervals for all subtypes
│   ├── confusion_matrix_RF.png            # Multi-class confusion matrix (Random Forest)
│   ├── confusion_matrix_SVM.png           # Multi-class confusion matrix (Linear SVM)
│   ├── confusion_matrix_KNN.png           # Multi-class confusion matrix (KNN)
│   ├── cv_accuracy_vs_k_anova.png         # Sensitivity analysis curve for ANOVA (k=500)
│   ├── cv_accuracy_vs_k_relieff.png       # Sensitivity analysis curve for ReliefF (k=150)
│   ├── cv_accuracy_vs_k_mrmr.png          # Sensitivity analysis curve for mRMR (k=50)
│   ├── pathway_enrichment_reactome_rna.csv# Significant Reactome pathways for RNA markers (FDR < 0.05)
│   ├── pathway_enrichment_reactome_meth.csv# Reactome pathways for Methylation markers
│   ├── ppi_network_summary_rna.json       # STRING PPI network interactome statistics
│   └── ppi_network_summary_meth.json      # STRING PPI network summary for Methylation
│
└── 📝 logs/
    └── pipeline_log.txt                   # Complete terminal log with exact execution timestamps
```

---

## 🔍 In-Depth Artifact Specifications

### 1. 🧬 Pickle Files (`.pkl`) — Feature Lists & Metadata

Pickle (`.pkl`) is Python's native binary serialization format. In this project, `.pkl` files store Python lists, dictionaries, string arrays, and scikit-learn model objects.

| File Name | Size | Type / Contents | Description & Role |
| :--- | :--- | :--- | :--- |
| `patient_ids.pkl` | 11 KB | `list[str]` ($N=549$) | Ordered TCGA barcodes (e.g. `'TCGA-A1-A0SD'`) of primary tumor patients synchronized across all 3 data layers. |
| `rna_feature_names.pkl` | 175 KB | `list[str]` ($N=20,531$) | Canonical HGNC gene symbols representing the unscaled mRNA features (e.g. `ESR1`, `ERBB2`, `PGR`, `MKI67`). |
| `meth_feature_names.pkl` | 176 KB | `list[str]` ($N=20,107$) | Gene symbols representing promoter-associated CpG island methylation probes from Infinium 450K. |
| `feat_names_rna_var.pkl` | 140 KB | `list[str]` ($N=16,104$) | RNA gene names retained after Phase 1 Variance Thresholding (top 80% most variable genes). |
| `feat_names_meth_var.pkl` | 142 KB | `list[str]` ($N=16,104$) | Methylation gene names retained after Phase 1 Variance Thresholding. |
| `feat_names_rna_anova.pkl` | 4.2 KB | `list[str]` ($N=500$) | Top 500 RNA genes ranked by ANOVA F-test statistical significance across PAM50 classes. |
| `feat_names_meth_anova.pkl` | 4.2 KB | `list[str]` ($N=500$) | Top 500 Methylation genes ranked by ANOVA F-test. |
| `feat_names_rna_relief.pkl` | 1.3 KB | `list[str]` ($N=150$) | Top 150 RNA genes selected by multi-class ReliefF nearest-neighbor wrapper. |
| `feat_names_meth_relief.pkl` | 1.3 KB | `list[str]` ($N=150$) | Top 150 Methylation genes selected by ReliefF. |
| `Final_Feature_Names.pkl` | 640 B | `list[str]` ($N=50$) | Final 50 multi-omics biomarkers selected by mRMR (`RNA_...` and `Meth_...` prefixes). |

#### 💻 How to Load `.pkl` Files:
```python
import pickle

# Example 1: Load final biomarker names
with open('outputs/Final_Feature_Names.pkl', 'rb') as f:
    biomarkers = pickle.load(f)
print(f"Top 5 Biomarkers: {biomarkers[:5]}")

# Example 2: Load patient IDs
with open('outputs/patient_ids.pkl', 'rb') as f:
    patients = pickle.load(f)
print(f"Total synchronized patients: {len(patients)}")
```

---

### 2. 🤖 Pre-Trained Classifiers (`outputs/models/*.pkl`)

These files contain scikit-learn estimator instances trained on the balanced, feature-selected multi-omics training set ($N=439$ patients, 50 features) using optimal hyperparameters found via in-fold GridSearchCV:

| Model File | Classifier | In-Fold CV BalAcc | Test Accuracy [95% CI] | Test Macro F1 [95% CI] |
| :--- | :--- | :--- | :--- | :--- |
| `models/random_forest_model.pkl` | `RandomForestClassifier` (100 trees, balanced) | **90.43%** | **87.27%** [80.91% - 93.64%] | **89.59%** [83.20% - 93.67%] |
| `models/svm_model.pkl` | `SVC(kernel='linear', C=0.1)` | **91.50%** | **84.55%** [78.18% - 90.00%] | **88.65%** [83.53% - 92.85%] |
| `models/knn_model.pkl` | `KNeighborsClassifier(k=5, weights='distance')` | **85.17%** | **78.18%** [70.89% - 85.45%] | **81.36%** [71.53% - 87.76%] |

#### 💻 How to Load and Use Pre-Trained Models:
```python
import pickle
import numpy as np

# 1. Load trained Random Forest model
with open('outputs/models/random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 2. Load held-out test data (110 samples, 50 features)
X_test = np.load('outputs/Final_X_test.npz')['data']

# 3. Predict PAM50 subtypes (0: LumA, 1: LumB, 2: Her2, 3: Basal)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

class_names = ['LumA', 'LumB', 'Her2', 'Basal']
print(f"Patient 0 Predicted Subtype: {class_names[y_pred[0]]}")
print(f"Subtype Probabilities: {dict(zip(class_names, y_proba[0].round(3)))}")
```

---

### 3. 📦 Compressed Feature Matrices (`.npz`)

NumPy Compressed Archive (`.npz`) format stores continuous high-dimensional arrays using zlib compression. Each archive contains an array accessible via key `'data'`.

| File Name | Shape ($N \times P$) | Uncompressed Size | Compressed Size | Description |
| :--- | :--- | :--- | :--- | :--- |
| `X_train_rna_imp.npz` | $439 \times 20,531$ | ~72 MB | ~22 MB | Log2-transformed, median-imputed RNA training matrix. |
| `X_test_rna_imp.npz` | $110 \times 20,531$ | ~18 MB | ~5.5 MB | Imputed RNA test matrix (imputed strictly using train medians). |
| `X_train_meth_imp.npz` | $439 \times 20,107$ | ~70 MB | ~23 MB | Median-imputed Methylation beta training matrix. |
| `X_test_meth_imp.npz` | $110 \times 20,107$ | ~17 MB | ~5.8 MB | Imputed Methylation test matrix. |
| `X_train_rna_var.npz` | $439 \times 16,104$ | ~56 MB | ~23 MB | Variance-filtered & MinMax normalized RNA training matrix $[0, 1]$. |
| `X_train_meth_var.npz` | $439 \times 16,104$ | ~56 MB | ~23 MB | Variance-filtered & MinMax normalized Methylation training matrix. |
| `X_train_rna_anova.npz` | $439 \times 500$ | ~1.7 MB | ~752 KB | Top 500 ANOVA RNA features for training. |
| `X_train_meth_anova.npz` | $439 \times 500$ | ~1.7 MB | ~758 KB | Top 500 ANOVA Methylation features for training. |
| `X_train_rna_relief.npz` | $439 \times 150$ | ~526 KB | ~226 KB | Top 150 ReliefF RNA features for training. |
| `X_train_meth_relief.npz`| $439 \times 150$ | ~526 KB | ~224 KB | Top 150 ReliefF Methylation features for training. |
| `Final_X_train.npz` | $439 \times 50$ | ~175 KB | ~77 KB | Final combined Multi-Omics training matrix (31 RNA + 19 Meth). |
| `Final_X_test.npz` | $110 \times 50$ | ~44 KB | ~19 KB | Final combined Multi-Omics held-out test matrix. |

#### 💻 How to Load `.npz` Matrices:
```python
import numpy as np

# Load final training matrix
with np.load('outputs/Final_X_train.npz') as f:
    X_train = f['data']

print(f"X_train shape: {X_train.shape}")  # (439, 50)
```

---

### 4. 📊 Tabular Data (`.parquet`)

Apache Parquet is a columnar storage format optimized for analytical queries with fast decompression and automatic preservation of DataFrame column types and indexes.

| File Name | Dimensions | Description |
| :--- | :--- | :--- |
| `X_rna_raw.parquet` | $549 \times 20,531$ | Raw, synchronized mRNA normalized count matrix with patient barcodes as index. |
| `X_meth_raw.parquet` | $549 \times 20,107$ | Raw, synchronized DNA Methylation beta-value matrix with patient barcodes as index. |
| `y_labels.parquet` | $549 \times 1$ | Target PAM50 subtype integer labels (0: LumA, 1: LumB, 2: Her2, 3: Basal). |
| `y_train.parquet` | $439 \times 1$ | Training set target labels. |
| `y_test.parquet` | $110 \times 1$ | Isolated held-out test set target labels. |

#### 💻 How to Load `.parquet` Files:
```python
import pandas as pd

# Load labels and raw matrices
y_labels = pd.read_parquet('outputs/y_labels.parquet')
print("Label Distribution:")
print(y_labels.value_counts())
```

---

### 5. 🔬 Research Results (`outputs/results/`)

This subfolder houses all statistical evidence, validation tables, and publication-ready figures:

1. **`final_50_features.csv`**:
   - Ranked list of the 50 selected biomarkers.
   - Includes feature name, omics modality (`RNA-Seq` vs `Methylation`), mRMR selection rank (Rank 1: `Meth_LRRC6`), and mutual information relevance score.
2. **`ablation_study.csv`**:
   - Full 6-configuration controlled ablation study using 5-fold cross-validation with a fixed classifier (Linear SVM).
   - Confirms that adding mRMR provides an essential **+10.02% accuracy jump** over filter/wrapper methods alone.
3. **`per_class_metrics_bootstrap.csv`**:
   - 1,000 non-parametric bootstrap resamples on the held-out test set ($N=110$).
   - Provides 95% Confidence Intervals for Precision, Recall, and F1 across all 4 subtypes.
4. **Sensitivity Curves (`cv_accuracy_vs_k_*.png`)**:
   - Line plots demonstrating cross-validation accuracy across varying numbers of selected features ($k$), mathematically justifying the cutoffs $k=500$ (ANOVA), $k=150$ (ReliefF), and $k=50$ (mRMR).
5. **Confusion Matrices (`confusion_matrix_*.png`)**:
   - Publication-quality multi-class confusion matrices with absolute patient counts and class-specific accuracy annotations.
6. **Pathway & Network Enrichment (`pathway_enrichment_reactome_*.csv` & `ppi_network_summary_*.json`)**:
   - Reactome pathway overrepresentation results confirming significant enrichment of canonical estrogen receptor signaling (`ESR-mediated signaling`, FDR = $3.91 \times 10^{-8}$).
   - STRING PPI network analysis confirming statistically significant interactome connectivity ($p = 3.33 \times 10^{-16}$).

---

## 🛠️ Unified Loading with `src.utils.load_object`

The `src` package provides a unified loader that handles `.parquet`, `.npz`, and `.pkl` automatically without needing to specify file extensions:

```python
from src.utils import load_object

# Automatically loads NPZ, Parquet, or PKL
X_train = load_object('Final_X_train')       # Loads outputs/Final_X_train.npz
y_train = load_object('Final_y_train')       # Loads outputs/Final_y_train.parquet
features = load_object('Final_Feature_Names') # Loads outputs/Final_Feature_Names.pkl

print(f"Loaded: {X_train.shape[0]} samples, {X_train.shape[1]} features.")
```
