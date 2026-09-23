# 🧬 Breast Cancer Subtype Prediction via Multi-Omics Integration

## 📌 Project Overview

This project presents a machine learning pipeline for predicting **Breast Cancer Subtypes** (Luminal A, Luminal B, Basal, and HER2) using **multi-omics data integration**:

* **Gene Expression (RNA-Seq)**
* **DNA Methylation**

The main contribution of this project is a **Multi-Stage Hybrid Feature Selection pipeline** designed to address the **curse of dimensionality** in bioinformatics datasets.

From tens of thousands of biological features, the pipeline extracts a robust subset of **50 biomarkers**, enabling accurate classification using a **Random Forest classifier**.

---

## ⚙️ Methodology & Pipeline

The workflow is implemented in **6 sequential Jupyter Notebooks**, ensuring modularity and reproducibility:

1. **Data Harmonization**

   * Synchronize patients across clinical, RNA, and methylation datasets

2. **Preprocessing**

   * Train/Test split (to prevent data leakage)
   * Missing value imputation
   * Log2 transformation
   * MinMax normalization

3. **Feature Selection – Phase 1 (Variance Threshold)**

   * Remove quasi-constant features

4. **Feature Selection – Phase 2 (ANOVA Filter)**

   * Select top 500 statistically significant features using **ANOVA (F-test)**

5. **Feature Selection – Phase 3 (ReliefF Wrapper)**

   * Select top 150 context-aware features based on nearest neighbors

6. **Feature Selection – Phase 4 (Hybrid + Modeling)**

   * Apply **mRMR (Minimum Redundancy Maximum Relevance)**
   * Select final 50 features
   * Apply **SMOTE** for class balancing
   * Train and evaluate models: Random Forest, SVM, KNN

---

## 📂 Repository Architecture

```text
Breast-Cancer-MultiOmics/
├── data/                    # Primary raw TCGA-BRCA datasets & detailed documentation
│   └── README.md
├── notebooks/               # 6 sequential interactive Jupyter Notebooks
│   ├── Reading.ipynb
│   ├── preprocessing.ipynb
│   ├── Feature_Selection_1.ipynb
│   ├── Feature Selection_2.ipynb
│   ├── Feature Selection_3.ipynb
│   ├── Feature Selection_4.ipynb
│   └── README.md            # Detailed step-by-step notebook execution guide
├── src/                     # Shared core Python modules and feature selection algorithms
│   ├── __init__.py          # Clean public API exports
│   ├── utils.py             # IO, ReliefF, mRMR, Nested CV, Metrics, and Visualizations
│   └── README.md            # Module API documentation and code examples
├── outputs/                 # Artifacts, pre-trained models, matrices, and evaluation logs
│   ├── models/              # Pre-trained models (Random Forest, Linear SVM, KNN)
│   ├── results/             # Metrics, confusion matrices, ablation study, sensitivity curves
│   ├── logs/                # Chronological execution logs
│   └── README.md            # Output directory structure and format specifications
├── run_pipeline.py          # Standalone end-to-end pipeline runner
├── requirements.txt         # Pinned Python package dependencies
├── LICENSE                  # Open-source license
└── README.md                # Main repository documentation
```

### Notebook Execution Sequence

| Step | Notebook                    | Phase Description                                            | Primary Output Artifacts |
| :--- | :-------------------------- | :----------------------------------------------------------- | :----------------------- |
| 01   | `Reading.ipynb`             | Data Ingestion & Patient Barcode Synchronization ($N=549$)    | `X_*_raw.parquet`, `y_labels.parquet` |
| 02   | `preprocessing.ipynb`       | Stratified Train/Test Split (80/20) & In-Train Median Imputation | `X_train/test_*_imp.npz`, `y_train/test` |
| 03   | `Feature_Selection_1.ipynb` | Variance Threshold (20th percentile) & MinMax Scaling $[0, 1]$ | `X_train/test_*_var.npz` |
| 04   | `Feature Selection_2.ipynb` | Univariate ANOVA F-test Filter (Top 500 per modality)         | `X_train/test_*_anova.npz` |
| 05   | `Feature Selection_3.ipynb` | Multi-Class ReliefF Nearest-Neighbor Wrapper (Top 150)       | `X_train/test_*_relief.npz` |
| 06   | `Feature Selection_4.ipynb` | mRMR (Top 50), Nested CV, In-Fold Tuning, and Held-Out Test   | `models/*.pkl`, `results/*` |

---

## 📊 Results

Models were evaluated using **True In-Fold Nested Cross-Validation (RepeatedStratifiedKFold, 10 folds)** to eliminate selection bias, and finally evaluated on a fresh, held-out **test set (N = 110 samples)** with **95% Bootstrap Confidence Intervals**.

| Model | Test Accuracy (95% CI) | Balanced Acc | Macro F1 (95% CI) | Weighted F1 | Nested CV Accuracy | Features |
|-------|------------------------|--------------|-------------------|-------------|--------------------|----------|
| 🏆 **Random Forest** | **87.27%** [80.91% - 93.64%] | **89.14%** | **89.59%** [83.91% - 94.31%] | **87.31%** | 89.75% ± 3.01% | 50 Multi-Omics |
| 🥈 **SVM (Linear)** | **84.55%** [78.18% - 90.00%] | **90.18%** | **88.65%** [83.53% - 92.85%] | **85.25%** | 90.09% ± 2.40% | 50 Multi-Omics |
| 🥉 **KNN (k=5)** | **78.18%** [70.89% - 85.45%] | **85.17%** | **81.36%** [72.40% - 88.24%] | **79.22%** | 85.76% ± 2.35% | 50 Multi-Omics |

### 🔬 Systematic Ablation Study (Fixed Classifier: SVM, 5-Fold CV)

| Configuration | RNA Features | Methylation Features | CV Accuracy | CV Balanced Accuracy | CV Macro F1 |
|---------------|--------------|----------------------|-------------|----------------------|-------------|
| **Multi-Omics: mRMR MID Variant (Top 50)** | 43 | 7 | **93.16%** | **93.49%** | **92.71%** |
| **RNA-Seq Only (Top 50 mRMR)** | 50 | 0 | **93.40%** | **93.19%** | 92.51% |
| **Multi-Omics: Direct (ANOVA → mRMR)** | 31 | 19 | **92.03%** | 91.20% | 91.27% |
| **Multi-Omics: Full (ANOVA → ReliefF → mRMR)** | 31 | 19 | 90.88% | **91.50%** | 91.04% |
| **Multi-Omics: ANOVA Only (Top 50)** | 25 | 25 | 82.47% | 85.50% | 84.75% |
| **Methylation Only (Top 50 mRMR)** | 0 | 50 | 82.00% | 82.13% | 81.49% |
| **Multi-Omics: Without mRMR (ReliefF 50)** | 25 | 25 | 80.86% | 81.04% | 80.52% |

> ℹ️ **Key Insights:**
> 1. In the final 50 biomarkers, exactly **19 features (38.0%)** are DNA Methylation and **31 features (62.0%)** are RNA.
> 2. Integrating **mRMR boosts classification accuracy by +10.02% to +12.30%** over filter/wrapper stages alone.
> 3. Canonical estrogen and luminal drivers (*ESR1*, *FOXA1*, *GATA3*, *MLPH*) are preserved among the top 50 biomarkers, while *ERBB2* and *MKI67* were filtered out during ANOVA as multi-omics features (e.g. *Meth_LRRC6*, *RNA_CENPA*) provided stronger non-redundant predictive power.
> 4. Biological pathway analysis confirms significant enrichment of estrogen-receptor signaling (*ESR-mediated signaling*, Reactome FDR = $3.91 \times 10^{-8}$) and dense protein interaction interactome (STRING PPI $p = 3.33 \times 10^{-16}$).

---

## 🛠️ Installation

Make sure you have Python (>= 3.9) installed, then install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Option A: End-to-End Execution via Standalone Script (Recommended)
Run the entire pipeline automatically from start to finish:

```bash
python run_pipeline.py
```
This will harmonize data, preprocess, perform all 4 feature selection phases, run true in-fold nested CV, train and evaluate final models, save all outputs, metrics CSV, confusion matrix plots, and execution logs.

### Option B: Step-by-Step Jupyter Notebooks
Execute notebooks sequentially from **01 → 06**:
1. `notebooks/Reading.ipynb`
2. `notebooks/preprocessing.ipynb`
3. `notebooks/Feature_Selection_1.ipynb`
4. `notebooks/Feature Selection_2.ipynb`
5. `notebooks/Feature Selection_3.ipynb`
6. `notebooks/Feature Selection_4.ipynb`

📁 Intermediate and final files will be saved in:
```
outputs/
```




