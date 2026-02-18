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

## 📂 Project Structure

Run notebooks in the following order:

| Step | Notebook                    | Description                            |
| ---- | --------------------------- | -------------------------------------- |
| 01   | `Reading.ipynb`             | Load datasets and synchronize patients |
| 02   | `preprocessing.ipynb`       | Data split, imputation, normalization  |
| 03   | `Feature_Selection-1.ipynb` | Variance Threshold                     |
| 04   | `Feature Selection_2.ipynb` | ANOVA feature selection                |
| 05   | `Feature Selection_3.ipynb` | ReliefF selection                      |
| 06   | `Feature Selection_4.ipynb` | mRMR + SMOTE + modeling                |

---

## 📊 Results

Models were evaluated on a held-out **test set**.

| Model | Accuracy | Features | Pipeline |
|------|---------|---------|----------|
| 🏆 **Random Forest** | **84.55%** | 50 | Variance → ANOVA → ReliefF → mRMR |
| SVM | 82.73% | 50 | Same pipeline |
| KNN | 68.18% | 50 | Same pipeline |


> ℹ️ The **mRMR step** significantly improved performance by removing redundant and highly correlated genes.

---

## 🛠️ Installation

Make sure you have Python installed, then run:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

If you used a specific mRMR library, also install:

```bash
pip install pymrmr
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YourUsername/Breast-Cancer-Subtype-Prediction.git
cd Breast-Cancer-Subtype-Prediction
```

### 2. Prepare data

Place the following datasets in the project root (or update paths inside `Reading.ipynb`):

* RNA-Seq data
* DNA Methylation data
* Clinical labels

### 3. Run notebooks

Execute notebooks in order from **01 → 06**

📁 Intermediate processed files will be automatically saved in:

```
outputs/
```




