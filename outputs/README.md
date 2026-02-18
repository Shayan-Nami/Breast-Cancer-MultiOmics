# 📊 Outputs Directory

This folder contains all **intermediate files**, **selected features**, and **final model results** generated throughout the pipeline.

---

## 📁 Folder Structure

```
outputs/
│
├── preprocessed/
│   ├── X_train.pkl
│   ├── X_test.pkl
│   ├── y_train.pkl
│   └── y_test.pkl
│
├── feature_selection/
│   ├── variance_filtered.pkl
│   ├── anova_top500.pkl
│   ├── reliefF_top150.pkl
│   └── mrmr_top50.pkl
│
├── models/
│   ├── random_forest_model.pkl
│   ├── svm_model.pkl
│   └── knn_model.pkl
│
├── results/
│   ├── evaluation_metrics.csv
│   ├── confusion_matrix_RF.png
│   ├── confusion_matrix_SVM.png
│   └── confusion_matrix_KNN.png
│
└── logs/
    └── pipeline_log.txt
```

---

## 📌 Description

### 🔹 `preprocessed/`

Contains cleaned and normalized datasets after preprocessing:

* Train/Test split
* Imputation
* Normalization

---

### 🔹 `feature_selection/`

Contains outputs from each feature selection phase:

* Variance Threshold
* ANOVA
* ReliefF
* mRMR (final 50 selected features)

---

### 🔹 `models/`

Trained machine learning models:

* Random Forest
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)

All models are saved as `.pkl` files for reuse.

---

### 🔹 `results/`

Evaluation outputs for each model:

* Accuracy and performance metrics
* Confusion matrices
* Final comparison results

---

### 🔹 `logs/`

Contains logs generated during pipeline execution for reproducibility and debugging.

---

## ▶️ Usage

These files are **automatically generated** when running the notebooks in order:

```
01 → 02 → 03 → 04 → 05 → 06
```

You can directly load saved objects using Python:

```python
import pickle

with open("outputs/models/random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)
```

---

## ⚠️ Notes

* Do **not manually edit** files in this directory
* You can safely delete this folder to re-run the pipeline from scratch
* Large files are ignored in `.gitignore`
