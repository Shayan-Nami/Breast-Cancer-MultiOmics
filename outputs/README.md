# Outputs directory and artifact specifications

This directory stores intermediate matrices, feature lists, trained classifiers, statistical validation tables, figures, and biological database validation artifacts generated directly by the 6 notebooks (and identically reproducible via `run_pipeline.py`).

## Directory layout

```text
outputs/
├── README.md
├── final_50_features.csv                    # Final 50 mRMR-MID biomarkers (43 RNA, 7 Methylation)
├── *.parquet                                # Raw matrices, clinical metadata, and train/test labels
├── *.npz                                    # Raw splits, imputed, variance-filtered, ANOVA, ReliefF, and final matrices
├── *.pkl                                    # Patient ID lists and stage-wise feature name lists
├── models/
│   ├── svm_model.pkl                        # Tuned Linear SVM (CV winner, class_weight='balanced')
│   ├── random_forest_model.pkl              # Tuned Random Forest (class_weight='balanced')
│   ├── logreg_model.pkl                     # Tuned L2 Logistic Regression (class_weight='balanced')
│   └── knn_model.pkl                        # Tuned KNN classifier
├── results/
│   ├── nested_cv_summary.csv                # 5x2 repeated nested CV across 4 models x 2 strategies
│   ├── evaluation_metrics.csv               # Single-pass held-out test metrics (Acc, BalAcc, F1, ROC-AUC, PR-AUC)
│   ├── per_class_metrics_bootstrap.csv      # 95% bootstrap CIs for overall and per-subtype Recall, Precision, F1
│   ├── ablation_study.csv                   # Leak-free 7-configuration ablation study
│   ├── feature_stability_frequency.csv      # Full-chain 5-fold feature selection frequency
│   ├── permutation_feature_importance.csv   # Held-out permutation importance (balanced accuracy, 25 repeats)
│   ├── cis_rna_meth_correlation.csv         # Cis RNA-methylation Spearman correlation and BH-FDR
│   ├── tumor_purity_audit.csv               # Biomarker vs TCGA tumor purity Spearman audit
│   ├── integrated_biomarker_scorecard.csv   # Unified 50-biomarker ML + clinical + biological scorecard
│   ├── biomarker_ensembl_annotation.csv     # Ensembl GRCh38 symbol harmonization and coordinates
│   ├── pathway_enrichment_reactome_*.csv    # Reactome overrepresentation (RNA, Methylation, Combined)
│   ├── ppi_network_summary_*.json           # STRING v12 PPI enrichment summaries (Taxon 9606)
│   ├── ppi_network_edges_combined.csv       # STRING v12 network edges (score >= 0.4)
│   ├── string_functional_enrichment.csv     # STRING v12 GO/KEGG/Reactome/literature terms (FDR < 0.05)
│   ├── biomarker_opentargets_druggability.csv # Open Targets breast carcinoma scores and tractability
│   ├── biomarker_breast_cancer_drugs.csv    # Open Targets breast cancer target-drug associations
│   ├── biomarker_hpa_protein.csv            # Human Protein Atlas normal breast IHC and localization
│   ├── biomarker_uniprot_annotations.csv    # UniProtKB reviewed human protein annotations
│   ├── biomarker_alphafold_summary.json     # AlphaFold DB pLDDT and PAE domain analysis (ESR1, AURKA)
│   ├── biomarker_clinical_trials.csv        # ClinicalTrials.gov Phase 2-3 breast cancer trials
│   ├── biomarker_pubmed_evidence.csv        # Verified PubMed literature references (PMIDs and URLs)
│   └── *.png                                # Confusion matrices, ROC, sensitivity, ablation, KM, purity, PPI, Reactome
└── logs/
    └── pipeline_log.txt
```

Three large or secondary local figures (`outputs/results/reactome_top_pathway_diagram.png`, `outputs/results/string_ppi_network_highres.png`, and `outputs/results/kaplan_meier_survival_test_predicted.png`) are kept locally and excluded in `.gitignore`.

## Trained classifiers (`outputs/models/*.pkl`)

All 4 classifiers are tuned via leak-free 3-fold stratified `GridSearchCV` using the CV-selected imbalance strategy (`Class_Weight_Balanced`) on the 50 mRMR-MID biomarkers (`N_train = 439`) and evaluated once on the held-out test set (`N_test = 110`):

| Model File | Classifier | Nested CV BalAcc | Test Accuracy [95% CI] | Test Macro F1 [95% CI] | Test Macro ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `models/svm_model.pkl` | `SVC(kernel='linear', probability=True)` *(CV winner)* | **91.49%** | 85.45% [78.18% - 91.82%] | 87.90% [81.10% - 93.22%] | **0.9853** |
| `models/random_forest_model.pkl` | `RandomForestClassifier` | 91.01% | **90.00%** [83.64% - 95.45%] | **90.69%** [82.87% - 95.65%] | 0.9782 |
| `models/logreg_model.pkl` | `LogisticRegression(max_iter=2000)` | 91.35% | 88.18% [81.82% - 93.64%] | 89.48% [82.28% - 94.77%] | 0.9837 |
| `models/knn_model.pkl` | `KNeighborsClassifier` | 85.80% | 85.45% [79.09% - 91.82%] | 86.85% [80.23% - 91.68%] | 0.9654 |

## Loading artifacts with `src.utils`

```python
from src.utils import load_object

X_test = load_object('Final_X_test')
y_test = load_object('Final_y_test')
features = load_object('Final_Feature_Names')
model = load_object('svm_model', input_dir='outputs/models')
```
