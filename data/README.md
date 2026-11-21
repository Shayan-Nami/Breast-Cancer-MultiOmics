# TCGA-BRCA multi-omics dataset

This directory stores the primary multi-omics and clinical datasets for the TCGA Breast Invasive Carcinoma (BRCA) cohort from the Broad Institute GDAC Firehose release (`2016_01_28`, LinkedOmics formatting).

## Required raw data files

| File Name | Data Type | Dimensions / Size | Platform | Measured Scale and Missingness |
| :--- | :--- | :--- | :--- | :--- |
| `Human_TCGA_BRCA_UNC_RNAseq_HiSeq_RNA_01_28_2016_BI_Gene_Firehose.gz` | Transcriptomics (RNA-Seq) | 20,155 genes x 1,093 samples (~52 MB) | Illumina HiSeq 2000 (RSEM normalized) | Already `log2(RSEM + 1)` transformed in `[0.0, 20.98]`, 0 NaNs, 7 zero-variance genes |
| `Human_TCGA_BRCA_JHU_USC_Methylation_Meth450_01_28_2016_BI_Gene_Firehose.gz` | Epigenomics (DNA Methylation) | 20,106 genes x 783 samples (~33 MB) | Illumina Infinium HumanMethylation450 | 0.5-centered gene-level beta values in `[-0.4969, +0.4964]` (median `-0.0756`), 4,293 NaNs across 1,327 genes in the 549-patient cohort (max 5.6% missing per gene) |
| `Human_TCGA_BRCA_MS_Clinical_Clinical_01_28_2016_BI_Clinical_Firehose.tsi` | Clinical and Phenotypic Data | 20 attributes x 1,097 patients (~150 KB) | Curated TCGA Clinical Annotations | PAM50 labels (826 non-null), tumor purity (546/549), overall survival and vital status (544/549), pathologic stage (542/549), age (538/549), sparse IHC ER/PR/HER2 (31/549) |

## Patient harmonization and overlap

Intersecting non-null PAM50 clinical records with RNA-Seq and DNA methylation samples yields 549 synchronized primary tumor patients (`TCGA.XX.XXXX` identifiers):

- **LumA:** 296 samples (53.9%)
- **LumB:** 122 samples (22.2%)
- **Basal:** 96 samples (17.5%)
- **Her2:** 35 samples (6.4%)

Across the two molecular layers, 18,957 gene symbols overlap directly and all 50/50 canonical PAM50 genes are present in both RNA-Seq and Methylation (`ORC6` appears under its 2016 Firehose alias `ORC6L`).

## Preprocessing summary

1. **Stratified train/test split.** 80% train (`N = 439`) and 20% held-out test (`N = 110`) with `SPLIT_SEED = 100` in `notebooks/preprocessing.ipynb`.
2. **Median imputation.** `SimpleImputer(strategy='median', keep_empty_features=True)` fit exclusively on training data and applied to test data.
3. **Unscaled variance filtering.** Removes the bottom 20% lowest-variance features per modality before scaling (`0.2690` RNA, `0.00062` Methylation).
4. **Clipped MinMax normalization.** `MinMaxScaler(clip=True)` fit strictly on training data so held-out test values stay within `[0.0, 1.0]`.

## Source acknowledgement

- Broad Institute TCGA GDAC Firehose (https://gdac.broadinstitute.org/)
- Cancer Genome Atlas Network. *Comprehensive molecular portraits of human breast tumours.* Nature 490, 61-70 (2012). https://doi.org/10.1038/nature11412
