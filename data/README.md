## 🧬 Dataset Information

### How to run the code

Download the dataset files from one of the following sources:

* GDC Data Portal: https://portal.gdc.cancer.gov/
* cBioPortal: https://www.cbioportal.org/
* UCSC Xena: https://xenabrowser.net/datapages/

---

### 📥 Required Files

Place the following files inside the `data/` directory:

* `Human_TCGA_BRCA_UNC_RNAseq_....` → RNA-Seq Gene Expression
* `Human_TCGA_BRCA_JHU_USC_Methylation_....` → DNA Methylation
* `Human_TCGA_BRCA_MS_Clinical_....` → Clinical Data

---

### 📁 Directory Structure

```
project-root/
│
├── data/
│   ├── Human_TCGA_BRCA_UNC_RNAseq_....
│   ├── Human_TCGA_BRCA_JHU_USC_Methylation_....
│   └── Human_TCGA_BRCA_MS_Clinical_....
│
├── notebooks/
├── outputs/
└── README.md
```

---

### ⚙️ Notes

* The notebooks are already configured to **automatically read files from the `data/` folder**
* If file names are different, update the file paths inside:

  ```
  Reading.ipynb
  ```
* Make sure all three datasets contain **matching patient IDs**
