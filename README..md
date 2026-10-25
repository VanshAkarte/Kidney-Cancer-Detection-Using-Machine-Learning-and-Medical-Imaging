# Kidney Tumor Detection on CT using a Hybrid Deep-Radiomic Stacking Ensemble

Mini Project – Healthcare Analytics
**Author:** `<Your Name>` | **Roll No:** `<Roll No>` | **Course/Institute:** `<Course, Institute>`
**Guide:** Dr. Prerna

---

## 1. Problem Statement
Kidney cancer is often found incidentally on CT scans, and manual slice-by-slice review is time-consuming. This project builds a machine-learning system that classifies axial CT slices as **tumor** or **normal kidney**, and explains *where* the model is looking.

- **Target variable:** slice label (`tumor` = 1, `normal kidney` = 0)
- **Task:** binary image classification (predictive analytics)

## 2. Dataset
- **Name:** C4KC-KiTS (Kidney and Kidney Tumor Segmentation Challenge 2019 training set)
- **Source:** The Cancer Imaging Archive (TCIA) – https://www.cancerimagingarchive.net/collection/c4kc-kits/
- **Content:** contrast-enhanced abdominal CT with expert kidney/tumor segmentations (DICOM CT + DICOM-SEG), 210 patients in the full collection
- **Used in this project:** `<N>` patients, `<N>` slices (`<N>` tumor / `<N>` normal) — *fill from `dataset_summary.csv`*
- **Labels:** derived from the expert segmentations. A slice is `tumor` if tumor pixels ≥ 50; `normal` if it shows kidney with no tumor/cyst. Slices without kidney are excluded.
- **Citation:** Papanikolopoulos N., Weight C. (2019) *Data from C4KC-KiTS*. TCIA. DOI: 10.7937/TCIA.2019.IX49E8NX. Heller N. et al. (2021) *Medical Image Analysis*, 67, 101821.

> The dataset is **not** included in this repository (size and TCIA usage policy). See "How to run" to download it.

## 3. Novelty of the Proposed Approach
1. **Hybrid features:** frozen ResNet18 deep features combined with handcrafted texture features (intensity statistics, GLCM, LBP).
2. **Tuned feature selection:** mutual-information selection with the number of features chosen by patient-grouped cross-validation.
3. **Stacking ensemble:** Random Forest + SVM + Gradient Boosting combined by a logistic-regression meta-learner.
4. **Patient-level evaluation:** train/test split and cross-validation are done by patient, so slices from one patient never appear in both sets (prevents data leakage).
5. **Explainability:** Grad-CAM heatmaps, plus an analysis of how much of the decision comes from deep vs. handcrafted features.
6. **Comparison with baselines:** (A) handcrafted features + logistic regression, (B) fine-tuned ResNet18.

## 4. Pipeline
```
TCIA DICOM (CT + SEG) -> label slices from segmentations -> de-duplication
 -> patient-level train/test split -> CLAHE contrast enhancement
 -> feature extraction (CNN + handcrafted) -> feature selection -> stacking ensemble
 -> evaluation (Acc, Precision, Recall, Specificity, F1, ROC-AUC) -> Grad-CAM + plots
```

## 5. Repository Structure
```
.
├── Kidney_Cancer_Colab.ipynb      # Colab notebook that runs everything
├── dl.py                          # background downloader for TCIA (via tcia_utils)
├── prepare_dataset.py             # DICOM + SEG -> labelled PNG slices
├── kidney_cancer_pipeline.py      # baselines, proposed model, evaluation, plots
├── dataset_summary.csv            # slice list used (patient, file, label)
├── results/
│   ├── results.csv                # metrics of all models
│   ├── roc.png
│   ├── confusion_matrices.png
│   ├── model_comparison.png
│   ├── feature_importance.png
│   ├── correlation.png
│   └── gradcam.png
└── README.md
```

## 6. How to Run (Google Colab, GPU recommended)
1. Open `Kidney_Cancer_Colab.ipynb` in Google Colab and set **Runtime → T4 GPU**.
2. Install dependencies:
   ```
   pip install pydicom scikit-image seaborn tcia_utils
   ```
3. Download a subset of C4KC-KiTS (example: 25 patients, ~5 GB):
   ```
   python dl.py 25
   ```
4. Convert DICOM + segmentations to labelled slices:
   ```
   python prepare_dataset.py --src /content/tciaDownload --out data --per-class 12
   ```
5. Train and evaluate:
   ```
   python kidney_cancer_pipeline.py --data data --out results --epochs 8
   ```
6. Optional code check on synthetic data (numbers are meaningless):
   ```
   python kidney_cancer_pipeline.py --demo
   ```

## 7. Tools and Versions
Fill in from `pip list` in your Colab session:

| Tool | Version |
|---|---|
| Python | `<x.x>` |
| PyTorch / torchvision | `<x.x>` |
| scikit-learn | `<x.x>` |
| scikit-image | `<x.x>` |
| pydicom | `<x.x>` |
| NumPy / Pandas | `<x.x>` |
| Matplotlib / Seaborn | `<x.x>` |

## 8. Results
*Replace with your own numbers from `results/results.csv` (test set, patient-level split).*

| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Baseline A: Handcrafted + LR | | | | | | |
| Baseline B: Fine-tuned ResNet18 | | | | | | |
| **Proposed: Hybrid + FS + Stacking** | | | | | | |

Plots are in `results/`. Key observations: *write your own interpretation here.*

## 9. Limitations
- Single-institution data; no external validation.
- Only a sample of slices per patient is used; 2D slices lose 3D context.
- Labels come from segmentation-derived rules, not per-slice radiologist diagnosis.
- This is an academic prototype, **not** a clinical diagnostic tool.

## 10. Future Scope
- 3D / multi-slice models and patient-level aggregation of slice predictions
- External validation on another TCIA kidney collection (e.g., CPTAC-CCRCC)
- Tumor segmentation and tumor-subtype or grade prediction

## 11. References
List the papers you cite in your report here (dataset papers above, plus your literature review).

## 12. License / Data Use
Code: `<choose a license, e.g. MIT>`. Imaging data is subject to the TCIA Data Usage Policy and CC BY 3.0 and must be downloaded from TCIA directly.
