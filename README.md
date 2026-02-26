# Official Implementation for MICCAI 2026 Submission

This repository contains the source code and dataset preview for our MICCAI 2026 submission.

## 📊 Dataset Preview & Anonymization

To adhere to the **double-blind review process**, we provide a blinded preview version. The full dataset will be released upon completion of the review process.

* **`images.zip`**: Contains 4,036 fundus images downsampled to 32x32 pixels. Original high-resolution images will be released post-review.
* **`miccai_review_sanitized.csv`**: Contains labels and clinical parameters. Currently, approximately 80% of the data is randomly masked.
* **`analysis.ipynb`**: Data analysis performed on the original, unmasked dataset.
* **`annotation_guideline.pdf`**: Documentation defining disease categories, systemic conditions, image quality criteria, and labeling policies.

---

## 💻 Codebase

The implementation is built using **PyTorch**, **timm**, and **PyTorch Lightning**.

* **`main.py`**: Main entry point for training.
* **`models.py`**: Model architecture definitions.
* **`dataset.py`**: Data loading and preprocessing pipelines.
* **`utils.py`**: Helper functions.
