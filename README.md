# MAC Dataset (MICCAI 2026 Submission)

This repository contains the code and dataset preview for our MICCAI 2026 submission.

## 📊 Dataset Preview & Anonymization

To adhere to the **double-blind review process**, we provide a blinded preview version. The full dataset will be released upon completion of the review process.

* **`images.zip`**: Contains 4,036 fundus images downsampled to 32x32 pixels. Original high-resolution images will be released post-review.
* **`miccai_review_sanitized.csv`**: Contains labels and clinical parameters. Currently, approximately 90% of the data is randomly masked. The original CSV file will be released post-review.
* **`analysis.ipynb`**: Data analysis performed on the original, unmasked dataset.
* **`annotation_guideline.pdf`**: Documentation defining disease categories, systemic conditions, image quality criteria, and labeling policies.

---

## 💻 Code
This code was used for the experiments in Sections 4.2 and 4.3.

* **`main.py`**: Main entry point for training.
* **`models.py`**: Model architecture definitions.
* **`dataset.py`**: Data loading and preprocessing pipelines.
* **`utils.py`**: Helper functions.
* **`requirements.txt`**: List of required Python packages and their versions needed to reproduce the experiments and run the code.
