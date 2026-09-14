# Multimodal Cancer Classification

A project from the **Advanced Deep Learning for Image Processing** course, exploring cancer classification with paired bright-field (BF) and fluorescence (FL) microscopy images for the **[Kaggle competition](https://www.kaggle.com/competitions/multimodal-cancer-classification-challenge-2026)**.

**[View the notebook](Refactored_Cancer_Classification_Pipeline.ipynb)**

## Approach

- Combine paired images into a six-channel input for a pretrained **EfficientNet-B0** model in PyTorch.
- Train with paired image augmentation and class-weighted loss.
- Use patient-level validation and track performance with ROC-AUC.

## Project context

This was a **group project**. Our team, **Group10**, placed **2nd** on the **[competition leaderboard](https://www.kaggle.com/competitions/multimodal-cancer-classification-challenge-2026/leaderboard)**.

This notebook showcases my work on the project.

## Data and usage

The dataset is available on **[Kaggle](https://www.kaggle.com/competitions/multimodal-cancer-classification-challenge-2026/data)** and is not included in this repository. Sign in to Kaggle and accept the competition rules if prompted.

**On Kaggle:** Import the notebook, attach the competition dataset, enable a GPU and internet access, and run the cells in order.

**Locally:**

1. Download and extract the dataset from Kaggle **outside this repository**. Preserve its layout: `train.csv`, `sampleSubmission.csv`, `BF/train/`, `BF/test/`, `FL/train/`, and `FL/test/` under one dataset directory.
2. From the repository directory, install dependencies and launch JupyterLab:

   ```bash
   python -m pip install -r requirements.txt
   jupyter lab
   ```

3. Open the notebook and change the dataset initialization to `BASE_DIR = find_dataset_dir("/absolute/path/to/dataset")`. Run the cells in order. Internet access is needed for the initial pretrained-weight download; a CUDA GPU is recommended for training.

The notebook produces a model checkpoint, learning curves, and a submission CSV in `/kaggle/working/` on Kaggle or the current directory locally.
