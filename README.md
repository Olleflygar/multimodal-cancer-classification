# Multimodal Cancer Classification

A portfolio project from the **Advanced Deep Learning for Image Processing** course, developed for the **[Kaggle competition](https://www.kaggle.com/competitions/multimodal-cancer-classification-challenge-2026)**. The task is binary cancer classification using paired bright-field (BF) and fluorescence (FL) microscopy images.

**[View the notebook](Refactored_Cancer_Classification_Pipeline.ipynb)**

## Approach

- Combine the two RGB images into a six-channel input for a pretrained **EfficientNet-B0**, adapted in PyTorch for binary classification.
- Apply shared geometric augmentation to preserve alignment between image pairs, and use class-weighted binary cross-entropy during training.
- Use a stratified **80/20 patient-level split**, keeping each patient's images in one partition. Evaluate with ROC-AUC and select the checkpoint with the best validation AUC.

## Results and contribution

This was a **group project**. Our team, **Group10**, placed **2nd** on the **[competition leaderboard](https://www.kaggle.com/competitions/multimodal-cancer-classification-challenge-2026/leaderboard)**.

The notebook contains **my own implementation** and may differ from the team's best-performing submission. The ranking reflects the team's result; no separate performance score is reported for this notebook.

## Data and usage

The dataset is available on **[Kaggle](https://www.kaggle.com/competitions/multimodal-cancer-classification-challenge-2026/data)** and is **not included in this repository**. Sign in to Kaggle and accept the competition rules if prompted.

**On Kaggle:** Import the notebook, attach the competition dataset as an input, enable a GPU and internet access, and run the cells in order. The notebook installs its dependencies and discovers the dataset automatically.

**Locally:**

1. Download and extract the dataset from Kaggle **outside this repository**. Preserve its layout: `train.csv`, `sampleSubmission.csv`, `BF/train/`, `BF/test/`, `FL/train/`, and `FL/test/` under one dataset directory.
2. From the repository directory, install dependencies and launch JupyterLab:

   ```bash
   python -m pip install -r requirements.txt
   jupyter lab
   ```

3. Open the notebook and change the dataset initialization to `BASE_DIR = find_dataset_dir("/absolute/path/to/dataset")`. Run the cells in order. Internet access is needed for the initial pretrained-weight download; a CUDA GPU is recommended for training.

The pipeline saves a model checkpoint, learning curves, and predictions in `sampleSubmission_p.csv`. Outputs go to `/kaggle/working/` on Kaggle or the current directory locally.
