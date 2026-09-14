# Multimodal Cancer Classification

A project from the **Advanced Deep Learning for Image Processing** course, exploring cancer classification with paired bright-field (BF) and fluorescence (FL) microscopy images for the **[Kaggle competition](https://www.kaggle.com/competitions/multimodal-cancer-classification-challenge-2026)**. The **[notebook](Cancer_Classification_Pipeline.ipynb)** captures my work on the classification pipeline.

This was a **group project**. Our team, **Group10**, placed **2nd** on the **[competition leaderboard](https://www.kaggle.com/competitions/multimodal-cancer-classification-challenge-2026/leaderboard)**.

## Project focus

The project connects **scientific image analysis**, **multimodal learning**, and **statistical model evaluation**. A central challenge is bringing two imaging modalities into one predictive model while preserving the correspondence between them. The workflow also accounts for class balance and for multiple images belonging to the same patient.

These considerations shape the approach:

- **Multimodal data fusion:** Combine paired BF and FL images into a six-channel input, with shared geometric augmentation to keep the modalities aligned.
- **Transfer learning:** Adapt a pretrained **EfficientNet-B0** model in **PyTorch** to the classification task, using class-weighted loss to account for class imbalance.
- **Generalisation and validation:** Keep each patient's images within one data partition to avoid overlap between training and validation. Track ROC-AUC and learning curves to assess performance on held-out patients.

The notebook follows the workflow from image preparation through model training and evaluation to competition predictions, making the modelling choices and their implementation available for inspection.

## Run the project

The dataset is available on **[Kaggle](https://www.kaggle.com/competitions/multimodal-cancer-classification-challenge-2026/data)** and is not included in this repository. Sign in to Kaggle and accept the competition rules if prompted.

**On Kaggle:** Import the notebook, attach the competition dataset, enable a GPU and internet access, and run the cells in order.

<details>
<summary>Local setup and dataset layout</summary>

1. Download and extract the dataset from Kaggle **outside this repository**. Preserve its layout: `train.csv`, `sampleSubmission.csv`, `BF/train/`, `BF/test/`, `FL/train/`, and `FL/test/` under one dataset directory.
2. From the repository directory, install dependencies and launch JupyterLab:

   ```bash
   python -m pip install -r requirements.txt
   jupyter lab
   ```

3. Open the notebook and change the dataset initialization to `BASE_DIR = find_dataset_dir("/absolute/path/to/dataset")`. Run the cells in order. Internet access is needed for the initial pretrained-weight download; a CUDA GPU is recommended for training.

</details>

The notebook produces a model checkpoint, learning curves, and a submission CSV in `/kaggle/working/` on Kaggle or the current directory locally.
