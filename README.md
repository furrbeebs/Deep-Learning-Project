# 50.039 Deep Learning Project

**Group 14**
Team members:
-  Phoebe
-  Lei Lei

### Dataset Setup

Due to file size limitations, the dataset is not included in this repository. To run the training or evaluation scripts, please follow these steps:

1. **Download the Data**: Obtain the `voraus-ad-dataset-100hz.parquet` file from (https://www.kaggle.com/datasets/marcorudolph/vorausad?select=train.py).
2. **Placement**: Place the file in the root directory of this project.
3. **Verification**: Ensure the file name matches exactly as referenced in the notebooks (`voraus-ad-dataset-100hz.parquet`).

The project is configured with a `.gitignore` to ensure this dataset is not accidentally uploaded to GitHub.

# Anomaly Detection — Data Preparation

## Overview
This notebook prepares raw robot sensor data for unsupervised LSTM autoencoder training.
The model is trained exclusively on normal samples and detects anomalies via reconstruction error.

## Requirements
source venv/bin/activate
pip install -r requirements.txt

## Setup
1. Download the raw parquet file and place it under root directory:
   voraus-ad-dataset-100hz.parquet

3. Run all 9 cells in order in:
   01_data_preparation.ipynb

4. Verify Cell 9 prints "✓ All checks passed" before proceeding.

## Output Files (written to data/processed/)
| File                  | Description                                      |
|-----------------------|--------------------------------------------------|
| splits.npz            | Scaled arrays: X_train, X_val, X_test, y_test   |
| scaler.pkl            | Fitted StandardScaler (train only)               |
| train_meta.parquet    | Sample IDs in training split                     |
| val_meta.parquet      | Sample IDs in validation split                   |
| test_meta.parquet     | Sample IDs in test split                         |

## Data Structure
- Raw file: 2,122 samples × ~1,094 timesteps × 137 columns
- 755 anomalous samples (categories 0–11), 1,367 normal samples (category 12)
- 130 sensor feature columns used (meta columns dropped)

## Key Design Decisions
| Decision                        | Reason                                              |
|---------------------------------|-----------------------------------------------------|
| Train on normal samples only    | Unsupervised — model learns normal behaviour        |
| MAX_LEN = 1028                  | Minimum timesteps across all normal samples         |
| Tail truncation (seq[-1028:])   | Most recent timesteps are most diagnostic           |
| Scaler fit on train only        | Prevents data leakage into val/test                 |
| Zero-pad only at test time      | 8 short sequences are all anomalies — not in train  |
| Test set = normals + anomalies  | Realistic evaluation of detection performance       |

## Splits
| Split | Normal | Anomaly | Total |
|-------|--------|---------|-------|
| Train | 956    | 0       | 956   |
| Val   | 205    | 0       | 205   |
| Test  | 206    | 755     | 961   |