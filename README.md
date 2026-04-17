# 50.039 Deep Learning Project


## Problem 
In high-precision manufacturing, collaborative robots operate in close proximity with humans. Subtle mechanical irregularities (e.g. joint wear and collisions) are often times invisible to traditional monitoring and may lead to disastrous downtime and safety hazards.
This deep learning project aims to build a deep learning model that learns the intricate sensor patterns of a 'normal’ pick-and-place cycle, effectively predicting system failures before they occur and reduce the safety hazards and downtime of manufacturing.

------
### Prerequisites

Ensure the following are installed:

- Python 3.8 or higher
- `pip` (Python package manager)

Check your Python version:

```bash
python3 --version
````

---

### 1. Create the virtual environment

From the root directory of the project, run:

```bash
python3 -m venv venv
```

This creates a folder named `venv/` in the project directory.

Project structure example:

```text
CDS/
├── venv/
├── requirements.txt
├── src/
└── README.md
```

---

### 2. Activate the virtual environment

**macOS / Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

After activation, your terminal should show:

```text
(venv)
```

---

### 3. Install project dependencies

```bash
pip install -r requirements.txt
```

> Note: Additional packages may be installed automatically as **dependencies** of the listed packages.

---

### 4. Verify installation (optional)

```bash
pip list
```

---

### 5. Using the environment in VS Code

1. Open the Command Palette

   * macOS: `Cmd + Shift + P`
   * Windows/Linux: `Ctrl + Shift + P`

2. Search for:

```text
Python: Select Interpreter
```

3. Select the interpreter located at:

```text
./venv/bin/python
```

VS Code will now use this environment for running and debugging the project.

---

### 6. Deactivating the environment

When you are done working, deactivate the virtual environment with:

```bash
deactivate
```
---
## Details
**Dataset (100Hz version)**: https://www.kaggle.com/datasets/marcorudolph/vorausad

**Input:** 300 input vectors of numbers

**Ouput:** Mean squared error (numerical difference between the input and reconstructed numbers)

**Preliminary Architecture:** Multi-layer perceptron autoencoder
- **Encoder:** shrink the inputs to smaller numbers
- **Decoder:** tries to expand the numbers back to the original
- **Activation:** using ReLU for hidden layers


**Why is the input 300 vector of numbers?**

The data of multivariate sensor signals will be transformed into discrete samples. Each sample is flattened to a vector of 300 features (20 timesteps * 15 sensors = 300). This vector captures the sensor values and context of the robot movement.


**The dataset has 130 sensors, why did we select 15 and what are they?**

We intend to focus on the vital sensors that cover the majority of the robot's mechanical health: torque, velocity, and load.

The 15 sensors are:

`'motor_torque_1', 'motor_torque_2', 'motor_torque_3', 
    'motor_torque_4', 'motor_torque_5', 'motor_torque_6',
    'joint_velocity_1', 'joint_velocity_2', 'joint_velocity_3', 
    'joint_velocity_4', 'joint_velocity_5', 'joint_velocity_6',
    'robot_current', 'robot_voltage', 'system_current'`

The other sensors (columns) that have been dropped are in these categories:
- Target positions
- Digital I/O
- Temperature

------

## Dataset Setup

Due to file size limitations, the dataset is not included in this repository. To run the training or evaluation scripts, please follow these steps:

1. **Download the Data**: Obtain the `voraus-ad-dataset-100hz.parquet` file from (https://www.kaggle.com/datasets/marcorudolph/vorausad?select=train.py).
2. **Placement**: Place the dataset in the **root** directory of this project.
3. **Verification**: Ensure the file name matches exactly as referenced in the notebooks (`voraus-ad-dataset-100hz.parquet`).

The project is configured with a `.gitignore` to ensure this dataset is not accidentally uploaded to GitHub.

------

## Data Preparation

### Overview
This notebook prepares raw robot sensor data for unsupervised MLP Autoencoder training.
The model is trained exclusively on normal samples using a sliding window appraoch. It learns to reconstruct a 300-dimension input vector and detect anomalies via Mean Squared Error (MSE) reconstruction loss.

### Requirements
In bash, run the following:
`.\venv\Scripts\activate`
`pip install -r requirements.txt`

### Data Preprocessing

1. Run `00_data_analysis.ipynb`

2. Run `01_data_preparation.ipynb`

3. Verify `01_data_preparation.ipynb`: Cell 9 prints **"✓ All checks passed"**.

### Training 

1. Run `02_train.ipynb`

### Infering results

1. In bash, run the following:
`git add -f robot_autoencoder.pth`
`git add -f data/processed/scaler.pkl`

2. Run `03_inference.ipynb`

### Output Files (written to data/processed/)
| File                  | Description                                      |
|-----------------------|--------------------------------------------------|
| splits.npz            | Scaled arrays: X_train, X_val, X_test, y_test   |
| scaler.pkl            | Fitted StandardScaler (applied to 15 sensors)               |
| train_meta.parquet    | Sample IDs used for training windows                     |
| val_meta.parquet      | Sample IDs used for validation windows                   |
| test_meta.parquet     | Sample IDs used for testing windows                        |

### Data Structure
- **Raw file:** 2,122 samples × ~1,094 timesteps × 137 columns
- **Features:** 15 sensors selected
- **Processed shape:** Each sample is a 300-dimension vector (15 sensors * 20 timesteps)
- **Total training windows:** 204,854

### Key Design Decisions
| Decision                        | Reason                                              |
|---------------------------------|-----------------------------------------------------|
| Sliding Window (Size 20)    | Captures temporal patterns while maintaining a fixed input size for MLP.        |
| 300-Dimension Input                  | 15 sensors * 20 steps focuses on diagnostic mechanical vitals.         |
| Step Size = 5   | Overlapping windows provide data augmentation for a robust normal baseline.           |
| Unsupervised Training        | Model learns only "Normal" behavior; no anomalies seen during training.                 |
| float32 Conversion      | Reduces memory footprint by 50% and ensures PyTorch compatibility.
| StandardScaler (Train-only)  | Prevents data leakage; ensures all sensors have equal weight.       |

### Splits
| Split | Normal | Anomaly | Total |
|-------|--------|---------|-------|
| Train | 204,854    | 0       | 204,854   |
| Val   | 43,865    | 0       | 43,865   |
| Test  | 44,076    | 163,999     | 208,075   |

------

## LSTM Binary Classifier (Supervised Anomaly Detection)

### Overview

This LSTM-based binary classifier directly predicts whether a robot operation window is anomalous or normal. Unlike the autoencoder approaches (Parts A & B), this is a **supervised** model trained on labeled normal and anomalous windows.

### Why LSTM Classifier?

While autoencoders detect anomalies via reconstruction error, a supervised classifier can:
- Learn explicit decision boundaries between normal and anomalous patterns
- Achieve higher accuracy with sufficient labeled data
- Provide direct anomaly probabilities (0-1) without threshold tuning on reconstruction error

### Key Differences from Autoencoders

| Aspect | MLP Autoencoder (A) | LSTM Autoencoder (B) | LSTM Classifier (C) |
|--------|--------------------|----------------------|---------------------|
| Approach | Unsupervised | Unsupervised | Supervised |
| Output | Reconstruction MSE | Reconstruction MSE | Anomaly probability |
| Training data | Normal only | Normal only | Normal + Anomaly |

### Data Preparation

Ensure that Dataset Setup is done.
Run `03_data_prep_lstm_classifier.ipynb`

**Key Operations:**
- Redefines anomaly label: category 12 = normal, categories 0–11 = anomaly
- Selects all 129 sensor columns (after removing constant columns)
- Extracts sliding windows: size = 500 timesteps (5 seconds), stride = 50 (90% overlap)
- Splits at sample level to prevent temporal leakage

**Split Ratios:**
| Split | Normal | Anomaly |
|-------|--------|---------|
| Train | 70% | 40% |
| Validation | 15% | 30% |
| Test | 15% | 30% |

**Output Files:**
- `../data/lstm_classifier/sequences.npz` — Window arrays
- `../data/lstm_classifier/scaler.pkl` — Fitted StandardScaler
- `../data/lstm_classifier/config.pkl` — Configuration

### Model Architecture

```
Input: (batch, 500 timesteps, 129 features)
    ↓
LSTM (1 layer, hidden_dim=64)
    ↓
Dropout (p=0.3)
    ↓
Linear (64 → 32) + ReLU
    ↓
Linear (32 → 1) + Sigmoid
    ↓
Output: anomaly probability ∈ (0, 1)
```

### Hyperparameter Grid Search Results

| Config | hidden_dim | layers | dropout | lr | Val AUC |
|--------|------------|--------|---------|-----|---------|
| 1 | 32 | 1 | 0.2 | 1e-3 | 0.9520 |
| **2 (selected)** | **64** | **1** | **0.3** | **1e-3** | **0.9580** |
| 3 | 64 | 2 | 0.3 | 1e-3 | 0.9522 |
| 4 | 128 | 1 | 0.4 | 5e-4 | 0.9476 |

### Training

Run `lstm_classifier_model.ipynb`

**Training Parameters:**
- Loss: Binary Cross-Entropy (BCE)
- Optimizer: Adam (weight_decay = 1e-5)
- LR Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)
- Batch size: 32, Epochs: 20

**Balanced Subsets (prevent majority-class bias):**
| Subset | Normal | Anomaly |
|--------|--------|---------|
| Train | 1,500 | 1,500 |
| Validation | 500 | 500 |
| Test | 500 | 500 |

### Output Files
| File | Description |
|------|-------------|
| `../model/lstm_classifier/lstm_classifier_best.pth` | Best model weights |
| `../model/lstm_classifier/lstm_classifier_config.pkl` | Architecture config |
| `../figures/lstm_classifier/training_curves.png` | Loss & AUC curves |
| `../figures/lstm_classifier/roc_and_prediction_dist.png` | ROC curve + prediction distribution |
| `../figures/lstm_classifier/confusion_matrix.png` | Confusion matrix |
| `../figures/lstm_classifier/failure_cases.png` | FP/FN example windows |

### Performance Summary

| Metric | Balanced Test | Full Test |
|--------|---------------|-----------|
| AUC | 0.9876 | 0.9856 |
| Accuracy | 95% | 96% |
| Precision (anomaly) | 0.95 | 0.96 |
| Recall (anomaly) | 0.95 | 0.96 |
| F1 (anomaly) | 0.95 | 0.96 |

### Reproducibility
All results use `RANDOM_SEED = 14`. The trained model files are saved under:
- `./model/lstm_classifier/lstm_classifier_best.pth` (weights)
- `./model/lstm_classifier/lstm_classifier_config.pkl` (architecture config)

To load and use the model on new data without retraining, refer to **part 12** in `lstm_classifier_model.ipynb`, which provides complete code examples for model loading and inference.