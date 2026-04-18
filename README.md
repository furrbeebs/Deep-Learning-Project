# 50.039 Deep Learning Project


## Problem 
In high-precision manufacturing, collaborative robots operate in close proximity with humans. Subtle mechanical irregularities (e.g. joint wear and collisions) are often times invisible to traditional monitoring and may lead to disastrous downtime and safety hazards.
This deep learning project aims to build a deep learning model that learns the intricate sensor patterns of a 'normal’ pick-and-place cycle, effectively predicting system failures before they occur and reduce the safety hazards and downtime of manufacturing.

------
### Overview
This project implements three deep learning architectures to detect mechanical anomalies in robot sensor data.

### Model Architectures
- **MLP Autoencoder:** Baseline model using flattened 300 dimension vectors (20 timesteps x 15 features)
- **LSTM Autoencoder:** Temporal aware model to capture sequence dependecies in sensor data.
- **LSTM Classifier:** A supervised many-to-one model that processes 5-second multivariate sensor sequences to predict a single binary anomaly probability. 

------
### Prerequisites

Ensure the following are installed:

- Python 3.8 or higher
- `pip` (Python package manager)

Check your Python version:

```bash
python3 --version
```

---

### 1. Create the virtual environment

From the root directory of the project, run:

```bash
python3 -m venv venv
```

This creates a folder named `venv/` in the project directory.

Project structure example:

```text

├── data/
├── model/
├── notebooks/
│   ├── model_lstm_autoencoder.ipynb
│   └── model_mlp_autoencoder.ipynb
│   └── 02_data_prep_lstm_classifier.ipynb
│   └── lstm_classifier_model.ipynb
├── venv/
├── .gitignore
├── 00_data_analysis.ipynb
├── 01_data_preparation.ipynb
├── README.md
└── requirements.txt
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

### Requirements
In bash, run the following:
`.\venv\Scripts\activate`
`pip install -r requirements.txt`

### Data Preprocessing

1. Run `00_data_analysis.ipynb`

2. Run `01_data_preparation.ipynb`

3. Verify `01_data_preparation.ipynb`: Cell 9 prints **"✓ All checks passed"**.


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

### Splits
| Split | Normal | Anomaly | Total |
|-------|--------|---------|-------|
| Train | 204,854    | 0       | 204,854   |
| Val   | 43,865    | 0       | 43,865   |
| Test  | 44,076    | 163,999     | 208,075   |

------
## MLP Autoencoder (Baseline Unsupervised)

### Overview

The MLP Autoencoder serves as the baseline supervised model. It learns a compressed representation of "Normal" robot behaviour. By reconstructing input data, it identifies anomalies as windows with high Mean Squared Error (MSE).

### Key Design Decisions

| Decision                  | Reason                                      |
|-----------------------|--------------------------------------------------|
| Sliding Window (Size 20)            | Captures temporal snapshots while maintaining fixed input size for MLP   |
| 300-dimension input            | 15 sensors x 20 steps focused on mechanical vitals              |
| Bottleneck (300 → 64 → 300)    | Forces the model to learn the most important features of normal operation                     |
| Zero false alarm threshold      | Calibrated to Max-Val MSE (0.2801) to prioritise industrial uptime                   |

### Model Architecture
```
Input: (batch, 300)
    ↓
Linear (300 → 128) + ReLU
    ↓
Linear (128 → 64) + ReLU (Bottleneck)
    ↓
Linear (64 → 128) + ReLU
    ↓
Linear (128 → 300)
    ↓
Output: Reconstructed 300-dim vector
```

------
## LSTM Autoencoder (Temporal Unsupervised)

### Overview

Unlike the MLP, LSTM Autoencoder treats the sensor data as a sequence, using the endoer-decoder architecture to reconstruct order of events, making it more sensitive to timing-based anomalies.

### Key Design Decisions

| Decision                  | Reason                                      |
|-----------------------|--------------------------------------------------|
| Sequential input            | Maintains data as (20, 15) to preserve temporal dependecies   |
| Hidden Dim (64)            | Balances model capacity with memory constraints of sequential processing              |
| Batch size (128)    | Optimised for stable gradient descent                     |
| Cycle-level evaluation      | Aggregates window errros to flag anomalies that emerge over long robot tasks                   |

### Model Architecture
```
Input: (batch, 20, 15)
    ↓
LSTM Encoder (Hidden: 64) -> Returns Last Hidden State
    ↓
Repeat Vector (20 times)
    ↓
LSTM Decoder (Hidden: 64) -> Returns Sequence
    ↓
TimeDistributed Linear (64 → 15)
    ↓
Output: Reconstructed (batch, 20, 15)
```

-----
### Thresholds of MLP and LSTM Autoencoders
| Model | Input Shape | Architecture | Calibration Threshold |
|--------|--------------------|----------------------|---------------------|
| MLP Autoencoder | Flattened (300) | 4 layer dense (Bottleneck: 64) | **0.2801** (Max-Val MSE) |
| LSTM Autoencoder | Sequential (20, 15) | Encoder-Decoder LSTM (64 units) | **2.7171** (Max-Val MSE) |

### Performance

- **The "Detection Gap":** Evalutation revealed that while both models achieved **zero false alarms**, they suffered from low recall (0% at window-level).
- **Cycle-level lift:** By aggregating window errors across a full robot cycle, the MLP achieved **14% recall**, proving temporal aggregation is superior to isolated window analysis.

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
Run `02_data_prep_lstm_classifier.ipynb`

**Split Ratios:**
| Split | Normal | Anomaly |
|-------|--------|---------|
| Train | 70% | 40% |
| Validation | 15% | 30% |
| Test | 15% | 30% |

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

### Performance Summary

| Metric | Balanced Subset Test | Full Test |
|--------|---------------|-----------|
| AUC | 0.9876 | 0.9856 |
| Accuracy | 95% | 96% |
| Precision (anomaly) | 0.95 | 0.96 |
| Recall (anomaly) | 0.95 | 0.96 |
| F1 (anomaly) | 0.95 | 0.96 |

### Reproducibility
All results use `RANDOM_SEED = 14`. The pre-trained model files are saved under:
- `./model/lstm_classifier/lstm_classifier_best.pth` (weights)
- `./model/lstm_classifier/lstm_classifier_config.pkl` (architecture config)

To load and use the model on new data without retraining, refer to **part 12** in `lstm_classifier_model.ipynb`, which provides complete code examples for model loading and inference.