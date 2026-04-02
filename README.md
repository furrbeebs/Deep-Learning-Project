# 50.039 Deep Learning Project

**Group 14**
Team members:
-  Phoebe
-  Lei Lei

### Problem 
In high-precision manufacturing, collaborative robots operate in close proximity with humans. Subtle mechanical irregularities (e.g. joint wear and collisions) are often times invisible to traditional monitoring and may lead to disastrous downtime and safety hazards.
This deep learning project aims to build a deep learning model that learns the intricate sensor patterns of a 'normal’ pick-and-place cycle, effectively predicting system failures before they occur and reduce the safety hazards and downtime of manufacturing.

#### Details
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

We intend to focus on the vital sensors that cover the majority of the robot's mechanical health: torque, veloctiy, and load.

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

### Dataset Setup

Due to file size limitations, the dataset is not included in this repository. To run the training or evaluation scripts, please follow these steps:

1. **Download the Data**: Obtain the `voraus-ad-dataset-100hz.parquet` file from (https://www.kaggle.com/datasets/marcorudolph/vorausad?select=train.py).
2. **Placement**: Place the dataset in the **root** directory of this project.
3. **Verification**: Ensure the file name matches exactly as referenced in the notebooks (`voraus-ad-dataset-100hz.parquet`).

The project is configured with a `.gitignore` to ensure this dataset is not accidentally uploaded to GitHub.

# Anomaly Detection — Data Preparation

## Overview
This notebook prepares raw robot sensor data for unsupervised MLP Autoencoder training.
The model is trained exclusively on normal samples using a sliding window appraoch. It learns to reconstruct a 300-dimension input vector and detect anomalies via Mean Squared Error (MSE) reconstruction loss.

## Requirements
`source venv/bin/activate`
`pip install -r requirements.txt`

## Data Preprocessing

1. Run `00_data_analysis.ipynb`

2. Run `01_data_preparation.ipynb`

3. Verify `01_data_preparation.ipynb`: Cell 9 prints **"✓ All checks passed"**.

## Output Files (written to data/processed/)
| File                  | Description                                      |
|-----------------------|--------------------------------------------------|
| splits.npz            | Scaled arrays: X_train, X_val, X_test, y_test   |
| scaler.pkl            | Fitted StandardScaler (applied to 15 sensors)               |
| train_meta.parquet    | Sample IDs used for training windows                     |
| val_meta.parquet      | Sample IDs used for validation windows                   |
| test_meta.parquet     | Sample IDs used for testing windows                        |

## Data Structure
- **Raw file:** 2,122 samples × ~1,094 timesteps × 137 columns
- **Features:** 15 sensors selected
- **Processed shape:** Each sample is a 300-dimension vector (15 sensors * 20 timesteps)
- **Total training windows:** 204,854

## Key Design Decisions
| Decision                        | Reason                                              |
|---------------------------------|-----------------------------------------------------|
| Sliding Window (Size 20)    | Captures temporal patterns while maintaining a fixed input size for MLP.        |
| 300-Dimension Input                  | 15 sensors * 20 steps focuses on diagnostic mechanical vitals.         |
| Step Size = 5   | Overlapping windows provide data augmentation for a robust normal baseline.           |
| Unsupervised Training        | Model learns only "Normal" behavior; no anomalies seen during training.                 |
| float32 Conversion      | Reduces memory footprint by 50% and ensures PyTorch compatibility.
| StandardScaler (Train-only)  | Prevents data leakage; ensures all sensors have equal weight.       |

## Splits
| Split | Normal | Anomaly | Total |
|-------|--------|---------|-------|
| Train | 204,854    | 0       | 204,854   |
| Val   | 43,865    | 0       | 43,865   |
| Test  | 44,076    | 163,999     | 208,075   |