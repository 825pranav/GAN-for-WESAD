# GAN-for-WESAD

This project is an implementation of a TimeGAN (Time-series Generative Adversarial Network) in PyTorch, trained on the WESAD (Wearable Stress and Affect Detection) dataset.

The goal is to generate realistic synthetic physiological time-series data (ECG, EDA, Respiration).

## 🚀 Running the Project

Here is the step-by-step guide to set up the environment and run the complete TimeGAN pipeline.

### 📋 Prerequisites

* An **NVIDIA GPU** is required for training.
* **WSL2 (Windows Subsystem for Linux)** with an Ubuntu distribution (required for GPU access on Windows).
* **Python 3.10+**
* The **WESAD Dataset** — download it and place the unzipped folder here:
    ```
    GAN-for-WESAD/WESAD/
    ```

### 1. Setup & Installation

1.  Open your **WSL/Ubuntu terminal**.
2.  Clone this repository:
    ```bash
    git clone [https://github.com/825pranav/GAN-for-WESAD.git](https://github.com/825pranav/GAN-for-WESAD.git)
    cd GAN-for-WESAD
    ```
3.  Install the required Python packages:
    ```bash
    pip install torch torchvision numpy scikit-learn joblib matplotlib
    ```
    (Note: Installing `torch` inside WSL2 automatically installs the correct CUDA version.)

### 2. Running the Pipeline

The project must be run in this specific order.

#### Step 1: Preprocess the Data

Run the `main.py` script.

This will:
* Load the raw WESAD data
* Downsample & filter
* Save clean sequences:
    * `baseline_seq.npy`
    * `stress_seq.npy`
* Save normalization scaler:
    * `scaler.gz`

Run:
```bash
python3 main.py
```

#### Step 2: Train the TimeGAN

This script:

* Loads `.npy` sequences
* Trains the Autoencoder → saves `autoencoder.pth`
* Runs adversarial TimeGAN training
* Saves:
    * `generator.pth`
    * `discriminator.pth`
* Generates 1000 synthetic samples → `synthetic_wesad_data.npy`

Run:
```bash
python3 timegan.py
