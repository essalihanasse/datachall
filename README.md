# ECG Heartbeat Classification
[![Build status](https://github.com/ramp-kits/ecg-heartbeat-classification/actions/workflows/test.yml/badge.svg)](https://github.com/ramp-kits/ecg-heartbeat-classification/actions/workflows/test.yml)

## Introduction

Electrocardiogram (ECG) is a critical tool for detecting cardiac abnormalities. This challenge aims to automatically classify heartbeats from the MIT-BIH Arrhythmia Database into five standard classes defined by the Association for the Advancement of Medical Instrumentation (AAMI):

- **N**: Normal beats
- **S**: Supraventricular ectopic beats
- **V**: Ventricular ectopic beats
- **F**: Fusion beats
- **Q**: Unknown beats

Accurate classification of heartbeats is essential for:
- Early detection of cardiac arrhythmias
- Long-term heart monitoring
- Automated ECG analysis in clinical settings
- Reducing the workload of healthcare professionals

## Getting started

### Install

To run a submission and the notebook, you will need to install the required dependencies. You have two options:

#### Option 1: Using pip

You can install the dependencies listed in `requirements.txt` with the following command:

```bash
pip install -U -r requirements.txt
```

#### Option 2: Using conda

If you prefer using conda, you can create an environment with all the required dependencies:

```bash
conda env create -f environment.yml
conda activate ecg-heartbeat-classification
```

This will create a new conda environment named `ecg-heartbeat-classification` with all the necessary packages installed.


### Test a submission

The submissions need to be located in the `submissions` folder. For instance, for `auto_encoder`, it should be located in `submissions/auto_encoder`.

To run a specific submission, you can use the `ramp-test` command line:

```bash
ramp-test --submission auto_encoder
```

You can get more information regarding this command line:

```bash
ramp-test --help
```

## Dataset

The challenge uses the MIT-BIH Arrhythmia Database, which contains 48 half-hour excerpts of two-channel ambulatory ECG recordings from 47 subjects. The database includes annotations for each heartbeat, which are mapped to the AAMI classes.

### MIT-BIH to AAMI Mapping

The original MIT-BIH annotations are mapped to AAMI classes as follows:

- **N (Normal)**: N, L, R, e, j
- **S (Supraventricular ectopic)**: A, a, J, S
- **V (Ventricular ectopic)**: V, E
- **F (Fusion)**: F
- **Q (Unknown/paced)**: /, f, Q

## Data Preprocessing

The `problem.py` script handles the data preprocessing pipeline:

1. **Download MIT-BIH Database**: The database is automatically downloaded if not already present.
2. **Extract Heartbeats**: Individual heartbeats are extracted using a temporal window around the R-peak.
3. **Map Annotations**: Heartbeat annotations are mapped to the AAMI classes.
4. **Split Data**: Data is split into training and testing sets using a patient-based approach.

To run the data preprocessing:

```python
python -c "from problem import process_all_records; process_all_records()"
```

## Feature Extraction Approaches

### 1. Traditional Feature Extraction (Starting Kit)

The basic approach extracts time-domain and frequency-domain features from the ECG signals:

- Statistical features (mean, standard deviation, min, max)
- Morphological features (peak-to-peak amplitude, zero crossings)
- Frequency domain features using Welch's method

### 2. Autoencoder-based Feature Extraction

The advanced approach uses a convolutional autoencoder to learn a compact representation of the ECG signals:

- Convolutional layers to capture temporal patterns
- Latent space representation used as features
- Unsupervised learning to capture inherent signal characteristics

## Performance Metrics

The performance of the models is evaluated using multiple metrics:

- **Accuracy**: Overall classification accuracy
- **F1 Score (Macro)**: Balanced F1 score across all classes
- **Balanced Accuracy**: Accuracy that accounts for class imbalance

## Class Imbalance

The MIT-BIH dataset has significant class imbalance, with the Normal (N) class being much more frequent than others. The challenge addresses this through:

1. Patient-based stratified splitting to maintain class distribution
2. Using `class_weight='balanced'` in the SVM classifier
3. Evaluating with balanced metrics (F1 macro, balanced accuracy)

## Advanced Features

### Autoencoder Architecture

The autoencoder uses a convolutional architecture:

- **Encoder**: Conv1D layers with max pooling to reduce dimensionality
- **Latent Space**: Dense layer with 32 units
- **Decoder**: Conv1D layers with upsampling to reconstruct the original signal

The autoencoder is trained to minimize reconstruction error, and the latent space representation serves as compact features for classification.

### Patient-wise Cross-validation

To ensure robust evaluation, the challenge implements patient-wise cross-validation using `StratifiedGroupKFold`. This ensures that heartbeats from the same patient don't appear in both training and validation sets.

## Project Structure

```
.
├── data/                        # Directory for storing the MIT-BIH database
├── submissions/                 # Model implementations
│   ├── auto_encoder/           # Autoencoder-based feature extraction
│   │   └── estimator.py        # Implementation of the autoencoder pipeline
│   └── starting_kit/           # Basic feature extraction
│       └── estimator.py        # Implementation of the baseline pipeline
├── .gitignore                  # Git ignore file
├── environment.yaml            # Yaml file for conda users
├── problem.py                  # Core functionality for data processing
├── README.md                   # Project documentation
└── requirements.txt            # Project dependencies
```

## To go further

You can find more information regarding `ramp-workflow` in the [dedicated documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html)

### Future Improvements

Potential areas for enhancement:

1. **Deep Learning Models**: Implement end-to-end deep learning architectures
2. **Data Augmentation**: Address class imbalance through augmentation techniques
3. **Explainability**: Add model interpretation to understand important features
4. **Real-time Classification**: Optimize for real-time processing
5. **Cross-database Validation**: Test on additional ECG databases



## Acknowledgments

This project uses the MIT-BIH Arrhythmia Database from PhysioNet https://physionet.org/content/mitdb/1.0.0/.

This project also uses the RAMP framework (Rapid Analytics and Model Prototyping) https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/index.html .
