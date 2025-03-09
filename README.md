# ECG Heartbeat Classification

This project implements a machine learning pipeline for classifying ECG heartbeats into different categories according to the Association for the Advancement of Medical Instrumentation (AAMI) standards using the MIT-BIH Arrhythmia Database.

## Project Overview

Electrocardiogram (ECG) is a critical tool for detecting cardiac abnormalities. This project aims to automatically classify heartbeats into five AAMI standard classes:

- **N**: Normal beats
- **S**: Supraventricular ectopic beats
- **V**: Ventricular ectopic beats
- **F**: Fusion beats
- **Q**: Unknown beats

The classification process involves extracting individual heartbeats from ECG recordings, preprocessing the data, extracting relevant features, and training machine learning models for classification.

## Dataset

The project uses the MIT-BIH Arrhythmia Database, which contains 48 half-hour excerpts of two-channel ambulatory ECG recordings from 47 subjects. The database includes annotations for each heartbeat, which are mapped to the AAMI classes for this project.

### MIT-BIH to AAMI Mapping

The original MIT-BIH annotations are mapped to AAMI classes as follows:

- **N (Normal)**: N, L, R, e, j
- **S (Supraventricular ectopic)**: A, a, J, S
- **V (Ventricular ectopic)**: V, E
- **F (Fusion)**: F
- **Q (Unknown/paced)**: /, f, Q

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
├── problem.py                  # Core functionality for data processing
├── README.md                   # Project documentation
└── requirements.txt            # Project dependencies
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/username/ecg-heartbeat-classification.git
   cd ecg-heartbeat-classification
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

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

## Training Models

To train and evaluate a model:

```python
from problem import get_train_data, get_test_data
from submissions.auto_encoder.estimator import get_estimator

# Load data
X_train, y_train = get_train_data()
X_test, y_test = get_test_data()

# Initialize and train the model
model = get_estimator()
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"Accuracy: {score:.4f}")

# Make predictions
predictions = model.predict(X_test)
```

## Performance Metrics

The performance of the models is evaluated using multiple metrics:

- **Accuracy**: Overall classification accuracy
- **F1 Score (Macro)**: Balanced F1 score across all classes
- **Balanced Accuracy**: Accuracy that accounts for class imbalance

## Class Imbalance

The MIT-BIH dataset has significant class imbalance, with the Normal (N) class being much more frequent than others. The project addresses this through:

1. Patient-based stratified splitting to maintain class distribution
2. Using `class_weight='balanced'` in the SVM classifier
3. Evaluating with balanced metrics (F1 macro, balanced accuracy)

## Exploratory Data Analysis

Before building models, it's beneficial to perform exploratory data analysis:

1. **Visualize Label Distribution**: Understand the class imbalance
2. **Display Sample Heartbeats**: Observe morphological differences between classes
3. **Perform PCA**: Visualize potential class separation in lower dimensions
4. **Calculate Basic Statistics**: Analyze statistical properties of each class

## Advanced Features

### Autoencoder Architecture

The autoencoder uses a convolutional architecture:

- **Encoder**: Conv1D layers with max pooling to reduce dimensionality
- **Latent Space**: Dense layer with 32 units
- **Decoder**: Conv1D layers with upsampling to reconstruct the original signal

The autoencoder is trained to minimize reconstruction error, and the latent space representation serves as compact features for classification.

### Patient-wise Cross-validation

To ensure robust evaluation, the project implements patient-wise cross-validation using `StratifiedGroupKFold`. This ensures that heartbeats from the same patient don't appear in both training and validation sets.

## Contributing

To contribute to the project:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Implement your changes
4. Run tests
5. Submit a pull request

## Future Improvements

Potential areas for enhancement:

1. **Deep Learning Models**: Implement end-to-end deep learning architectures
2. **Data Augmentation**: Address class imbalance through augmentation techniques
3. **Explainability**: Add model interpretation to understand important features
4. **Real-time Classification**: Optimize for real-time processing
5. **Cross-database Validation**: Test on additional ECG databases


## Acknowledgments

This project uses the MIT-BIH Arrhythmia Database from PhysioNet https://physionet.org/content/mitdb/1.0.0/ and ramp framework https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/index.html

