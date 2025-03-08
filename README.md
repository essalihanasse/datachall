# ECG Heartbeat Classification

This document describes a process for ECG heartbeat classification using the MIT-BIH Arrhythmia Database. The goal is to categorize each heartbeat into one of the classes defined by the Association for the Advancement of Medical Instrumentation (AAMI).

The process involves several steps, from data extraction and preprocessing to exploratory data analysis and feature extraction.

## Data Preprocessing

The first step is to **extract individual heartbeats** from the raw ECG recordings and their annotations. This is done using a temporal window around the R-peak of each heartbeat. The annotations in the MIT-BIH database are **mapped to the five main AAMI classes**: Normal (N), Supraventricular (S), Ventricular (V), Fusion (F), and Unclassifiable (Q).

The preprocessed data, including the heartbeat segments, their AAMI labels, and the record identifiers, are then saved. A **distribution of AAMI classes** is also calculated to understand the distribution of different types of heartbeats in the database.

The data is divided into training and test sets using a **patient-based split** approach. This means that all heartbeats from one set of patients are used for training, while heartbeats from another set of patients are reserved for testing. A **70% training ratio** is used, and a random seed is set to ensure reproducibility of the split.

## Exploratory Data Analysis (EDA)

An exploratory data analysis is performed to better understand the characteristics of the database and the different types of heartbeats. The main steps of the EDA include:

*   **Loading MIT-BIH data** from a specified directory.
*   **Visualizing the label distribution** of heartbeats to identify majority and minority classes.
*   **Displaying sample heartbeats** for each label to observe morphological differences between classes.
*   Performing **Principal Component Analysis (PCA)** to reduce the dimensionality of the heartbeat data and visualize the potential separation between classes.
*   **Calculating and displaying basic statistics** (mean, standard deviation, min, max, median) for each heartbeat label.
*   **Visualizing correlations** between the mean heartbeats of different labels to identify similarities and differences.
*   Creating **2D histograms** to visualize the distribution of ECG signal amplitudes over time for specific classes of heartbeats.

## Feature Extraction

To improve the performance of classification models, **feature extraction** techniques can be applied. The use of an **ECG autoencoder** is considered. An autoencoder is a type of neural network that learns a compressed representation (latent space) of the input data. This latent representation can then be used as a set of features for classification. The autoencoder can be pre-trained on ECG data to capture important temporal patterns present in the heartbeat signals.

In summary, this process allows for the extraction, preprocessing, and analysis of ECG heartbeat data from the MIT-BIH database for automatic classification according to AAMI standards, potentially using features extracted by an autoencoder.

