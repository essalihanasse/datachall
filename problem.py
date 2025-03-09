import os
import numpy as np
import rampwf as rw
import wfdb
from collections import defaultdict
from sklearn.model_selection import StratifiedGroupKFold

# Problem definition
problem_title = 'ECG Heartbeat Classification from MIT-BIH Arrhythmia Database'
_target_column_name = 'heartbeat_class'
_prediction_label_names = [0, 1, 2, 3, 4]  # 0: Normal, 1: Supraventricular, 2: Ventricular, 3: Fusion, 4: Unknown

# AAMI heartbeat classes
BEAT_CLASSES = {
    'N': 0,  # Normal beat 
    'S': 1,  # Supraventricular premature beat
    'V': 2,  # Ventricular premature beat
    'F': 3,  # Fusion beat
    'Q': 4   # Unknown beat
}

# AAMI mapping from MIT-BIH annotations to AAMI classes
MITBIH_TO_AAMI = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',  # Normal
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',            # Supraventricular ectopic
    'V': 'V', 'E': 'V',                                # Ventricular ectopic
    'F': 'F',                                          # Fusion
    '/': 'Q', 'f': 'Q', 'Q': 'Q'                       # Unknown/paced
}

# Define the prediction type
# We're dealing with a multi-class classification task
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names
)

# Define the workflow
workflow = rw.workflows.Estimator()

# Define score metrics
score_types = [
    rw.score_types.Accuracy(name='accuracy', precision=4),
    rw.score_types.F1Above(name='f1_macro', precision=4),
    rw.score_types.BalancedAccuracy(name='Balanced Accuracy', precision=4),
]

def get_cv(X, y):
    """Return stratified cross-validation splits for training."""
    # Load record IDs to ensure patient-wise splitting
    data_path = os.path.join('.', 'data')
    record_ids = np.load(os.path.join(data_path, 'record_ids.npy'))
    record_ids = record_ids[:len(y)]  # Ensure matching length
    
    # Use StratifiedGroupKFold from scikit-learn for stratified patient-wise splitting
    n_splits = 5
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Return the stratified splits
    for train_indices, test_indices in cv.split(X, y, groups=record_ids):
        yield train_indices, test_indices

def download_mitbih_data():
    """Download the MIT-BIH Arrhythmia Database if not already present."""
    data_path = os.path.join('.', 'data')
    os.makedirs(data_path, exist_ok=True)
    
    # Check if the data is already downloaded
    if not os.path.exists(os.path.join(data_path, 'RECORDS')):
        print("Downloading MIT-BIH Arrhythmia Database...")
        wfdb.dl_database('mitdb', data_path)
        print("Download complete!")
    else:
        print("MIT-BIH Arrhythmia Database already downloaded.")

def extract_heartbeats(record_id, data_path, window_size=250):
    """
    Extract heartbeats from a record with associated annotations.
    
    Parameters:
    -----------
    record_id : str
        The record ID to process
    data_path : str
        Path to the data directory
    window_size : int
        The window size around the R-peak
        
    Returns:
    --------
    beats : list of dict
        List of dictionaries containing beat information
    """
    # Read the record
    record_path = os.path.join(data_path, record_id)
    record = wfdb.rdrecord(record_path)
    
    # Get annotations (R-peak locations and beat types)
    annotation = wfdb.rdann(record_path, 'atr')
    
    # Filter to keep only beat annotations
    beat_indices = [i for i, label in enumerate(annotation.symbol) if label in MITBIH_TO_AAMI]
    beat_samples = [annotation.sample[i] for i in beat_indices]
    beat_symbols = [annotation.symbol[i] for i in beat_indices]
    
    # Map to AAMI classes
    beat_classes = [MITBIH_TO_AAMI[symbol] for symbol in beat_symbols]
    
    # Extract beats - fixed window around each R-peak
    beats = []
    signal = record.p_signal[:, 0]  # Use the first channel (MLII)
    
    for sample, symbol, aami_class in zip(beat_samples, beat_symbols, beat_classes):
        # Skip if window extends beyond signal boundaries
        if sample < window_size // 2 or sample >= len(signal) - window_size // 2:
            continue
            
        # Extract the segment
        start = sample - window_size // 2
        end = sample + window_size // 2
        segment = signal[start:end]
        
        # Store beat data
        beat = {
            'record_id': record_id,
            'sample': sample,
            'original_class': symbol,
            'aami_class': aami_class,
            'target': BEAT_CLASSES[aami_class],
            'segment': segment
        }
        beats.append(beat)
    
    return beats

def process_all_records():
    """Process all records in the MIT-BIH database and create dataset."""
    data_path = os.path.join('.', 'data')
    os.makedirs(data_path, exist_ok=True)
    
    # Get record IDs
    with open(os.path.join(data_path, 'RECORDS'), 'r') as f:
        record_ids = [line.strip() for line in f]
    
    # Process each record
    all_beats = []
    for record_id in record_ids:
        print(f"Processing record {record_id}...")
        beats = extract_heartbeats(record_id, data_path)
        all_beats.extend(beats)
    
    print(f"Total extracted beats: {len(all_beats)}")
    
    # Save processed data
    X = np.array([beat['segment'] for beat in all_beats])
    y = np.array([beat['target'] for beat in all_beats])
    record_ids = np.array([beat['record_id'] for beat in all_beats])
    
    np.save(os.path.join(data_path, 'X.npy'), X)
    np.save(os.path.join(data_path, 'y.npy'), y)
    np.save(os.path.join(data_path, 'record_ids.npy'), record_ids)
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print("Class distribution:")
    for class_idx, count in class_dist.items():
        class_name = list(BEAT_CLASSES.keys())[list(BEAT_CLASSES.values()).index(class_idx)]
        print(f"Class {class_name} (AAMI {class_idx}): {count} beats ({count/len(y)*100:.2f}%)")
    
    return X, y, record_ids

def _read_data(path):
    """Read data from processed numpy files."""
    data_path = os.path.join(path, 'data')
    
    # Check if data exists, otherwise process it
    if not os.path.exists(os.path.join(data_path, 'X.npy')):
        download_mitbih_data()
        X, y, _ = process_all_records()
    else:
        X = np.load(os.path.join(data_path, 'X.npy'))
        y = np.load(os.path.join(data_path, 'y.npy'))
    
    return X, y

def stratified_group_split(X, y, groups, test_size=0.3, random_state=42):
    """
    Custom function to perform a stratified group split, ensuring that:
    1. The same groups (patients) are not in both train and test
    2. Class distribution is preserved as much as possible
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Target labels
    groups : array-like
        Group identifiers (patient/record IDs)
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    train_mask, test_mask : boolean arrays
        Masks to apply to the original data
    """
    # Get unique groups and their class distributions
    unique_groups = np.unique(groups)
    np.random.seed(random_state)
    np.random.shuffle(unique_groups)
    
    # Calculate overall class distribution
    unique_classes, overall_class_counts = np.unique(y, return_counts=True)
    overall_class_dist = overall_class_counts / len(y)
    
    # For each group, get the class distribution
    group_class_counts = defaultdict(lambda: np.zeros(len(unique_classes), dtype=int))
    for group in unique_groups:
        group_mask = (groups == group)
        group_y = y[group_mask]
        for i, cls in enumerate(unique_classes):
            group_class_counts[group][i] = np.sum(group_y == cls)
    
    # Greedy algorithm to select groups for test set while maintaining distribution
    test_groups = []
    train_groups = list(unique_groups)  # Start with all groups in train
    test_count = np.zeros(len(unique_classes), dtype=int)
    
    # Target counts for test set
    target_test_count = overall_class_counts * test_size
    
    # Keep selecting groups until we reach desired test size for all classes
    while len(train_groups) > 0 and np.any(test_count < target_test_count):
        best_group = None
        best_fit = float('inf')
        
        for group in train_groups:
            # Calculate how this group would affect the test distribution
            tentative_test_count = test_count + group_class_counts[group]
            # Check if adding this group would exceed target for any class
            if np.any(tentative_test_count > target_test_count * 1.1):  # Allow 10% overflow
                continue
                
            # Calculate the fit as the sum of squared differences from target
            fit = np.sum(((tentative_test_count / np.sum(tentative_test_count) if np.sum(tentative_test_count) > 0 else 0) - 
                           overall_class_dist)**2)
            
            if fit < best_fit:
                best_fit = fit
                best_group = group
        
        # If we found a good group, add it to test set
        if best_group is not None:
            test_groups.append(best_group)
            train_groups.remove(best_group)
            test_count += group_class_counts[best_group]
        else:
            # If no suitable group found, just break
            break
    
    # Create masks
    test_mask = np.isin(groups, test_groups)
    train_mask = ~test_mask
    
    # Print statistics
    print(f"Train set: {np.sum(train_mask)} samples, Test set: {np.sum(test_mask)} samples")
    print("Class distribution in train set:")
    train_class_counts = np.zeros(len(unique_classes), dtype=int)
    for i, cls in enumerate(unique_classes):
        train_class_counts[i] = np.sum(y[train_mask] == cls)
        print(f"Class {cls}: {train_class_counts[i]} ({train_class_counts[i]/np.sum(train_mask)*100:.2f}%)")
    
    print("Class distribution in test set:")
    for i, cls in enumerate(unique_classes):
        test_class_count = np.sum(y[test_mask] == cls)
        print(f"Class {cls}: {test_class_count} ({test_class_count/np.sum(test_mask)*100:.2f}%)")
    
    return train_mask, test_mask

def get_train_data(path='.'):
    """Get the training data using stratified split."""
    X, y = _read_data(path)
    
    # Get record IDs for stratified group split
    data_path = os.path.join(path, 'data')
    record_ids = np.load(os.path.join(data_path, 'record_ids.npy'))
    record_ids = record_ids[:len(y)]  # Ensure matching length
    
    # Perform stratified split
    train_mask, _ = stratified_group_split(X, y, record_ids, test_size=0.3, random_state=42)
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    
    return X_train, y_train

def get_test_data(path='.'):
    """Get the test data using stratified split."""
    X, y = _read_data(path)
    
    # Get record IDs for stratified group split
    data_path = os.path.join(path, 'data')
    record_ids = np.load(os.path.join(data_path, 'record_ids.npy'))
    record_ids = record_ids[:len(y)]  # Ensure matching length
    
    # Perform stratified split
    _, test_mask = stratified_group_split(X, y, record_ids, test_size=0.3, random_state=42)
    
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    return X_test, y_test
