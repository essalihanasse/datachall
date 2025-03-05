import os
import numpy as np
import rampwf as rw
import wfdb

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
    rw.score_types.NegativeLogLikelihood(name='nll', precision=4),
]

def get_cv(X, y):
    """Return cross-validation splits for training."""
    # Load record IDs to ensure patient-wise splitting
    data_path = os.path.join('.', 'data')
    record_ids = np.load(os.path.join(data_path, 'record_ids.npy'))
    record_ids = record_ids[:len(y)]  # Ensure matching length
    
    # Get unique patient IDs
    unique_records = np.unique(record_ids)
    
    # Create 5-fold cross-validation
    cv = []
    n_splits = 5
    
    # Shuffle patient IDs
    np.random.seed(42)
    np.random.shuffle(unique_records)
    
    # Split into folds
    folds = np.array_split(unique_records, n_splits)
    
    # Create train/test indices for each fold
    for i in range(n_splits):
        test_records = folds[i]
        train_records = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        
        # Get indices for train and test
        test_indices = np.where(np.isin(record_ids, test_records))[0]
        train_indices = np.where(np.isin(record_ids, train_records))[0]
        
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

def get_train_data(path='.'):
    """Get the training data."""
    X, y = _read_data(path)
    
    # Split data into train/test
    data_path = os.path.join(path, 'data')
    record_ids = np.load(os.path.join(data_path, 'record_ids.npy'))
    unique_records = np.unique(record_ids)
    
    # Train/test split by patient ID
    np.random.seed(42)
    np.random.shuffle(unique_records)
    split_idx = int(len(unique_records) * 0.7)
    train_records = unique_records[:split_idx]
    
    # Create mask for training data
    train_mask = np.isin(record_ids, train_records)
    X_train = X[train_mask]
    y_train = y[train_mask]
    
    return X_train, y_train

def get_test_data(path='.'):
    """Get the test data."""
    X, y = _read_data(path)
    
    # Split data into train/test
    data_path = os.path.join(path, 'data')
    record_ids = np.load(os.path.join(data_path, 'record_ids.npy'))
    unique_records = np.unique(record_ids)
    
    # Train/test split by patient ID
    np.random.seed(42)
    np.random.shuffle(unique_records)
    split_idx = int(len(unique_records) * 0.7)
    test_records = unique_records[split_idx:]
    
    # Create mask for test data
    test_mask = np.isin(record_ids, test_records)
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    return X_test, y_test