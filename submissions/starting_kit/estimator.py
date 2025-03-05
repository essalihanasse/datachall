import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class HeartbeatFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Feature extraction for heartbeat classification.
    
    This transformer extracts time-domain and frequency-domain features
    from raw ECG heartbeat segments.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        """Fit feature extractor."""
        features = self._extract_features(X)
        self.scaler.fit(features)
        return self
    
    def transform(self, X):
        """Transform data by extracting features."""
        features = self._extract_features(X)
        return self.scaler.transform(features)
    
    def _extract_features(self, X):
        """Extract features from heartbeat segments."""
        # Initialize feature array
        n_samples = X.shape[0]
        n_features = 12  # Adjust based on your feature extraction
        features = np.zeros((n_samples, n_features))
        
        for i, segment in enumerate(X):
            # Basic statistical features
            features[i, 0] = np.mean(segment)
            features[i, 1] = np.std(segment)
            features[i, 2] = np.min(segment)
            features[i, 3] = np.max(segment)
            features[i, 4] = np.max(segment) - np.min(segment)  # Peak-to-peak amplitude
            
            # Shape features
            features[i, 5] = np.mean(np.abs(np.diff(segment)))  # Mean absolute difference
            features[i, 6] = np.sum(segment**2)  # Energy
            
            # Higher-order statistics
            features[i, 7] = np.mean((segment - np.mean(segment))**3) / (np.std(segment)**3)  # Skewness
            features[i, 8] = np.mean((segment - np.mean(segment))**4) / (np.std(segment)**4)  # Kurtosis
            
            # Zero crossings and peaks
            zero_crossings = np.where(np.diff(np.signbit(segment)))[0]
            features[i, 9] = len(zero_crossings)
            
            # Find peaks (simplistic approach)
            try:
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(segment)
                features[i, 10] = len(peaks)
            except:
                # Fallback if scipy is not available
                features[i, 10] = np.sum((segment[1:-1] > segment[:-2]) & (segment[1:-1] > segment[2:]))
            
            # R-peak amplitude (assuming R-peak is at the center)
            center = len(segment) // 2
            features[i, 11] = segment[center]
            
        return features

def get_estimator():
    """Return the model pipeline."""
    # Feature extraction
    feature_extractor = HeartbeatFeatureExtractor()
    
    # Classifier - SVM instead of Random Forest
    classifier = SVC(
        C=10.0,                   # Regularization parameter
        kernel='rbf',             # Radial basis function kernel
        gamma='scale',            # Kernel coefficient
        probability=True,         # Enable probability estimates
        class_weight='balanced',  # Adjust weights inversely proportional to class frequencies
        random_state=42
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('features', feature_extractor),
        ('classifier', classifier)
    ])
    
    return pipeline