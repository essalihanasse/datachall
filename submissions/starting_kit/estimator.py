import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks, welch
from scipy.stats import skew, kurtosis

class HeartbeatFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Feature extraction for heartbeat classification.
    
    This transformer extracts time-domain and frequency-domain features
    from raw ECG heartbeat segments.
    """
    
    def __init__(self, sampling_rate=250):
        self.scaler = StandardScaler()
        self.sampling_rate = sampling_rate
    
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
        n_features = 23  # Increased number of features
        features = np.zeros((n_samples, n_features))
        
        for i, segment in enumerate(X):
            # Time domain features
            # Basic statistical features
            features[i, 0] = np.mean(segment)
            features[i, 1] = np.std(segment)
            features[i, 2] = np.min(segment)
            features[i, 3] = np.max(segment)
            features[i, 4] = np.max(segment) - np.min(segment)  # Peak-to-peak amplitude
            
            # Shape features
            features[i, 5] = np.mean(np.abs(np.diff(segment)))  # Mean absolute difference
            features[i, 6] = np.sum(segment**2)  # Energy
            
            # Higher-order statistics (using scipy functions for stability)
            features[i, 7] = skew(segment)  # Skewness
            features[i, 8] = kurtosis(segment)  # Kurtosis
            
            # Zero crossings
            zero_crossings = np.where(np.diff(np.signbit(segment)))[0]
            features[i, 9] = len(zero_crossings)
            
            # Peak detection
            peaks, peak_properties = find_peaks(segment, height=0)
            features[i, 10] = len(peaks)  # Number of peaks
            
            # Calculate peak heights and widths if peaks exist
            if len(peaks) > 0:
                features[i, 11] = np.mean(peak_properties['peak_heights'])  # Mean peak height
                features[i, 12] = np.std(peak_properties['peak_heights'])   # Std of peak heights
            else:
                features[i, 11] = 0
                features[i, 12] = 0
            
            # R-peak amplitude (assuming R-peak is at the center)
            center = len(segment) // 2
            window_size = len(segment) // 10  # 10% of the segment
            r_region = segment[max(0, center - window_size):min(len(segment), center + window_size)]
            features[i, 13] = np.max(r_region)
            
            # RR interval variability (if multiple heartbeats in the segment)
            if len(peaks) > 1:
                rr_intervals = np.diff(peaks)
                features[i, 14] = np.mean(rr_intervals)
                features[i, 15] = np.std(rr_intervals)
                features[i, 16] = np.max(rr_intervals)
                features[i, 17] = np.min(rr_intervals)
            else:
                features[i, 14:18] = 0
            
            # Frequency domain features (using Welch's method)
            try:
                freqs, psd = welch(segment, fs=self.sampling_rate, nperseg=min(256, len(segment)))
                
                # Total power
                features[i, 18] = np.sum(psd)
                
                # Power in specific frequency bands
                # VLF: 0.003-0.04 Hz, LF: 0.04-0.15 Hz, HF: 0.15-0.4 Hz
                vlf_mask = (freqs >= 0.003) & (freqs < 0.04)
                lf_mask = (freqs >= 0.04) & (freqs < 0.15)
                hf_mask = (freqs >= 0.15) & (freqs < 0.4)
                
                features[i, 19] = np.sum(psd[vlf_mask]) if np.any(vlf_mask) else 0
                features[i, 20] = np.sum(psd[lf_mask]) if np.any(lf_mask) else 0
                features[i, 21] = np.sum(psd[hf_mask]) if np.any(hf_mask) else 0
                
                # LF/HF ratio (autonomic balance indicator)
                lf_power = np.sum(psd[lf_mask]) if np.any(lf_mask) else 0
                hf_power = np.sum(psd[hf_mask]) if np.any(hf_mask) else 0
                features[i, 22] = lf_power / hf_power if hf_power > 0 else 0
                
            except:
                features[i, 18:23] = 0
                
        return features

def get_estimator():
    """Return the model pipeline."""
    # Feature extraction
    feature_extractor = HeartbeatFeatureExtractor()
    
    # Classifier - SVM
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
