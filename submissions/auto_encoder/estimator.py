import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import os

class AutoencoderFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Feature extraction for heartbeat classification using an autoencoder.
    
    This transformer uses a convolutional autoencoder to extract latent space 
    representations from raw ECG heartbeat segments.
    """
    
    def __init__(self, latent_dim=32, input_shape=(250, 1), epochs=50, 
                 batch_size=32, model_path=None, pretrained=False):
        """
        Initialize the autoencoder feature extractor.
        
        Parameters:
        -----------
        latent_dim : int
            Dimension of the latent space (encoded features)
        input_shape : tuple
            Shape of input ECG segments (length, channels)
        epochs : int
            Number of epochs to train the autoencoder
        batch_size : int
            Batch size for training
        model_path : str
            Path to save/load the autoencoder model
        pretrained : bool
            Whether to use a pretrained model or train a new one
        """
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_path = model_path if model_path else 'ecg_autoencoder.h5'
        self.pretrained = pretrained
        self.scaler = StandardScaler()
        self.autoencoder = None
        self.encoder = None
    
    def _build_autoencoder(self):
        """Build the autoencoder architecture."""
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # Encoding layers
        # Using padding='same' and careful pooling to maintain dimensions
        # Calculate pooling factors to ensure we can reconstruct the exact input size
        # For 250 samples, we'll use pooling factors that multiply to a power of 2
        # 250 -> 125 -> 25 -> 5 (using pool sizes of 2, 5, 5)
        
        x = Conv1D(32, 5, activation='relu', padding='same')(inputs)
        x = MaxPooling1D(2, padding='same')(x)  # 250 -> 125
        
        x = Conv1D(16, 5, activation='relu', padding='same')(x)
        x = MaxPooling1D(5, padding='same')(x)  # 125 -> 25
        
        x = Conv1D(8, 5, activation='relu', padding='same')(x)
        encoded_conv = MaxPooling1D(5, padding='same')(x)  # 25 -> 5
        
        # Store shapes for reconstruction
        shape_before_flatten = tf.keras.backend.int_shape(encoded_conv)
        
        # Flatten and encode to latent dimension
        x = Flatten()(encoded_conv)
        encoded = Dense(self.latent_dim, activation='relu', name='encoded')(x)
        
        # Start decoding
        x = Dense(shape_before_flatten[1] * shape_before_flatten[2], activation='relu')(encoded)
        x = Reshape((shape_before_flatten[1], shape_before_flatten[2]))(x)
        
        # Decoding layers with corresponding upsampling
        x = Conv1D(8, 5, activation='relu', padding='same')(x)
        x = UpSampling1D(5)(x)  # 5 -> 25
        
        x = Conv1D(16, 5, activation='relu', padding='same')(x)
        x = UpSampling1D(5)(x)  # 25 -> 125
        
        x = Conv1D(32, 5, activation='relu', padding='same')(x)
        x = UpSampling1D(2)(x)  # 125 -> 250
        
        # Output layer
        decoded = Conv1D(1, 5, activation='linear', padding='same')(x)
        
        # Create models
        autoencoder = Model(inputs, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Create encoder model (for feature extraction)
        encoder = Model(inputs, autoencoder.get_layer('encoded').output)
        
        return autoencoder, encoder
    
    def fit(self, X, y=None):
        """
        Fit the autoencoder on ECG data.
        
        Parameters:
        -----------
        X : numpy.ndarray
            ECG segments, shape (n_samples, segment_length)
        y : numpy.ndarray, optional
            Class labels (not used for training the autoencoder)
        
        Returns:
        --------
        self : object
            Returns self
        """
        # Reshape data for the autoencoder if needed
        X_reshaped = self._prepare_input(X)
        
        if self.pretrained and os.path.exists(self.model_path):
            # Load pretrained model
            print(f"Loading pretrained autoencoder from {self.model_path}")
            self.autoencoder = load_model(self.model_path)
            # Create encoder model
            encoder_layer = self.autoencoder.get_layer('encoded')
            input_layer = self.autoencoder.input
            self.encoder = Model(input_layer, encoder_layer.output)
        else:
            # Build and train a new autoencoder
            print("Training new autoencoder model")
            self.autoencoder, self.encoder = self._build_autoencoder()
            
            # Print model summary to debug dimensions
            print("Autoencoder Summary:")
            self.autoencoder.summary()
            
            # Callbacks for training
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ModelCheckpoint(self.model_path, save_best_only=True)
            ]
            
            # Train the autoencoder
            self.autoencoder.fit(
                X_reshaped, X_reshaped,  # Using same data for input and output
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
            
            # Save the trained model
            self.autoencoder.save(self.model_path)
        
        # Extract features for scaling
        features = self._extract_encoded_features(X_reshaped)
        self.scaler.fit(features)
        
        return self
    
    def transform(self, X):
        """
        Transform data by extracting features from the autoencoder's latent space.
        
        Parameters:
        -----------
        X : numpy.ndarray
            ECG segments, shape (n_samples, segment_length)
        
        Returns:
        --------
        features : numpy.ndarray
            Extracted features, shape (n_samples, latent_dim)
        """
        X_reshaped = self._prepare_input(X)
        features = self._extract_encoded_features(X_reshaped)
        return self.scaler.transform(features)
    
    def _prepare_input(self, X):
        """Reshape input data for the autoencoder if needed."""
        if len(X.shape) == 2:
            # Reshape to (n_samples, segment_length, 1) for Conv1D
            return X.reshape(X.shape[0], X.shape[1], 1)
        return X
    
    def _extract_encoded_features(self, X):
        """Extract features from the autoencoder's latent space."""
        if self.encoder is None:
            raise ValueError("Autoencoder model is not fitted yet. Call fit() first.")
        
        return self.encoder.predict(X)

def get_estimator(pretrained=False, model_path='ecg_autoencoder.h5'):
    """
    Return the model pipeline with autoencoder feature extraction.
    
    Parameters:
    -----------
    pretrained : bool
        Whether to use a pretrained autoencoder model
    model_path : str
        Path to the pretrained model file
    
    Returns:
    --------
    pipeline : sklearn.pipeline.Pipeline
        The complete model pipeline
    """
    # Feature extraction with autoencoder
    feature_extractor = AutoencoderFeatureExtractor(
        latent_dim=32,
        input_shape=(250, 1),
        epochs=20,
        batch_size=32,
        model_path=model_path,
        pretrained=pretrained
    )
    
    # Classifier - SVM (same as in the provided code)
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
