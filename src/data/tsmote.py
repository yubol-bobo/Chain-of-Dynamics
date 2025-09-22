import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array
import warnings

class TSMOTE:
    """
    Temporal SMOTE for time series data.
    Maintains temporal structure while oversampling minority classes.
    """
    
    def __init__(self, random_state=42, k_neighbors=5):
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.minority_samples = None
        self.minority_neighbors = None
        
    def fit_resample(self, X, y):
        """
        Fit and resample the dataset.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_timesteps, n_features)
            Time series data
        y : array-like of shape (n_samples,)
            Target labels
            
        Returns:
        --------
        X_resampled : array-like
            Resampled features
        y_resampled : array-like
            Resampled labels
        """
        X = check_array(X, allow_nd=True)
        y = np.array(y)
        
        # Find minority class
        unique_labels, counts = np.unique(y, return_counts=True)
        minority_label = unique_labels[np.argmin(counts)]
        majority_label = unique_labels[np.argmax(counts)]
        
        # Separate minority and majority samples
        minority_indices = np.where(y == minority_label)[0]
        majority_indices = np.where(y == majority_label)[0]
        
        X_minority = X[minority_indices]
        y_minority = y[minority_indices]
        
        # Calculate how many samples to generate
        n_minority = len(minority_indices)
        n_majority = len(majority_indices)
        n_samples_to_generate = n_majority - n_minority
        
        if n_samples_to_generate <= 0:
            return X, y
        
        # Find k nearest neighbors for each minority sample
        # Flatten time series for distance calculation
        X_minority_flat = X_minority.reshape(X_minority.shape[0], -1)
        nbrs = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, n_minority), 
                               algorithm='auto').fit(X_minority_flat)
        distances, indices = nbrs.kneighbors(X_minority_flat)
        
        # Remove self from neighbors
        indices = indices[:, 1:]  # Remove first column (self)
        distances = distances[:, 1:]
        
        # Generate synthetic samples
        np.random.seed(self.random_state)
        synthetic_samples = []
        
        for i in range(n_samples_to_generate):
            # Randomly select a minority sample
            idx = np.random.randint(0, n_minority)
            selected_sample = X_minority[idx]
            
            # Randomly select a neighbor
            if len(indices[idx]) > 0:
                neighbor_idx = np.random.choice(indices[idx])
                neighbor_sample = X_minority[neighbor_idx]
                
                # Generate synthetic sample with temporal interpolation
                alpha = np.random.random()
                synthetic_sample = self._temporal_interpolation(selected_sample, neighbor_sample, alpha)
            else:
                # If no neighbors, add small noise to original sample
                noise = np.random.normal(0, 0.01, selected_sample.shape)
                synthetic_sample = selected_sample + noise
            
            synthetic_samples.append(synthetic_sample)
        
        # Combine original and synthetic samples
        X_synthetic = np.array(synthetic_samples)
        y_synthetic = np.full(len(synthetic_samples), minority_label)
        
        X_resampled = np.vstack([X, X_synthetic])
        y_resampled = np.hstack([y, y_synthetic])
        
        print(f"[SUCCESS] TSMOTE: Generated {len(synthetic_samples)} synthetic samples")
        print(f"[SUCCESS] Original: {len(y)} samples, Resampled: {len(y_resampled)} samples")
        
        return X_resampled, y_resampled
    
    def _temporal_interpolation(self, sample1, sample2, alpha):
        """
        Perform temporal interpolation between two time series samples.
        Maintains temporal structure while interpolating features.
        
        Parameters:
        -----------
        sample1 : array-like of shape (n_timesteps, n_features)
            First time series sample
        sample2 : array-like of shape (n_timesteps, n_features)
            Second time series sample
        alpha : float
            Interpolation factor (0 to 1)
            
        Returns:
        --------
        interpolated_sample : array-like
            Interpolated time series sample
        """
        # Linear interpolation between samples
        interpolated = alpha * sample1 + (1 - alpha) * sample2
        
        # Add small temporal noise to maintain realistic patterns
        temporal_noise = np.random.normal(0, 0.005, interpolated.shape)
        interpolated += temporal_noise
        
        return interpolated
    
    def fit(self, X, y):
        """Fit the TSMOTE model (for sklearn compatibility)."""
        return self
    
    def sample(self, X, y):
        """Sample method (for sklearn compatibility)."""
        return self.fit_resample(X, y) 