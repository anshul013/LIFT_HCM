import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class HardClusterAssigner(nn.Module):
    """Module for hard clustering of channels"""
    def __init__(self, n_vars, num_clusters, method='kmeans', device='cuda', random_state=42):
        super().__init__()
        self.n_vars = n_vars
        self.num_clusters = num_clusters
        self.method = method
        self.device = device
        print(f"HardClusterAssigner initialized with device: {device}")
        
        # Initialize clusterer with fixed random state
        if method == 'kmeans':
            self.clusterer = KMeans(
                n_clusters=num_clusters,
                random_state=random_state  # Add fixed random state
            )
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Initialize cluster assignments on specified device
        self.register_buffer('cluster_assignments', 
                           torch.zeros(n_vars, dtype=torch.long, device=device))
        self.fitted = False
        self.inertia = None
        self.silhouette = None
        
    def extract_features(self, x):
        """Extract features for clustering"""
        print(f"Extracting features from tensor (temporarily moving to CPU)")
        
        # Remember original device
        original_device = x.device
        
        # Temporarily move to CPU for sklearn operations
        x_cpu = x.cpu() if x.is_cuda else x
        x_numpy = x_cpu.numpy()
        
        # Extract features for clustering
        features = []
        for i in range(self.n_vars):
            channel_data = x_numpy[:, :, i]
            features.append([
                np.mean(channel_data),
                np.std(channel_data),
                np.mean(np.abs(np.diff(channel_data, axis=1)))
            ])
        
        # Create numpy array of features
        features_array = np.array(features)
        print(f"Extracted features shape: {features_array.shape}")
        
        return features_array
    
    def fit(self, x):
        """Fit clustering on data"""
        print(f"Fitting clustering on data (original device: {self.device})")
        
        # Save original device - we'll return outputs to this device
        original_device = self.device
        
        # Extract features (this happens on CPU)
        features = self.extract_features(x)
        
        # Fit KMeans clustering (this happens on CPU)
        print(f"Running KMeans on features of shape {features.shape}")
        cluster_idx = self.clusterer.fit_predict(features)
        
        # Move cluster assignments back to the original device
        print(f"Moving cluster assignments to original device: {original_device}")
        self.cluster_assignments = torch.tensor(
            cluster_idx, device=original_device, dtype=torch.long)
        
        # Compute metrics
        self.inertia = self.clusterer.inertia_
        if self.n_vars > self.num_clusters:
            self.silhouette = silhouette_score(features, cluster_idx)
        self.fitted = True
        
        # Print device confirmation
        print(f"Cluster assignments now on device: {self.cluster_assignments.device}")
        print(f"Cluster distribution: {torch.bincount(self.cluster_assignments)}")
        
        return self.cluster_assignments
    
    def forward(self, x, if_update=False):
        """Forward pass returns cluster assignments"""
        # CRITICAL: Make sure we're using the model's device (GPU) and not changing based on input
        
        # If input is on different device than our model, we need to handle that
        # In initialization, the input should be moved to our device (in tsmixer.py)
        # so this is just a safety check
        input_device = x.device
        if input_device != self.device:
            print(f"Warning: Input on {input_device}, but model on {self.device}")
            print(f"You should ensure input is moved to {self.device} before calling this module")
            # No need to move inputs here - that should be done by the caller
        
        # If this is the first time or we need to update, fit the clustering
        if if_update or not self.fitted:
            # When fitting, we'll handle device changes within the fit method
            # but maintain our original device 
            self.fit(x)
        
        # Ensure assignments are on the right device (matching the model's device)
        if self.cluster_assignments.device != self.device:
            print(f"Moving cluster_assignments from {self.cluster_assignments.device} to {self.device}")
            self.cluster_assignments = self.cluster_assignments.to(self.device)
            
        # Return cluster assignments on the model's device
        return self.cluster_assignments
    
    def get_metrics(self):
        """Get clustering metrics"""
        return {
            'inertia': self.inertia,
            'silhouette': self.silhouette
        } 