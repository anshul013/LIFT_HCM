import torch
import torch.nn as nn
from models.hcm1.preprocessor import HardClusterAssigner
from models.RevIN import RevIN
from models.hcm1.blocks import TSMixerBlock, TMixerBlock

# class ClusterTSMixer(nn.Module):
#     """TSMixer model adapted for cluster-specific processing"""
#     def __init__(self, num_features, args):
#         super().__init__()
#         self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
#         self.model = TSMixerBlock(
#             in_len=args.seq_len,
#             out_len=args.pred_len,
#             d_ff=args.d_ff,
#             n_layers=args.n_layers,
#             enc_in=num_features,
#             dropout=args.dropout
#         ).to(self.device)
        
#     def forward(self, x):
#         return self.model(x)

class TSMixerH(nn.Module):
    """Hard Clustering variant of TSMixer"""
    def __init__(self, args):
        super().__init__()
        # Basic parameters
        self.n_vars = args.batch_size if args.data in ["M4", "stock"] else args.enc_in
        self.in_len = args.seq_len
        self.out_len = args.pred_len
        self.num_clusters = args.num_clusters
        self.d_ff = args.d_ff
        
        # CUDA check at initialization
        print(f"CUDA available at model init: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            
        # Consistently use the same device throughout
        if hasattr(args, 'cuda'):
            self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
        print(f"TSMixerH initialized with device: {self.device}")
        self.enc_in = args.enc_in
        self.args = args
        
        # Create all components with correct device from the start
        # Normalization layer for full input
        self.rev_in = RevIN(num_features=args.enc_in).to(self.device)
        
        # Clustering module - explicitly use the same device
        self.cluster_assigner = HardClusterAssigner(
            n_vars=self.n_vars,
            num_clusters=args.num_clusters,
            method=args.clustering_method,
            device=self.device
        ).to(self.device)
        
        # Initialize empty ModuleList for cluster models
        self.cluster_models = nn.ModuleList()
        self.cluster_sizes = {}
        
        # Move main module to device - components will be added later
        self.to(self.device)
        
    def to_device(self, device):
        """Explicitly move all components to a specific device"""
        # Only change devices if we need to
        if device != self.device:
            print(f"Moving TSMixerH from {self.device} to {device}")
            self.device = device
            self.to(device)
            self.rev_in.to(device)
            self.cluster_assigner.device = device
            self.cluster_assigner.to(device)
            if hasattr(self.cluster_assigner, 'cluster_assignments'):
                self.cluster_assigner.cluster_assignments = self.cluster_assigner.cluster_assignments.to(device)
                
            # Move all cluster models
            for i, model in enumerate(self.cluster_models):
                self.cluster_models[i] = model.to(device)
        
        return self
        
    def initialize_clusters(self, full_data):
        """Initialize clusters using full training data"""
        print(f"Initializing clusters on {self.device}")
        
        # Ensure data is on correct device
        if full_data.device != self.device:
            print(f"Moving input data from {full_data.device} to {self.device}")
            full_data = full_data.to(self.device)
        
        # Force cluster update using full data
        cluster_assignments = self.cluster_assigner(full_data, if_update=True)
        
        # CRITICAL: After clustering operations, explicitly move the cluster_assigner to GPU
        self.cluster_assigner = self.cluster_assigner.to(self.device)
        
        # Verify cluster assignments are on correct device
        if cluster_assignments.device != self.device:
            print(f"Moving cluster assignments from {cluster_assignments.device} to {self.device}")
            cluster_assignments = cluster_assignments.to(self.device)
            self.cluster_assigner.cluster_assignments = cluster_assignments
        
        # Create cluster models based on assignments
        self.cluster_models = nn.ModuleList()
        self.cluster_sizes = {}
        
        # Create models directly on target GPU
        for cluster_idx in range(self.num_clusters):
            cluster_mask = (cluster_assignments == cluster_idx)
            if cluster_mask.any():
                cluster_channels = torch.where(cluster_mask)[0]
                num_channels = len(cluster_channels)
                self.cluster_sizes[cluster_idx] = num_channels
                
                # Create cluster-specific model directly on target device
                mixer_block = TSMixerBlock(
                    in_len=self.in_len,
                    out_len=self.out_len,
                    d_ff=self.d_ff,
                    n_layers=self.args.n_layers,
                    enc_in=num_channels,
                    dropout=self.args.dropout
                ).to(self.device)  # Explicitly move each module to the device
                
                # Add to ModuleList
                self.cluster_models.append(mixer_block)
        
        # Verify devices are consistent
        print(f"Cluster assignments device: {cluster_assignments.device}")
        
        # print(f"Initial cluster assignments: {cluster_assignments.cpu().numpy()}")
        # print(f"Cluster sizes: {self.cluster_sizes}")
        # Register cluster_models as a module
        print(f"Created {len(self.cluster_models)} cluster models on {self.device}")
        
        # Final device check after initialization
        print(f"Model parameters device after init: {next(self.parameters()).device}")
        
        return cluster_assignments
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Ensure input tensor is on correct device
        if x.device != self.device:
            x = x.to(self.device)
            
        # Apply RevIN normalization
        x = self.rev_in(x, 'norm')
        
        # Get cluster assignments
        cluster_assignments = self.cluster_assigner.cluster_assignments
        
        # Initialize output tensor with correct shape [batch_size, out_len, enc_in]
        outputs = torch.zeros(batch_size, self.out_len, self.enc_in, device=self.device)
        
        # Process each cluster
        model_idx = 0
        for cluster_idx in range(self.num_clusters):
            cluster_mask = (cluster_assignments == cluster_idx)
            if not cluster_mask.any():
                continue
                
            # Select data for current cluster
            cluster_channels = torch.where(cluster_mask)[0]
            cluster_input = x[:, :, cluster_channels]
            
            # Process with corresponding model
            model = self.cluster_models[model_idx]
            cluster_output = model(cluster_input)
            model_idx += 1
            
            # Place outputs back in correct positions
            outputs[:, :, cluster_channels] = cluster_output
        
        # Apply inverse normalization
        outputs = self.rev_in(outputs, 'denorm')
        return outputs
    
    def get_current_assignments(self):
        """Get current cluster assignments"""
        return self.cluster_assigner.cluster_assignments.cpu().numpy()
    
    def get_clustering_metrics(self):
        """Get clustering metrics"""
        return self.cluster_assigner.get_metrics() 
    
class TMixerH(nn.Module):
    """Hard Clustering variant of TSMixer"""
    def __init__(self, args):
        super().__init__()
        # Basic parameters
        self.n_vars = args.batch_size if args.data in ["M4", "stock"] else args.enc_in
        self.in_len = args.seq_len
        self.out_len = args.pred_len
        self.num_clusters = args.num_clusters
        self.d_ff = args.d_ff
        
        # CUDA check at initialization
        print(f"CUDA available at model init: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            
        # Consistently use the same device throughout
        if hasattr(args, 'cuda'):
            self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
        print(f"TMixerH initialized with device: {self.device}")
        self.enc_in = args.enc_in
        self.args = args
        
        # Create all components with correct device from the start
        # Normalization layer for full input
        self.rev_in = RevIN(num_features=args.enc_in).to(self.device)
        
        # Clustering module - explicitly use the same device
        self.cluster_assigner = HardClusterAssigner(
            n_vars=self.n_vars,
            num_clusters=args.num_clusters,
            method=args.clustering_method,
            device=self.device
        ).to(self.device)
        
        # Initialize empty ModuleList for cluster models
        self.cluster_models = nn.ModuleList()
        self.cluster_sizes = {}
        
        # Move main module to device - components will be added later
        self.to(self.device)
        
    def to_device(self, device):
        """Explicitly move all components to a specific device"""
        # Only change devices if we need to
        if device != self.device:
            print(f"Moving TMixerH from {self.device} to {device}")
            self.device = device
            self.to(device)
            self.rev_in.to(device)
            self.cluster_assigner.device = device
            self.cluster_assigner.to(device)
            if hasattr(self.cluster_assigner, 'cluster_assignments'):
                self.cluster_assigner.cluster_assignments = self.cluster_assigner.cluster_assignments.to(device)
                
            # Move all cluster models
            for i, model in enumerate(self.cluster_models):
                self.cluster_models[i] = model.to(device)
        
        return self
        
    def initialize_clusters(self, full_data):
        """Initialize clusters using full training data"""
        print(f"Initializing clusters on {self.device}")
        
        # Ensure data is on correct device
        if full_data.device != self.device:
            print(f"Moving input data from {full_data.device} to {self.device}")
            full_data = full_data.to(self.device)
        
        # Force cluster update using full data
        cluster_assignments = self.cluster_assigner(full_data, if_update=True)
        
        # CRITICAL: After clustering operations, explicitly move the cluster_assigner to GPU
        self.cluster_assigner = self.cluster_assigner.to(self.device)
        
        # Verify cluster assignments are on correct device
        if cluster_assignments.device != self.device:
            print(f"Moving cluster assignments from {cluster_assignments.device} to {self.device}")
            cluster_assignments = cluster_assignments.to(self.device)
            self.cluster_assigner.cluster_assignments = cluster_assignments
        
        # Create cluster models based on assignments
        self.cluster_models = nn.ModuleList()
        self.cluster_sizes = {}
        
        # Create models directly on target GPU
        for cluster_idx in range(self.num_clusters):
            cluster_mask = (cluster_assignments == cluster_idx)
            if cluster_mask.any():
                cluster_channels = torch.where(cluster_mask)[0]
                num_channels = len(cluster_channels)
                self.cluster_sizes[cluster_idx] = num_channels
                
                # Create cluster-specific model directly on target device
                mixer_block = TMixerBlock(
                    in_len=self.in_len,
                    out_len=self.out_len,
                    n_layers=self.args.n_layers,
                    enc_in=num_channels,
                    dropout=self.args.dropout
                ).to(self.device)
                
                # Add to ModuleList
                self.cluster_models.append(mixer_block)
        
        # Verify devices are consistent
        print(f"Cluster assignments device: {cluster_assignments.device}")
        
        # print(f"Initial cluster assignments: {cluster_assignments.cpu().numpy()}")
        # print(f"Cluster sizes: {self.cluster_sizes}")
        # Register cluster_models as a module
        print(f"Created {len(self.cluster_models)} cluster models on {self.device}")
        
        # Final device check after initialization
        print(f"Model parameters device after init: {next(self.parameters()).device}")
        
        return cluster_assignments
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Ensure input tensor is on correct device
        if x.device != self.device:
            x = x.to(self.device)
            
        # Apply RevIN normalization
        x = self.rev_in(x, 'norm')
        
        # Get cluster assignments
        cluster_assignments = self.cluster_assigner.cluster_assignments
        
        # Initialize output tensor with correct shape [batch_size, out_len, enc_in]
        outputs = torch.zeros(batch_size, self.out_len, self.enc_in, device=self.device)
        
        # Process each cluster
        model_idx = 0
        for cluster_idx in range(self.num_clusters):
            cluster_mask = (cluster_assignments == cluster_idx)
            if not cluster_mask.any():
                continue
                
            # Select data for current cluster
            cluster_channels = torch.where(cluster_mask)[0]
            cluster_input = x[:, :, cluster_channels]
            
            # Process with corresponding model
            model = self.cluster_models[model_idx]
            cluster_output = model(cluster_input)
            model_idx += 1
            
            # Place outputs back in correct positions
            outputs[:, :, cluster_channels] = cluster_output
        
        # Apply inverse normalization
        outputs = self.rev_in(outputs, 'denorm')
        return outputs
    
    def get_current_assignments(self):
        """Get current cluster assignments"""
        return self.cluster_assigner.cluster_assignments.cpu().numpy()
    
    def get_clustering_metrics(self):
        """Get clustering metrics"""
        return self.cluster_assigner.get_metrics() 