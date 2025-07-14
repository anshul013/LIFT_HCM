import torch
import torch.nn as nn
from .hcm1.tsmixer import TSMixerH as _TSMixerH
from .hcm1.blocks import TSMixerBlock

class Model(_TSMixerH):
    def __init__(self, args):
        super().__init__(args)
        
    def load_state_dict(self, state_dict, strict=True):
        # Call parent's load_state_dict
        super().load_state_dict(state_dict, strict)
        
        # Re-initialize cluster models if needed
        if not hasattr(self, 'cluster_models') or len(self.cluster_models) == 0:
            # Create cluster models based on saved assignments
            cluster_assignments = self.cluster_assigner.cluster_assignments
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
                    ).to(self.device)
                    
                    # Add to ModuleList
                    self.cluster_models.append(mixer_block) 