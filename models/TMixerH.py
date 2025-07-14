import torch
import torch.nn as nn
from .hcm1.tsmixer import TMixerH as _TMixerH
from .hcm1.blocks import TMixerBlock

class Model(_TMixerH):
    def __init__(self, args):
        super().__init__(args)
        
    def load_state_dict(self, state_dict, strict=True):
        # Call parent's load_state_dict
        result = super().load_state_dict(state_dict, strict)
        
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
                    mixer_block = TMixerBlock(
                        in_len=self.in_len,
                        out_len=self.out_len,
                        n_layers=self.args.n_layers,
                        enc_in=num_channels,
                        dropout=self.args.dropout
                    ).to(self.device)
                    
                    # Add to ModuleList
                    self.cluster_models.append(mixer_block)
        
        return result
        
    # def requires_grad_(self, requires_grad: bool = True):
    #     # If we're using LIFT and freezing, only freeze the backbone
    #     if hasattr(self.args, 'lift') and self.args.lift and not requires_grad:
    #         # First freeze everything
    #         for param in self.parameters():
    #             param.requires_grad = False
                
    #         # Then unfreeze specific LIFT components if they exist
    #         if hasattr(self, 'lead_refiner'):
    #             # Unfreeze temperature parameter for leader selection
    #             if hasattr(self.lead_refiner, 'temperature'):
    #                 self.lead_refiner.temperature.requires_grad = True
                    
    #             # Unfreeze mix_layer (ComplexLinear) parameters
    #             if hasattr(self.lead_refiner, 'mix_layer'):
    #                 for param in self.lead_refiner.mix_layer.parameters():
    #                     param.requires_grad = True
                        
    #             # Unfreeze FilterFactory parameters
    #             if hasattr(self.lead_refiner, 'factory'):
    #                 factory = self.lead_refiner.factory
                    
    #                 # Unfreeze classifier, basic_state, and bias if they exist
    #                 if factory.num_state > 1 and factory.need_classifier:
    #                     if hasattr(factory, 'classifier'):
    #                         for param in factory.classifier.parameters():
    #                             param.requires_grad = True
    #                     if hasattr(factory, 'basic_state'):
    #                         factory.basic_state.requires_grad = True
    #                     if hasattr(factory, 'bias'):
    #                         factory.bias.requires_grad = True
                    
    #                 # Unfreeze mix_head parameters
    #                 if factory.num_state == 1:
    #                     if hasattr(factory, 'mix_head'):
    #                         for param in factory.mix_head.parameters():
    #                             param.requires_grad = True
    #                 else:
    #                     if hasattr(factory, 'mix_head_w'):
    #                         factory.mix_head_w.requires_grad = True
    #                     if hasattr(factory, 'mix_head_b'):
    #                         factory.mix_head_b.requires_grad = True
    #     else:
    #         # Otherwise use normal freezing behavior
    #         for param in self.parameters():
    #             param.requires_grad = requires_grad
    #     return self 