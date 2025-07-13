import torch
import torch.nn as nn

class MlpBlockFeatures(nn.Module):
    """MLP for features"""
    def __init__(self, channels, mlp_dim, dropout_factor, activation, single_layer_mixer=False):
        super().__init__()
        self.normalization_layer = nn.BatchNorm1d(channels)
        self.single_layer_mixer = single_layer_mixer
        if self.single_layer_mixer:
            self.linear_layer1 = nn.Linear(channels, channels)
        else:
            self.linear_layer1 = nn.Linear(channels, mlp_dim)
            self.linear_layer2 = nn.Linear(mlp_dim, channels)
            
        if activation == "gelu":
            self.activation_layer = nn.GELU()
        elif activation == "relu":
            self.activation_layer = nn.ReLU()
        else:
            self.activation_layer = None
        self.dropout_layer = nn.Dropout(dropout_factor)

    def forward(self, x):
        y = torch.swapaxes(x, 1, 2)
        y = self.normalization_layer(y)
        y = torch.swapaxes(y, 1, 2)
        y = self.linear_layer1(y)
        if self.activation_layer is not None:
            y = self.activation_layer(y)
        if not self.single_layer_mixer:
            y = self.dropout_layer(y)
            y = self.linear_layer2(y)
        y = self.dropout_layer(y)
        return x + y

class MlpBlockTimesteps(nn.Module):
    """MLP for timesteps with 1 layer"""
    def __init__(self, seq_len, dropout_factor, activation):
        super().__init__()
        self.normalization_layer = nn.BatchNorm1d(seq_len)
        self.linear_layer = nn.Linear(seq_len, seq_len)
        if activation == "gelu":
            self.activation_layer = nn.GELU()
        elif activation == "relu":
            self.activation_layer = nn.ReLU()
        else:
            self.activation_layer = None
        self.dropout_layer = nn.Dropout(dropout_factor)

    def forward(self, x):
        y = self.normalization_layer(x)
        y = torch.swapaxes(y, 1, 2)
        y = self.linear_layer(y)
        y = self.activation_layer(y)
        y = self.dropout_layer(y)
        y = torch.swapaxes(y, 1, 2)
        return x + y

class MixerBlock(nn.Module):
    """Mixer block layer"""
    def __init__(self, channels, features_block_mlp_dims, seq_len, dropout_factor, activation, single_layer_mixer):
        super().__init__()
        self.channels = channels
        self.seq_len = seq_len
        # Timesteps mixing block
        self.timesteps_mixer = MlpBlockTimesteps(seq_len, dropout_factor, activation)
        # Features mixing block
        self.channels_mixer = MlpBlockFeatures(channels, features_block_mlp_dims, dropout_factor, activation, single_layer_mixer)

    def forward(self, x):
        # Ensure same device
        device = x.device
        if next(self.parameters()).device != device:
            self.to(device)
        y = self.timesteps_mixer(x)
        y = self.channels_mixer(y)
        return y

class TSMixerBlock(nn.Module):
    """TSMixer block for processing clustered variables"""
    def __init__(self, in_len, out_len, d_ff, n_layers, enc_in, dropout=0.1):
        super().__init__()
        self.num_layers = n_layers
        self.channels = enc_in
        self.pred_len = out_len
        self.in_len = in_len
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Create mixer layers
        self.mixer_layers = nn.ModuleList([
            MixerBlock(
                channels=self.channels,
                features_block_mlp_dims=self.d_ff,
                seq_len=self.in_len,
                dropout_factor=self.dropout,
                activation='relu',
                single_layer_mixer=False
            ) for _ in range(self.num_layers)
        ])
        
        # RevIN normalization
        # self.rev_norm = RevIN(num_features=self.channels)
        
        # Output projection
        # self.output_linear_layers = nn.ModuleList([
        #     nn.Linear(self.in_len, self.pred_len) for _ in range(self.channels)
        # ])
        self.output_linear_layers = nn.Linear(self.in_len, self.pred_len)

    def forward(self, x):
        # Verify device
        # if not x.is_cuda:
        #     print("Warning: Input tensor is not on GPU")
        # Ensure model is on same device as input
        device = x.device
        if next(self.parameters()).device != device:
            self.to(device)
        # Apply mixer layers sequentially
        for layer in self.mixer_layers:
            # if not next(layer.parameters()).is_cuda:
            #     print(f"Warning: Layer {layer} is not on GPU")
            x = layer(x)
        
        # Project to output length
        y = torch.zeros([x.size(0), self.channels, self.pred_len], 
                       dtype=x.dtype, device=device)
        x = torch.swapaxes(x, 1, 2)
        
        # for c in range(self.channels):
        #     if not self.output_linear_layers[c].weight.is_cuda:
        #         print(f"Warning: Output layer {c} is not on GPU")
        #     y[:, c, :] = self.output_linear_layers[c](x[:, c, :])
        y = self.output_linear_layers(x.clone())
        y = torch.swapaxes(y, 1, 2)
        return y

class T_MixerBlock(nn.Module):
    """T-Mixer block layer"""
    def __init__(self, channels, seq_len, dropout_factor, activation):
        super().__init__()
        self.channels = channels
        self.seq_len = seq_len
        # Timesteps mixing block
        self.timesteps_mixer = MlpBlockTimesteps(seq_len, dropout_factor, activation)

    def forward(self, x):
        # Ensure same device
        device = x.device
        if next(self.parameters()).device != device:
            self.to(device)
        y = self.timesteps_mixer(x)
        return y

class TMixerBlock(nn.Module):
    """TMixer block for processing clustered variables"""
    def __init__(self, in_len, out_len, n_layers, enc_in, dropout=0.1):
        super().__init__()
        self.num_layers = n_layers
        self.channels = enc_in
        self.pred_len = out_len
        self.in_len = in_len
        self.dropout = dropout
        
        # Create mixer layers
        self.mixer_layers = nn.ModuleList([
            T_MixerBlock(
                channels=self.channels,
                seq_len=self.in_len,
                dropout_factor=self.dropout,
                activation='relu'
            ) for _ in range(self.num_layers)
        ])
        
        # RevIN normalization
        # self.rev_norm = RevIN(num_features=self.channels)
        
        # Output projection
        # self.output_linear_layers = nn.ModuleList([
        #     nn.Linear(self.in_len, self.pred_len) for _ in range(self.channels)
        # ])
        self.output_linear_layers = nn.Linear(self.in_len, self.pred_len)

    def forward(self, x):
        # Verify device
        # if not x.is_cuda:
        #     print("Warning: Input tensor is not on GPU")
        # Ensure model is on same device as input
        device = x.device
        if next(self.parameters()).device != device:
            self.to(device)
        # Apply mixer layers sequentially
        for layer in self.mixer_layers:
            # if not next(layer.parameters()).is_cuda:
            #     print(f"Warning: Layer {layer} is not on GPU")
            x = layer(x)
        
        # Project to output length
        y = torch.zeros([x.size(0), self.channels, self.pred_len], 
                       dtype=x.dtype, device=device)
        x = torch.swapaxes(x, 1, 2)
        
        # for c in range(self.channels):
        #     if not self.output_linear_layers[c].weight.is_cuda:
        #         print(f"Warning: Output layer {c} is not on GPU")
        #     y[:, c, :] = self.output_linear_layers[c](x[:, c, :])
        y = self.output_linear_layers(x.clone())
        y = torch.swapaxes(y, 1, 2)
        return y 