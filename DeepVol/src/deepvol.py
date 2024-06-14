import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalLayer(nn.Module):
    def __init__(self, residual_channels: int = 32, dilation_channels: int = 32, skip_channels: int = 64, 
                 kernel_size: int = 3, dilation_factor: int = 2, use_bias: bool = True, causal: bool = True) -> None:
        super().__init__()

        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.kernel_size = kernel_size
        self.dilation_factor = dilation_factor
        self.use_bias = use_bias
        self.causal = causal

        # Padding for causal convolutions
        self.padding_size = (self.kernel_size - 1) * self.dilation_factor if self.causal else ((self.kernel_size - 1) * self.dilation_factor) // 2

        # Dilated convolutions for gated activation
        self.filter_convolution = nn.Conv1d(in_channels=self.residual_channels, out_channels=self.dilation_channels,
                                            kernel_size=self.kernel_size, padding=self.padding_size, dilation=self.dilation_factor,
                                            bias=self.use_bias)
        self.gate_convolution = nn.Conv1d(in_channels=self.residual_channels, out_channels=self.dilation_channels,
                                          kernel_size=self.kernel_size, padding=self.padding_size, dilation=self.dilation_factor,
                                          bias=self.use_bias)

        # Batch normalization layers
        self.batch_norm = nn.BatchNorm1d(self.dilation_channels)
        self.skip_batch_norm = nn.BatchNorm1d(self.skip_channels)
        self.residual_batch_norm = nn.BatchNorm1d(self.residual_channels)

        # 1x1 convolutions for residual and skip connections
        self.residual_convolution = nn.Conv1d(in_channels=self.dilation_channels, out_channels=self.residual_channels, kernel_size=1,
                                              bias=self.use_bias)
        self.skip_convolution = nn.Conv1d(in_channels=self.dilation_channels, out_channels=self.skip_channels, kernel_size=1,
                                          bias=self.use_bias)

    def apply_causal_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a causal mask to the output of a convolutional layer.
        :param x: Input tensor.
        :return: Causally masked tensor.
        """
        return x if not self.causal else x[..., :-self.padding_size]

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass for ConvolutionalLayer.
        :param x: Input tensor.
        :return: Residual output and skip connection.
        """
        filter_output = torch.tanh(self.filter_convolution(x))
        gate_output = torch.sigmoid(self.gate_convolution(x))
        activation_output = self.apply_causal_mask(filter_output) * self.apply_causal_mask(gate_output)

        skip_output = self.skip_convolution(activation_output)
        residual_output = self.residual_convolution(activation_output) + x

        return residual_output, skip_output

class ResidualBlock(nn.Module):
    def __init__(self, num_layers: int = 2, residual_channels: int = 32, dilation_channels: int = 32, 
                 skip_channels: int = 64, kernel_size: int = 3, use_bias: bool = True, causal: bool = True) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.causal = causal

        self.layers = nn.ModuleList([
            ConvolutionalLayer(self.residual_channels, self.dilation_channels, self.skip_channels, self.kernel_size, dilation_factor=(2 ** i),
                               use_bias=self.use_bias, causal=self.causal) for i in range(self.num_layers)
        ])

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass for ResidualBlock.
        :param x: Input tensor.
        :return: Residual output and stacked skip connections.
        """
        skip_connections = []
        for layer in self.layers:
            x, skip = layer(x)
            skip_connections.append(skip)

        return x, torch.stack(skip_connections)

class AttentionMechanism(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.attention_layer = nn.Linear(self.size, self.size)

    def forward(self, base: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for AttentionMechanism.
        :param base: Base for attention.
        :param context: If None, self-attention is used. Otherwise, context is attended by base.
        :return: Attention-weighted output.
        """
        return (base if context is None else context) * F.softmax(self.attention_layer(base), dim=-1)

class DeepVol(nn.Module):
    def __init__(self, num_blocks: int = 2, num_layers: int = 2, num_classes: int = 4, output_len: int = 16, 
                 ch_start: int = 16, ch_residual: int = 32, ch_dilation: int = 32, ch_skip: int = 64, 
                 ch_end: int = 32, kernel_size: int = 3, bias: bool = True, causal: bool = True) -> None:
        super().__init__()

        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.output_len = output_len

        self.ch_start = ch_start
        self.ch_residual = ch_residual
        self.ch_dilation = ch_dilation
        self.ch_skip = ch_skip
        self.ch_end = ch_end
        self.kernel_size = kernel_size
        self.bias = bias
        self.causal = causal

        # Initial 1x1 convolution
        self.initial_convolution = nn.Conv1d(in_channels=1, out_channels=self.ch_residual, kernel_size=1, bias=self.bias)

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(self.num_layers, self.ch_residual, self.ch_dilation, self.ch_skip,
                          self.kernel_size, self.bias, self.causal) for _ in range(self.num_blocks)
        ])

        # Output convolutions
        self.output_convolution = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(in_channels=self.ch_skip, out_channels=self.ch_end, kernel_size=1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv1d(in_channels=self.ch_end, out_channels=self.num_classes, kernel_size=1, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for DeepVol.
        :param x: Input tensor.
        :return: Output tensor.
        """
        residual = self.initial_convolution(x)

        skip_connections = []
        for block in self.residual_blocks:
            residual, skip = block(residual)
            skip_connections.append(skip)

        summed_skip = torch.cat(skip_connections).sum(dim=0)
        output = self.output_convolution(summed_skip)

        return output[..., -self.output_len:]  # Important for correct indexing


if __name__ == '__main__':
    # Test the DeepVol model
    batch_size, channels, seq_len = 16, 1, 64
    model = DeepVol()
    input_tensor = torch.randn(batch_size, channels, seq_len)
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
