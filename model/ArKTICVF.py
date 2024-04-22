import torch
import torch.nn as nn
from model.SEP_STS_Encoder import ResBlock

class ArKTICVF(nn.Module):
    def __init__(self, num_inputs=4, num_outputs=1, joinType="concat", kernel_scale=5, dilation=1):
        super().__init__()

        num_features = [192, 128, 64, 32]
        spatial_window_sizes = [(1, 8, 8), (1, 8, 8), (1, 8, 8), (1, 8, 8)] # For Sep-STS (Separated-Spatio-Temporal-SWIN) Encoder
        num_heads = [2, 4, 8, 16]
        self.joinType = joinType
        self.n_inputs = num_inputs
        self.n_outputs = num_outputs

        growth = 2 if joinType == "concat" else 1
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        from model.SEP_STS_Encoder import SepSTSEncoder
        self.encoder = SepSTSEncoder(num_features, num_inputs, spatial_window_sizes, num_heads)

        self.decoder = nn.Sequential(
            #define upSplit
        )


class upSplit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)

    def forward(self, x, output_size):
        x = self.upconv(x, output_size=output_size)
        return x