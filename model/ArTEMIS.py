import torch
import torch.nn as nn
from model.SEP_STS_Encoder import ResBlock


class ArTEMIS(nn.Module):
    def __init__(self, num_inputs=4, joinType="concat", kernel_scale=5, dilation=1, delta_t=0.5):
        super().__init__()

        num_features = [192, 128, 64, 32]
        # For Sep-STS (Separated-Spatio-Temporal-SWIN) Encoder
        spatial_window_sizes = [(1, 8, 8), (1, 8, 8), (1, 8, 8), (1, 8, 8)]
        num_heads = [2, 4, 8, 16]  # For Multi-Head Attention
        self.joinType = joinType
        self.n_inputs = num_inputs
        self.n_outputs = num_outputs

        growth = 2 if joinType == "concat" else 1
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        from model.SEP_STS_Encoder import SepSTSEncoder
        self.encoder = SepSTSEncoder(
            num_features, num_inputs, spatial_window_sizes, num_heads)

        self.decoder = nn.Sequential(
            upSplit(num_features[0], num_features[1]),
            upSplit(num_features[1]*growth, num_features[2]),
            upSplit(num_features[2]*growth, num_features[3]),
        )

        def SmoothNet(in_channels, out_channels):
            return nn.Sequential(
                Conv3d(in_channels, out_channels, kernel_size=3,
                       stride=1, padding=1, batchnorm=False),
                ResBlock(out_channels, kernel_size=3),
            )

        num_features_out = 64
        self.smooth1 = SmoothNet(num_features[1]*growth, num_features_out)
        self.smooth2 = SmoothNet(num_features[2]*growth, num_features_out)
        self.smooth3 = SmoothNet(num_features[3]*growth, num_features_out)

        self.predict1 =  # define SynBlock

    def forward(self, frames, t=0.5):
        images = torch.stack(frames, dim=2)
        _, _, _, H, W = images.shape

        # Batch mean normalization works slightly better than global mean normalization (hence the repeated calls to .mean() below)
        mean_ = images.mean(2, keepdim=True).mean(
            3, keepdim=True).mean(4, keepdim=True)
        images = images - mean_


class upSplit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)

    def forward(self, x, output_size):
        x = self.upconv(x, output_size=output_size)
        return x


class Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, batchnorm=False):
        super().__init__()
        self.conv = [nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias)]

        if batchnorm:
            self.conv += [nn.BatchNorm3d(out_channels)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        return self.conv(x)
