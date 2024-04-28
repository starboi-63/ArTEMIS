import torch
import torch.nn as nn
from model.SEP_STS_Encoder import ResBlock


class ArTEMIS(nn.Module):
    def __init__(self, num_inputs=4, num_outputs=1, joinType="concat", kernel_size=5, dilation=1):
        super().__init__()

        num_features = [192, 128, 64, 32]
        # For Sep-STS (Separated-Spatio-Temporal-SWIN) Encoder
        spatial_window_sizes = [(1, 8, 8), (1, 8, 8), (1, 8, 8), (1, 8, 8)]
        num_heads = [2, 4, 8, 16]  # For Multi-Head Attention
        self.joinType = joinType
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        # delta_t: the perceived timestep between each frame
        # We treat all input and output frames as spaced out evenly
        self.delta_t = 1 / (num_outputs + 1)

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

        self.predict1 =  # TODO: define SynBlock

    def forward(self, frames):
        '''
        Performs the forward pass for each output frame needed, a number of times equal to num_outputs.
        Returns the interpolated frames as a list of outputs: [interp1, interp2, interp3, ...]
        frames: input frames
        '''
        images = torch.stack(frames, dim=2)
        _, _, _, H, W = images.shape

        # Batch mean normalization works slightly better than global mean normalization (hence the repeated calls to .mean() below)
        mean_ = images.mean(2, keepdim=True).mean(
            3, keepdim=True).mean(4, keepdim=True)
        images = images - mean_

        out = []
        out_l = []
        out_ll = []
        for i in range(self.num_outputs):
            # __________________________________________________________________
            # TODO: Modify VFIT architecture below to incorporate time
            x_0, x_1, x_2, x_3, x_4 = self.encoder(images)

            dx_3 = self.lrelu(self.decoder[0](x_4, x_3.size()))
            dx_3 = joinTensors(dx_3, x_3, type=self.joinType)

            dx_2 = self.lrelu(self.decoder[1](dx_3, x_2.size()))
            dx_2 = joinTensors(dx_2, x_2, type=self.joinType)

            dx_1 = self.lrelu(self.decoder[2](dx_2, x_1.size()))
            dx_1 = joinTensors(dx_1, x_1, type=self.joinType)

            fea3 = self.smooth_ll(dx_3)
            fea2 = self.smooth_l(dx_2)
            fea1 = self.smooth(dx_1)

            curr_out_ll = self.predict_ll(fea3, frames, x_2.size()[-2:])

            curr_out_l = self.predict_l(fea2, frames, x_1.size()[-2:])
            curr_out_l = F.interpolate(out_ll, size=out_l.size()
                                       [-2:], mode='bilinear') + out_l

            curr_out = self.predict(fea1, frames, x_0.size()[-2:])
            curr_out = F.interpolate(out_l, size=out.size()
                                     [-2:], mode='bilinear') + out
            out_ll.append(curr_out_ll)
            out_l.append(curr_out_l)
            out.append(curr_out)
        if self.training:
            return out_ll, out_l, out
        else:
            return out


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


def joinTensors(X1, X2, type="concat"):

    if type == "concat":
        return torch.cat([X1, X2], dim=1)
    elif type == "add":
        return X1 + X2
    else:
        return X1
