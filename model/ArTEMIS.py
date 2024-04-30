import torch
import torch.nn as nn
import torch.nn.functional as F
from model.SEP_STS_Encoder import ResBlock
import ChronoSynth


class TimeEmbedding(nn.Module):
    def __init__(self, embedding_size, height_dim, width_dim):
        '''
        Inspired by the positional encoding from Attention is All You Need, we calculate
        Time encodings for each output vector and concatenate it to the channel dimension
        of the latent representation.

        embedding_size: hyperparameter for the size of the embedding along the channel dimension
        height_dim: height dimension
        width_dim: width dimension
        '''
        self.dense1 = nn.Linear(1, embedding_size)
        self.dense2 = nn.Linear(embedding_size, embedding_size * height_dim)
        self.dense3 = nn.Linear(
            embedding_size * height_dim, embedding_size * height_dim * width_dim)
        self.activation = nn.ReLU()

    def forward(self, output_frame_time, latent):
        '''
        output_frame_time: value between 0-1 representing the time of the output frame
        latent: the encoded representation of input frames
        '''
        assert output_frame_time > 0 and output_frame_time < 1
        # Batch * Time * Channel * Height * Width
        B, T, C, H, W = latent.shape

        # Feed Forward: [1] --> [embedding_size * height_dim * width_dim]
        output_frame_time = torch.tensor(output_frame_time)
        time_embedding = self.dense1(output_frame_time)
        time_embedding = self.activation(time_embedding)
        time_embedding = self.dense2(time_embedding)
        time_embedding = self.activation(time_embedding)
        time_embedding = self.dense3(time_embedding)
        time_embedding = self.activation(time_embedding)

        # Reshape: [embedding_size * height_dim * width_dim] --> [embedding_size, height_dim, width_dim]
        time_embedding = torch.reshape(
            time_embedding, (-1, H, W))
        # Copy values across Batch and Temporal dimensions: [embedding_size, height_dim, width_dim] --> [B, T, embedding_size, height_dim, width_dim]
        time_embedding = time_embedding.unsqueeze(
            0).unsqueeze(0).expand(B, T, -1, -1, -1)
        # Concat embedding to the latent representation's channel dimension: [B, T, embedding_size, height_dim, width_dim] --> [B, T, C + embedding_size, H, W]
        return torch.cat([latent, time_embedding], 2)


class ArTEMIS(nn.Module):
    def __init__(self, num_inputs=4, joinType="concat", kernel_size=5, dilation=1, time_embedding_size=64):
        super().__init__()

        num_features = [192, 128, 64, 32]
        # For Sep-STS (Separated-Spatio-Temporal-SWIN) Encoder
        spatial_window_sizes = [(1, 8, 8), (1, 8, 8), (1, 8, 8), (1, 8, 8)]
        num_heads = [2, 4, 8, 16]  # For Multi-Head Attention
        self.joinType = joinType
        self.num_inputs = num_inputs
        # self.num_outputs = num_outputs
        # delta_t: the perceived timestep between each frame
        # We treat all input and output frames as spaced out evenly
        # self.delta_t = 1 / (num_outputs + 1)

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

        # TODO: Figure out what HEIGHT_DIM and WIDTH_DIM are
        HEIGHT_DIM = ??
        WIDTH_DIM = ??
        self.time_embedding = TimeEmbedding(
            time_embedding_size, height_dim=HEIGHT_DIM, width_dim=WIDTH_DIM)
        # TODO: define SynBlocks/ChronoSynths
        self.predict1 = ChronoSynth(
            num_inputs, num_features, kernel_size, dilation, apply_softmax=True)
        self.predict2 = ChronoSynth(
            num_inputs, num_features, kernel_size, dilation, apply_softmax=False)
        self.predict3 = ChronoSynth(
            num_inputs, num_features, kernel_size, dilation, apply_softmax=False)

    def forward(self, frames, num_outputs=1):
        '''
        Performs the forward pass for each output frame needed, a number of times equal to num_outputs.
        Returns the interpolated frames as a list of outputs: [interp1, interp2, interp3, ...]
        frames: input frames
        '''
        images = torch.stack(frames, dim=2)
        B, T, C, H, W = images.shape

        # Batch mean normalization works slightly better than global mean normalization (hence the repeated calls to .mean() below)
        mean_ = images.mean(2, keepdim=True).mean(
            3, keepdim=True).mean(4, keepdim=True)
        images = images - mean_

        out = []
        out_l = []
        out_ll = []

        # __________________________________________________________________
        # TODO: Modify VFIT architecture below to incorporate time

        # Only need to generate latent representation once
        x0, x1, x2, x3, x4 = self.encoder(images)

        dx3 = self.lrelu(self.decoder[0](x4, x3.size()))
        dx3 = joinTensors(dx3, x3, type=self.joinType)

        dx2 = self.lrelu(self.decoder[1](dx3, x2.size()))
        dx2 = joinTensors(dx2, x2, type=self.joinType)

        dx1 = self.lrelu(self.decoder[2](dx2, x1.size()))
        dx1 = joinTensors(dx1, x1, type=self.joinType)

        low_scale_features = self.smooth1(dx3)
        mid_scale_features = self.smooth2(dx2)
        high_scale_features = self.smooth3(dx1)

        # Generate multiple output frames
        for i in range(1, num_outputs + 1):
            delta_t = 1 / (num_outputs + 1)
            time = i * delta_t
            low_scale_features_with_time = self.time_embedding(
                time, low_scale_features)

            curr_out_ll = self.predict1(
                low_scale_features_with_time, frames, x2.size()[-2:])

            curr_out_l = self.predict2(
                mid_scale_features, frames, x1.size()[-2:])
            curr_out_l = F.interpolate(curr_out_ll, size=curr_out_l.size()
                                       [-2:], mode='bilinear') + curr_out_l

            curr_out = self.predict3(
                high_scale_features, frames, x0.size()[-2:])
            curr_out = F.interpolate(curr_out_l, size=curr_out.size()
                                     [-2:], mode='bilinear') + curr_out

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
