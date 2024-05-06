import torch
import torch.nn as nn
from model.sep_sts_encoder import ResBlock, SepSTSEncoder
from model.chrono_synth import ChronoSynth
from model.helper_modules import upSplit, joinTensors, Conv_3d


class ArTEMIS(nn.Module):
    def __init__(self, num_inputs=4, joinType="concat", kernel_size=5, dilation=1): 
        super().__init__()

        # TODO: Change num_features to [512, 256, 128, 64] if we want to scale up model 
        num_features = [192, 128, 64, 32]
        # For Sep-STS (Separated-Spatio-Temporal-SWIN) Encoder
        spatial_window_sizes = [(1, 8, 8), (1, 8, 8), (1, 8, 8), (1, 8, 8)]
        num_heads = [2, 4, 8, 16]  # For Multi-Head Attention
        self.joinType = joinType
        self.num_inputs = num_inputs

        growth = 2 if joinType == "concat" else 1
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.encoder = SepSTSEncoder(
            num_features, num_inputs, spatial_window_sizes, num_heads)

        self.decoder = nn.Sequential(
            upSplit(num_features[0], num_features[1]),
            upSplit(num_features[1]*growth, num_features[2]),
            upSplit(num_features[2]*growth, num_features[3]),
        )

        def SmoothNet(in_channels, out_channels):
            return nn.Sequential(
                Conv_3d(in_channels, out_channels, kernel_size=3,
                        stride=1, padding=1, batchnorm=False),
                ResBlock(out_channels, kernel_size=3),
            )

        num_features_out = 64
        self.smooth1 = SmoothNet(num_features[1]*growth, num_features_out)
        self.smooth2 = SmoothNet(num_features[2]*growth, num_features_out)
        self.smooth3 = SmoothNet(num_features[3]*growth, num_features_out)

        self.predict1 = ChronoSynth(
            num_inputs, num_features_out, kernel_size, dilation, apply_softmax=True)
        self.predict2 = ChronoSynth(
            num_inputs, num_features_out, kernel_size, dilation, apply_softmax=False)
        self.predict3 = ChronoSynth(
            num_inputs, num_features_out, kernel_size, dilation, apply_softmax=False)
        
    
    def forward(self, frames, output_frame_times):
        '''
        Performs the forward pass for each output frame needed, a number of times equal to num_outputs.
        Returns the interpolated frames as a list of outputs: [interp1, interp2, interp3, ...]
        frames: input frames
        output_frame_times: batch of arbitrary 't' from 0 to 1
        '''

        images = torch.stack(frames, dim=2)
        # Sanity check that the input frames are in the correct shape
        B, C, T, H, W = images.shape

        # Batch mean normalization works slightly better than global mean normalization (hence the repeated calls to .mean() below)
        mean_ = images.mean(2, keepdim=True).mean(
            3, keepdim=True).mean(4, keepdim=True)
        images = images - mean_

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

        curr_out_ll = self.predict1(low_scale_features, frames, x2.size()[-2:], output_frame_times)

        curr_out_l = self.predict2(mid_scale_features, frames, x1.size()[-2:], output_frame_times)
        curr_out_l = nn.functional.interpolate(curr_out_ll, size=curr_out_l.size()[-2:], mode='bilinear') + curr_out_l

        curr_out = self.predict3(high_scale_features, frames, x0.size()[-2:], output_frame_times)
        curr_out = nn.functional.interpolate(curr_out_l, size=curr_out.size()[-2:], mode='bilinear') + curr_out

        return curr_out_ll, curr_out_l, curr_out
