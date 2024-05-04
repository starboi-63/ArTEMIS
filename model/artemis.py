import torch
import torch.nn as nn
import torch.multiprocessing as mp
from model.sep_sts_encoder import ResBlock, SepSTSEncoder
from model.chrono_synth import ChronoSynth
from model.helper_modules import upSplit, joinTensors, Conv_3d


class ArTEMIS(nn.Module):

    def __init__(self, num_inputs=4, joinType="concat", kernel_size=5, dilation=1, num_outputs=3): 
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

        num_features_plus_time = num_features_out + 1

        self.predict1 = ChronoSynth(
            num_inputs, num_features_out, kernel_size, dilation, self.delta_t, apply_softmax=True)
        self.predict2 = ChronoSynth(
            num_inputs, num_features_out, kernel_size, dilation, self.delta_t, apply_softmax=False)
        self.predict3 = ChronoSynth(
            num_inputs, num_features_out, kernel_size, dilation, self.delta_t, apply_softmax=False)

    @staticmethod 
    def generate_single_frame(frames, frame_index, output_queue, 
                              delta_t, predict1, predict2, predict3,
                              low_scale_features, mid_scale_features, high_scale_features,
                              x0_size, x1_size, x2_size):
        """
        Use a worker thread to generate A SINGLE frame 
        frame_index: the index of the frame, (0, 1, 2, ...)
        """

        time_step = frame_index * delta_t

        curr_out_ll = predict1(
            low_scale_features, frames, x2_size.size()[-2:], time_step)

        curr_out_l = predict2(
            mid_scale_features, frames, x1_size.size()[-2:], time_step)
        curr_out_l = nn.functional.interpolate(curr_out_ll, size=curr_out_l.size()
                                   [-2:], mode='bilinear') + curr_out_l

        curr_out = predict3(
            high_scale_features, frames, x0_size.size()[-2:], time_step)
        curr_out = nn.functional.interpolate(curr_out_l, size=curr_out.size()
                                 [-2:], mode='bilinear') + curr_out

        curr_out_ll = curr_out_ll.share_memory_()
        curr_out_l =  curr_out_l.share_memory_()
        curr_out = curr_out.share_memory_()

        # queue the three output frames, along with an index for later sorting
        output_queue.put(frame_index, curr_out_ll, curr_out_l, curr_out)

    def forward(self, frames):
        '''
        Performs the forward pass for each output frame needed, a number of times equal to num_outputs.
        Returns the interpolated frames as a list of outputs: [interp1, interp2, interp3, ...]
        frames: input frames
        '''

        images = torch.stack(frames, dim=2)
        B, C, T, H, W = images.shape

        # Batch mean normalization works slightly better than global mean normalization (hence the repeated calls to .mean() below)
        mean_ = images.mean(2, keepdim=True).mean(
            3, keepdim=True).mean(4, keepdim=True)
        images = images - mean_

        # define the three output lists for each of the generated frame sizes
        out_list = [None]*self.num_outputs
        out_l_list = [None]*self.num_outputs
        out_ll_list = [None]*self.num_outputs

        # set the spawn method for multiprocessing
        mp.set_start_method('spawn', force = True)

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

        # share the features across threads 
        low_scale_features = low_scale_features.share_memory_()
        mid_scale_features = mid_scale_features.share_memory_()
        high_scale_features = high_scale_features.share_memory_()

        # define a Queue for workers to send their output
        output_queue = mp.Queue()
        # keep track of our process
        processes = []

        # NOTE: we detach the frames tensors, as they are not needed for grad. descent 
        frames = [frame.detach() for frame in frames]

        # Spawn threads to generate each frame
        for i in range(1, self.num_outputs + 1):
            # set up the arguments for the worker. NOTE that these should be mostly pass by reference
            worker_args = (frames, i, output_queue, 
                           self.delta_t, self.predict1, self.predict2, self.predict3,
                           low_scale_features, mid_scale_features, high_scale_features,
                           x0.size(), x1.size(), x2.size())
            # set up the process to generate a single frame with the provided args
            process = mp.Process(target=ArTEMIS.generate_single_frame, args=worker_args)
            # spawn the process
            process.start()
            # add it to the list of processes 
            processes.append(process)

        # wait for each thread to finish
        for process in processes:
            process.join()

        # gather the outputs; sort them by the indices
        while not output_queue.empty():
            index, out_ll, out_l, out = output_queue.get()
            out_ll_list[index] = out_ll 
            out_l_list[index] = out_l 
            out_list[index] = out 

        if self.training:
            return out_ll_list, out_l_list, out_list
        else:
            return out_list
