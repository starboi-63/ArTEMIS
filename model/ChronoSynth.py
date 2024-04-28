import torch
import Conv_2d from ArTEMIS

class ChronoSynth(nn.Module):
    def __init__(self, num_inputs, num_features, kernel_size, dilation, apply_softmax=True):
        super(ChronoSynth, self).__init__()

        # Subnetwork to learn vertical and horizontal offsets during convolution
        def Subnet_offset(kernel_size):
            return MySequential(
                torch.nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Conv2d(in_channels=num_features, out_channels=kernel_size, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.ConvTranspose2d(kernel_size, kernel_size, kernel_size=3, stride=2, padding=1),
                torch.nn.Conv2d(in_channels=kernel_size, out_channels=kernel_size, kernel_size=3, stride=1, padding=1)
            )

        # Subnetwork to learn weights for each pixel in the kernel
        def Subnet_weight(kernel_size):
            return MySequential(
                torch.nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Conv2d(in_channels=num_features, out_channels=kernel_size, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.ConvTranspose2d(kernel_size, kernel_size, kernel_size=3, stride=2, padding=1),
                torch.nn.Conv2d(in_channels=kernel_size, out_channels=kernel_size, kernel_size=3, stride=1, padding=1),
                nn.Softmax(1) if apply_softmax else nn.Identity()
            )

        # Subnetwork to learn occlusion masks
        def Subnet_occlusion():
            return MySequential(
                torch.nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.ConvTranspose2d(num_features, num_features, kernel_size=3, stride=2, padding=1),
                torch.nn.Conv2d(in_channels=num_features, out_channels=num_inputs, kernel_size=3, stride=1, padding=1),
                torch.nn.Softmax(dim=1)
            )

        self.num_inputs = num_inputs
        self.kernel_size = kernel_size
        self.kernel_pad = int(((kernel_size - 1) * dilation) / 2.0)
        self.dilation = dilation

        self.modulePad = torch.nn.ReplicationPad2d([self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad])

        import cupy_module.synth as synth
        self.moduleSynth = synth.FunctionSynth.apply

        self.ModuleWeight = Subnet_weight(kernel_size ** 2)
        self.ModuleAlpha = Subnet_offset(kernel_size ** 2)
        self.ModuleBeta = Subnet_offset(kernel_size ** 2)
        self.moduleOcclusion = Subnet_occlusion()

        self.feature_fuse = Conv_2d(num_features * num_inputs, num_features, kernel_size=1, stride=1, batchnorm=False, bias=True)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, features, frames, output_size):
        H, W = output_size

        occlusion = torch.cat(torch.unbind(features, 1), 1)
        occlusion = self.lrelu(self.feature_fuse(occlusion))
        Occlusion = self.moduleOcclusion(occlusion, (H, W))

        B, C, T, cur_H, cur_W = features.shape
        features = features.transpose(1, 2).reshape(B*T, C, cur_H, cur_W)
        weights = self.ModuleWeight(features, (H, W)).view(B, T, -1, H, W)
        alphas = self.ModuleAlpha(features, (H, W)).view(B, T, -1, H, W)
        betas = self.ModuleBeta(features, (H, W)).view(B, T, -1, H, W)

        warp = []
        for i in range(self.n_inputs):
            weight = weights[:, i].contiguous()
            alpha = alphas[:, i].contiguous()
            beta = betas[:, i].contiguous()
            occlusion = Occlusion[:, i:i+1]
            frame = F.interpolate(frames[i], size=weight.size()[-2:], mode='bilinear')

            warp.append(
                occlusion * self.moduleAdaCoF(self.modulePad(frame), weight, alpha, beta, self.dilation)
            )

        framet = sum(warp)
        return framet