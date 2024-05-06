import torch
import torch.nn as nn


class upSplit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)

    def forward(self, x, output_size):
        x = self.upconv(x, output_size=output_size)
        return x
    
def joinTensors(X1, X2, type="concat"):

    if type == "concat":
        return torch.cat([X1, X2], dim=1)
    elif type == "add":
        return X1 + X2
    else:
        return X1


class Conv_2d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=False, batchnorm=False):
        super().__init__()
        self.conv = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]

        if batchnorm:
            self.conv += [nn.BatchNorm2d(out_ch)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        return self.conv(x)


class Conv_3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, batchnorm=False):
        super().__init__()
        self.conv = [nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias)]

        if batchnorm:
            self.conv += [nn.BatchNorm3d(out_channels)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        return self.conv(x)
    

class MySequential(nn.Sequential):
    def forward(self, input, output_size):
        for module in self:
            if isinstance(module, nn.ConvTranspose2d):
                input = module(input, output_size)
            else:
                input = module(input)
        return input