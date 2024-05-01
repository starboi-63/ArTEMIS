# Sourced from https://github.com/myungsub/CAIN/blob/master/loss.py, who sourced from https://github.com/thstkdgus35/EDSR-PyTorch/tree/master/src/loss
# Added Huber loss in addition.
import torch
import torch.nn.functional as F
import torchvision.models as models


class Charbonnier(torch.nn.Module):
    def __init__(self, epsilon=0.001):
        # VFIT uses e=0.001
        # EDSC uses e=0.000001
        super(Charbonnier, self).__init__()
        self.epsilon = epsilon

    # interp :: interpolated frame
    # gt :: ground truth frame
    def forward(self, interp, gt):
        diff = (interp - gt) ** 2
        ep = self.epsilon ** 2
        # mean?
        return torch.mean(torch.sqrt(diff + ep))

# based off of this implementation of Gradient Difference Loss https://github.com/mmany/pytorch-GDL/blob/main/custom_loss_functions.py


class GDL(torch.nn.Module):
    def __init__(self):  # can add alpha term to be faithful to originally proposed paper or go without (consistent with edsc)
        super(GDL, self).__init__()

    # interp :: interpolated frame
    # gt :: ground truth frame
    # add alpha term?
    def forward(self, interp, gt):
        # assuming dimensions are B x C x T x H x W
        x = torch.abs(torch.diff(interp, dim=0) - torch.diff(gt, dim=0))
        y = torch.abs(torch.diff(interp, dim=1) - torch.diff(gt, dim=1))

        # take simple sum?
        return torch.mean(x + y)
    
class MeanShift(torch.nn.Conv2d):
    def __init__(self, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False
    
class VGG(torch.nn.Module):
    def __init__(self, loss_type):
        super(VGG, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        conv_index = loss_type[-2:]
        if conv_index == '22':
            self.vgg = torch.nn.Sequential(*modules[:8])
        elif conv_index == '33':
            self.vgg = torch.nn.Sequential(*modules[:16])
        elif conv_index == '44':
            self.vgg = torch.nn.Sequential(*modules[:26])
        elif conv_index == '54':
            self.vgg = torch.nn.Sequential(*modules[:35])
        elif conv_index == 'P':
            self.vgg = torch.nn.ModuleList([
                torch.nn.Sequential(*modules[:8]),
                torch.nn.Sequential(*modules[8:16]),
                torch.nn.Sequential(*modules[16:26]),
                torch.nn.Sequential(*modules[26:35])
            ])
        self.vgg = torch.nn.DataParallel(self.vgg).cuda()

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229, 0.224, 0.225)
        self.sub_mean = MeanShift(vgg_mean, vgg_std)
        self.vgg.requires_grad = False
        # self.criterion = nn.L1Loss()
        self.conv_index = conv_index

    def forward(self, sr, hr):
        def _forward(x):
            x = self.sub_mean(x)
            x = self.vgg(x)
            return x
        def _forward_all(x):
            feats = []
            x = self.sub_mean(x)
            for module in self.vgg.module:
                x = module(x)
                feats.append(x)
            return feats

        if self.conv_index == 'P':
            vgg_sr_feats = _forward_all(sr)
            with torch.no_grad():
                vgg_hr_feats = _forward_all(hr.detach())
            loss = 0
            for i in range(len(vgg_sr_feats)):
                loss_f = F.mse_loss(vgg_sr_feats[i], vgg_hr_feats[i])
                #print(loss_f)
                loss += loss_f
            #print()
        else:
            vgg_sr = _forward(sr)
            with torch.no_grad():
                vgg_hr = _forward(hr.detach())
            loss = F.mse_loss(vgg_sr, vgg_hr)

        return loss

# class wrapper for loss functions

class Loss(torch.nn.modules.loss._Loss):
    def __init__(self, args):
        super(Loss, self).__init__()

        self.loss = []
        self.loss_module = torch.nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            match loss_type:
                case'L1':
                    loss_function = torch.nn.L1Loss()
                case 'Charb':
                    loss_function = Charbonnier()
                case 'VGG': 
                    loss_function = VGG()
                case 'GDL':
                    loss_function = GDL()

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )

        for loss_info in self.loss:
            if loss_info['function'] is not None:
                print(
                    '{:.3f} * {}'.format(loss_info['weight'], loss_info['type']))
                self.loss_module.append(loss_info['function'])

        device = torch.device('cuda' if args.cude else 'cpu')
        self.loss_module.to(device)
        if args.cuda:
            self.loss_module = torch.nn.DataParallel(self.loss_module)

    # interp :: interpolated frame
    # gt :: ground truth frame
    def forward(self, interp, gt):
        avg_loss = 0
        loss_record = {}

        for loss_info in self.loss:
            if loss_info['function'] is not None:
                loss = loss_info['function'](interp, gt)
                weighted_loss = loss_info['weight'] * loss
                # type : weighted_loss (str : num) ?
                loss_record[loss_info['type']] = weighted_loss
                avg_loss += weighted_loss

        return avg_loss, loss_record
