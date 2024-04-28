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
    # interp_flipped :: interpolated frame
    # gt :: ground truth frame 
    def forward(self, interp, interp_flipped, gt): 
        diff = (interp - gt) ** 2
        diff_flipped = (interp_flipped - gt) ** 2
        ep = self.epsilon ** 2
        # mean? 
        return torch.mean(torch.sqrt(diff + ep) + torch.sqrt(diff_flipped + ep))

# based off of this implementation of Gradient Difference Loss https://github.com/mmany/pytorch-GDL/blob/main/custom_loss_functions.py
class GDL(torch.nn.Module): 
    def __init__(self): # can add alpha term to be faithful to originally proposed paper or go without (consistent with edsc)  
        super(GDL, self).__init__()

    # interp :: interpolated frame
    # interp_flipped :: interpolated frame
    # gt :: ground truth frame 
    # add alpha term? 
    def forward(self, interp, interp_flipped, gt): 
        # assuming dimensions are B x T x C x H x W
        x = torch.abs(torch.diff(interp, dim=0) - torch.diff(gt, dim=0))
        y = torch.abs(torch.diff(interp, dim=1) - torch.diff(gt, dim=1))

        xf = torch.abs(torch.diff(interp_flipped, dim=0) - torch.diff(gt, dim=0))
        yf = torch.abs(torch.diff(interp_flipped, dim=0) - torch.diff(gt, dim=0))
    
        # take simple sum?  
        return torch.mean(x + y + xf + yf)

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
                case 'GDL': 
                    loss_function = GDL()

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )

        for loss_info in self.loss: 
            if loss_info['function'] is not None: 
                print('{:.3f} * {}'.format(loss_info['weight'], loss_info['type']))
                self.loss_module.append(loss_info['function'])
        
        device = torch.device('cuda' if args.cude else 'cpu')
        self.loss_module.to(device)
        if args.cuda: 
            self.loss_module = torch.nn.DataParallel(self.loss_module)

    # interp :: interpolated frame
    # interp_flipped :: interpolated frame
    # gt :: ground truth frame 
    def forward(self, interp, interp_flipped, gt): 
        avg_loss = 0
        loss_record = {}

        for loss_info in self.loss: 
            if loss_info['function'] is not None: 
                loss = loss_info['function'](interp, interp_flipped, gt)   
                weighted_loss = loss_info['weight'] * loss
                loss_record[loss_info['type']] = weighted_loss # type : weighted_loss (str : num) ? 
                avg_loss += weighted_loss

        return avg_loss, loss_record
        