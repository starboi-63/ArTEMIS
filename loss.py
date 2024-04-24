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

    # output :: interpolated frame
    # gt :: ground truth frame 
    def call(self, output, output_flipped, gt): 
        diff = (output - gt) ** 2
        diff_flipped = (output_flipped - gt) ** 2
        ep = self.epsilon ** 2
        return torch.mean(torch.sqrt(diff + ep) + torch.sqrt(diff_flipped + ep))
    
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
                case '': 
                    loss_function = ... # combo edsc loss function

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )