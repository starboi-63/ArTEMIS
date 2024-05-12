import torch 
import torch.nn as nn


class Loss(nn.modules.loss._Loss):
    def __init__(self, args):
        super(Loss, self).__init__()
        # Just use simple L1 loss.
        self.loss_function = nn.L1Loss(reduction='mean')
        self.device = torch.device('cuda' if args.cuda else 'cpu')

    def forward(self, output, ground_truth):
        """
        calculate loss for given outputs and ground_truth 
        outputs: tuple of interpolated frames in three lists for ll, l, and output)
        """
        _, _, out_img = output 
        return self.loss_function(out_img, ground_truth)
