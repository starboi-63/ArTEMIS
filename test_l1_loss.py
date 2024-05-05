import torch 
import torch.nn as nn
import torchvision.models as models

class Loss(nn.modules.loss._Loss):
    def __init__(self, args):
        super(Loss, self).__init__()
        # just use simple L1 loss.
        self.loss_function = nn.L1Loss(reduction='mean')
        self.device = torch.device('cuda' if args.cuda else 'cpu')

    # def forward(self, outputs, ground_truths):
    #     """
    #     calculate loss for given outputs and ground_truth 
    #     outputs: tuple of interpolated frames in three lists for ll, l, and output)
    #     """
    #     # out_list and ground_truths are [(B,C,H,W), (B,C,H,W), (B,C,H,W)]
    #     _, _, out_list = outputs 
    #
    #     outputs = torch.cat(out_list, dim=0)            # (B*3, C, H, W)
    #     ground_truths = torch.cat(ground_truths, dim=0) # (B*3, C, H, W)
    #
    #     return self.loss_function(outputs, ground_truths)

    def forward(self, output, ground_truth):
        """
        calculate loss for given outputs and ground_truth 
        outputs: tuple of interpolated frames in three lists for ll, l, and output)
        """
        # out_list and ground_truths are [(B,C,H,W), (B,C,H,W), (B,C,H,W)]
        _, _, out_img = output 

        # outputs = torch.cat(out_list, dim=0)            # (B*3, C, H, W)
        # ground_truths = torch.cat(ground_truths, dim=0) # (B*3, C, H, W)

        return self.loss_function(output, ground_truth)
