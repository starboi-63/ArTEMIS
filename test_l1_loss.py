import torch 
import torch.nn as nn
import torchvision.models as models

class Loss(nn.modules.loss._Loss):
    def __init__(self, args):
        super(Loss, self).__init__()
        # just use simple L1 loss.
        self.loss_function = nn.L1Loss(reduction='mean')
        self.device = torch.device('cuda' if args.cuda else 'cpu')

    def forward(self, outputs, ground_truths):
        """
        calculate loss for given outputs and ground_truth 
        outputs: tuple of interpolated frames in three lists for ll, l, and output)
        """
        # out_list and ground_truths are [(B,C,H,W), (B,C,H,W), (B,C,H,W)]
        _, _, out_list = outputs 

        print("outputs shape: List of", out_list[0].shape)
        print("ground_truths shape: List of", ground_truths[0].shape)

        outputs = torch.cat(out_list, dim=0)            # (B*3, C, H, W)
        ground_truths = torch.cat(ground_truths, dim=0) # (B*3, C, H, W)

        print("outputs shape (after cat): ", outputs.shape)
        print("ground_truths shape (after cat): ", ground_truths.shape)

        return self.loss_function(outputs, ground_truths)

