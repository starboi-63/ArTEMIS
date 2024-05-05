import torch 
import torch.nn as nn
import torchvision.models as models

class Loss(nn.modules.loss._Loss):
    def __init__(self, args):
        super(Loss, self).__init__()
        # just use simple L1 loss. I don't think we need CUDA for this
        self.loss_function = nn.L1Loss()
        self.device = torch.device('cuda' if args.cuda else 'cpu')

    def forward(self, outputs, ground_truths):
        """
        calculate loss for given outputs and ground_truth 
        outputs: tuple of interpolated frames in three lists for ll, l, and output)
        """
        _, _, out_list = outputs

        total_loss, num_outputs = 0, 0

        for out_image, gt_image in zip(out_list, ground_truths):
            # cuda-ify those images 
            out_image = out_image.to(self.device)
            gt_image = gt_image.to(self.device)
            total_loss += self.loss_function(out_image, gt_image)
            num_outputs += 1

        return total_loss / num_outputs

