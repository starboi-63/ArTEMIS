import time
import config
import os

import cv2
import numpy as np

# from PIL import Image
import lightning as L
import torch

from lightning.pytorch.loggers import TensorBoardLogger
from model.artemis import ArTEMIS
from torch.optim import Adamax
from torch.optim.lr_scheduler import MultiStepLR
# from loss import Loss
from loss import Loss
from metrics import eval_metrics
from data.preprocessing.vimeo90k_septuplet_process import get_loader


# Parse command line arguments
args, unparsed = config.get_args()
save_location = os.path.join(args.checkpoint_dir, "checkpoints")

# Initialize CUDA & set random seed
device = torch.device('cuda' if args.cuda else 'cpu')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.random_seed)

if args.cuda:
    torch.cuda.manual_seed(args.random_seed)

# Initialize DataLoaders
if args.dataset == "vimeo90K_septuplet":
    t0 = time.time()
    train_loader = get_loader('train', args.data_root, args.batch_size, shuffle=True, num_workers=args.num_workers)
    t1 = time.time()
    test_loader = get_loader('test', args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers)
    t2 = time.time()
else:
    raise NotImplementedError


def save_images(output, gt_image, batch_index, epoch_index):
    """
    Given some outputs and ground truths, save them all locally 
    outputs are, like always, a triple of ll, l, and output

    """
    _, _, output_img = output

    for sample_num, (gt, output_image) in enumerate(zip(gt_image, output_img)):
        # Convert to numpy and scale to 0-255
        gt_image_color = gt.permute(1, 2, 0).cpu().clamp(0.0, 1.0).detach().numpy() * 255.0
        output_image_color = output_image.permute(1, 2, 0).cpu().clamp(0.0, 1.0).detach().numpy() * 255.0

        # Convert to BGR for OpenCV
        gt_image_result = cv2.cvtColor(gt_image_color.squeeze().astype(np.uint8), cv2.COLOR_RGB2BGR)
        output_image_result = cv2.cvtColor(output_image_color.squeeze().astype(np.uint8), cv2.COLOR_RGB2BGR)

        gt_image_name = f"gt_epoch{epoch_index}_batch{batch_index}_sample{sample_num}.png"
        output_image_name = f"pred_epoch{epoch_index}_batch{batch_index}_sample{sample_num}.png"

        # Create directories for each epoch, batch, sample, and frame
        gt_write_path = os.path.join(
            args.output_dir, f"epoch_{epoch_index}", f"batch_{batch_index}", f"sample_{sample_num}", gt_image_name
        )

        output_write_path = os.path.join(
            args.output_dir, f"epoch_{epoch_index}", f"batch_{batch_index}", f"sample_{sample_num}", output_image_name
        )

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(gt_write_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_write_path), exist_ok=True)

        # Write images to disk
        cv2.imwrite(gt_write_path, gt_image_result)
        cv2.imwrite(output_write_path, output_image_result)


class ArTEMISModel(L.LightningModule):
    def __init__(self, cmd_line_args=args):
        super().__init__()
        # Call this to save command line arguments to checkpoints
        self.save_hyperparameters()
        # Initialize instance variables
        self.args = args
        self.model = ArTEMIS(num_inputs=args.nbr_frame, joinType=args.joinType, kernel_size=args.kernel_size, dilation=args.dilation)
        self.optimizer = Adamax(self.model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        self.loss = Loss(args)
        self.validation = eval_metrics


    def forward(self, images, output_frame_times):
        return self.model(images, output_frame_times)

    
    def training_step(self, batch, batch_idx):
        images, gt_image, output_frame_times = batch
        output = self(images, output_frame_times)
        loss = self.loss(output, gt_image)

        # every collection of batches, save the outputs
        if batch_idx % args.log_iter == 0:
            save_images(output, gt_image, batch_index = batch_idx, epoch_index = self.current_epoch)
 
        # log metrics for each step
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    
    def test_step(self, batch, batch_idx):
        images, gt_image, output_frame_times = batch
        output = self.model(images, output_frame_times)
        loss = self.loss(output, gt_image)
        psnr, ssim = self.validation(output, gt_image)

        # log metrics for each step
        self.log_dict({'test_loss': loss, 'psnr': psnr, 'ssim': ssim})
        
    
    def configure_optimizers(self):
        training_schedule = [40, 60, 75, 85, 95, 100]
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": MultiStepLR(optimizer = self.optimizer, milestones = training_schedule, gamma = 0.5),
            }
        }
    

""" Entry Point """


def main(args):
    # Set the precision for the model to fully utilize the GPU tensor cores
    torch.set_float32_matmul_precision('medium')

    # Train with Lightning 
    model = ArTEMISModel(args)
    logger = TensorBoardLogger(args.log_dir, name="ArTEMIS")
    trainer = L.Trainer(max_epochs=args.max_epoch, log_every_n_steps=args.log_iter, default_root_dir=args.checkpoint_dir, logger=logger)
    trainer.fit(model, train_loader)

    # Test the model with Lightning
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main(args)
