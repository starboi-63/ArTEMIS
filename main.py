import time
import config
import os

import  cv2
import numpy as np

# from PIL import Image
import lightning as L
import torch

from lightning.pytorch.loggers import TensorBoardLogger
from model.artemis import ArTEMIS
from torch.optim import Adamax
from torch.optim.lr_scheduler import MultiStepLR
# from loss import Loss
from test_l1_loss import Loss
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
    print("Time to load train loader: ", t1-t0)
    test_loader = get_loader('test', args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers)
    t2 = time.time()
    print("Time to load test loader", t2-t1)
else:
    raise NotImplementedError


def save_images(outputs, gt_images, batch_index, epoch_index = 0):
    """
    Given some outputs and ground truths, save them all locally 
    outputs are, like always, a triple of ll, l, and output

    """
    _, _, output_list = outputs

    for frame_index, (gt_image_batch, output_batch) in enumerate(zip(gt_images, output_list)):
        print("output image shape", output_batch.shape)
        print("ground_truth image shape", gt_image_batch.shape)

        for sample_num, (gt_image, output_image) in enumerate(zip(gt_image_batch, output_batch)):
            gt_image_color = gt_image.permute(1, 2, 0).cpu().clamp(0.0, 1.0).numpy() * 255.0
            output_image_color = output_image.permute(1, 2, 0).cpu().clamp(0.0, 1.0).numpy() * 255.0

            gt_image_result = cv2.cvtColor(gt_image_color.squeeze().astype(np.uint8), cv2.COLOR_RGB2BGR)
            output_image_result = cv2.cvtColor(output_image_color.squeeze().astype(np.uint8), cv2.COLOR_RGB2BGR)

            gt_image_name = f"gt_epoch{epoch_index}_batch{batch_index}_sample{sample_num}_frame{frame_index}.png"
            output_image_name = f"pred_epoch{epoch_index}_batch{batch_index}_sample{sample_num}_frame{frame_index}.png"

            gt_write_path = os.path.join(args.out_dir, "out", f"epoch_{epoch_index}", f"batch_{batch_index}", f"sample_{sample_num}", f"frame_{frame_index}", gt_image_name)
            output_write_path = os.path.join(args.out_dir, "out", f"epoch_{epoch_index}", f"batch_{batch_index}", f"sample_{sample_num}", f"frame_{frame_index}", output_image_name)

            cv2.imwrite(gt_write_path, gt_image_result)
            cv2.imwrite(output_write_path, output_image_result)


class ArTEMISModel(L.LightningModule):
    def __init__(self, cmd_line_args=args):
        super().__init__()
        # Call this to save command line arguments to checkpoints
        self.save_hyperparameters()
        # Initialize instance variables
        self.args = args
        self.model = ArTEMIS(num_inputs=args.nbr_frame, joinType=args.joinType, kernel_size=args.kernel_size, dilation=args.dilation, num_outputs=args.num_outputs)
        self.optimizer = Adamax(self.model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        self.loss = Loss(args)
        self.validation = eval_metrics


    def forward(self, images):
        return self.model(images)

    
    def training_step(self, batch, batch_idx):
        images, gt_images = batch
        outputs = self(images)
        loss = self.loss(outputs, gt_images)

        # every collection of batches, save the outputs
        if batch_idx % args.log_iter == 0:
            save_images(outputs, gt_images, batch_index = batch_idx)
            print("saved images on batch ", batch_idx)
 
        # log metrics for each step
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    
    def test_step(self, batch, batch_idx):
        images, gt_images = batch
        outputs = self.model(images)
        loss = self.loss(outputs, gt_images)
        psnr, ssim = self.validation(outputs, gt_images)

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
    # load_checkpoint(args, model, optimizer, save_location+'/epoch20/model_best.pth')

    # Train with Lightning 
    model = ArTEMISModel(args)
    logger = TensorBoardLogger(args.log_dir, name="ArTEMIS")
    trainer = L.Trainer(max_epochs=args.max_epoch, log_every_n_steps=args.log_iter, default_root_dir=args.checkpoint_dir, logger=logger)
    trainer.fit(model, train_loader)

    # Test the model with Lightning
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main(args)
