import time
import config
import os

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


def save_images(outputs, gt_images):
    """
    Given some outputs and ground truths, save them all locally 
    """


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
            # save_images(outputs, gt_images)
            print("a single output frame shape", outputs[0].shape)
            print("a single reference frame shape ", gt_images[0].shape)
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
