import config
import os
import lightning as L
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from model.artemis import ArTEMIS
from torch.optim import Adamax
from torch.optim.lr_scheduler import MultiStepLR
from loss import Loss
from metrics import eval_metrics
from data.preprocessing.vimeo90k_septuplet_process import get_loader
from tqdm import tqdm
from utils import read_image, save_image, save_images, read_video, save_video


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
    data_loader = get_loader(args.mode, args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
else:
    print("Custom Dataset Detected")


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
        """
        Run a forward pass of the model:
        images: a list of 4 tensors, each of shape (batch_size, 3, 256, 256)
        output_frame_times: a batch of time steps: (batch_size, 1)
        """
        return self.model(images, output_frame_times)

    def training_step(self, batch, batch_idx):
        images, gt_image, output_frame_times = batch

        output = self(images, output_frame_times)
        loss = self.loss(output, gt_image)

        # every collection of batches, save the outputs
        if batch_idx % args.log_iter == 0:
            save_images(output, gt_image, batch_idx, images, args.output_dir, epoch_index = self.current_epoch)
 
        # log metrics for each step
        learning_rate = self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[0]["lr"]
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('lr', learning_rate, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, gt_image, output_frame_times = batch
        output = self.model(images, output_frame_times)
        loss = self.loss(output, gt_image)
        psnr, ssim = self.validation(output, gt_image)

        # log metrics for each step
        self.log_dict({'test_loss': loss, 'psnr': psnr, 'ssim': ssim})

        if batch_idx % args.log_iter == 0:
            save_images(output, gt_image, batch_idx, images, args.output_dir, testing=True)
        
        # return metrics dictionary
        return {'loss': loss, 'psnr': psnr, 'ssim': ssim}
    
    def configure_optimizers(self):
        training_schedule = [40, 60, 75, 85, 95, 100] 
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": MultiStepLR(optimizer = self.optimizer, milestones = training_schedule, gamma = 0.5),
            }
        }


def interpolate(args):
    """
    Generate interpolated frames between a single set of four context frames.
    """
    device = torch.device('cuda' if args.cuda else 'cpu')

    # Load the pre-trained model
    model = ArTEMISModel.load_from_checkpoint(args.model_path)
    model.to(device)
    model.eval()

    # Read in the context frames
    paths = [args.frame1_path, args.frame2_path, args.frame3_path, args.frame4_path]
    context_frames = [read_image(path).to(device) for path in paths]

    # Run the forward pass of the model to generate the interpolated frames
    timesteps = "".join(args.timesteps.split()).split(",")
    timesteps = [float(t) for t in timesteps]
    timesteps = torch.tensor(timesteps).to(device)

    with tqdm(timesteps, desc="Interpolating frames") as pbar:
        for timestep in pbar:
            with torch.no_grad():
                _, _, out_batch = model(context_frames, timestep)
    
            # Save the interpolated frame
            save_image(out_batch[0], f"frame_t={timestep}.png", args.save_path)
    

def main(args):
    torch.set_float32_matmul_precision("medium")
    
    logger = TensorBoardLogger(args.log_dir, name="ArTEMIS")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model = ArTEMISModel(args)
    trainer = L.Trainer(max_epochs=args.max_epoch, log_every_n_steps=args.log_iter, logger=logger, enable_checkpointing=args.use_checkpoint, callbacks=[lr_monitor])
    
    if args.mode == "interpolate":
        return interpolate(args)
    
    if args.mode == "train":
        if args.use_checkpoint:
            return trainer.fit(model, data_loader, ckpt_path=args.checkpoint_dir)
        else:
            return trainer.fit(model, data_loader)
    
    if args.mode == "test":
        if args.use_checkpoint:
            return trainer.test(model, data_loader, ckpt_path=args.checkpoint_dir)
        else:
            return trainer.test(model, data_loader)


if __name__ == "__main__":
    main(args)
