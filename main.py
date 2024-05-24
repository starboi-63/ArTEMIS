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
from utils import save_images, read_video, save_video


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


def video_interpolation(args):
    """
    Run an interpolation on a video of frames: 
    By default, generates a frame between each pair of input frames
    """
    # Read the video file and send it to the GPU
    device = torch.device('cuda' if args.cuda else 'cpu')
    input_frames, input_frame_rate = read_video(args.input_path)
    input_frames = [frame.to(device) for frame in input_frames]
    
    # Duplicate the first and last input frames
    input_frames = [input_frames[0]] + input_frames + [input_frames[-1]]

    # Load the pre-trained model
    model = ArTEMISModel.load_from_checkpoint(args.model_path)
    model.to(device)
    model.eval()

    # Initialize a list to store interpolated frames
    interpolated_frames = []
    
    # Iterate through every window of 4 frames
    with tqdm(range(len(input_frames) - 3), desc="Interpolating frames") as pbar:
        for i in pbar:
            # Extract the 4 frames and set the interpolated frame time to 0.5
            context_frames = input_frames[i:i+4]
            interpolated_frame_time = torch.tensor([0.5]).to(device)

            # Interpolate in the exact center of the 4 frames
            with torch.no_grad():
                _, _, out_batch = model(context_frames, interpolated_frame_time)

            # Extract the output frame
            interpolated_frames.append(out_batch[0])

    # Remove the first and last frames from the input
    input_frames = input_frames[1:-1]

    # Alternate between the input and output frames
    output_frames = []

    for i in range(len(interpolated_frames)):
        output_frames.append(input_frames[i])
        output_frames.append(interpolated_frames[i])
    
    interpolated_frames.append(input_frames[-1])

    # Save the output frames to a video file
    save_video(output_frames, args.save_path, input_frame_rate * 2)
    print("Saved video to: ", args.save_path)


def main(args):
    torch.set_float32_matmul_precision("medium")
    
    logger = TensorBoardLogger(args.log_dir, name="ArTEMIS")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model = ArTEMISModel(args)
    trainer = L.Trainer(max_epochs=args.max_epoch, log_every_n_steps=args.log_iter, logger=logger, enable_checkpointing=args.use_checkpoint, callbacks=[lr_monitor])

    if args.mode == "interpolate":
        return video_interpolation(args)
    
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
