import time
import config
import os

import cv2
import numpy as np

from PIL import Image
import lightning as L
import torch
from torchvision import transforms

from lightning.pytorch.loggers import TensorBoardLogger
from model.artemis import ArTEMIS
from torch.optim import Adamax
from torch.optim.lr_scheduler import MultiStepLR
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
    print("Custom Dataset Detected")


def save_image(image, name, path):
    """
    Save an image to disk
    """
    # Convert to numpy and scale to 0-255
    image = image.permute(1, 2, 0).cpu().clamp(0.0, 1.0).detach().numpy() * 255.0
    # Convert to BGR for OpenCV
    image = cv2.cvtColor(image.squeeze().astype(np.uint8), cv2.COLOR_RGB2BGR)
    # Create directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    # Write image to disk
    cv2.imwrite(os.path.join(path, name), image)


def save_images(output, gt_image, batch_index, context_frames, epoch_index):
    """
    Given an output and ground truth, save them all locally along with context frames
    outputs are, like always, a triple of ll, l, and output
    """
    _, _, output_img = output

    # context_frames is a list of 4 tensors: [(16, 3, 256, 256), (16, 3, 256, 256), (16, 3, 256, 256), (16, 3, 256, 256)]
    # Want 16 lists of lists of 4 tensors instead: [[(3, 256, 256), (3, 256, 256), (3, 256, 256), (3, 256, 256)], ...]
    context_frames = [list(context_frame) for context_frame in zip(*context_frames)]

    for sample_num, (gt, output_image, contexts) in enumerate(zip(gt_image, output_img, context_frames)):
        # Create image names
        gt_image_name = f"gt_epoch{epoch_index}_batch{batch_index}_sample{sample_num}.png"
        output_image_name = f"pred_epoch{epoch_index}_batch{batch_index}_sample{sample_num}.png"

        # Create directories for each epoch, batch, sample, and frame
        gt_write_path = os.path.join(args.output_dir, f"epoch_{epoch_index}", f"batch_{batch_index}", f"sample_{sample_num}")
        output_write_path = os.path.join(args.output_dir, f"epoch_{epoch_index}", f"batch_{batch_index}", f"sample_{sample_num}")

        # Save the ground-truth and prediction images
        save_image(gt, gt_image_name, gt_write_path)
        save_image(output_image, output_image_name, output_write_path)

        # Save the context frames
        for i, context in enumerate(contexts):
            context_image_name = f"context_epoch{epoch_index}_batch{batch_index}_sample{sample_num}_frame{i}.png"
            context_write_path = os.path.join(args.output_dir, f"epoch_{epoch_index}", f"batch_{batch_index}", f"sample_{sample_num}")
            save_image(context, context_image_name, context_write_path)


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
            save_images(output, gt_image, batch_index = batch_idx, context_frames=images, epoch_index = self.current_epoch)
 
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

def test_and_train(args):
    torch.set_float32_matmul_precision("medium")
    logger = TensorBoardLogger(args.log_dir, name="ArTEMIS")
    model = ArTEMISModel(args)
    trainer = L.Trainer(max_epochs=args.max_epoch, log_every_n_steps=args.log_iter, logger=logger, enable_checkpointing=args.use_checkpoint)

    # Train with Lightning: Load from checkpoint if specified
    if args.use_checkpoint:
        trainer.fit(model, train_loader, ckpt_path=args.checkpoint_dir)
    else:
        trainer.fit(model, train_loader)

    # Test the model with Lightning
    trainer.test(model, test_loader)


def single_interpolation(args):
    """
    Run an interpolation on a single input of 4 frames: 
    Produces a single output frame at an arbitrary time step
    """
    img_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    device = torch.device('cuda' if args.cuda else 'cpu')

    t = args.time_step
    input_image_paths = [args.f1_path, args.f2_path, args.f3_path, args.f4_path]
    input_images = [torch.unsqueeze(img_transforms(Image.open(path)), 0).to(device) for path in input_image_paths]
    output_frame_times = torch.tensor([t]).to(device)
    model = ArTEMISModel.load_from_checkpoint(args.parameter_path)
    model.eval()
    _, _, out_batch = model.forward(input_images, output_frame_times)
    # the image is currently batched, so we need to extract it
    out_img = out_batch[0]
    out_name = "output.png"
    save_image(out_img, out_name, args.eval_output_path)


def read_video(video_path):
    """
    Read a video file and return a numpy array of individual frames
    
    Returns:
    - video_frames: a list of tensors, each of shape (1, 3, 256, 256)
    - frame_rate: the frame rate of the video
    """
    # Define the transform to apply to each frame
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load the video file
    capture = cv2.VideoCapture(args.video_path)
   
    if not capture.isOpened():
        raise Exception("Error opening video file")
    
    frame_rate = capture.get(cv2.CAP_PROP_FPS)
    video_frames = []

    # Read the video frame by frame
    while True:
        ret, frame = capture.read()

        if not ret:
            break 

        # Convert the frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert the numpy array to a PIL image
        frame = Image.fromarray(frame)
        # Add transforms and a batch dimension to the frame
        frame = transform(frame).unsqueeze(0)
        video_frames.append(frame)

    # Release the video capture object
    capture.release()
    return video_frames, frame_rate


def save_video(frames, output_path, frame_rate):
    """
    Save a list of frames to a video file
    """
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    size = (frames[0].shape[2], frames[0].shape[1])
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, size)

    # Convert each frame to a numpy array and write it to the video file
    for frame in frames:
        # Convert the frame to a numpy array
        frame = frame.permute(1, 2, 0).cpu().numpy() * 255.0
        frame = frame.astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Write the frame to the video file
        out.write(frame)

    # Release the VideoWriter object
    out.release()


def video_interpolation(args):
    """
    Run an interpolation on a video of frames: 
    By default, generates a frame between each pair of input frames
    """
    # Read the video file and send it to the GPU
    device = torch.device('cuda' if args.cuda else 'cpu')
    input_frames, input_frame_rate = read_video(args.video_path)
    input_frames = [frame.to(device) for frame in input_frames]

    print("Input frames: ", len(input_frames))
    
    # Duplicate the first and last input frames
    input_frames = [input_frames[0]] + input_frames + [input_frames[-1]]

    print("Input frames post duplication: ", len(input_frames))

    # Load the pre-trained model
    model = ArTEMISModel.load_from_checkpoint(args.parameter_path)
    model.to(device)
    model.eval()

    # Initialize a list to store interpolated frames
    interpolated_frames = []
    
    # Iterate through every window of 4 frames
    for i in range(len(input_frames) - 3):
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

    print("Input frames post removal: ", len(input_frames))
    print("Interpolated frames: ", len(interpolated_frames))

    # Alternate between the input and output frames
    output_frames = []

    for i in range(len(interpolated_frames)):
        output_frames.append(input_frames[i])
        output_frames.append(interpolated_frames[i])
    
    interpolated_frames.append(input_frames[-1])

    print("Output frames: ", len(output_frames))

    # Save the output frames to a video file
    save_video(output_frames, args.output_path, input_frame_rate * 2)
    print("Saved video to: ", args.output_path)


def main(args):
    if args.eval:
        single_interpolation(args)
    else:
        test_and_train(args)


if __name__ == "__main__":
    main(args)
