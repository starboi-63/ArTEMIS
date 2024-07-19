import os
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


def read_image(path):
    """
    Read an image from disk and return it as a tensor
    """
    image = Image.open(path).convert('RGB')
    transform = transforms.Compose([transforms.ToTensor()])
    image = transform(image).unsqueeze(0)
    return image


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


def save_images(output, gt_image, batch_index, context_frames, output_dir, epoch_index=None, testing=False):
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
        if testing:
            gt_image_name = f"gt_batch{batch_index}_sample{sample_num}_testset.png"
            output_image_name = f"pred_batch{batch_index}_sample{sample_num}_testset.png"
        else:
            gt_image_name = f"gt_epoch{epoch_index}_batch{batch_index}_sample{sample_num}.png"
            output_image_name = f"pred_epoch{epoch_index}_batch{batch_index}_sample{sample_num}.png"

        # Create directories for each epoch, batch, sample, and frame
        if testing:
            gt_write_path = os.path.join(output_dir, f"test_set", f"batch_{batch_index}", f"sample_{sample_num}")
            output_write_path = os.path.join(output_dir, f"test_set", f"batch_{batch_index}", f"sample_{sample_num}")
        else:
            gt_write_path = os.path.join(output_dir, f"epoch_{epoch_index}", f"batch_{batch_index}", f"sample_{sample_num}")
            output_write_path = os.path.join(output_dir, f"epoch_{epoch_index}", f"batch_{batch_index}", f"sample_{sample_num}")

        # Save the ground-truth and prediction images
        save_image(gt, gt_image_name, gt_write_path)
        save_image(output_image, output_image_name, output_write_path)

        # Save the context frames
        for i, context in enumerate(contexts):
            if testing:
                context_image_name = f"context_batch{batch_index}_sample{sample_num}_frame{i}_testset.png"
                context_write_path = os.path.join(output_dir, f"test_set", f"batch_{batch_index}", f"sample_{sample_num}")
            else:
                context_image_name = f"context_epoch{epoch_index}_batch{batch_index}_sample{sample_num}_frame{i}.png"
                context_write_path = os.path.join(output_dir, f"epoch_{epoch_index}", f"batch_{batch_index}", f"sample_{sample_num}")
            
            save_image(context, context_image_name, context_write_path)
