# from https://github.com/myungsub/CAIN/blob/master/utils.py,
# but removed the errenous normalization and quantization steps from computing the PSNR.
# Contains support for calculating/tracking accuracy and loss metrics, called in main.py and test.py

# Originally Supported:
#
# Peak Signal-to-Noise ratio (PSNR): essentially a per-pixel inverse MSE calculation.
# Ranges 0-60 where higher is better.
# great for evaluating image compression, not great for picking up on quality/blurriness.
#
# Structural Similarity Index Measure (SSIM): compares luminance, contrast, and structure of the images.
# Ranges 0-1 where 1 is best
# Sensitive to spatial shifts, rotations, distortions. Bad at picking up hue and images colors.

# Additional: VMAF
# https://github.com/Netflix/vmaf
# https://github.com/Netflix/vmaf/blob/master/resource/doc/python.md

from pytorch_msssim import ssim_matlab as calc_ssim
import math

def eval_metrics(output, gt_image):
    """
    Average the metrics across an interpolated frame

    outputs: list of interpolated images produced by model
    gt_images: list ground truth images. What output will be compared against.

    PSNR should be calculated for each image, since sum(log) =/= log(sum).
    """
    print("output shape", output.shape)
    print("gt_image shape", gt_image.shape)
    total_psnr, total_ssim = 0, 0

    batch_size = gt_image.size(0)

    for b in range(batch_size):
        psnr = calc_psnr(output[b], gt_image[b])
        total_psnr += psnr

        # unsqueeze(0) to add batch dimension
        ssim = calc_ssim(output[b].unsqueeze(0).clamp(0,1), gt_image[b].unsqueeze(0).clamp(0,1) , val_range=1.)
        total_ssim += ssim

    return total_psnr / batch_size, total_ssim / batch_size


def calc_psnr(pred, gt):
    print("pred shape", pred.shape)
    print("gt shape", gt.shape)
    diff = (pred - gt).pow(2).mean() + 1e-8
    return -10 * math.log10(diff)
