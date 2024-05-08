import argparse
import os

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Dataset selection and paths
data_arg = add_argument_group("Dataset")
data_arg.add_argument("--dataset", type=str, default="vimeo90K_septuplet")

# Model parameters
model_arg = add_argument_group("Model")
model_choices = ["ArTEMIS"]
model_arg.add_argument("--model", choices=model_choices, type=str, default="ArTEMIS")
model_arg.add_argument("--nbr_frame", type=int, default=4)
model_arg.add_argument("--joinType", choices=["concat", "add", "none"], default="concat")
model_arg.add_argument("--kernel_size", type=int, default=5)
model_arg.add_argument("--dilation", type=int, default=1)
model_arg.add_argument("--num_outputs", type=int, default=3)

# Evaluation Parameters
eval_arg = add_argument_group("Evaluation")
eval_arg.add_argument("--interpolate", action=argparse.BooleanOptionalAction, help="Run the model on custom inputs.")
eval_arg.add_argument("--time_step", type=float, default=0.5, help ="Arbitrary time step from 0-1.")
eval_arg.add_argument("--model_path", type=str, help="Path to the pretrained model parameters.")
eval_arg.add_argument("--input_path", type=str, help="Path to the input video that will be interpolated.")
eval_arg.add_argument("--save_path", type=str, help="Path to save the interpolated video output.")
eval_arg.add_argument("--test", action=argparse.BooleanOptionalAction, help="Run the model on the test set initially to calculate PSNR and SSIM.")

# Training parameters
learn_arg = add_argument_group("Learning")
learn_arg.add_argument("--lr", type=float, default=2e-4)
learn_arg.add_argument("--beta1", type=float, default=0.9)
learn_arg.add_argument("--beta2", type=float, default=0.999)
learn_arg.add_argument("--batch_size", type=int, default=4)
learn_arg.add_argument("--test_batch_size", type=int, default=12)
learn_arg.add_argument("--start_epoch", type=int, default=0)
learn_arg.add_argument("--max_epoch", type=int, default=100)

# Directories
dir_arg = add_argument_group("Directories")
dir_arg.add_argument("--data_root", type=str, default=os.path.join(os.getcwd(), "data", "sources", "vimeo_septuplet"))
dir_arg.add_argument("--checkpoint_dir", type=str, default=os.path.join(os.getcwd(), "training"))
dir_arg.add_argument("--load_from", type=str, default=os.path.join(os.getcwd(), "training", "checkpoints", "ArTEMIS", "model_best.pth"))
dir_arg.add_argument("--use_checkpoint", action=argparse.BooleanOptionalAction)
dir_arg.add_argument("--log_dir", type=str, default=os.path.join(os.getcwd(), "training", "logs"))
dir_arg.add_argument("--output_dir", type=str, default=os.path.join(os.getcwd(), "training", "output"))
    
# Miscellaneous
misc_arg = add_argument_group("Miscellaneous")
misc_arg.add_argument("--log_iter", type=int, default=100)
misc_arg.add_argument("--num_gpu", type=int, default=1)
misc_arg.add_argument("--random_seed", type=int, default=103)
misc_arg.add_argument("--num_workers", type=int, default=12)


def get_args():
    """Parses all of the arguments above"""
    args, unparsed = parser.parse_known_args()
    if args.num_gpu > 0:
        setattr(args, "cuda", True)
    else:
        setattr(args, "cuda", False)
    if len(unparsed) > 1:
        print("Unparsed args: {}".format(unparsed))
    return args, unparsed
