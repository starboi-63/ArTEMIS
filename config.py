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
model_modes = ["train", "test", "interpolate"]
model_arg.add_argument("--mode", choices=model_modes, type=str, default="interpolate")
model_arg.add_argument("--nbr_frame", type=int, default=4)
model_arg.add_argument("--joinType", choices=["concat", "add", "none"], default="concat")
model_arg.add_argument("--kernel_size", type=int, default=5)
model_arg.add_argument("--dilation", type=int, default=1)
model_arg.add_argument("--num_outputs", type=int, default=3)

# Interpolation parameters
interpolate_arg = add_argument_group("Interpolation")
interpolate_arg.add_argument("--model_path", type=str, help="Path to the pretrained model.")
interpolate_arg.add_argument("--save_path", type=str, help="Path to save the interpolated output.")
interpolate_arg.add_argument("--frame1_path", type=str, help="Path to the first context frame.")
interpolate_arg.add_argument("--frame2_path", type=str, help="Path to the second context frame.")
interpolate_arg.add_argument("--frame3_path", type=str, help="Path to the third context frame.")
interpolate_arg.add_argument("--frame4_path", type=str, help="Path to the fourth context frame.")
interpolate_arg.add_argument("--timesteps", type=str, default="0.5", help ="Comma-separated list of timesteps from 0-1 to interpolate (e.g. '0.25, 0.5, 0.75').")

# Training parameters
learn_arg = add_argument_group("Learning")
learn_arg.add_argument("--lr", type=float, default=2e-4)
learn_arg.add_argument("--beta1", type=float, default=0.9)
learn_arg.add_argument("--beta2", type=float, default=0.999)
learn_arg.add_argument("--batch_size", type=int, default=4)
learn_arg.add_argument("--start_epoch", type=int, default=0)
learn_arg.add_argument("--max_epoch", type=int, default=100)

# Directories
dir_arg = add_argument_group("Directories")
dir_arg.add_argument("--data_dir", type=str, default="")
dir_arg.add_argument("--use_checkpoint", action=argparse.BooleanOptionalAction)
dir_arg.add_argument("--checkpoint_dir", type=str, default=os.path.join(os.getcwd(), "training"))
dir_arg.add_argument("--load_from", type=str, default=os.path.join(os.getcwd(), "model.ckpt"))
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
