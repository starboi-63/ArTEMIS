# ArTEMIS

Arbitrary timestep Transformer for Enhanced Multi-frame Interpolation and Synthesis.

## Getting Started

ArTEMIS is a deep learning model that interleaves [VFIT](https://github.com/zhshi0816/Video-Frame-Interpolation-Transformer) and [EDSC](https://github.com/Xianhang/EDSC-pytorch) together in order to enable the synthesis of intermediate video frames at arbitrary timesteps while using multiple frames on either side of the target timestep as context. The model is trained on the [Vimeo-90K Septuplet dataset](http://toflow.csail.mit.edu/).

### Prerequisites

ArTEMIS requires CUDA to execute. If your GPU does not support CUDA, then we also provide a [notebook file](train_google_colab.ipynb) which can be uploaded to Google Colab and executed there. If executing locally, ensure that you have [Python](https://www.python.org) installed on your system. To use ArTEMIS, you need to set up a Python environment with the necessary packages installed. You can do this by running the following commands in your terminal.

First, clone the repository to your local machine.

```bash
git clone https://github.com/starboi-63/ArTEMIS.git
```

Next, navigate to the project directory.

```bash
cd ArTEMIS
```

Then, install `virtualenv` if you don't already have it.

```bash
pip install virtualenv
```

Create a new virtual environment named `artemis-env` in the project directory.

```bash
python -m venv artemis-env
```

Activate the virtual environment.

```bash
source artemis-env/bin/activate
```

Install the required packages, which are listed in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

Finally, depending on your hardware, you will need to install the appropriate versions of the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit), [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/latest/installation/overview.html), and [cupy](https://docs.cupy.dev/en/stable/install.html). To see what the maximum version of CUDA your GPU supports, you can check the output of the following command:

```bash
nvidia-smi
```

### Vimeo-90K Septuplet Dataset

To train or test the model with default settings, you will need to download the _"The original training + test set (82GB)"_ version of the Vimeo-90K Septuplet dataset. This data can be found on the official website at [http://toflow.csail.mit.edu/](http://toflow.csail.mit.edu/).

You can also run the following command in your terminal to download the dataset. This will take some time, as the dataset is quite large.

```bash
wget http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip
```

Then, unzip the downloaded file. This will also take a few minutes.

```bash
unzip vimeo_septuplet.zip
```

## Usage

To use ArTEMIS, you can run `main.py` in your terminal with the appropriate command line arguments. For a full list of command line arguments, execute:

```bash
python main.py --help
```

There are four modes in which you can run the model: `train`, `test`, `interpolate_video`, and `interpolate_singleton`. The `train` and `test` modes are used to train/test the model on the Vimeo-90K Septuplet dataset respectively. The `interpolate_video` mode is used to upsample an inputted video to a higher frame rate. Finally, the `interpolate_singleton` mode is used to generate interpolated frames between a single window of four context frames.

For the `train` and `test` modes, the following command line arguments will be critical.

- `--model`: The model to use. Right now, we have only implemented the `ArTEMIS` model.
- `--mode`: The mode in which to run the model. This can be either `train` or `test`.
- `--dataset`: The dataset to use. Right now, we have only implemented the `vimeo90K_septuplet` dataset.
- `--data_dir`: The directory containing the Vimeo-90K Septuplet dataset.
- `--output_dir`: The directory to periodically save some output frames to while training or testing.
- `--use_checkpoint`: Whether to use a checkpoint to initialize the model.
- `--checkpoint_dir`: The directory containing the checkpoint file.
- `--log_dir`: The directory to save logs to while training/testing.
- `--log_iter`: The frequency at which to log training information and save outputs (default = 100 steps).
- `--batch_size`: The batch size to use while training or testing.

For the `interpolate_video` mode, the following command line arguments will be important.

- `--model`: The model to use. Right now, we have only implemented the `ArTEMIS` model.
- `--mode`: The mode in which to run the model. Should be set to `interpolate_video`.
- `--model_path`: The path to the pre-trained model checkpoint. We provide a `model.ckpt` file in the project root directory.
- `--input_path`: The path to the video file to interpolate frames for.
- `--save_path`: The directory to save the interpolated frames to.

For the `interpolate_singleton` mode, the following command line arguments must be used.

- `--model`: The model to use. Right now, we have only implemented the `ArTEMIS` model.
- `--mode`: The mode in which to run the model. Should be set to `interpolate_singleton`.
- `--model_path`: The path to the pre-trained model checkpoint. We provide a `model.ckpt` file in the project root directory.
- `--frame1_path`: The path to the first context frame (before the interpolated frame in time).
- `--frame2_path`: The path to the second context frame (before the interpolated frame in time).
- `--frame3_path`: The path to the third context frame (after the interpolated frame in time).
- `--frame4_path`: The path to the fourth context frame (after the interpolated frame in time).
- `--timesteps`: A list of timesteps in the range (0,1) to interpolate frames for.
- `--save_path`: The directory to save the interpolated frames to.

For example, to train the model, you can run the following command:

```bash
python main.py --model ArTEMIS --mode train --data_dir <data_dir> --output_dir <output_dir> --log_dir <log_dir> --use_checkpoint --checkpoint_dir <checkpoint_dir> --batch_size <batch_size>
```

Alternatively, to generate intermediate frames for a video using the pre-trained model, you can run:

```bash
python main.py --model ArTEMIS --mode interpolate_video --model_path <model_path> --input_path <input_path> --save_path <save_path>
```

Finally, to generate intermediate frames for a single window of context frames, you can run:

```bash
python main.py --model ArTEMIS --mode interpolate_singleton --model_path <model_path> --frame1_path <frame1_path> --frame2_path <frame2_path> --frame3_path <frame3_path> --frame4_path <frame4_path> --timesteps <timesteps> --save_path <save_path>
```
