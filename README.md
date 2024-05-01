# ArTEMIS

Arbitrary timestep Transformer for Enhanced Multi-frame Interpolation and Synthesis

## Getting Started

ArTEMIS is a deep learning model that interleaves [VFIT](https://github.com/zhshi0816/Video-Frame-Interpolation-Transformer) and [EDSC](https://github.com/Xianhang/EDSC-pytorch) together in order to enable the synthesis of intermediate video frames at arbitrary timesteps while using multiple frames on either side of the target timestep as context. The model is trained on the [Vimeo-90K Septuplet dataset](http://toflow.csail.mit.edu/).

### Prerequisites

Ensure that you have Python installed on your system. To use ArTEMIS, you need to set up a Python environment with the necessary packages installed. You can do this by running the following commands in your terminal.

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

Finally, install the required packages, which are listed in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## Usage

## Training

To train the model yourself, you will need to download the _"The original training + test set (82GB)"_ version of the Vimeo-90K Septuplet dataset. This data can be found on the official website at [http://toflow.csail.mit.edu/](http://toflow.csail.mit.edu/). The resulting data folder should be placed in the `data/sources/` directory.
