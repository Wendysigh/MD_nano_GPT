# Nano-GPT for Molecular Dynamics

A deep learning framework using Nano-GPT architecture for learning molecular dynamics and generating trajectories on alanine dipeptide and simulated 4-state systems. The model can effectively capture molecular state transitions and generate realistic molecular dynamics trajectories.

## Features

- Implementation of Nano-GPT for molecular dynamics trajectory generation
- Support for multiple systems:
  - Alanine dipeptide system
  - 4-state simulated system
  - FiP35 protein system
- Post-analysis tools for:
  - Mean First Passage Time (MFPT)
  - Implied Timescales (ITS)
  - Causal tracing
  - Free energy landscape analysis

## Requirements

The code has been tested on Linux with Python 3.6+ and PyTorch on NVIDIA 2080Ti GPU. To set up the environment:

```bash
# Create and activate conda environment
conda create -n tf_env python=3.6
conda activate tf_env

# Install CUDA dependencies
conda install conda-forge::cudatoolkit=11.2.2
conda install conda-forge::cudnn=8.1.0.77

# Install required packages
pip install tensorflow-gpu==2.10.0
pip install keras-nlp==0.3.1 --no-deps
pip install "numpy<2"

# Install MSMBuilder (required for analysis)
git clone https://github.com/msmbuilder/msmbuilder2022.git
python -m pip install ./msmbuilder2022
```

## Usage

1. Training:

python scripts/train.py --config configs/default.yaml

2. Generation:

python scripts/generate.py --checkpoint path/to/checkpoint