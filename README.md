# README #

### What is this repository for? ###

Nano-GPT for learning molecular dynamics and generating trajectories on alanine depeptide and simulated 4-state. 
Code are tested on Linux and Python3.6 with Pytorch on 2080Ti GPU.


### Env Requirement ###
conda env create -f environment.yml
OR
pip install -r requirements.txt

### Basic usage ###

It contains the following files and subfolders:

4states/ : for experiments on 4-state simulated system.

alanine/ : for experiments on alanine system.

post_ana/: for post analysis like MFPT, ITS and causal tracing.

data/ : dataset

code running:
cd $HOME/MD_code/


# using nano-GPT on alanine
python -u alanine/train_gpt.py --data_type='RMSD' --gpu_id=5 --interval=1 --trans_block=2

# post-analysis
python -u post_ana/Fig_causal_detection.py --data_type='RMSD' 