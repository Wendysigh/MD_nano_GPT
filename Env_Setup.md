
### Env Requirement ###
```
conda create -n tf_env python=3.9
conda activate tf_env
conda install conda-forge::cudatoolkit=11.2.2
conda install conda-forge::cudnn=8.1.0.77
pip install tensorflow-gpu==2.10.0
pip install keras-nlp==0.3.1 --no-deps
pip install "numpy<2"
<!-- python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))" -->
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
nano $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
<!-- # add below line  -->
<!-- #!/bin/bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH -->

git clone https://github.com/msmbuilder/msmbuilder2022.git # old msmbuilder only works for python<3.6>
python -m pip install ./msmbuilder2022
```
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