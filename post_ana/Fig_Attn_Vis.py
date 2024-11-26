from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.compat.v1.enable_eager_execution() 
import os
import re
import numpy as np
from model_util.utils import *
from model_util.models import *
from tqdm import tqdm, trange

import argparse
np.random.seed(7)
tf.random.set_seed(7)
from tensor2tensor.visualization import attention

parser = argparse.ArgumentParser(description='Attention Visulization')
parser.add_argument('--state', default=True, action='store_false')
parser.add_argument('--data_type', type=str, default='RMSD',choices=['RMSD', 'MacroAssignment'])
parser.add_argument('--sample_strategy', default='category', choices = ['category', 'argmax', 'top_p', 'top_k'])
parser.add_argument('--gpu_id', type=str, default='4')
parser.add_argument('--ckpt_task', type=str, default='Label0.0_window50_interval2_lr0.0005_emb_dim128_l100_block3_scheduled')
# parser.add_argument('--interval', type=int, default=1)
parser.add_argument('--task', type=str, default='trans_gpt')
# parser.add_argument('--seq_lenth', type=int, default=100)
parser.add_argument('--reset_thresh', type=int, default=2000)
parser.add_argument('--gen_files', type=int, default=20)
parser.add_argument('--ckpt_choice', type=str, default='epoch150')
parser.add_argument('--preprocess_type', type=str, default='count', choices=['count', 'ordered_count', 'dense'])
parser.add_argument('--seed', type=str, default='valid', choices=['train', 'valid'])

args = parser.parse_args()
sample_strategy = args.sample_strategy
ckpt_task = args.ckpt_task
seq_lenth = int(re.search(r'_l(\d+)', ckpt_task).group(1))
interval = int(re.search(r'_interval(\d+)', ckpt_task).group(1))
num_block = int(re.search(r'_block(\d+)', ckpt_task).group(1))
gen_files = args.gen_files
ckpt_choice = args.ckpt_choice
task = args.task
state = args.state
reset_thresh = args.reset_thresh
# state = not ckpt_task.endswith('stateless')
preprocess_type = args.preprocess_type
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
embedding_dim = 128 
samples = 10
data_type = args.data_type
seed = args.seed
include_pos = False 
pretrained_emb = True if 'transE' in ckpt_task else False


physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

datapath = f'/home/wzengad/projects/MD_code/data/{data_type}/'
checkpoint_dir = f'/home/wzengad/projects/MD_code/LSTM/checkpoint/{data_type}/{task}/{ckpt_task}/'
save_dir = f'/home/wzengad/projects/MD_code/LSTM/results/{data_type}/{task}/{ckpt_task}/{sample_strategy}/{ckpt_choice}_{seed}_interval{interval}'
os.makedirs(save_dir, exist_ok=True)

emb = None

train0 = np.loadtxt(datapath+'train',dtype=int).reshape(-1)
train = train0.reshape(-1, interval).T.flatten()
valid0 = np.loadtxt(datapath+'test',dtype=int).reshape(-1)
valid = valid0.reshape(-1, interval).T.flatten()
vocab_size=len(np.unique(train))
pos_size = 100


model = trans_gpt(vocab_size, pos_size, embedding_dim, gen_files, seq_lenth, num_layers=num_block ,inference=True)
if ckpt_choice == 'best_loss':
    model.load_weights(checkpoint_dir + 'minTestLoss')
elif 'epoch' in ckpt_choice:
    model.load_weights(checkpoint_dir + ckpt_choice)



# find input start from alpha_l and end with alpha_r 
alpha_l_idx = np.where(valid==4)[0]
alpha_r_idx = np.where(valid==68)[0]
beta_idx = np.where(valid==87)[0]
c5_state_idx = np.where(valid==16)[0]

def find_end_seq(end_idx, seq_lenth):
    end = []
    for idx in end_idx:
        if idx < seq_lenth:
            continue
        sequence = valid[idx-seq_lenth:idx+1]
        if (68 not in sequence) and (87 not in sequence) and (16 not in sequence):
            continue
        end.append(sequence)
    return end

l_end = find_end_seq(alpha_l_idx, seq_lenth)
l_end2 = l_end[0][:100].reshape(1,-1)

# gen_pos = tf.map_fn(cal_pos, l_end2)
# gen_trans = tf.map_fn(cal_trans, l_end2)

# x = model.emb(l_end2)
predictions = model(l_end2, None, None, None, None, include_pose=include_pos, training=False)


from keras_nlp.layers.transformer_layer_utils import compute_causal_mask
x_decoder = model.emb(l_end2)
choice = 2 # choose i-th block in decoder stacks
for block in model.decoder[:choice]:
      x_decoder = block(x_decoder) 

target_decoder = model.decoder[choice]
decoder_mask = tf.cast(
    compute_causal_mask(x_decoder),
    dtype=tf.int32,
)

self_attended, attn_score = target_decoder._self_attention_layer(
    x_decoder,
    x_decoder,
    x_decoder,
    attention_mask=decoder_mask,
)

multi_attn = model.decoder[0]._self_attention_layer(return_attention=True)
# idx = find_input_range(alpha_l_idx, alpha_r_idx, seq_lenth)
# gen_input = valid[idx[0]:idx[1]].reshape(1,-1)
# answer = valid[idx[1]+1]
# sampler = tf.random.categorical


alpha_l = (50, 30)
alpha_r = (-70, -20)
beta = (-130,170)
c5 = (-90, 170)
c7eq = (-80,70)
c7ax = (70,-50)

import pandas as pd
state_info = pd.read_csv('/home/wzengad/projects/MD_code/LSTM/rama_state_index.txt',sep='\s+')
def nearest_state(df, target):
    df['distance'] = (df['phi'] - target[0])**2 + (df['psi'] - target[1])**2
    state = df.sort_values(by='distance')['state']
    return state
alpha_l_state = nearest_state(state_info, alpha_l).iloc[0]
alpha_r_state = nearest_state(state_info, alpha_r).iloc[0]
beta_state = nearest_state(state_info, beta).iloc[0]
c5_state= nearest_state(state_info, c5).iloc[0]
c7eq_state = nearest_state(state_info, c7eq).iloc[0]
c7ax_state = nearest_state(state_info, c7ax).iloc[0]

