
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.compat.v1.enable_eager_execution() 
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from model_util.utils import *
from model_util.models import *
from tqdm import tqdm, trange
import argparse
import pandas as pd
np.random.seed(7)

parser = argparse.ArgumentParser(description='md generation')
parser.add_argument('--state', default=True, action='store_false')
parser.add_argument('--data_type', type=str, default='4state',choices=['RMSD', '4state'])
parser.add_argument('--sample_strategy', default='category', choices = ['category', 'argmax', 'top_p', 'top_k'])
parser.add_argument('--gpu_id', type=str, default='4')
parser.add_argument('--ckpt_task', type=str, default='Label0.0_sparse50_interval1_lr0.0001_emb_dim64_l100_block1_scheduled')
parser.add_argument('--task', type=str, default='trans_gpt')
parser.add_argument('--reset_thresh', type=int, default=2000)
parser.add_argument('--gen_files', type=int, default=1)
parser.add_argument('--ckpt_choice', type=str, default='epoch10')
parser.add_argument('--preprocess_type', type=str, default='count', choices=['count', 'ordered_count', 'dense'])
parser.add_argument('--seed', type=str, default='valid', choices=['train', 'valid'])

args = parser.parse_args()
sample_strategy = args.sample_strategy
ckpt_task = args.ckpt_task
seq_lenth = int(re.search(r'_l(\d+)', ckpt_task).group(1))
block_num = int(re.search(r'_block(\d+)', ckpt_task).group(1))
interval = int(re.search(r'_interval(\d+)', ckpt_task).group(1))
gen_files = args.gen_files
ckpt_choice = args.ckpt_choice
task = args.task
state = args.state
reset_thresh = args.reset_thresh
# state = not ckpt_task.endswith('stateless')
preprocess_type = args.preprocess_type
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
embedding_dim = int(re.search(r'_dim(\d+)', ckpt_task).group(1))
rnn_units = 64
data_type = args.data_type
seed = args.seed
include_pos = False 
pretrained_emb = True if 'transE' in ckpt_task else False

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

datapath = f'/home/zengwenqi/projects/MD_code2/data/4state_discrete.txt'
checkpoint_dir = f'/home/wzengad/projects/MD_code/LSTM/checkpoint/{data_type}/{task}/{ckpt_task}/'

# num_bins=4
# sm_length=50
# threshold=100
X=[2.0, 0.5, -0.5, -2.0] 

# input_x, input_y = np.loadtxt(datapath, unpack=True, usecols=(0,1), skiprows=1)
# input_x = running_mean(input_x, sm_length) # average on sm_length states
# idx_x = map(lambda x: find_nearest(X, x), input_x) # clustering

# idx_2d = list(idx_x)
# idx_2d = Rm_peaks_steps(idx_2d, threshold) # actually threshold=100 is too large to filter out peaks 
# text = np.array(idx_2d)

text = np.loadtxt(datapath, dtype=int)
md = text[:int(len(text)*0.8)]
vocab_size=len(X)

def calculate_commit_times(states, from_state, to_state):
    transition_times = []
    commit_time = 0
    
    for i in range(1, len(states)):
        if states[i-1] == from_state and states[i] == to_state:
            transition_times.append(commit_time)
            commit_time = 0  
        elif states[i-1] == from_state:
            commit_time += 1
        else:
            commit_time = 0
    return transition_times


def get_times(ckpt_task, ckpt_choice, from_state=0, to_state=3):
    gpt_all_times = []
    lstm_all_times = []

    for file in range(0, 50, 1):
        # Load predictions
        gpt_path = f'./results/4state/trans_gpt/{ckpt_task}/category/{ckpt_choice}_120000_valid_interval1/prediction_{file}'
        gpt = np.loadtxt(gpt_path, dtype=int).reshape(-1)
        lstm_path = f'./results/4state/lstm/lr0.001_interval1_seq100/category/{ckpt_choice}_120000_valid_interval1/prediction_{file}'
        lstm = np.loadtxt(lstm_path, dtype=int).reshape(-1)
        
        gpt_times = calculate_commit_times(gpt, from_state, to_state)
        lstm_times = calculate_commit_times(lstm, from_state, to_state)
        
        gpt_all_times.extend(gpt_times)
        lstm_all_times.extend(lstm_times)
    return gpt_all_times, lstm_all_times


def count_commit_times(input_times, commit_times):
    counts = np.zeros(len(commit_times))

    for time in input_times:
        if time <= 1500:  # Only count times within our range
            bin_index = (time - 1) // 50  # Integer division to find the correct bin
            if bin_index < len(counts):  # Make sure we're within bounds
                counts[bin_index] += 1
    return counts

from_state = 0
to_state = 3
gpt_all_times, lstm_all_times = get_times(ckpt_task, 'epoch10', from_state=from_state, to_state=to_state)
gpt_all_times2, lstm_all_times2 = get_times(ckpt_task, 'epoch20', from_state=from_state, to_state=to_state)
gpt_all_times3, lstm_all_times3 = get_times(ckpt_task, 'epoch30', from_state=from_state, to_state=to_state)
md_times = calculate_commit_times(md, from_state=from_state, to_state=to_state)

commit_times = np.arange(1, 1501, 50)
gpt_counts1 = count_commit_times(gpt_all_times, commit_times)
gpt_counts2 = count_commit_times(gpt_all_times2, commit_times)
gpt_counts3 = count_commit_times(gpt_all_times3, commit_times)

# LSTM counts for each epoch
lstm_counts1 = count_commit_times(lstm_all_times, commit_times)
lstm_counts2 = count_commit_times(lstm_all_times2, commit_times)
lstm_counts3 = count_commit_times(lstm_all_times3, commit_times)


gpt_counts_all = np.vstack((gpt_counts1, gpt_counts2, gpt_counts3))
gpt_mean = np.mean(gpt_counts_all, axis=0)
gpt_std = np.std(gpt_counts_all, axis=0)

# Calculate mean and std for LSTM
lstm_counts_all = np.vstack(( lstm_counts2, lstm_counts3))
lstm_mean = np.mean(lstm_counts_all, axis=0)
lstm_std = np.std(lstm_counts_all, axis=0)


md_counts = count_commit_times(md_times, commit_times)

df = pd.DataFrame({
    'commit_time': commit_times,
    'gpt_mean': gpt_mean,
    'gpt_std': gpt_std,
    'lstm_mean': lstm_mean,
    'lstm_std': lstm_std,
    'md_counts': md_counts
})


# Save to CSV
df.to_csv(f'./fig_data/transition_commit_time_from{from_state}_to{to_state}.csv', index=False)

# Plot results
plt.figure(figsize=(10, 6))
plt.errorbar(commit_times, gpt_mean, yerr=gpt_std, label='GPT', marker='o', markersize=3, capsize=3)
plt.errorbar(commit_times, lstm_mean, yerr=lstm_std, label='LSTM', marker='s', markersize=3, capsize=3)
plt.plot(commit_times, md_counts, label='MD', marker='s', markersize=3)
plt.xlabel('Commit Time')
plt.ylabel('Number of Transitions')
plt.title(f'Transition Count vs Commit Time (State {from_state} → {to_state})')
plt.xlim(0, 1500)
plt.legend()
plt.grid(True)
plt.savefig(f'./results/4state/commit_time_from{from_state}_to{to_state}.png')
plt.close()
