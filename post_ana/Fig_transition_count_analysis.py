
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.compat.v1.enable_eager_execution() 
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/wzengad/projects/MD_code/LSTM')
from train.utils import *
from models import *
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

datapath = f'/home/wzengad/projects/MD_code/data/4state.txt'
checkpoint_dir = f'/home/wzengad/projects/MD_code/LSTM/checkpoint/{data_type}/{task}/{ckpt_task}/'

num_bins=4
sm_length=50
threshold=100
X=[2.0, 0.5, -0.5, -2.0] 

input_x, input_y = np.loadtxt(datapath, unpack=True, usecols=(0,1), skiprows=1)
input_x = running_mean(input_x, sm_length) # average on sm_length states
idx_x = map(lambda x: find_nearest(X, x), input_x) # clustering

idx_2d = list(idx_x)
idx_2d = Rm_peaks_steps(idx_2d, threshold) # actually threshold=100 is too large to filter out peaks 
text = np.array(idx_2d)
md = text[:int(len(text)*0.8)]
vocab_size=len(X)

def state_count(data):
    total=pd.DataFrame(data)
    state_count_p=total.value_counts()
    state_count_p=state_count_p.reset_index(name='count')
    state_count_p.columns=['state','count']
    state_count_p['count']=state_count_p['count']/state_count_p['count'].sum()
    return state_count_p

def get_transition_count(md, T):
    M = [[0]*vocab_size for _ in range(vocab_size)]
    dataset = data_as_input(md, 1, T  , BUFFER_SIZE = 100000, shuffle=False, type = 'sparse')
    for item in  dataset:
        sequence = item[0].numpy()[0]
        # drop consecutive same states
        sequence = [sequence[i] for i in range(len(sequence)) if i == 0 or sequence[i] != sequence[i-1]]
        for i in range(len(sequence)-1):
            for j in range(i+1, len(sequence)):
                M[sequence[i]][sequence[j]] += 1
    return M

def get_adjcent_count(md, T):
    M = [[0]*vocab_size for _ in range(vocab_size)]
    dataset = data_as_input(md, 1, T  , BUFFER_SIZE = 100000, shuffle=False, type = 'sparse')
    for item in  dataset:
        sequence = item[0].numpy()[0]
        # drop consecutive same states
        sequence = [sequence[i] for i in range(len(sequence)) if i == 0 or sequence[i] != sequence[i-1]]
        if len(sequence)>1:
            for (i,j) in zip(sequence,sequence[1:]):
                M[i][j] += 1
                # print(f'{i} to {j} in {sequence}')
    return M

for T in range(50,1500,50):
    # T = 500
    g_dis=[]
    l_dis=[]
    for file in range(0,50,1):
        g=[]
        l=[]
        gpt_path = f'/home/wzengad/projects/MD_code/LSTM/results/4state/trans_gpt/{ckpt_task}/category/{ckpt_choice}_120000_valid_interval1/prediction_{file}'
        gpt = np.loadtxt(gpt_path, dtype=int).reshape(-1)
        lstm_path = f'/home/wzengad/projects/MD_code/LSTM/results/4state/lstm/lr0.001_interval1_seq100/category/{ckpt_choice}_120000_valid_interval1/prediction_{file}'
        lstm = np.loadtxt(lstm_path, dtype=int).reshape(-1)

        M_g = get_adjcent_count(gpt, T)
        M_l = get_adjcent_count(lstm, T)
        for i in range(vocab_size):
            for j in range(i+1,vocab_size):
                g.append(M_g[i][j])
                l.append(M_l[i][j])
                g.append(M_g[j][i])
                l.append(M_l[j][i])
        g_dis.append(g)
        l_dis.append(l)
        
    g_dis = np.array(g_dis)
    l_dis = np.array(l_dis)
    np.savetxt(f'/home/wzengad/projects/MD_code/Fig_data/gpt_{T}', g_dis, fmt='%i')
    np.savetxt(f'/home/wzengad/projects/MD_code/Fig_data/lstm_{T}', l_dis, fmt='%i')

transition = ['AB','BA','AC','CA','AD','DA','BC','CB','BD','DB','CD','DC']
plots = [4,5,6,7]
import matplotlib.style as style 
import matplotlib
style.available
style.use('seaborn-paper') #sets the size of the charts
# style.use('ggplot')
# style.use('_mpl-gallery')
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]


for plot_choice in plots:
    M_plot = []
    g_plot=[]
    l_plot=[]

    for T in range(50,1500,50):
        # T = 500
        num_sequence = md.shape[0]//T   
        M = get_adjcent_count(md, T)
        M_dis=[]
        for i in range(vocab_size):
                for j in range(i+1,vocab_size):
                    M_dis.append(M[i][j])
                    M_dis.append(M[j][i])
        g_dis = np.loadtxt(f'/home/wzengad/projects/MD_code/Fig_data/gpt_lstm/gpt_{T}',dtype=int)
        l_dis = np.loadtxt(f'/home/wzengad/projects/MD_code/Fig_data/gpt_lstm/lstm_{T}',dtype=int)

        M_plot.append(M_dis[plot_choice])
        g_plot.append(g_dis[:,plot_choice])
        l_plot.append(l_dis[:,plot_choice])

    g_plot = np.array(g_plot).T
    l_plot = np.array(l_plot).T
    M_plot = [i//10 for i in M_plot]

    draw = pd.DataFrame({'x':np.arange(len(M_plot)),'baseline':M_plot, 'nano-GPT_mean':g_plot.mean(axis=0),'GPT_std': g_plot.std(axis=0),'LSTM_mean':l_plot.mean(axis=0),'LSTM_std':l_plot.std(axis=0)})
    draw.to_csv(f'/home/wzengad/projects/MD_code/Fig_data/4state_transition_{transition[plot_choice]}.csv', index=False)

    
    fig, ax = plt.subplots(figsize=(7,5)) 
    ax.scatter(np.arange(len(M_plot)), M_plot, s=40, marker='s', facecolor='none', edgecolor='r', c='darkorange', label= 'Baseline')
    ax.errorbar(np.arange(len(M_plot)), g_plot.mean(axis=0), yerr=g_plot.std(axis=0), fmt='o', c='mediumblue',  lw=2, markersize=5,markeredgewidth=0.5, capsize=8, label='nano-GPT')
    ax.errorbar(np.arange(len(M_plot)), l_plot.mean(axis=0), yerr=l_plot.std(axis=0), fmt='o',   c='darkgreen', lw=2, markersize=5, markeredgewidth=0.5, capsize=8, label='LSTM')

    ax.tick_params(axis='both', which='both', direction='in', labelsize=15)
    ax.set_xlabel('Commit Time', size=15)
    ax.set_ylabel('Count', size=15)
    # set y limit to 0-35
    ax.set_ylim(-3,35)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    # reset the value in x-axis
    ax.set_xticklabels(['',50,500,1000,1500])
    
    ax.legend(loc='center right', fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    # fig.tight_layout()
    plt.savefig(f'/home/wzengad/projects/MD_code/Fig/adjacent_count/{transition[plot_choice]}.pdf', format='pdf', dpi=600, pad_inches = 0.05)



