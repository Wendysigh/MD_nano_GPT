from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.compat.v1.enable_eager_execution() 
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
import sys
sys.path.append('/home/wzengad/projects/MD_code/LSTM')
from train.utils import *
from models import *
from tqdm import tqdm, trange
import argparse
import pandas as pd
np.random.seed(7)
from sklearn import preprocessing
import matplotlib.style as style 
import matplotlib
style.available
style.use('seaborn-paper') #sets the size of the charts
# style.use('ggplot')
matplotlib.rcParams['font.family'] = "serif"

parser = argparse.ArgumentParser(description='md generation')
parser.add_argument('--state', default=True, action='store_false')
parser.add_argument('--data_type', type=str, default='4state',choices=['RMSD', '4state'])
parser.add_argument('--sample_strategy', default='category', choices = ['category', 'argmax', 'top_p', 'top_k'])
parser.add_argument('--gpu_id', type=str, default='4')
parser.add_argument('--ckpt_task', type=str, default='Label0.0_sparse50_interval1_lr0.0005_emb_dim128_l100_block1_scheduled')
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
save_dir = f'/home/wzengad/projects/MD_code/Fig/kinetic_distance/{ckpt_task}/'
os.makedirs(save_dir, exist_ok=True)

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
    # calculate state transtion, A ->... ->B
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
    # calculate adjacent transtion, A -> B
    M = [[0]*vocab_size for _ in range(vocab_size)]
    dataset = data_as_input(md, 1, T  , BUFFER_SIZE = 100000, shuffle=False, type = 'sparse')
    for item in  dataset:
        sequence = item[0].numpy()[0]
        # drop consecutive same states
        sequence = [sequence[i] for i in range(len(sequence)) if i == 0 or sequence[i] != sequence[i-1]]
        if len(sequence)>1:
            for (i,j) in zip(sequence,sequence[1:]):
                M[i][j] += 1
    return M

def q(embs,l,m ):
    # softmax prob from l to m
    m_emb = embs[m]
    l_emb = embs[l]
    q_ml = tf.reduce_sum(tf.multiply(m_emb, l_emb)).numpy()
    q_ml = (np.exp(q_ml))
    scale = 0
    for i in range(embs.shape[0]):
        i_emb = embs[i]
        q_i = (np.exp(tf.reduce_sum(tf.multiply(i_emb, l_emb)).numpy()))
        scale += q_i
    return q_ml/scale

def mix_q(embs,out_emb, l,m ):
    # softmax prob from l to m
    m_emb = out_emb[m]
    l_emb = embs[l]
    q_ml = tf.reduce_sum(tf.multiply(m_emb, l_emb)).numpy()
    q_ml = (np.exp(q_ml))
    scale = 0
    for i in range(embs.shape[0]):
        i_emb = out_emb[i]
        q_i = (np.exp(tf.reduce_sum(tf.multiply(i_emb, l_emb)).numpy()))
        scale += q_i
    return q_ml/scale

def k_ml(embs,out_emb, i, j, botz_dis, type = 'mix'):
    if type == 'mix':
        q_ij = mix_q(embs, out_emb, i, j)
        q_ji = mix_q(embs, out_emb, j, i)
    elif type == 'raw':
        q_ij = q(embs, i, j)
        q_ji = q(embs, j, i)
    k_ml = q_ij * botz_dis[i] + q_ji * botz_dis[j]
    return k_ml

def NormalizeData(data):
    data = np.array(data)
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def fig_kinetic(ckpt, T, save_path, scale = 'normalize'):
    ckpt_choice = f'epoch{ckpt}'
    model_lstm = LSTM (vocab_size,8, gen_files, rnn_units, state, seq_lenth,return_sequences=True )
    model_lstm.load_weights(f'/home/wzengad/projects/MD_code/LSTM/checkpoint/4state/lstm/lr0.001_interval1_seq100/{ckpt_choice}')
    
    model = trans_gpt(vocab_size, 100, embedding_dim, gen_files, seq_lenth, num_layers=block_num ,inference=True)
    model.load_weights(checkpoint_dir + ckpt_choice)

    test = tf.convert_to_tensor([0,1,2,3])
    embs = model.emb.token_embedding(test)
    input = tf.expand_dims(test, axis=0)
    # generate a tensor of shape(1,100)
    input = tf.tile(input, [1, 25])
    run_gpt = model(input, None, None, None, None, training=False)
    out_emb_gpt  = model.out_emb

    lstm_embs = model_lstm.emb(test)
    run_lstm = model_lstm(input, training=False)
    out_emb_lstm = model_lstm.out_emb

    g_dis=[]
    l_dis=[]
    for file in range(0,50,1):
        gpt_path = f'/home/wzengad/projects/MD_code/LSTM/results/4state/trans_gpt/{ckpt_task}/category/{ckpt_choice}_120000_valid_interval1/prediction_{file}'
        gpt = np.loadtxt(gpt_path, dtype=int).reshape(-1)
        lstm_path = f'/home/wzengad/projects/MD_code/LSTM/results/4state/lstm/lr0.001_interval1_seq100/category/{ckpt_choice}_120000_valid_interval1/prediction_{file}'
        lstm = np.loadtxt(lstm_path, dtype=int).reshape(-1)

        md_count=state_count(md)
        gpt_count=state_count(gpt)
        lstm_count=state_count(lstm)
        if gpt_count.shape[0]<4 or lstm_count.shape[0]<4:
            continue

        name_list = [0,1,2,3]
        num_list = md_count['count']
        botz_dis=[gpt_count[gpt_count.state==name_list[i]]['count'].squeeze() for i in range(4)]
        md_botz_dis=[md_count[md_count.state==name_list[i]]['count'].squeeze() for i in range(4)]
        lstm_botz_dis=[lstm_count[lstm_count.state==name_list[i]]['count'].squeeze() for i in range(4)]

        t_estimate =[]
        lstm_estimate = []
        for i in range(4):
            for j in range(i+1,4):
                # t_ij equals to t_ji due to symmetry
                t_ij = 1/k_ml(embs,out_emb_gpt[0],i,j, botz_dis, type = 'mix')
                l_ij = 1/k_ml(lstm_embs,out_emb_lstm[0], i,j, lstm_botz_dis, type='raw')
                t_estimate.append(t_ij)
                lstm_estimate.append(l_ij)
        if scale == 'normalize':
            norm_t = preprocessing.normalize(np.array(t_estimate).reshape(1,-1), norm = 'l2') 
            norm_l = preprocessing.normalize(np.array(lstm_estimate).reshape(1,-1), norm = 'l2')      
        elif scale == 'scale':
            norm_t = NormalizeData(t_estimate)
            norm_l = NormalizeData(lstm_estimate)
        g_dis.append(norm_t.reshape(-1))
        l_dis.append(norm_l.reshape(-1))

    g_dis = np.array(g_dis)
    l_dis = np.array(l_dis)

    num_sequence = md.shape[0]//T   
    M = get_transition_count(md, T)
    # M = get_adjcent_count(md, T)
    tau = []
    for i in range(4):
        for j in range(i+1,4):
            if M[i][j] != 0:
                tau.append(T/(M[i][j]/num_sequence))

    assert g_dis.shape[1] == len(tau)

    if scale == 'normalize':
        tau_scale = preprocessing.normalize(np.array(tau).reshape(1,-1), norm = 'l2').reshape(-1)
    elif scale == 'scale':
        tau_scale = NormalizeData(tau)
    elif scale == 'prob':
        tau_scale = tau/np.sum(tau)

    fig, ax = plt.subplots(figsize=(6,4)) 
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    labels = ['','AB', 'AC', 'AD', 'BC', 'BD', 'CD']
    ax.scatter(np.arange(len(labels)-1), tau_scale, s=80, c='darkorange',marker='s', facecolor='none', edgecolor='r', label='Baseline')
    ax.errorbar(np.arange(len(labels)-1), g_dis.mean(axis=0), c='mediumblue', yerr=g_dis.std(axis=0), fmt='o',  lw=1, markersize=5, markeredgewidth=0.5, capsize=8, label='nano-GPT')
    ax.errorbar(np.arange(len(labels)-1), l_dis.mean(axis=0), c='darkgreen', yerr=l_dis.std(axis=0), fmt='o',  lw=1, markersize=5, markeredgewidth=0.5, capsize=8, label='LSTM')

    ax.tick_params(axis='both', which='both', direction='in', labelsize=15)
    ax.set_xlabel('States Transition', size=15)
    ax.set_ylabel('Scaled Distanced', size=15)
    ax.set_xticklabels(labels)

    ax.set_ylim(-0.2, 1.2)
    ax.legend(loc='upper right', fontsize=15)

    fig.tight_layout()
    # plt.savefig('/home/wzengad/projects/MD_code/Fig/kinetic_distance/test_T.pdf', format='pdf', dpi=600, pad_inches = 0.05)
    plt.savefig(save_path + f'ckpt{ckpt}_{T}.pdf', format='pdf', dpi=600, pad_inches = 0.05)
    print(f'kinetic_dis_ckpt{ckpt}_length{T} complete')

for ckpt in (10,20,30,40,50,80, 90, 100, 200, 300):
    T =1900
    fig_kinetic(ckpt, T,save_dir, scale = 'normalize')
    # ckpt = 10
    # for T in range(100,2000,100):
        # fig(ckpt, T, save_dir, scale = 'normalize')
        # fig_kinetic(ckpt, T, '/home/wzengad/projects/MD_code/Fig/kinetic_distance/test_T/', scale = 'scale')
        # T =1900