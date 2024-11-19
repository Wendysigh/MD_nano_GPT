from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.compat.v1.enable_eager_execution() 
import os
import re
import numpy as np
import sys
sys.path.append('/home/wzengad/projects/MD_code/LSTM')
from utils import *
from models import *
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
import argparse
np.random.seed(7)
tf.random.set_seed(7)

def collect_embedding_std(mt, subjects):
    embs = model.emb(subjects)  
    noise_level = tf.math.reduce_std(embs).numpy()
    return noise_level

def get_traced(inp, trace_layers, noise, kind, return_type = 'argmax'):
    predictions = model(inp, None, None, None, emb,
                        include_pose=include_pos, training=False, 
                        trace_layers = trace_layers,
                        kind = kind, noise = noise)[1:,-1,:]
    
    probs = tf.reduce_sum(tf.nn.softmax(predictions, axis=1), axis=0).numpy()

    # if return_type == 'argmax':
    #     preds_prob = tf.reduce_max(probs, axis=1).numpy()
    #     preds_idx = tf.math.argmax(probs, axis=1).numpy()
    # elif return_type == 'category':
    #     tf.random.set_seed(0)
    #     preds_idx = sampler(predictions[:,-1,:], 1).numpy().reshape(-1)
    #     probs = probs.numpy()
    #     preds_prob = [probs[i,preds_idx[i]] for i in range(len(preds_idx)) ]
    return probs
def find_input_range(alpha_l , alpha_r, diff):
    for i in alpha_l:
        num = alpha_r - i
        min_diff = np.abs(num).min()
        r_idx = num[np.argmin(np.abs(num))]  + i
        if min_diff < diff:
            idx = (np.min((i, r_idx)), np.max((i, r_idx)))
            diff = min_diff
    return idx


parser = argparse.ArgumentParser(description='LSTM generation')
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

if pretrained_emb:
    emb = np.load('/home/wzengad/projects/OpenKE/checkpoint/entity2vec.npy')
    emb = tf.convert_to_tensor(emb, dtype=tf.float32)
else:
    emb = None


train0 = np.loadtxt(datapath+'train',dtype=int).reshape(-1)
train = train0.reshape(-1, interval).T.flatten()
valid0 = np.loadtxt(datapath+'test',dtype=int).reshape(-1)
valid = valid0.reshape(-1, interval).T.flatten()
vocab_size=len(np.unique(train))
pos_size = 100


model = trans_gpt_causal(vocab_size, pos_size, embedding_dim, gen_files, seq_lenth, inference=True, num_blocks=num_block)
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


# idx = find_input_range(alpha_l_idx, alpha_r_idx, seq_lenth)
# gen_input = valid[idx[0]:idx[1]].reshape(1,-1)
# answer = valid[idx[1]+1]
# sampler = tf.random.categorical

def calculate_flow_each(gen_input,noise_level, kind='mlp'):
    # the input size is fixed by model.emb
    # set samples to test different random noise
    assert kind in ['emb', 'mlp', 'attn']
    inp = np.array(gen_input.tolist()* (samples + 1))
    
    base_prob = get_traced(inp, {}, None, None, return_type = 'argmax')  # no noise 
    worst_prob = get_traced(inp, {'corrupt_emb': range(0,inp.shape[1])}, 
                            noise_level, None, return_type = 'argmax') # noise to embedding and no recover
    # noise and recover selected states in emb, mlp or attention
    tokens = np.unique(gen_input)
    probs=np.zeros((len(tokens), seq_lenth))
    if kind: 
        for i in range(len(tokens)): 
            state = tokens[i]
            idx = np.where(gen_input == state)[1]
            for i_idx in idx:
                trace_layers = { f'layer{j}' : [i_idx] for j in range(num_block)}
                trace_layers['corrupt_emb']= range(0, gen_input.shape[1])
                trace_layers['restore_emb'] = [i_idx]

                r = get_traced(inp, trace_layers, noise_level, kind=kind, return_type = 'argmax')
                probs[i][i_idx] = r[answer]

    final =  dict(
            scores=np.array(probs), # probabilty
            low_score=worst_prob[answer], # probablity when corrupt all tokens
            high_score=base_prob[answer], # probablity without any intervention
            input_ids=tokens, # input ids
            input_tokens=tokens,
            subject_range=None,
            answer=answer,
            window=10,
            kind=kind,
        )
    return final


def plot_trace_heatmap(result, n, savepdf=None, title=None, xlabel=None, modelname=None):
    differences = result["scores"]
    low_score = result["low_score"]
    answer = result["answer"]
    kind = (
        None
        if (not result["kind"] or result["kind"] == "None")
        else str(result["kind"])
    )
    window = result.get("window", 10)
    # labels = result["input_tokens"]
    labels = result["label"]

    # with plt.rc_context(rc={"font.family": "Times New Roman"}):
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    fig, ax = plt.subplots(figsize=(3.5, 2), dpi=600)
    # save data for drawing figure
    # draw = pd.DataFrame(differences)
    # draw.to_csv(f'/home/wzengad/projects/MD_code/Fig_data/causal_single_{kind}.csv', index=False)
    h = ax.pcolor(
        differences,
        cmap={None: "Purples", "emb": "Purples", "mlp": "Greens", "attn": "Reds"}[
            kind
        ],
        vmin=low_score,
    )
    ax.invert_yaxis()
    ax.set_yticks([0.5 + i for i in range(len(differences))])
    # ax.set_xticks([0.5 + i for i in range(0, differences.shape[1])])
    # ax.set_xticklabels(list(range(0, differences.shape[1])))
    ax.set_yticklabels(labels, fontsize=5)
    ax.xaxis.set_tick_params(labelsize=5)

    if not modelname:
        modelname = "nano-GPT"
    if not kind:
        ax.set_title(f"Impact of restoring state after corrupted input", fontsize=7)
        ax.set_xlabel(f"Restored position within {modelname}")
    else:
        kindname = "Attn" if kind == "attn" else "Emb"
        ax.set_title(f"Impact of restoring {kindname} after corrupted input", fontsize=7)
        ax.set_xlabel(f"Sequence Positions", fontsize=7)
    cb = plt.colorbar(h)
    cb.ax.tick_params(labelsize=5)
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    elif answer is not None:
        # The following should be cb.ax.set_xlabel, but this is broken in matplotlib 3.5.1.
        # cb.ax.set_title(f"p({str(answer).strip()})", y=-0.13, fontsize=8)
        cb.ax.set_title(r'p($\alpha_L$)', y=-0.13, fontsize=7)
    if savepdf:
        os.makedirs(os.path.dirname(savepdf), exist_ok=True)
        plt.savefig(savepdf, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


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



noise_ratio = 2
# n,i=2,5; 4,1; 6,6; 8,4;8,6;8,10
# for n in [1,2,4,6,8]:
#     for i in range(len(l_end)):
def gen_lable(tokens):
    # tokens = result_emb ['input_tokens']
    label = tokens.tolist()
    for i in range(len(tokens)):

        if tokens[i] == alpha_l_state:
            label[i] = r'$\alpha_L$'
            # print(tokens[i])
        elif tokens[i] == alpha_r_state:
            label[i] = (  r'$\alpha_R$')
        elif tokens[i] == beta_state:
            label[i] = (  r'$\beta$')
        elif tokens[i] == c5_state:
            label[i] = (  r'$C5$')
        elif tokens[i] == c7eq_state:
            label[i] = (  r'$C7eq$')
        elif tokens[i] == c7ax_state:
            label[i] = ( r'$C7ax$')
    return label


for (n,i) in [(1,6), (6,6), (8,6)]:
        (n,i) =(8,6)
        noise_level = n * collect_embedding_std(model, train.reshape(80, -1)[:, :100])
        data = l_end[i]
        gen_input = data[:seq_lenth].reshape(1,-1)
        answer = data[-1]
        sampler = tf.random.categorical
        
        result_attn = calculate_flow_each(gen_input, noise_level, kind='attn')
        result_emb = calculate_flow_each(gen_input, noise_level, kind='emb')

        result_attn['label']  = gen_lable(result_attn['input_tokens'])
        result_emb['label']  = gen_lable(result_emb['input_tokens'])
        # os.makedirs(f"/home/wzengad/projects/MD_code/Fig/causal/noise_{n}", exist_ok=True)
        plot_trace_heatmap(result_attn, n, savepdf=f"/home/wzengad/projects/MD_code/Fig/causal/single_attn_{n}_{i}.pdf")
        plot_trace_heatmap(result_emb, n, savepdf=f"/home/wzengad/projects/MD_code/Fig/causal/single_emb_{n}_{i}.pdf")
        print(f"noise: {n}, sample: {i} done")