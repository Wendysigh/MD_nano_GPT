from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.compat.v1.enable_eager_execution() 
import os
import re
import numpy as np
import pandas as pd
from utils.utils import *
from models.lstm import *
from models.transformer import *
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
        # if (68 not in sequence) and (87 not in sequence) and (16 not in sequence):
        #     continue
        end.append(sequence)
    return end

l_end = find_end_seq(alpha_l_idx, seq_lenth)


# given valid, return sequences (length=100) containing alpha_l and alpha_r
# dataset = data_as_input(valid, 1, seq_lenth, BUFFER_SIZE = 100000, shuffle=False, type = 'sparse')
# for item in  dataset:
#     sequence = item[0].numpy()[0]
#     sequence = [sequence[i] for i in range(len(sequence)) if i == 0 or sequence[i] != sequence[i-1]]
#     if (68 not in sequence) and (87 not in sequence) and (16 not in sequence):
#         continue
#     if 4 in sequence:
#         break


# gen_input = l_end[0][:seq_lenth].reshape(1,-1)
# answer = l_end[0][-1]
# sampler = tf.random.categorical

# idx = find_input_range(alpha_l_idx, alpha_r_idx, seq_lenth)
# gen_input = valid[idx[0]:idx[1]].reshape(1,-1)
# answer = valid[idx[1]+1]
# sampler = tf.random.categorical
noise_ratio = 2
noise_level = noise_ratio * collect_embedding_std(model, train.reshape(80, -1)[:, :100])

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


# result = calculate_flow_each(gen_input, noise_level, kind='attn')

for i in range(len(l_end)):
    data = l_end[i]
    gen_input = data[:seq_lenth].reshape(1,-1)
    answer = data[-1]
    sampler = tf.random.categorical
    result_attn = calculate_flow_each(gen_input, noise_level, kind='attn')
    result_mlp = calculate_flow_each(gen_input, noise_level, kind='mlp')
    result_emb = calculate_flow_each(gen_input, noise_level, kind='emb')
    # save result as .npz
    os.makedirs(f'/home/wzengad/projects/MD_code/Fig_data/causal/noise{noise_ratio}', exist_ok=True)
    np.savez(f'/home/wzengad/projects/MD_code/Fig_data/causal/noise{noise_ratio}/{i}_attn.npz', **result_attn)
    np.savez(f'/home/wzengad/projects/MD_code/Fig_data/causal/noise{noise_ratio}/{i}_mlp.npz', **result_mlp)
    np.savez(f'/home/wzengad/projects/MD_code/Fig_data/causal/noise{noise_ratio}/{i}_emb.npz', **result_emb)
    # result_attn = np.load(f'/home/wzengad/projects/MD_code/Fig_data/causal/{i}_attn.npz', allow_pickle=True)
    # {k: v for k, v in result_attn.items()}
    print(f'finish {i}th sequence')


def load_result(path):
    result = np.load(path, allow_pickle=True)
    result = {k: v for k, v in result.items()}
    return result

class Avg:
    def __init__(self):
        self.d = []

    def add(self, v):
        self.d.append(v[None])

    def add_all(self, vv):
        self.d.append(vv)

    def avg_non_zero(self):
        matrix = np.concatenate(self.d)
        return np.array([np.mean(col[col!=0]) for col in matrix.T])

    def avg(self):
        return np.concatenate(self.d).mean(axis=0)

    def std(self):
        return np.concatenate(self.d).std(axis=0)

    def size(self):
        return sum(datum.shape[0] for datum in self.d)


def read_knowlege(count=150, kind='emb', noise_ratio=2):
    dirname = f"/home/wzengad/projects/MD_code/Fig_data/causal/noise{noise_ratio}/"
    (
        avg_fe,
        avg_ee,
        avg_le,
        avg_fa,
        avg_la,
        avg_hs,
        avg_ls,
        avg_fs,
        avg_fle,
        avg_all
    ) = [Avg() for _ in range(10)]
    for i in range(count):
        data = load_result(dirname + f"{i}_{kind}.npz")
        scores = data["scores"]
        tokens = data["input_tokens"]

        # original prediction
        avg_hs.add(data["high_score"])
        # prediction after subject is corrupted
        avg_ls.add(data["low_score"])
        avg_fs.add(scores.max())
        # last state
        avg_fle.add(scores[-1].max())

        first_pos = np.where(tokens <= 33)[0]
        middle_pos = np.where((tokens > 33) & (tokens <= 66))[0]
        last_pos = np.where(tokens > 66)[0]
        # First/ middle/last pos subjet.
        avg_fe.add_all(scores[first_pos])
        avg_ee.add_all(scores[middle_pos])
        avg_le.add_all(scores[last_pos])
        # First/last pos subjet.
        avg_fa.add(scores[0])
        avg_la.add(scores[-1])
        avg_all.add_all(scores)

    result = np.stack(
        [
            avg_fe.avg_non_zero(),
            avg_ee.avg_non_zero(),
            avg_le.avg_non_zero(),
            avg_fa.avg_non_zero(),
            avg_la.avg_non_zero(),
            avg_all.avg_non_zero(),
        ]
    )
    result_std = np.stack(
        [
            avg_fe.std(),
            avg_ee.std(),
            avg_le.std(),
            avg_fa.std(),
            avg_la.std(),
        ]
    )
    print("Average Total Effect", avg_hs.avg() - avg_ls.avg())
    print(
        "Best average indirect effect on late position",
        avg_le.avg_non_zero().max() - avg_ls.avg(),
    )
    print(
        "Best average indirect effect on last state", avg_la.avg_non_zero().max() - avg_ls.avg()
    )
    print("Average best-fixed score", avg_fs.avg())
    print("Average best-fixed on last state", avg_fle.avg())
    print("Argmax at late positions", np.argmax(avg_le.avg_non_zero()))
    print("Max at late positions", np.max(avg_le.avg_non_zero()))
    print("Argmax at last state", np.argmax(avg_la.avg_non_zero()))
    print("Max at last state", np.max(avg_la.avg_non_zero()))
    return dict(
        low_score=avg_ls.avg(), result=result, result_std=result_std, size=avg_fe.size()
    )

def plot_array(
    differences,
    kind=None,
    savepdf=None,
    title=None,
    low_score=None,
    high_score=None,
):
    if low_score is None:
        low_score = differences.min()
    if high_score is None:
        high_score = differences.max()
    answer = "AIE"
    labels = [
        "Front states",
        "Middle states",
        "Far back states",
        "First state",
        "Last state",
    ]

    fig, ax = plt.subplots(figsize=(3.5, 2), dpi=600)
    h = ax.pcolor(
        differences,
        cmap={'emb': "Purples", "mlp": "Greens", "attn": "Reds"}[kind],
        vmin=low_score,
        vmax=high_score,
    )
    if title:
        ax.set_title(title)
    ax.invert_yaxis()
    ax.set_yticks([0.5 + i for i in range(len(differences))])
    # ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
    # ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
    ax.set_yticklabels(labels)
    # ax.set_xticklabels(ax.get_xticks(), rotation = 50)
    ax.tick_params(axis='both', which='minor', labelsize=7)
    ax.set_xlabel(f"Sequence Positions", fontsize=7)
    cb = plt.colorbar(h)
    # The following should be cb.ax.set_xlabel(answer), but this is broken in matplotlib 3.5.1.
    if answer:
        cb.ax.set_title(str(answer).strip(), y=-0.13, fontsize=7)

    if savepdf:
        os.makedirs(os.path.dirname(savepdf), exist_ok=True)
        plt.savefig(savepdf, bbox_inches="tight")
    plt.show()


high_score = None  # Scale all plots according to the y axis of the first plot

# for plotting array
for kind in ['emb', "mlp", "attn"]:
    d = read_knowlege(len(l_end), kind, noise_ratio=8)
    count = len(l_end)
    what = {
        'emb': "Indirect Effect of Emb",
        "mlp": "Indirect Effect of MLP",
        "attn": "Indirect Effect of Attn",
    }[kind]
    title = f"Avg {what} over {count} prompts"
    result = np.clip(d["result"] - d["low_score"], 0, None)[:5]
    if kind not in ["mlp", "attn"]:
        high_score = result.max()
    # plot_array(
    #     result,
    #     kind=kind,
    #     title=title,
    #     low_score=0.0,
    #     high_score=high_score,
    #     savepdf=f"/home/wzengad/projects/MD_code/Fig/causal/{kind}.pdf",
    # )

# for plotting bar
# for noise_ratio in [1,2,4,6,8]:
    noise_ratio = 4
    plots = []
    for kind in ['emb', "mlp", "attn"]:
        d = read_knowlege(len(l_end), kind, noise_ratio=noise_ratio)
        count = len(l_end)
        title = f"Avg {what} over {count} prompts"
        result = np.clip(d["result"] - d["low_score"], 0, None)[-1]
        plots.append(result)
    # save data for drawing figure    
    draw = pd.DataFrame({'x':np.arange(len(plots[0])),'emb':plots[0], 'mlp':plots[1]})
    draw.to_csv(f'/home/wzengad/projects/MD_code/Fig_data/draw/AIE_bar.csv', index=False)

    fig, ax = plt.subplots(1, figsize=(6, 4), dpi=600)
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    ax.bar(
        [i - 0.3 for i in range(100)],
        plots[0],
        width=0.3,
        color="#7261ab",
        label="Effect with State Emb Restored",
        
    )
    # set font size for bar
    ax.bar(
        [i for i in range(100)],
        plots[1],
        width=0.3,
        color="#f3201b",
        label="Effect with Attn Restored",
    )
    # ax.bar(
    #     [i + 0.3 for i in range(48)],
    #     avg_no_mlp,
    #     width=0.3,
    #     color="#20b020",
    #     label="Effect with Attn Restored",
    # )
    # ax.set_title(
    #     title
    # )  #'Impact of individual hidden state at last subject token with MLP disabled')
    ax.set_ylabel("Log Average Indirect Effect", fontsize=14)
    ax.set_xlabel("Position where state is restored", fontsize=14)
    # set font size for x and y axis
    ax.tick_params(axis='both', which='minor', labelsize=14)
    # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    # ax.set_ylim(None, max(0.025, 0.105))
    plt.yscale("log")
    ax.legend(frameon=False, fontsize=13)
    fig.savefig(f"/home/wzengad/projects/MD_code/Fig/causal/AIE_bar{noise_ratio}.pdf")