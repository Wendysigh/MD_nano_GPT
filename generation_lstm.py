from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.compat.v1.enable_eager_execution() 
import os
import numpy as np
import re
from tqdm import tqdm, trange
import argparse

from utils.utils import *
from models.lstm import *
from models.transformer import *

np.random.seed(7)
tf.random.set_seed(7)

parser = argparse.ArgumentParser(description='LSTM generation')
parser.add_argument('--state', default=True, action='store_false')
parser.add_argument('--data_type', type=str, default='Fip35_micro100',
                    choices=['RMSD', 'MacroAssignment', 'phi', 'psi', 
                            'Fip35_micro100', 'Fip35_macro5'])
parser.add_argument('--sample_strategy', default='category', choices = ['category', 'argmax', 'top_p', 'top_k'])
parser.add_argument('--gpu_id', type=str, default='4')
parser.add_argument('--ckpt_task', type=str, default='Label0.0_sparse50_interval1_lr0.001_emb_dim128_l100_units512_emb128_no_pos')
parser.add_argument('--interval', type=int, default=10)
parser.add_argument('--task', type=str, default='lstm')
parser.add_argument('--seq_lenth', type=int, default=100)
parser.add_argument('--reset_thresh', type=int, default=2000)
parser.add_argument('--gen_files', type=int, default=20)
parser.add_argument('--gen_length', type=int, default=100000)
parser.add_argument('--ckpt_choice', type=str, default='epoch20')
parser.add_argument('--preprocess_type', type=str, default='count', choices=['count', 'ordered_count', 'dense'])
parser.add_argument('--seed', type=str, default='valid', choices=['train', 'valid'])

args = parser.parse_args()


sample_strategy = args.sample_strategy
ckpt_task = args.ckpt_task
seq_lenth = int(re.search(r'_l(\d+)', ckpt_task).group(1))
interval = int(re.search(r'_interval(\d+)', ckpt_task).group(1))
gen_files = args.gen_files
gen_length = args.gen_length
ckpt_choice = args.ckpt_choice
task = args.task
state = args.state
reset_thresh = args.reset_thresh
# state = not ckpt_task.endswith('stateless')
preprocess_type = args.preprocess_type
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
embedding_dim = 128 
rnn_units = 512
data_type = args.data_type
seed = args.seed
include_pos = False if 'no_pos' in ckpt_task else True
pretrained_emb = True if 'transE' in ckpt_task else False
block_num = int(re.search(r'_block(\d+)', ckpt_task).group(1)) if '_block' in ckpt_task else 2

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

datapath = f'data/{data_type}/'
checkpoint_dir = f'checkpoint/{data_type}/{task}/{ckpt_task}/'
if not state:
    save_dir = f'results/{data_type}/{task}/{ckpt_task}/{sample_strategy}/{ckpt_choice}_stateless_{gen_length}_{seed}_interval{interval}_gen{gen_files}'
else:
    save_dir = f'results/{data_type}/{task}/{ckpt_task}/{sample_strategy}/{ckpt_choice}_{gen_length}_{seed}_interval{interval}_gen{gen_files}'
os.makedirs(save_dir, exist_ok=True)

train0 = np.loadtxt(datapath+'train',dtype=int).reshape(-1)
train = train0.reshape(-1, interval).T.flatten()
valid0 = np.loadtxt(datapath+'test',dtype=int).reshape(-1)
valid = valid0.reshape(-1, interval).T.flatten()

vocab_size=len(np.unique(train))
pos_size = 100

if pretrained_emb:
    emb = np.load('/home/wzengad/projects/OpenKE/checkpoint/entity2vec.npy')
    emb = tf.convert_to_tensor(emb, dtype=tf.float32)
else:
    emb = None

# gen_files as batchsize
if task == 'lstm':
    model = LSTM(vocab_size, embedding_dim, gen_files, rnn_units, state, seq_lenth)
elif task == 'trans_gpt':
    model = trans_gpt(vocab_size, pos_size, embedding_dim, gen_files, seq_lenth, 
                     num_layers=block_num, inference=True)
    
if ckpt_choice == 'best_loss':
    model.load_weights(checkpoint_dir + 'minTestLoss')
elif 'epoch' in ckpt_choice:
    model.load_weights(checkpoint_dir + ckpt_choice)

if include_pos:
    save_name = f'prediction_'
else:
    save_name = f'no_gen_pos_prediction_'

# vdata_ite=iter(vdataset)
# initialize gen_input, text_generated as seqs from test dataset
if seed == 'valid':
    gen_input = valid.reshape(20,-1)[:gen_files, :seq_lenth]
elif seed == 'train':
    gen_input = train.reshape(80,-1)[:gen_files, :seq_lenth]
# np.random.shuffle(gen_input)

gen_pos = tf.map_fn(cal_pos, gen_input)
gen_trans = tf.map_fn(cal_trans, gen_input)
# gen_input, gen_pos, gen_transition = vdata_ite.get_next()[0], vdata_ite.get_next()[2], vdata_ite.get_next()[3]  
text_generated = gen_input

if sample_strategy == 'category':
    sampler = tf.random.categorical
elif sample_strategy == 'argmax':
    sampler = tf.argmax
elif sample_strategy == 'top_p':
    sampler = top_p_sampler    
elif sample_strategy == 'top_k':
    sampler = top_k_sampler  

for i in trange(gen_length):
    # predictions = model(gen_input, gen_pos, gen_transition, include_pose=include_pos, include_trans=False, training=False)
    
    predictions = model(gen_input, gen_pos, gen_trans, include_pose=include_pos, training=False)
    predicted_id = sampler(predictions[:,-1,:], 1)
    # reshape predicted_id. The first dimension is consistent with num of gen_files
    predicted_id = tf.reshape(predicted_id, [gen_files,1])
    gen_input = tf.concat([gen_input[:,1:], predicted_id], axis=1)
    gen_pos = tf.map_fn(cal_pos, gen_input)
    gen_trans = tf.map_fn(cal_trans, gen_input)

    text_generated = tf.concat([text_generated, predicted_id], axis=1)

    # if i % reset_thresh == 0:
    #     model.reset_states()

    # if i ==1:
    #     # test if saving works
    #     save_path = os.path.join(save_dir, save_name+str(1))
    #     np.savetxt(save_path, text_generated[1,:], fmt='%i')

# Save prediction:
for i in range(gen_files):
    save_path = os.path.join(save_dir, save_name+str(i))
    file = text_generated[i,:]
    np.savetxt(save_path, file, fmt='%i')    
    
