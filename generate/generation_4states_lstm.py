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
import argparse
np.random.seed(7)

parser = argparse.ArgumentParser(description='md generation')
parser.add_argument('--data_type', type=str, default='4state',choices=['RMSD', '4state'])
parser.add_argument('--sample_strategy', default='category', choices = ['category', 'argmax', 'top_p', 'top_k'])
parser.add_argument('--gpu_id', type=str, default='4')
parser.add_argument('--ckpt_task', type=str, default='lr0.001_interval1_seq100')
parser.add_argument('--task', type=str, default='lstm')
parser.add_argument('--reset_thresh', type=int, default=2000)
parser.add_argument('--gen_files', type=int, default=50)
parser.add_argument('--gen_length', type=int, default=120000)
parser.add_argument('--ckpt_choice', type=str, default='epoch10')
parser.add_argument('--preprocess_type', type=str, default='count', choices=['count', 'ordered_count', 'dense'])
parser.add_argument('--seed', type=str, default='valid', choices=['train', 'valid'])

args = parser.parse_args()
sample_strategy = args.sample_strategy
ckpt_task = args.ckpt_task
seq_lenth = int(re.search(r'_seq(\d+)', ckpt_task).group(1))
interval = int(re.search(r'_interval(\d+)', ckpt_task).group(1))
state = not ckpt_task.endswith('stateless')
gen_files = args.gen_files
gen_length = args.gen_length
ckpt_choice = args.ckpt_choice
task = args.task
reset_thresh = args.reset_thresh
# state = not ckpt_task.endswith('stateless')
preprocess_type = args.preprocess_type
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
embedding_dim = 8
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
if not state:
    save_dir = f'/home/wzengad/projects/MD_code/LSTM/results/{data_type}/{task}/{ckpt_task}/{sample_strategy}/{ckpt_choice}_stateless_{gen_length}_{seed}_interval{interval}'
else:
    save_dir = f'/home/wzengad/projects/MD_code/LSTM/results/{data_type}/{task}/{ckpt_task}/{sample_strategy}/{ckpt_choice}_{gen_length}_{seed}_interval{interval}'
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
valid_x = text[int(len(text)*0.8):]
gen_input= valid_x[:seq_lenth].reshape(1,-1)
gen_input = np.repeat(gen_input, gen_files, axis=0)
text_generated = gen_input
gen_pos = tf.map_fn(cal_pos, gen_input)
gen_trans = tf.map_fn(cal_trans, gen_input)
vocab_size=len(X)
if task == 'lstm':
    model = LSTM (vocab_size,embedding_dim, gen_files, rnn_units, state, seq_lenth,return_sequences=False )

if ckpt_choice == 'best_loss':
    model.load_weights(checkpoint_dir + 'minTestLoss')
elif 'epoch' in ckpt_choice:
    model.load_weights(checkpoint_dir + ckpt_choice)


for i in trange(gen_length):
    # predictions = model(gen_input, gen_pos, gen_trans, None, None, include_pose=include_pos, training=False)
    predictions = model(gen_input,  include_pose=include_pos, training=False)
    if 'trans' in task:
        predictions = predictions[:,-1,:]
    predicted_id = tf.random.categorical(predictions, num_samples=1)
    #predicted_id = tf.cast(predicted_id, tf.int32)

    gen_input = tf.concat([gen_input[:,1:], predicted_id], axis=1)
    text_generated = tf.concat([text_generated, predicted_id], axis=1)

   
# Save prediction:
for i in range(gen_files):
    save_path = os.path.join(save_dir, 'prediction_'+str(i))
    np.savetxt(save_path, text_generated[i,:], fmt='%i')    
    
