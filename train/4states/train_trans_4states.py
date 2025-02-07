from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.compat.v1.enable_eager_execution() 

import numpy as np
import os
import json
import pandas as pd
import argparse
import shutil
import sys
sys.path.append('/home/wzengad/projects/MD_code/LSTM')
from models.utils import *
from models.lstm import *
np.random.seed(7)
tf.random.set_seed(7)

parser = argparse.ArgumentParser(description='4 state model')
parser.add_argument('--task', type=str, default='trans_gpt')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--data_type', type=str, default='4state',choices=['RMSD', '4state'])
parser.add_argument('--interval', type=int, default=1)
parser.add_argument('--seq_length', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--gpu_id', type=str, default='2')
parser.add_argument('--EPOCHS', type=int, default=301)
parser.add_argument('--save_epoch', type=int, default=10)

parser.add_argument('--preprocess_type', type=str, default='count', choices=['count', 'ordered_count', 'dense'])
parser.add_argument('--include_transition', default=False, action='store_true')
parser.add_argument('--include_pos', default=False, action='store_true')
parser.add_argument('--batch_type', type=str, default='sparse', choices=['window', 'sparse'])
parser.add_argument('--window_shift', type=int, default=50)
parser.add_argument('--ss_infer', default=False, action='store_true')
parser.add_argument('--label_smoothing', type=float, default=0.0)
parser.add_argument('--pretrained_emb', default=False, action='store_true')
parser.add_argument('--trans_block', type=int, default=1)
parser.add_argument('--decay_lr', default=False, action='store_true')
parser.add_argument('--gradient_clip', default=False, action='store_true')

args = parser.parse_args()
task = args.task
interval = args.interval
seq_length = args.seq_length
BATCH_SIZE = args.batch_size
lr = args.learning_rate
EPOCHS = args.EPOCHS
save_epoch = args.save_epoch
preprocess_type = args.preprocess_type
include_transition = args.include_transition
decay_lr = args.decay_lr
embedding_dim = args.embedding_dim 
include_pos = args.include_pos
# include_pos =False
batch_type = args.batch_type
window_shift = args.window_shift
data_type = args.data_type
ss_infer = args.ss_infer
label_smoothing = args.label_smoothing
pretrained_emb = args.pretrained_emb
trans_block = args.trans_block
gradient_clip = args.gradient_clip

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

root_dir = '/home/wzengad/projects/MD_code/'
checkpoint_dir = root_dir + f'LSTM/checkpoint/{data_type}/{task}/Label{label_smoothing}_{batch_type}{window_shift}_interval{interval}_lr{lr}_emb_dim{embedding_dim}_l{seq_length}_block{trans_block}'
log_dir = root_dir + f'LSTM/logs/{data_type}/{task}/Label{label_smoothing}_{batch_type}{window_shift}_interval{interval}_lr{lr}_emb_dim{embedding_dim}_l{seq_length}_block{trans_block}'
if include_pos:
    checkpoint_dir += '_add_pos'
    log_dir += '_add_pos'
if not ss_infer:
    checkpoint_dir += '_scheduled'
    log_dir += '_scheduled'
if pretrained_emb:
    checkpoint_dir += '_transE_emb'
    log_dir += '_transE_emb'
if decay_lr:
    checkpoint_dir += '_decay_lr'
    log_dir += '_decay_lr'
if gradient_clip:
    checkpoint_dir += '_clip'
    log_dir += '_clip'
# make sure logdir has no other files
shutil.rmtree(log_dir,ignore_errors=True)
summary_writer = tf.summary.create_file_writer(log_dir)

# Number of bins and smoothen length
num_bins=4
sm_length=50
threshold=100
# x-values of the metastable states in the 4-state model potential.
X=[2.0, 0.5, -0.5, -2.0] 
# Labels of all possible states in the ranges we considered.
# For 2d systems, this is not the same as the number of representative values.
all_combs = [i for i in range(num_bins)]
vocab=sorted(all_combs)
vocab_size = len(vocab)
char2idx = {u:i for i, u in enumerate(vocab)} # Mapping from characters to indices
idx2char = np.array(vocab)


datapath = root_dir + f'/data/4state.txt'
input_x, input_y = np.loadtxt(datapath, unpack=True, usecols=(0,1), skiprows=1)
input_x = running_mean(input_x, sm_length) # average on sm_length states
idx_x = map(lambda x: find_nearest(X, x), input_x) # clustering

idx_2d = list(idx_x)
idx_2d = Rm_peaks_steps(idx_2d, threshold) # actually threshold=100 is too large to filter out peaks 
text = np.array(idx_2d)
# np.savetxt('/home/wzengad/projects/MD_code/data/4state_discrete.txt', text, fmt='%i')
train_x = text[:int(len(text)*0.8)]
valid_x = text[int(len(text)*0.8):]

dataset = data_as_input(train_x,  BATCH_SIZE, seq_length, BUFFER_SIZE = 100000, shuffle=True, type = batch_type)
vdataset = data_as_input(valid_x,  BATCH_SIZE, seq_length, BUFFER_SIZE = 100000, shuffle=False, type = batch_type)

model = trans_gpt(vocab_size,100, embedding_dim, BATCH_SIZE, seq_length,num_layers=trans_block, inference=ss_infer)
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing = label_smoothing)
if decay_lr:
    learning_rates = tf.keras.optimizers.schedules.CosineDecay(lr, decay_steps = 250000, alpha=0.1)
else:
    learning_rates = lr
optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rates, beta_1=0.9, beta_2=0.98)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
val_loss = tf.keras.metrics.Mean(name='test_loss')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(train_data, train_pos, train_transition, labels, include_pos, step):
    with tf.GradientTape() as tape:
    # train_data, train_pos, train_transition, labels=t_data, t_pos, t_trans, t_labels
        predictions = model(train_data, train_pos, train_transition, step, None, include_pose = include_pos, training=True)
        labels_one_hot = tf.one_hot(labels, vocab_size)
        loss = loss_object(labels_one_hot, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    if gradient_clip:
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def val_step(val_data, val_pos, val_transition, labels, include_pos, step):
    # val_data, val_pos, val_transition, labels = v_data, v_pos, v_trans, v_labels
    predictions = model(val_data, val_pos, val_transition, step, None, include_pose = include_pos, training=False)
    labels_one_hot = tf.one_hot(labels, vocab_size)
    t_loss = loss_object(labels_one_hot, predictions)
    # print(f'validation loss in steps: {t_loss}')

    val_loss(t_loss)
    val_accuracy(labels, predictions)

prev_val_loss = 1000

for epoch in range(EPOCHS):

    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    val_loss.reset_states()
    val_accuracy.reset_states()
    step_t = 0 
    step_v = 0
    
    for t_dataset_item in dataset:
        step_t += 1
        step_t = tf.convert_to_tensor(step_t, dtype=tf.int64)
        t_data, t_labels, t_trans, t_pos = t_dataset_item[0], t_dataset_item[1], t_dataset_item[2], t_dataset_item[3]
        train_step(t_data, t_pos, t_trans, t_labels, include_pos, step_t)


    for vdataset_item in vdataset:
        step_v += 1
        step_v = tf.convert_to_tensor(step_v, dtype=tf.int64)
        v_data, v_labels, v_trans, v_pos = vdataset_item[0], vdataset_item[1], vdataset_item[2], vdataset_item[3]
        val_step(v_data, v_pos, v_trans, v_labels, include_pos, step_v)


    with summary_writer.as_default():
        tf.summary.scalar('loss/train', train_loss.result(), step=epoch)
        tf.summary.scalar('accuracy/train', train_accuracy.result(), step=epoch)
        tf.summary.scalar('loss/val', val_loss.result(), step=epoch)
        tf.summary.scalar('accuracy/val', val_accuracy.result(), step=epoch)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, val Loss: {}, val Accuracy: {}'
    print (template.format(epoch+1,
                            train_loss.result(), 
                            train_accuracy.result()*100,
                            val_loss.result(), 
                            val_accuracy.result()*100))

    if val_loss.result() <= prev_val_loss:
        print('prev_val_loss', prev_val_loss)
        print('val_total_loss', val_loss.result())
        prev_val_loss = val_loss.result()
        model.save_weights(checkpoint_dir+f'/minTestLoss')      

    if epoch % save_epoch ==0:
        #model.save_weights(checkpoint_dir+f'/epoch{epoch}.h5')
        #model.save(checkpoint_dir+f'/epoch{epoch}.h5', save_format="tf")
        model.save_weights(checkpoint_dir+f'/epoch{epoch}')