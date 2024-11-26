from __future__ import absolute_import, division, print_function, unicode_literals
from ast import arg
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
from model_util.utils import *
from model_util.models import *

np.random.seed(7)
tf.random.set_seed(7)

parser = argparse.ArgumentParser(description='LSTM Task')
parser.add_argument('--task', type=str, default='lstm')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--data_type', type=str, default='MacroAssignment',choices=['RMSD', 'MacroAssignment'])
parser.add_argument('--interval', type=int, default=1)
parser.add_argument('--seq_length1', type=int, default=10000)
parser.add_argument('--seq_length2', type=int, default=200)
parser.add_argument('--state', default=True, action='store_false')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--rnn_units', type=int, default=128)
parser.add_argument('--gpu_id', type=str, default='2')
parser.add_argument('--EPOCHS', type=int, default=200)
parser.add_argument('--save_epoch', type=int, default=10)
parser.add_argument('--shuffle2', default=False, action='store_true')
parser.add_argument('--preprocess_type', type=str, default='count', choices=['count', 'ordered_count', 'dense'])
parser.add_argument('--include_transition', default=False, action='store_true')
parser.add_argument('--include_pos', default=True, action='store_false')
parser.add_argument('--batch_type', type=str, default='sparse', choices=['window', 'sparse'])

args = parser.parse_args()
task = args.task
interval = args.interval
seq_length1 = args.seq_length1
seq_length2 = args.seq_length2
state = args.state
BATCH_SIZE = args.batch_size
lr = args.learning_rate
EPOCHS = args.EPOCHS
save_epoch = args.save_epoch
preprocess_type = args.preprocess_type
include_transition = args.include_transition
shuffle2 = args.shuffle2
embedding_dim = args.embedding_dim 
rnn_units = args.rnn_units 
include_pos = args.include_pos
# include_pos =False
batch_type = args.batch_type
data_type = args.data_type

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


root_dir = '/home/wzengad/projects/MD_code/'
checkpoint_dir = root_dir + f'LSTM/checkpoint/{data_type}/{task}/{batch_type}_interval{interval}_TPemb_lr{lr}_units{rnn_units}_emb{embedding_dim}_l1_{seq_length1}_l2_{seq_length2}_bs{BATCH_SIZE}'
log_dir = root_dir + f'LSTM/logs/{data_type}/{task}/{batch_type}_interval{interval}_TPemb_lr{lr}_units{rnn_units}_emb{embedding_dim}_l1_{seq_length1}_l2_{seq_length2}_bs{BATCH_SIZE}'
if not state:
    checkpoint_dir += '_stateless'
    log_dir += '_stateless'
if include_transition:
    checkpoint_dir += '_include_trans'
    log_dir += '_include_trans'
if not include_pos:
    checkpoint_dir += '_no_pos'
    log_dir += '_no_pos'
if shuffle2:
    checkpoint_dir += '_shuffle_2nd_Batch'
    log_dir += '_shuffle_2nd_Batch'    

# make sure logdir has no other files
shutil.rmtree(log_dir,ignore_errors=True)
summary_writer = tf.summary.create_file_writer(log_dir)

datapath = root_dir + f'data/{data_type}/'
train0 = np.loadtxt(datapath+'train',dtype=int).reshape(-1)
train = train0.reshape(-1, interval).T.flatten()
valid0 = np.loadtxt(datapath+'test',dtype=int).reshape(-1)
valid = valid0.reshape(-1, interval).T.flatten()

train_batch1 = data_first_batch(train,  seq_length1, BATCH_SIZE, shuffle=True)
valid_batch1 = data_first_batch(valid,  seq_length1, BATCH_SIZE, shuffle=False)

vocab_size = len(np.unique(train))
pos_size = 100

if task == 'lstm_trans':
    model = LSTM_trans(vocab_size, pos_size, embedding_dim, BATCH_SIZE, rnn_units, state)
elif task == 'lstm':
    model = LSTM(vocab_size, pos_size, embedding_dim, BATCH_SIZE, rnn_units, state, seq_length2, emb_choice='TokenAndPosition')
elif task =='transformer':
    model = transformer_decoder(vocab_size, embedding_dim, BATCH_SIZE)
elif task =='bi_lstm':
    model = bi_LSTM(vocab_size, pos_size, embedding_dim, BATCH_SIZE, rnn_units, state)
elif task == 'transformer_encoder':
    model = transformer_encoder(vocab_size, embedding_dim, BATCH_SIZE)
elif task == 'tri_lstm':
    model = tri_LSTM(vocab_size, pos_size, embedding_dim, BATCH_SIZE, rnn_units, state)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

@tf.function
def train_step(train_data, train_pos, train_transition, labels, include_pos):
    with tf.GradientTape() as tape:
    # train_data, train_pos, train_transition, labels=t_data, t_pos, t_trans, t_labels
        predictions = model(train_data, train_pos, train_transition, include_pose = include_pos, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def val_step(val_data, val_pos, val_transition, labels, include_pos):
    # val_data, val_pos, val_transition, labels = v_data, v_pos, v_trans, v_labels
    predictions = model(val_data, val_pos, val_transition, include_pose = include_pos, training=False)
    t_loss = loss_object(labels, predictions)
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

    for t_dataset1 in train_batch1:
        t_dataset1 = data_second_batch(t_dataset1, seq_length2, shuffle=shuffle2, type=batch_type)
        for t_dataset_item in t_dataset1:
            t_data, t_labels, t_trans, t_pos = t_dataset_item[0], t_dataset_item[1], t_dataset_item[2], t_dataset_item[3]
            # if include_pos:
            #     t_pos = tf.map_fn(cal_pos, t_data)
            #     t_trans = tf.map_fn(cal_transition, t_data)
            train_step(t_data, t_pos, t_trans, t_labels, include_pos)
        model.reset_states()


    for vdataset1 in valid_batch1:
        vdataset1 = data_second_batch(vdataset1, seq_length2, shuffle=shuffle2, type=batch_type)
        for vdataset_item in vdataset1:
            v_data, v_labels, v_trans, v_pos = vdataset_item[0], vdataset_item[1], vdataset_item[2], vdataset_item[3]
            # if include_pos:
            #     v_pos = tf.map_fn(cal_pos, v_data)
            #     v_trans = tf.map_fn(cal_transition, v_data)
            val_step(v_data, v_pos, v_trans, v_labels, include_pos)
        model.reset_states()

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