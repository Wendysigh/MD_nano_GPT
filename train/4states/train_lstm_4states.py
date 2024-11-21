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
from train.utils import *
from models import *

parser = argparse.ArgumentParser(description='LSTM Task')
parser.add_argument('--task', type=str, default='4state',choices=['RMSD','phi','psi','4state'])
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--interval', type=int, default=1)
parser.add_argument('--seq_length', type=int, default=100)
parser.add_argument('--state', default=True, action='store_false')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--gpu_id', type=str, default='1')
parser.add_argument('--EPOCHS', type=int, default=200)
parser.add_argument('--save_epoch', type=int, default=10)
#args,_ = parser.parse_known_args()
args = parser.parse_args()
task = args.task
interval = args.interval
seq_length = args.seq_length
state = args.state
BATCH_SIZE = args.batch_size
lr = args.learning_rate
EPOCHS = args.EPOCHS
save_epoch = args.save_epoch
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

checkpoint_dir = f'/home/wzengad/projects/MD_code/LSTM/checkpoint/{task}/lstm/lr{lr}_interval{interval}_seq{seq_length}'
log_dir = f'/home/wzengad/projects/MD_code/LSTM/logs/{task}/lstm/lr{lr}_interval{interval}_seq{seq_length}'
if not state:
    checkpoint_dir += '_stateless'
    log_dir += '_stateless'

# make sure logdir has no other files
shutil.rmtree(log_dir,ignore_errors=True)
summary_writer = tf.summary.create_file_writer(log_dir)

# setting from tiwary code
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

# Sequence length and shift in step between past (input) & future (output)
seq_length = 100
shift=1

# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset.
BUFFER_SIZE = 50000

# Model parameters
embedding_dim = 8
rnn_units = 64

datapath = f'/home/wzengad/projects/MD_code/data/4state.txt'
input_x, input_y = np.loadtxt(datapath, unpack=True, usecols=(0,1), skiprows=1)
input_x = running_mean(input_x, sm_length) # average on sm_length states
idx_x = map(lambda x: find_nearest(X, x), input_x) # clustering

idx_2d = list(idx_x)
idx_2d = Rm_peaks_steps(idx_2d, threshold) # actually threshold=100 is too large to filter out peaks 
text = np.array(idx_2d)
# np.savetxt('/home/wzengad/projects/MD_code/data/4state_discrete.txt', text, fmt='%i')
train_x = text[:int(len(text)*0.8)]
valid_x = text[int(len(text)*0.8):]

dataset = data_as_input(train_x,  BATCH_SIZE, seq_length, BUFFER_SIZE = 100000, shuffle=True)
vdataset = data_as_input(valid_x,  BATCH_SIZE, seq_length, BUFFER_SIZE = 100000, shuffle=False)


model = LSTM (vocab_size,embedding_dim, BATCH_SIZE, rnn_units, state, seq_length,return_sequences=True)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(train_data, labels):
    with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
        predictions = model(train_data, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)
@tf.function
def test_step(test_data, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(test_data, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

prev_test_loss = 1000
for epoch in range(EPOCHS):

    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for dataset_item in dataset:
        train_data, train_labels = dataset_item[0], dataset_item[1]
        train_step(train_data, train_labels)

    for vdataset_item in vdataset:
        test_data, test_labels = vdataset_item[0], vdataset_item[1]
        test_step(test_data,  test_labels)

    with summary_writer.as_default():
        tf.summary.scalar('loss/train', train_loss.result(), step=epoch)
        tf.summary.scalar('accuracy/train', train_accuracy.result(), step=epoch)
        tf.summary.scalar('loss/test', test_loss.result(), step=epoch)
        tf.summary.scalar('accuracy/test', test_accuracy.result(), step=epoch)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print (template.format(epoch+1,
                            train_loss.result(), 
                            train_accuracy.result()*100,
                            test_loss.result(), 
                            test_accuracy.result()*100))
    
    if test_loss.result() <= prev_test_loss:
        print('prev_test_loss', prev_test_loss)
        print('test_total_loss', test_loss.result())
        prev_test_loss = test_loss.result()
        model.save_weights(checkpoint_dir+f'/minTestLoss')  


    if epoch % save_epoch ==0:
        #model.save_weights(checkpoint_dir+f'/epoch{epoch}.h5')
        #model.save(checkpoint_dir+f'/epoch{epoch}.h5', save_format="tf")
        model.save_weights(checkpoint_dir+f'/epoch{epoch}')