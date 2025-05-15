from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.compat.v1.enable_eager_execution() 

import numpy as np
import os
import time
import requests
import json
import pandas as pd
import argparse

def split_and_trans(input_seq):
    seq = input_seq[:, :-1]
    # target = input_seq[:, -1]
    target = input_seq[:, 1:]
    trans = tf.map_fn(cal_trans, seq)
    pos = tf.map_fn(cal_pos, seq)
    return seq, target, trans, pos

def split_and_trans_single(input_seq):
    seq = input_seq[:-1]
    # target = input_seq[-1]
    target = input_seq[1:]
    trans = cal_trans(seq)
    pos = cal_pos(seq)
    return seq, target, trans, pos

@tf.autograph.experimental.do_not_convert
def cal_trans(seq):
    seq_shift = tf.concat((seq[1:], [9999]), axis=0)
    mask = tf.not_equal(seq - seq_shift, 0)
    mask = tf.concat(([True], mask[:-1]), axis=0)
    # interstate transition count 
    trans = tf.cast(mask, tf.int64)
    return trans

@tf.autograph.experimental.do_not_convert
def cal_pos(seq):    
    '''
    count consecutives counts and transitions by tensorflow
    '''
    seq_shift = tf.concat((seq[1:], [9999]), axis=0)
    mask = tf.not_equal(seq - seq_shift, 0)
    mask = tf.concat(([True], mask[:-1]), axis=0)
    # intrastate transitoin count
    cnt_cumsum = tf.math.cumsum(tf.cast(mask, tf.int64))
    _, _, cnt = tf.unique_with_counts(cnt_cumsum)
    cnt = tf.cast(cnt, tf.int64)
    pos = tf.repeat(cnt, cnt)
    return pos


# def cal_transition(seq):
#     '''
#     count transitions
#     '''
#     df=pd.DataFrame(seq)
#     count = df[0].ne(df[0].shift(-1)).astype(int)
#     return count


# def cal_pos(seq):
#     '''
#     count consecutives and return counts
#     '''
#     df=pd.DataFrame(seq)
#     count = df.groupby(df[0].ne(df[0].shift()).cumsum())[0].transform('size')
#     return count

def cal_ordered_pos(seq):
    '''
    count consecutives and return ordered counts
    '''
    df=pd.DataFrame(seq)
    count = df.groupby(df[0].ne(df[0].shift()).cumsum())[0].cumcount() + 1
    return count

def cal_consecutive_duplicates(data, return_type = 'count'):
    '''
    data preprocess to calculate consecutive duplicates

    return_raw: 'raw': do not drop duplicates and return raw data trajectory, with consecutive counts
                'ordered_count': do not drop duplicates and return raw data trajectory, with ordered counts
                'dense': drop duplicates with consecutive counts
    '''
    df = pd.DataFrame(data)
    # add count for consecutive duplicates
    df['count'] = df.groupby(df[0].ne(df[0].shift()).cumsum())[0].transform('size')
    df['ordered_count'] = df.groupby(df[0].ne(df[0].shift()).cumsum())[0].cumcount() + 1
    
    if return_type == 'count': #do not drop duplicates
        return df[0], df['count']

    elif return_type == 'ordered_count':
        return df[0], df['ordered_count']

    elif return_type =='dense':
        # delete consecutive duplicates
        cols=df.columns
        de_dup = df[cols].loc[(df[cols].shift() != df[cols]).any(axis=1)]
        return de_dup[0], de_dup['count']



def data_as_input(data, BATCH_SIZE, seq_length, shift=20, BUFFER_SIZE = 500000, shuffle=True, type='sparse'):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    if type =='window':
        sequences = dataset.window(seq_length+1, shift=shift, drop_remainder=True)
        sequences = sequences.flat_map(lambda x: x.batch(seq_length+1))

    elif type == 'sparse':
        sequences = dataset.batch(seq_length+1, drop_remainder=True)

    final_dataset = sequences.map(split_and_trans_single)

    if shuffle:
        final_dataset = final_dataset.batch(BATCH_SIZE, drop_remainder=True).shuffle(BUFFER_SIZE)
    else:
        final_dataset = final_dataset.batch(BATCH_SIZE, drop_remainder=True)
    return final_dataset



# data=train
# seq_length=10000
# ds = data_first_batch(data, seq_length1, BATCH_SIZE=64)
# dataset = next(iter(ds))
# seq_length2=20
# shift=1
# input_seq = next(iter(final_dataset))

def data_first_batch(data, seq_length1, BATCH_SIZE, BUFFER_SIZE = 8000000, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    if shuffle:
        dataset = dataset.batch(seq_length1, drop_remainder=True).shuffle(BUFFER_SIZE)
    else:
        dataset = dataset.batch(seq_length1, drop_remainder=True)

    final_dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return final_dataset



def combine(seq, *args):
    '''
    combine sequences in TensorSliceDataset
    '''
    seq = tf.expand_dims(seq, axis=0)
    return tf.concat([seq, args], axis=0)


def data_second_batch(dataset, seq_length2, shift=1, shuffle=False, BUFFER_SIZE = 8000000, type='sparse'):
    
    '''
    dataset: tf tensors. shape: (batch_size, seq_length1)
    '''
    # Convert unstack dataset to TensorSliceDataset
    # TensorSliceDataset can batch() on seq_length1 dimenstion to create mini batches
    dataset = tf.unstack(dataset, dataset.shape[0])
    dataset = tf.data.Dataset.from_tensor_slices(tuple(dataset))

    # Combine batched to form tensor for model input, shape: (batch_size, seq_length2)
    dataset = dataset.batch(seq_length2+shift, drop_remainder=True)
    dataset = dataset.map(combine)
    # element in dataset: shape (batch_size, seq_length+shift)
    # final_dataset = dataset.map(lambda x: (x[:,:-1], x[:,-1]))
    final_dataset = dataset.map(split_and_trans)

    if shuffle:
            final_dataset = final_dataset.shuffle(BUFFER_SIZE)

    return final_dataset


def mask_data(data,cut_c=5,cut_l=100):
    fre,infre=pd.DataFrame(),pd.DataFrame()
    for sta in data.state.unique():
        data_t=data[data.state==sta]
        data_t=data_t.value_counts().reset_index(name='counts')

        #set to frequent or infrequent by cutoffs
        infre_t=data_t[(data_t.counts<cut_c) & (data_t.length>cut_l)]  
        fre_t=data_t.drop(infre_t.index)    
        fre=fre.append(fre_t,ignore_index=True)

        infre_t.reset_index(drop=True,inplace=True)  #reset index for slicing

        while len(infre_t.length)>0:    
            min_infre=infre_t.length.min()
            # cluster whose lengths difference are in 10 steps
            to_cluster=infre_t[(infre_t.length-min_infre)<11].copy() #without copy() there would be SettingWithCopyWarning
            # drop these in cluster
            infre_t=infre_t.drop(to_cluster.index) #use inplace=True will cause SettingWithCopyWarning

            to_cluster.reset_index(drop=True,inplace=True)
            # use the first one in cluster to substitute rest and add up their count for validation
            to_cluster.loc[0,'counts']=to_cluster.counts.sum()
            infre=infre.append(to_cluster[:1],ignore_index=True)
            for r in to_cluster.raw:
                data.loc[data.raw==r,'raw']=to_cluster.raw[0]
    return data
    
def drop_consecutive_duplicates(a):
    # function for drop_consecutive_duplicates in np.array 
    return a[np.concatenate(([True],a[:-1]!= a[1:]))]

def running_mean(x, N):
    """
    Convolution as running average. Smoothen data.
    """
    return np.convolve(x, np.ones((N,))/N, mode='valid')

def find_nearest(key_arr, target):
    """
    key_arr: array-like, storing keys.
    target: the representative value which we want to be closest to.
    """
    idx=np.abs(key_arr-target).argmin()
    return idx

def Rm_peaks_steps(traj, threshold):
    """
    Remove sudden changes in the trajectory such as peaks and small steps.
    Here the gradient is used to identify the changes. If two nonzero
    gradients are too close (< threshold), we treat it as noise.
    """
    traj=np.array(traj)
    grad_traj=np.gradient(traj) # gradient of trajectory
    idx_grad=np.where(grad_traj!=0)[0] # the index of nonzero gradient.
    max_idx = len(traj)-1
    idx_grad = idx_grad[idx_grad != max_idx]

    idx0=idx_grad[0]
    for idx in idx_grad:
        window=idx-idx0
        if window <= 1: # neighbor
            continue
        elif 1 < window <= threshold:
            traj[idx0:idx0+window//2+1]=traj[idx0]
            traj[idx0+window//2+1:idx+1]=traj[idx+1]
            idx0=idx
        elif window > threshold:
            idx0=idx
    return traj


# This function is copied from tf-models-official.
# https://github.com/tensorflow/models/blob/79da7eba8b3df52ced2ea9aa340d8c8843de951a/official/nlp/modeling/ops/sampling_module.py#L67
def sample_top_p(logits, top_p):
  """Chooses most probable logits with cumulative probabilities upto top_p.
  Sets the remaining logits to negative infinity.
  Args:
    logits: Input logits for next token.
    top_p: Float tensor with a value >=0 and < 1.0
  Returns:
    Logits with top_p filtering applied.
  """
  sorted_indices = tf.argsort(logits, direction="DESCENDING")
  # Flatten logits as tf.gather on TPU needs axis to be compile time constant.
  logits_shape = tf.shape(logits)
  range_for_gather = tf.expand_dims(tf.range(0, logits_shape[0]), axis=1)
  range_for_gather = tf.tile(range_for_gather * logits_shape[1],
                             [1, logits_shape[1]]) + sorted_indices
  flattened_logits = tf.reshape(logits, [-1])
  flattened_sorted_indices = tf.reshape(range_for_gather, [-1])
  sorted_logits = tf.reshape(
      tf.gather(flattened_logits, flattened_sorted_indices),
      [logits_shape[0], logits_shape[1]])
  cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)

  # Remove tokens with cumulative probability above the threshold.
  sorted_indices_to_remove = cumulative_probs > top_p

  # Shift the indices to the right to keep the first token above threshold.
  sorted_indices_to_remove = tf.roll(sorted_indices_to_remove, 1, axis=-1)
  sorted_indices_to_remove = tf.concat([
      tf.zeros_like(sorted_indices_to_remove[:, :1]),
      sorted_indices_to_remove[:, 1:]
  ], -1)

  # Scatter sorted indices to original indexes.
  indices_to_remove = scatter_values_on_batch_indices(sorted_indices_to_remove,
                                                      sorted_indices)
  top_p_logits = set_tensor_by_indices_to_value(logits, indices_to_remove,
                                                np.NINF)
  return top_p_logits

def scatter_values_on_batch_indices(values, batch_indices):
  """Scatter `values` into a tensor using `batch_indices`.
  Args:
    values: tensor of shape [batch_size, vocab_size] containing the values to
      scatter
    batch_indices: tensor of shape [batch_size, vocab_size] containing the
      indices to insert (should be a permutation in range(0, n))
  Returns:
    Tensor of shape [batch_size, vocab_size] with values inserted at
    batch_indices
  """
  tensor_shape = tf.shape(batch_indices)
  broad_casted_batch_dims = tf.reshape(
      tf.broadcast_to(
          tf.expand_dims(tf.range(tensor_shape[0]), axis=-1), tensor_shape),
      [1, -1])
  pair_indices = tf.transpose(
      tf.concat([broad_casted_batch_dims,
                 tf.reshape(batch_indices, [1, -1])], 0))
  return tf.scatter_nd(pair_indices, tf.reshape(values, [-1]), tensor_shape)

def set_tensor_by_indices_to_value(input_tensor, indices, value):
  """Where indices is True, set the value in input_tensor to value.
  Args:
    input_tensor: float (batch_size, dim)
    indices: bool (batch_size, dim)
    value: float scalar
  Returns:
    output_tensor: same shape as input_tensor.
  """
  value_tensor = tf.zeros_like(input_tensor) + value
  output_tensor = tf.where(indices, value_tensor, input_tensor)
  return output_tensor

def sample_top_k(logits, top_k):
  """Chooses top_k logits and sets the others to negative infinity.
  Args:
    logits: Input logits for next token.
    top_k: Tensor to specify the top_k values.
  Returns:
    Logits with top_k filtering applied.
  """
  top_k = tf.clip_by_value(
      top_k, clip_value_min=1, clip_value_max=tf.shape(logits)[-1])
  top_k_logits = tf.math.top_k(logits, k=top_k)
  indices_to_remove = logits < tf.expand_dims(top_k_logits[0][..., -1], -1)
  top_k_logits = set_tensor_by_indices_to_value(logits, indices_to_remove,
                                                np.NINF)
  return top_k_logits

def top_p_sampler(predictions, sample_num):
    top_p = 0.95
    prediction_top_p = sample_top_p(predictions, top_p)
    output = tf.random.categorical(prediction_top_p,1)
    return output

def top_k_sampler(predictions, sample_num):
    top_k = 20
    prediction_top_k = sample_top_k(predictions, top_k)
    output = tf.random.categorical(prediction_top_k,1)
    return output

def count_ml(text):
    num_01,num_10=0,0
    num_02,num_20=0,0
    num_03,num_30=0,0
    num_12,num_21=0,0
    num_31,num_13=0,0
    num_23,num_32=0,0

    for i in range(len(text)-1):
        if text[i] == 0 and text[i+1] == 1:
            num_01 += 1
        elif text[i] == 1 and text[i+1] == 0:
            num_10 += 1
        elif text[i] == 0 and text[i+1] == 2:
            num_02 += 1
        elif text[i] == 2 and text[i+1] == 0:
            num_20 += 1
        elif text[i] == 0 and text[i+1] == 3:
            num_03 += 1
        elif text[i] == 3 and text[i+1] == 0:
            num_30 += 1
        elif text[i] == 1 and text[i+1] == 2:
            num_12 += 1
        elif text[i] == 2 and text[i+1] == 1:
            num_21 += 1
        elif text[i] == 3 and text[i+1] == 1:
            num_31 += 1
        elif text[i] == 1 and text[i+1] == 3:
            num_13 += 1
        elif text[i] == 2 and text[i+1] == 3:
            num_23 += 1
        elif text[i] == 3 and text[i+1] == 2:
            num_32 += 1
    return num_01, num_10, num_02,num_20, num_03, num_30, num_12, num_21, num_13, num_31, num_23, num_32

def calculate_flow(gen_input,noise_level, kind='mlp', site='early'):
    # the input size is fixed by model.emb
    # set samples to test different random noise
    inp = np.array(gen_input.tolist()* (samples + 1))
    
    base_prob = get_traced(inp, {}, None, None, return_type = 'argmax')  # no noise 
    worst_prob = get_traced(inp, {'emb': range(0,inp.shape[1])}, 
                            noise_level, None, return_type = 'argmax') # noise to embedding and no recover
    # noise and recover selected states in mlp or attention
    tokens = np.unique(gen_input)
    probs=[]
    if kind: 
        for i in tokens: 
            i_prob_layers = []
            for j in range(num_block):
                idx = np.where(gen_input == i)[1]
                if site == 'early':
                    idx = idx[idx <= (seq_lenth//2)]
                elif site == 'late':
                    idx = idx[idx > (seq_lenth//2)]
                elif site == 'all':
                    idx = idx
                trace_layers = {'emb': range(0,gen_input.shape[1]), f'layer{j}':idx }
                r = get_traced(inp, trace_layers, noise_level, kind=kind, return_type = 'argmax')
                i_prob_layers.append(r[answer])
            probs.append(i_prob_layers)
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