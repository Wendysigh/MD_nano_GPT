import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
import keras_nlp
tf.compat.v1.enable_eager_execution() 
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from keras_nlp.layers.transformer_layer_utils import (  # isort:skip
    compute_causal_mask,
    merge_padding_and_attention_mask,
)
from models.scheduled_sampling import scheduled_sampler

class LSTM(Model):
  def __init__(self, vocab_size, embedding_dim, batch_size, rnn_units,
                                state, seq_length, emb_choice = 'Token', return_sequences=True):
    super().__init__()
    self.vocab_size = vocab_size
    if emb_choice == 'TokenAndPosition':
      self.emb = keras_nlp.layers.TokenAndPositionEmbedding(
    vocab_size, seq_length, embedding_dim,
    embeddings_initializer="glorot_uniform",
    mask_zero=False)
    elif emb_choice == 'Token':
      self.emb = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                batch_input_shape=[batch_size, None])
                                
    # self.pos_emb = tf.keras.layers.Embedding(pos_size, embedding_dim,
    #                           batch_input_shape=[batch_size, None])
    self.lstm = CuDNNLSTM(rnn_units, 
        return_sequences=return_sequences,
        recurrent_initializer='orthogonal',
        kernel_regularizer='l2',
        recurrent_regularizer='l2',
        stateful=state)
    self.dropout = tf.keras.layers.Dropout(0.2)    
    self.d = tf.keras.layers.Dense(vocab_size)

    
  def call(self, x, pos=None,trans=None, include_pose=False):
    x = self.emb(x)
    # pos = self.emb(pos)
    if include_pose:
      pos = self.pos_emb(pos)
      x += pos

    x = self.lstm(x)
    x = self.dropout(x)
    self.out_emb = x
    return self.d(x)

class LSTM_trans(Model):
  def __init__(self, vocab_size, pos_size, embedding_dim, batch_size, rnn_units,
                                state, return_sequences=True):
    super().__init__()
    self.vocab_size = vocab_size
    self.emb = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None])
    self.pos_emb = tf.keras.layers.Embedding(pos_size, embedding_dim,
                              batch_input_shape=[batch_size, None])
    self.trans_emb = tf.keras.layers.Embedding(2, embedding_dim,
                              batch_input_shape=[batch_size, None])

    self.lstm = CuDNNLSTM(rnn_units, 
        return_sequences=return_sequences,
        recurrent_initializer='orthogonal',
        kernel_regularizer='l2',
        recurrent_regularizer='l2',
        stateful=state)
    self.dropout = tf.keras.layers.Dropout(0.2)    
    self.d = tf.keras.layers.Dense(vocab_size)

    
  def call(self, x, pos,trans, include_pose=False):
    x = self.emb(x)
    # pos = self.emb(pos)
    if include_pose:
      pos = self.pos_emb(pos)
      trans = self.trans_emb(trans)
      x = x + pos + trans

    x = self.lstm(x)
    x = self.dropout(x)
    return self.d(x)

class LSTM_share_emb(Model):
  def __init__(self, vocab_size, pos_size, embedding_dim, batch_size, rnn_units,
                                state, return_sequences=True):
    super(LSTM_share_emb, self).__init__()
    self.vocab_size = vocab_size
    self.emb = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None])
    self.lstm = CuDNNLSTM(rnn_units, 
        return_sequences=return_sequences,
        recurrent_initializer='orthogonal',
        kernel_regularizer='l2',
        recurrent_regularizer='l2',
        stateful=state)
    self.dropout = tf.keras.layers.Dropout(0.2)    
    self.d = tf.keras.layers.Dense(vocab_size)


  def call(self, x, pos, include_pose=True):
    x = self.emb(x)
    if include_pose:
      pos = self.emb(pos)
      x += pos
    x = self.lstm(x)
    x = self.dropout(x)
    return self.d(x)


class bi_LSTM(Model):
  def __init__(self, vocab_size, pos_size, embedding_dim, batch_size, rnn_units,
                                state, return_sequences=True):
    super().__init__()
    self.vocab_size = vocab_size
    self.emb = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None])
    self.pos_emb = tf.keras.layers.Embedding(pos_size, embedding_dim,
                              batch_input_shape=[batch_size, None])
    self.bi_lstm = tf.keras.layers.Bidirectional(
      CuDNNLSTM(rnn_units, 
        return_sequences=return_sequences,
        recurrent_initializer='orthogonal',
        kernel_regularizer='l2',
        recurrent_regularizer='l2',
        stateful=state))

    self.dropout = tf.keras.layers.Dropout(0.2)    
    self.d = tf.keras.layers.Dense(vocab_size)

  def call(self, x, pos, include_pose=True):
    x = self.emb(x)
    if include_pose:
      pos = self.pos_emb(pos)
      x += pos
    x = self.bi_lstm(x)
    x = self.dropout(x)
    return self.d(x)

class tri_LSTM(Model):
  def __init__(self, vocab_size, pos_size, embedding_dim, batch_size, rnn_units,
                                state, seq_length, return_sequences=True):
    super().__init__()
    self.vocab_size = vocab_size
    self.emb = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None])
    self.pos_emb = tf.keras.layers.Embedding(pos_size, embedding_dim,
                              batch_input_shape=[batch_size, None])
    self.bi_lstm = tf.keras.layers.Bidirectional(
      CuDNNLSTM(rnn_units, 
        return_sequences=True,
        recurrent_initializer='orthogonal',
        kernel_regularizer='l2',
        recurrent_regularizer='l2',
        stateful=state))
    self.lstm = CuDNNLSTM(rnn_units, 
        return_sequences=False,
        recurrent_initializer='orthogonal',
        kernel_regularizer='l2',
        recurrent_regularizer='l2',
        stateful=state)
    self.dropout1 = tf.keras.layers.Dropout(0.2)    
    self.dropout2 = tf.keras.layers.Dropout(0.2)  
    self.d = tf.keras.layers.Dense(vocab_size)

  def call(self, x, pos=None, include_pose=True):
    x = self.emb(x)
    if include_pose:
      pos = self.pos_emb(pos)
      x += pos
    x = self.bi_lstm(x)
    x = self.dropout1(x)
    x = self.lstm(x)
    x = self.dropout2(x)
    return self.d(x)




class attn_LSTM(Model):
  def __init__(self, vocab_size, pos_size, embedding_dim, batch_size, rnn_units,
                                state, return_sequences=True):
    super().__init__()
    self.vocab_size = vocab_size
    self.emb = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None])
    self.pos_emb = tf.keras.layers.Embedding(pos_size, embedding_dim,
                              batch_input_shape=[batch_size, None])

    self.lstm = CuDNNLSTM(rnn_units, 
        return_sequences=return_sequences,
        recurrent_initializer='orthogonal',
        kernel_regularizer='l2',
        recurrent_regularizer='l2',
        stateful=state)
    self.dropout = tf.keras.layers.Dropout(0.2)    
    self.d = tf.keras.layers.Dense(vocab_size)

    
  def call(self, x, pos, include_pose=False):
    x = self.emb(x)
    # pos = self.emb(pos)
    if include_pose:
      pos = self.pos_emb(pos)
      x += pos

    x = self.lstm(x)
    x = self.dropout(x)
    return self.d(x)
  



