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

class transformer_decoder(Model):
    def __init__(self, vocab_size, pos_size, embedding_dim, batch_size, num_heads=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                batch_input_shape=[batch_size, None])
        self.pos_emb = tf.keras.layers.Embedding(pos_size, embedding_dim,
                                batch_input_shape=[batch_size, None])
        self.decoder = keras_nlp.layers.TransformerDecoder(intermediate_dim=embedding_dim, num_heads=num_heads)
        self.dropout = tf.keras.layers.Dropout(0.1)    
        self.d = tf.keras.layers.Dense(vocab_size)

    def call(self, x, pos, trans, include_pose=True):
        x = self.emb(x)
        # if include_pose:
        pos = self.pos_emb(pos)
        x += pos
        x = self.decoder(x)
        x = self.dropout(x)
        return self.d(x)

class trans_gpt(Model):
    def __init__(self, vocab_size, pos_size, embedding_dim,
                    batch_size, seq_length, num_heads=8, num_layers=1, inference=False, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.inference = inference

        self.emb = keras_nlp.layers.TokenAndPositionEmbedding(
        vocab_size, seq_length, embedding_dim,
        embeddings_initializer="glorot_uniform",
        mask_zero=False)

        self.pos_emb = tf.keras.layers.Embedding(pos_size, embedding_dim,
                                batch_input_shape=[batch_size, None])

        self.decoder = [keras_nlp.layers.TransformerDecoder(intermediate_dim=embedding_dim, 
                                                            num_heads=num_heads, **kwargs) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(0.1)    
        self.d = tf.keras.layers.Dense(vocab_size)

        self.sampler = scheduled_sampler
        
    def call(self, x, pos, trans, step, pretrained_emb, include_pose=False):
        
        if pretrained_emb is not None:
        # one_hot_x = tf.one_hot(x, depth=self.vocab_size)
        # # look up to the pretrained_emb for one_hot_x and add it to x
        # x = self.emb(x) + tf.matmul(one_hot_x, pretrained_emb)
            x = self.emb(x) + tf.nn.embedding_lookup(pretrained_emb, x)
        else:
            x = self.emb(x)

        if include_pose:
            pos = self.pos_emb(pos)
            x_emb = x + pos
            x_decoder = x + pos
        else:
            x_emb = x
            x_decoder = x

        for block in self.decoder:
            x_decoder = block(x_decoder)  
        
        # for inference: no scheduled sampling
        if self.inference:
            self.out_emb = self.dropout(x_decoder)
            return self.d(self.dropout(x_decoder))

        # scheduled sampling for train and validation
        x_sampled_decoder = self.sampler(x_emb, x_decoder, step)

        for block in self.decoder:
            x_sampled_decoder = block(x_sampled_decoder) 
        output = self.dropout(x_sampled_decoder)
        
        return self.d(output)


class trans_gpt_causal(Model):
    def __init__(self, vocab_size, pos_size, embedding_dim, batch_size, seq_length, num_heads=8, num_blocks=1, inference=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_blocks = num_blocks
        self.inference = inference
        self.emb = keras_nlp.layers.TokenAndPositionEmbedding(
        vocab_size, seq_length, embedding_dim,
        embeddings_initializer="glorot_uniform",
        mask_zero=False)

        self.pos_emb = tf.keras.layers.Embedding(pos_size, embedding_dim,
                                batch_input_shape=[batch_size, None])

        self.decoder = [keras_nlp.layers.TransformerDecoder(intermediate_dim=embedding_dim, num_heads=num_heads) for _ in range(num_blocks)]
        self.dropout = tf.keras.layers.Dropout(0.1)    
        self.d = tf.keras.layers.Dense(vocab_size)

        self.sampler = scheduled_sampler
    
    def call(self, x, pos, trans, step, pretrained_emb, include_pose=False, trace_layers = {'emb':[]}, kind='mlp', noise=0.1):
        if pretrained_emb is not None:
            x = self.emb(x) + tf.nn.embedding_lookup(pretrained_emb, x)
        else:
            x = self.emb(x)

        if include_pose:
            pos = self.pos_emb(pos)
            x_emb = x + pos
            x_decoder = x + pos
        else:
            x_emb = x
            x_decoder = x

        if 'corrupt_emb' in trace_layers.keys():
            # x_decoder = item_assignment(x_decoder, trace_layers['emb'])
            x_decoder = add_noise(x_decoder, trace_layers['corrupt_emb'], noise)

        if kind == 'emb':
                x_decoder = item_assignment(x_decoder, trace_layers[f'restore_emb'])

        for i, block in enumerate(self.decoder):
            # if f'layer{i}' in trace_layers.keys():
            if kind == 'attn' or kind == 'mlp':
                has_encoder_sequence = False
                if not block._built:
                    block._build(x_decoder.shape, has_encoder_sequence)

                is_cross_attention = block._cross_attention_layer is not None
                
                decoder_mask = merge_padding_and_attention_mask(
                    x_decoder, None, None
                )
                causal_mask = tf.cast(
                    compute_causal_mask(x_decoder),
                    dtype=tf.int32,
                )
                if decoder_mask is None:
                    decoder_mask = causal_mask
                else:
                    decoder_mask = tf.minimum(decoder_mask, causal_mask)

                # Decoder input self-attention.
                self_attended = block._self_attention_layer(
                    x_decoder,
                    x_decoder,
                    x_decoder,
                    attention_mask=decoder_mask,
                )
                self_attended = block._self_attention_dropout(self_attended)
                x_decoder = block._add_and_norm(
                    self_attended, x_decoder, block._decoder_attention_layernorm
                )
                if kind == 'attn':
                    x_decoder = item_assignment(x_decoder, trace_layers[f'layer{i}'])

                assert block._cross_attention_layer is None
                # Feedforward.
                feed_forward_output = block._feed_forward(x_decoder)
                x_decoder =  block._add_and_norm(
                    x_decoder,
                    feed_forward_output,
                    block._feedforward_layernorm,
                )
                if kind == 'mlp':
                    x_decoder = item_assignment(x_decoder, trace_layers[f'layer{i}'])

            else:
                x_decoder = block(x_decoder) 

        assert self.inference == True  # no need for scheduled sampling
        return self.d(self.dropout(x_decoder))


class transformer_encoder(Model):
    def __init__(self, vocab_size, embedding_dim, batch_size, num_heads=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                batch_input_shape=[batch_size, None])

        self.net = keras_nlp.layers.TransformerEncoder(intermediate_dim=embedding_dim, num_heads=num_heads)
        self.dropout = tf.keras.layers.Dropout(0.2)    
        self.d = tf.keras.layers.Dense(vocab_size)

    def call(self, x, pos, include_pose=True):
        x = self.emb(x)
        if include_pose:
            pos = self.emb(pos)
            x += pos
        x = self.net(x)
        x = self.dropout(x)
        return self.d(x)
    
def item_assignment(x, idx_to_change):
    '''
    for the item assignment in tf:
    x[:1, idx_to_change, :] = x[0, idx_to_change, :]
    '''
    replacement = tf.stack([x[0, idx, :] for idx in idx_to_change])
    for i in range(x.shape[0]-1):
        update_idx = []
        for j in idx_to_change:
            update_idx.append([i+1, j])
            # print('update_idx', update_idx)
        x = tf.tensor_scatter_nd_update(x, update_idx, replacement)
    return x


def add_noise(x, idx_to_change, noise):
    '''
    for the item assignment in tf:
    x[1:, idx_to_change] += noise * prng.randn(
                prng.randn(x.shape[0] - 1, idx_to_change, x.shape[2])
    '''
    prng = np.random.RandomState(1)
    updates = tf.stack([x[0, idx, :] for idx in idx_to_change])
    updates = updates + noise * prng.randn(len(idx_to_change), x.shape[2])
    for i in range(x.shape[0]-1):
        prng = np.random.RandomState(i)
        updates = tf.stack([x[0, idx, :] for idx in idx_to_change])
        updates = updates + noise * prng.randn(len(idx_to_change), x.shape[2])
        update_idx = []
        for j in idx_to_change:
            update_idx.append([i+1, j])
            # print(f'add noise for states {j}')
        x = tf.tensor_scatter_nd_update(x, update_idx, updates)
    return x