
trace_layers = {'emb':[0,1],'layer0':[1,2,3,4,5]}
kind='mlp'
x = inp

predictions = model(gen_input, None, None, None, None,
                        include_pose=False, training=False, 
                        trace_layers = {},
                        kind = None, noise = None)

x = model.emb(x)


x_emb = x
x_decoder = x


if 'emb' in trace_layers.keys():
    print('true')
    x_decoder = add_noise(x_decoder, trace_layers['emb'], noise=0.1)

for i, block in enumerate(model.decoder):
    if f'layer{i}' in trace_layers.keys():
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
            self_attended, x_decoder, model._decoder_attention_layernorm
        )
        if kind == 'attn':
            x_decoder = item_assignment(x_decoder, trace_layers[f'layer{i}'])

        assert block._cross_attention_layer is None
        # Feedforward.
        feed_forward_output = block._feed_forward(x_decoder)
        x_block_out =  block._add_and_norm(
            x_decoder,
            feed_forward_output,
            block._feedforward_layernorm,
        )
        if kind == 'mlp':
            x_decoder = item_assignment(x_decoder, trace_layers[f'layer{i}'])

    else:
        x_decoder = block(x_decoder) 

assert model.inference == True  # no need for scheduled sampling
model.d(model.dropout(x_decoder))

def item_assignment(x, idx_to_change):
    '''
    for the item assignment in tf:
    x[:1, idx_to_change, :] = x[0, idx_to_change, :]
    '''
    replacement = tf.stack([x[0, idx, :] for idx in idx_to_change])
    for i in range(3):
        update_idx = []
        for j in idx_to_change:
            update_idx.append([i+1, j])
        print(update_idx)
        x = tf.tensor_scatter_nd_update(x, update_idx, replacement)
    return x

idx_to_change = [1, 3, 5]

def add_noise(x, idx_to_change, noise=0.1):
    '''
    for the item assignment in tf:
    x[1:, idx_to_change] += noise * prng.randn(
                prng.randn(x.shape[0] - 1, idx_to_change, x.shape[2])
    '''
    prng = np.random.RandomState(1)
    updates = tf.stack([x[0, idx, :] for idx in idx_to_change])
    updates = updates + noise * prng.randn(len(idx_to_change), x.shape[2])
    for i in range(3):
        update_idx = []
        for j in idx_to_change:
            update_idx.append([i+1, j])
        print(update_idx)
        x = tf.tensor_scatter_nd_update(x, update_idx, updates)
    return x




