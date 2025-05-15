import tensorflow as tf
tf.compat.v1.enable_eager_execution() 

def scheduled_sampler(input_emb, decoder_outputs, step, train_scheduled_sampling='exp', infren_scheduled_sampling='exp',\
    trainstep_sigmoid_k=10000,trainstep_exp_radix=0.99995, exp_epsilon=0.95, sigmoid_k=20):

    input_len = decoder_outputs.shape[1]
    batch_size = decoder_outputs.shape[0]
    hidden_size = decoder_outputs.shape[-1]
    max_len = input_len

    step = tf.cast(step, tf.float64)
    # for training step
    if train_scheduled_sampling == 'exp':
        current_step_threshold = trainstep_exp_radix ** step
    elif train_scheduled_sampling == 'sigmoid':
        current_step_threshold = trainstep_sigmoid_k / (trainstep_sigmoid_k + tf.exp(step/trainstep_sigmoid_k)) 
    elif train_scheduled_sampling == "none":
        current_step_threshold = 1.0


    t = tf.expand_dims(tf.range(start=0, limit=input_len, dtype=tf.float64), 0)
    # for decoding sequence length
    current_step_threshold = tf.cast(current_step_threshold, tf.float64) 
    if infren_scheduled_sampling == "linear":
        threshold_table = 1 - t / max_len * (1 - current_step_threshold) 
    elif infren_scheduled_sampling == "inverse_linear":
        threshold_table = t / max_len * (1 - current_step_threshold)

    elif infren_scheduled_sampling == "exp":
        threshold_table = exp_epsilon ** (t * (1 - current_step_threshold))  
    elif infren_scheduled_sampling == "sigmoid":
        threshold_table = sigmoid_k / (sigmoid_k + tf.exp(t / sigmoid_k * (1 - current_step_threshold)))

    elif infren_scheduled_sampling == "inverse_exp":
        threshold_table = 1 - exp_epsilon ** t * (1 - current_step_threshold) 
    elif infren_scheduled_sampling == "inverse_sigmoid":
        threshold_table = 1 - sigmoid_k / (sigmoid_k + tf.exp(t / sigmoid_k * (1 - current_step_threshold)))     


    final_threshold = tf.tile(threshold_table, [batch_size,1])
    final_threshold = tf.expand_dims(final_threshold, -1)
    final_threshold = tf.tile(final_threshold, [1, 1, hidden_size])

    select_seed = tf.random.uniform([batch_size, input_len],minval=0, maxval=1) 
    select_seed = tf.expand_dims(select_seed, -1)
    select_seed = tf.tile(select_seed, [1, 1, hidden_size])
    select_seed = tf.cast(select_seed, tf.float64)

    # consider decoder outputs as next noken embedding
    # select_emb = tf.concat([input_emb[:,:1,:], decoder_outputs[:, :-1, :]],1)
    # consider decoder output as current token embedding
    select_emb = decoder_outputs

    new_input = tf.where(tf.math.less(select_seed, final_threshold), x= input_emb, y= select_emb)
    return new_input



# x = model.emb(x)
#     # if include_pose:
#     pos = model.pos_emb(pos)

#     x_emb = x + pos
#     x_decoder = x + pos

#     for block in model.decoder:
#       x_decoder = block(x_decoder)  
    
#     if model.inference:
#       return model.d(model.dropout(x_decoder))

#     # scheduled sampling for train and validation
#     x_sampled_decoder = model.sampler(x_emb, x_decoder, 1)
    
#     for block in model.decoder:
#       x_sampled_decoder = block(x_sampled_decoder) 
#     output = model.dropout(x_sampled_decoder)

#     return model.d(output)