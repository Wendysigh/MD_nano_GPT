import tensorflow as tf



# import tensorflow as tf
# from tensorflow.python.platform import build_info as tf_build_info

# print("CUDA version used by TensorFlow: ", tf_build_info.cuda_version)
# print("cuDNN version used by TensorFlow: ", tf_build_info.cudnn_version)
# print("Physical GPUs detected by TensorFlow: ", tf.config.list_physical_devices('GPU'))

# tf.debugging.set_log_device_placement(True)
# import os
# os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=0"

# with tf.device('/GPU:0'):
#     a = tf.constant([1.0, 2.0, 3.0])
#     b = tf.reduce_sum(a)
#     print("GPU computation result:", b.numpy())
# Dummy model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Dummy data
data = tf.random.normal((32, 10))
labels = tf.random.normal((32, 1))

# Optimizer and loss
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.MeanSquaredError()

# Training step
@tf.function
def train_step(data, labels):
    with tf.GradientTape() as tape:
        predictions = model(data, training=True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)

    for grad, var in zip(gradients, model.trainable_variables):
        print(f"Variable: {var.name}, Gradient: {grad}, Shape: {grad.shape}")
        if tf.reduce_any(tf.math.is_nan(grad)):
            print(f"NaN detected in gradients for variable {var.name}")
        if tf.reduce_any(tf.math.is_inf(grad)):
            print(f"Infinite value detected in gradients for variable {var.name}")

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Run a training step
train_step(data, labels)