import keras
import numpy as np
import tensorflow as tf


# # Clear all previously registered custom objects
keras.saving.get_custom_objects().clear()

def positional_encoding(length, depth):
    half_depth = depth / 2
    
    positions = np.arange(length)[:, np.newaxis]      # (length, 1)
    depths = np.arange(half_depth)[np.newaxis, :] / half_depth  # (1, depth/2)
    
    angle_rates = 1 / (10000**depths)                 # (1, depth/2)
    angle_rads = positions * angle_rates              # (length, depth/2)
    
    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)                                       # (length, depth)
    
    return tf.cast(pos_encoding, dtype=tf.float32)


@keras.saving.register_keras_serializable()
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, **kwargs):
        super().__init__(**kwargs)
        
        assert d_model % 2 == 0, "Embedding size must be even"
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, 
                                                output_dim=d_model,
                                                mask_zero=True
                                                )
        self.pos_encoding = positional_encoding(2048, depth=d_model)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def compute_output_shape(self, *args, **kwargs):
        return self.embedding.compute_output_shape(*args, **kwargs)

    # def get_config(self):
    #     config =super().get_config()
    #     config.update({
    #         'vocab_size':vocab_size,
    #         'd_model':d_model
    #     })
    #     return config
