import keras
import tensorflow as tf


@keras.saving.register_keras_serializable()
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ffn, dropout_rate=0.1, **kwargs):
        '''d_ffn: (first) feed forward dimension'''
        super().__init__(**kwargs)
        self.supports_masking = True
        
        self.seq = tf.keras.Sequential([
          tf.keras.layers.Dense(d_ffn, activation='relu'),
          tf.keras.layers.Dense(d_model),
          tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x) 
        return x

    def compute_mask(self, inputs, mask=None):
        return mask
