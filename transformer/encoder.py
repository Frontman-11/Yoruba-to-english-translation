import keras
import tensorflow as tf
from feedforward import FeedForward
from attention import GlobalSelfAttention
from positional_embedding import PositionalEmbedding


# # Clear all previously registered custom objects
keras.saving.get_custom_objects().clear()

@keras.saving.register_keras_serializable()
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, d_ffn, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking=True
        
        self.glob_self_attn = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)
        
        self.ffn = FeedForward(d_model, d_ffn)
        
    def call(self, x, training=False):
        x = self.glob_self_attn(x, training=training)
        x = self.ffn(x)
        return x


@keras.saving.register_keras_serializable()
class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, N, d_model, num_heads, d_ffn, vocab_size, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.N = N
        
        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)
        
        self.encoder_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         d_ffn=d_ffn,
                         dropout_rate=dropout_rate)
            for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        '''x is token-IDs shape: (batch, seq_len)'''
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.
        
        # Add dropout.
        x = self.dropout(x, training=training)
        
        for i in range(self.N):
            x = self.encoder_layers[i](x, training=training)
        
        return x  # Shape `(batch_size, seq_len, d_model)`.

    def compute_output_shape(self, *args, **kwargs):
        return self.pos_embedding.compute_output_shape(*args, **kwargs)
