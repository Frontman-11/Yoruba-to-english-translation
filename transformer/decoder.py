import keras
import tensorflow as tf
from transformer.feedforward import FeedForward
from transformer.positional_embedding import PositionalEmbedding
from transformer.attention import CausalSelfAttention, CrossAttention


@keras.saving.register_keras_serializable()
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, d_ffn, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        
        self.causal_self_attn = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)
        
        self.cross_attn = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)
        
        self.ffn = FeedForward(d_model, d_ffn)
        
    def call(self, x, encoder_output, training=False):
        x = self.causal_self_attn(x=x, training=training)
        x = self.cross_attn(x=(x, encoder_output), training=training)
        
        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attn.last_attn_scores
        
        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x
        

@keras.saving.register_keras_serializable()
class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, N, d_model, num_heads, d_ffn, vocab_size, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.N = N
        
        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                                 d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.decoder_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads,
                         d_ffn=d_ffn, dropout_rate=dropout_rate)
            for _ in range(N)]
        
        self.last_attn_scores = None

    def call(self, x, encoder_output, training=False):
        # `x` is token-IDs shape (batch, target_seq_len)
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)
        
        x = self.dropout(x, training=training)
        
        for i in range(self.N):
            x  = self.decoder_layers[i](x, encoder_output, training=training)
        
        self.last_attn_scores = self.decoder_layers[-1].last_attn_scores
        
        # The shape of x is (batch_size, target_seq_len, d_model).
        return x

    def compute_output_shape(self, *args, **kwargs):
        return self.pos_embedding.compute_output_shape(*args, **kwargs)

