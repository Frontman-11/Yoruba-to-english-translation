import keras
import tensorflow as tf

        
@keras.saving.register_keras_serializable()
class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.supports_masking = True
        
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()
        

@keras.saving.register_keras_serializable()
class GlobalSelfAttention(BaseAttention): 
    def call(self, x, mask=None, training=False):
        attn_output = self.mha(
            query=x,
            key=x,
            value=x,
            attention_mask=mask[:, tf.newaxis] if mask is not None else None,
            training=training
        )
        
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


@keras.saving.register_keras_serializable()
class CausalSelfAttention(BaseAttention):
    def call(self, x, mask=None, training=False):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            attention_mask=mask[:, tf.newaxis] if mask is not None else None,
            use_causal_mask = True,
            training=training
        )
        
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


@keras.saving.register_keras_serializable()
class CrossAttention(BaseAttention):
    def call(self, x, mask=None, training=False):
        x, encoder_output = x
        x_mask, encoder_output_mask = mask if mask is not None else (None, None)
        attn_output, attn_scores = self.mha(
            query=x,
            key=encoder_output,
            value=encoder_output,
            attention_mask=encoder_output_mask[:, tf.newaxis] if encoder_output_mask is not None else None,
            return_attention_scores=True,
            training=training
        )

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores
        
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        
        return x
