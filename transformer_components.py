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


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        
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
        
        

class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.supports_masking = True
        
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()
        

class GlobalSelfAttention(BaseAttention): 
    def call(self, x, mask=None, training=False):
        attn_output = self.mha(
            query=x,
            key=x,
            value=x,
            attention_mask=mask[:, tf.newaxis],
            training=training
        )
        
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class CausalSelfAttention(BaseAttention):
    def call(self, x, mask=None, training=False):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            attention_mask=mask[:, tf.newaxis],
            use_causal_mask = True,
            training=training
        )
        
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


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


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ffn, dropout_rate=0.1):
        '''d_ffn: (first) feed forward dimension'''
        super().__init__()
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


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, d_ffn, dropout_rate=0.1):
        super().__init__()
        self.supports_masking=True
        
        self.glob_self_attn = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model//num_heads,
            dropout=dropout_rate)
        
        self.ffn = FeedForward(d_model, d_ffn)
        
    def call(self, x, training=False):
        x = self.glob_self_attn(x, training=training)
        x = self.ffn(x)
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, N, d_model, num_heads,
                   d_ffn, vocab_size, dropout_rate=0.1):
        super().__init__()
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

    
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, d_ffn, dropout_rate=0.1):
        super().__init__()
        self.supports_masking = True
        
        self.causal_self_attn = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model//num_heads,
            dropout=dropout_rate)
        
        self.cross_attn = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model//num_heads,
            dropout=dropout_rate)
        
        self.ffn = FeedForward(d_model, d_ffn)
        
    def call(self, x, encoder_output, training=False):
        x = self.causal_self_attn(x=x, training=training)
        x = self.cross_attn(x=(x, encoder_output), training=training)
        
        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attn.last_attn_scores
        
        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x
        

class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, N, d_model, num_heads, d_ffn, vocab_size, dropout_rate=0.1):
        super().__init__()
        
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


# # Clear all previously registered custom objects
keras.saving.get_custom_objects().clear()

@keras.saving.register_keras_serializable()
class Transformer(tf.keras.Model):
    def __init__(self, *,
                 N,
                 d_model,
                 num_heads, 
                 d_ffn,
                 input_vocab_size,
                 target_vocab_size,
                 dropout_rate=0.1,
                 **kwargs
                ):
        super().__init__(**kwargs)

        self.N = N
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ffn = d_ffn
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.dropout_rate = dropout_rate
        
        self.encoder = Encoder(N=N,
                               d_model=d_model,
                               num_heads=num_heads,
                               d_ffn=d_ffn,
                               vocab_size=input_vocab_size,
                               dropout_rate=dropout_rate)
        
        self.decoder = Decoder(N=N,
                               d_model=d_model,
                               num_heads=num_heads,
                               d_ffn=d_ffn,
                               vocab_size=target_vocab_size,
                               dropout_rate=dropout_rate)
        
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        
    def call(self, inputs, training=False):
        encoder_input, x  = inputs
        
        encoder_output = self.encoder(encoder_input, training=training)  # (batch_size, seq_len, d_model)
        
        x = self.decoder(x, encoder_output, training=training)  # (batch_size, target_len, d_model)
        
        # Final linear layer output.
        logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)
        
        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            del logits._keras_mask
        except AttributeError as e:
            print(e)
        
        # Return the final output and the attention weights.
        return logits

    def get_config(self):
        config = super().get_config()
        config.update({
            'N': self.N,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_ffn': self.d_ffn,#           
            'input_vocab_size': self.input_vocab_size,
            'target_vocab_size': self.target_vocab_size,
            'dropout_rate': self.dropout_rate
        })
        return config
