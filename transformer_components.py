import tensorflow as tf


# ## Creating Positional Encoding

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.pos_encodings = None
        self.supports_masking = True
        
    def build(self, input_shape):
        _, seq_length, embed_size = input_shape
        assert embed_size % 2 == 0, "embed_size must be even"
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "pos_encodings": self.pos_encodings,
        })
        return config
              
    def call(self, inputs):
        seq_length = tf.shape(inputs)[1]
        embed_size = tf.shape(inputs)[-1]
        
        p = tf.range(seq_length, dtype=self.dtype)
        i = tf.range(embed_size // 2, dtype=self.dtype) * 2  
        angle_rates = 1 / tf.pow(10_000.0, (tf.cast(i, dtype=self.dtype) / tf.cast(embed_size, dtype=self.dtype)))
        
        pos_encodings = tf.concat([
            tf.sin(tf.tensordot(p, angle_rates, axes=0)),  # Apply sin to even indices
            tf.cos(tf.tensordot(p, angle_rates, axes=0))   # Apply cos to odd indices
        ], axis=-1)  # Shape: [max_seq_length, embed_size]
        
        self.pos_encodings = tf.expand_dims(pos_encodings, axis=0)
        batch_max_length = tf.shape(inputs)[1]
        return inputs + self.pos_encodings[:, :batch_max_length]

    def compute_mask(self, inputs, mask=None):
        return mask



# ## Decoder (Multi-Attention Head)

class EncoderTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, n_units=128, num_heads=8, dropout_rate=0.0, N=2, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.N = N
        self.n_units = n_units
        self.num_heads = num_heads
        self.epsilon = 1e-4
        self.activation = tf.keras.activations.gelu
        self.dropout_rate = dropout_rate
        
        self.dropout = [tf.keras.layers.Dropout(self.dropout_rate) for _ in range(N)]
        self.layer_norm1 = [tf.keras.layers.LayerNormalization(epsilon=self.epsilon) for _ in range(N)]
        self.layer_norm2 = [tf.keras.layers.LayerNormalization(epsilon=self.epsilon) for _ in range(N)]
        self.supports_masking = True 

    def build(self, input_shape):
        _, _, embed_size = input_shape

        self.self_attn_layer = [
            tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads,
                                               key_dim=embed_size // self.num_heads,
                                               dropout=self.dropout_rate) for _ in range(self.N)]
        
        self.dense1 = [tf.keras.layers.Dense(self.n_units, activation=self.activation) for _ in range(self.N)]
        self.dense2 =  [tf.keras.layers.Dense(embed_size) for _ in range(self.N)]
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_units": self.n_units,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "N": self.N
        })
        return config

    def call(self, inputs, attention_mask, training=False):
        Z = inputs
        for i in range(self.N):
            skip = Z
            Z = self.self_attn_layer[i](query=Z, key=Z, value=Z, attention_mask=attention_mask, training=training)
            Z = self.layer_norm1[i](Z + skip)
            skip = Z
            Z = self.dense1[i](Z)
            Z = self.dense2[i](Z)
            Z = self.dropout[i](Z, training=training)
            Z = self.layer_norm2[i](Z + skip)
        return Z

    def compute_mask(self, inputs, mask=None):
        return mask



# ## Encoder (Multi-Attention Head)

class DecoderTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, n_units=128, num_heads=8, dropout_rate=0.0, N=2, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.N = N
        self.n_units = n_units
        self.num_heads = num_heads
        self.epsilon = 1e-4
        self.activation = tf.keras.activations.gelu
        self.dropout_rate = dropout_rate
        
        self.layer_norm1 = [tf.keras.layers.LayerNormalization(epsilon=self.epsilon) for _ in range(N)]
        self.layer_norm2 = [tf.keras.layers.LayerNormalization(epsilon=self.epsilon) for _ in range(N)]
        self.layer_norm3 = [tf.keras.layers.LayerNormalization(epsilon=self.epsilon) for _ in range(N)]
        self.supports_masking = True

    def build(self, input_shape):
        _, _, embed_size = input_shape

        self.self_attn_layer = [
            tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads,
                                               key_dim=embed_size // self.num_heads,
                                               dropout=self.dropout_rate) for _ in range(self.N)]
        self.cross_attn_layer = [
            tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads,
                                               key_dim=embed_size // self.num_heads,
                                               dropout=self.dropout_rate) for _ in range(self.N)]
    
        self.dense1 = [tf.keras.layers.Dense(self.n_units, activation=self.activation) for _ in range(self.N)]
        self.dense2 = [tf.keras.layers.Dense(embed_size) for _ in range(self.N)]
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_units": self.n_units,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "N": self.N
        })
        return config
        
    def call(self, inputs, encoder_output, attention_mask1, attention_mask2, training=False):
        Z = inputs
        for i in range(self.N):
            skip = Z
            Z = self.self_attn_layer[i](query=Z, key=Z, value=Z, attention_mask=attention_mask1, training=training)
            Z = self.layer_norm1[i](Z + skip)
            skip = Z
            Z = self.cross_attn_layer[i](query=Z, key=encoder_output, value=encoder_output, attention_mask=attention_mask2, training=training)
            Z = self.layer_norm2[i](Z + skip)
            skip = Z
            Z = self.dense1[i](Z)
            Z = self.dense2[i](Z)
            Z = self.layer_norm3[i](Z + skip)
        return Z

    def compute_mask(self, inputs, mask=None):
        return mask
