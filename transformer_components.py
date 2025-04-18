import tensorflow as tf

# ## Creating Positional Encoding
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.supports_masking = True
        self.pos_encodings = None
        
    def build(self, input_shape):
        _, seq_length, embed_size = input_shape
        assert embed_size % 2 == 0, "Embedding size must be even"

        p = tf.range(seq_length, dtype=self.dtype)
        i = tf.range(embed_size // 2, dtype=self.dtype) * 2  
        angle_rates = 1 / tf.pow(10_000.0, (tf.cast(i, dtype=self.dtype) / tf.cast(embed_size, dtype=self.dtype)))
        angle_rads = tf.tensordot(p, angle_rates, axes=0)  # (seq_len, embed_size/2)

        # Apply sin to even indices, cos to odd indices
        pos_encodings = tf.stack([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)
        pos_encodings = tf.reshape(pos_encodings, (seq_length, embed_size))  # Reshape to (seq_len, embed_size)

        # Store as a non-trainable weight
        self.pos_encodings = self.add_weight(
            name="positional_encoding",
            shape=pos_encodings.shape,
            initializer=tf.constant_initializer(pos_encodings),  
            trainable=False
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs + self.pos_encodings
        
    def compute_mask(self, inputs, mask=None):
        return mask

    # def compute_output_shape(self, input_shape):
    #     return input_shape



# ## Decoder (Multi-Attention Head)

class EncoderTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, n_units=128, activation='relu', num_heads=8, dropout_rate=0.0, epsilon=1e-4, N=2, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.N = N
        self.n_units = n_units
        self.num_heads = num_heads
        self.epsilon = epsilon
        self.activation = activation 
        self.dropout_rate = dropout_rate
        self.supports_masking = True 

    def build(self, input_shape):
        _, _, embed_size = input_shape
        
        for i in range(self.N):
            layer = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=embed_size // self.num_heads,
                                               dropout=self.dropout_rate, name=f'encoder_attention_{i}')
            setattr(self, f"encoder_attention_{i}", layer)

            layer = tf.keras.layers.Dense(self.n_units, activation=self.activation, kernel_initializer="he_normal", name=f'encoder_dense1_{i}')
            setattr(self, f"encoder_dense1_{i}", layer)

            layer = tf.keras.layers.Dense(embed_size, kernel_initializer="glorot_normal", name=f'encoder_dense2_{i}')
            setattr(self, f"encoder_dense2_{i}", layer)
        
            layer = tf.keras.layers.Dropout(self.dropout_rate, name=f'encoder_dropout_{i}')
            setattr(self, f"encoder_dropout_{i}", layer)
            
            layer = tf.keras.layers.LayerNormalization(epsilon=self.epsilon, name=f'encoder_layer_norm1_{i}')
            setattr(self, f"encoder_layer_norm1_{i}", layer)
            
            layer = tf.keras.layers.LayerNormalization(epsilon=self.epsilon, name=f'encoder_layer_norm2_{i}')
            setattr(self, f"encoder_layer_norm2_{i}", layer)
        
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_units": self.n_units,
            'activation': self.activation,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "N": self.N,
            "epsilon": self.epsilon
        })
        return config

    def call(self, inputs, attention_mask, training=False):
        Z = inputs
        for i in range(self.N):
            skip = Z
            Z = getattr(self, f'encoder_attention_{i}')(query=Z, key=Z, value=Z, attention_mask=attention_mask, training=training)
            Z = getattr(self, f'encoder_layer_norm1_{i}')(Z + skip)
            skip = Z
            Z = getattr(self, f'encoder_dense1_{i}')(Z)
            Z = getattr(self, f'encoder_dense2_{i}')(Z)
            Z = getattr(self, f'encoder_dropout_{i}')(Z, training=training)
            Z = Z + skip 
            Z = getattr(self, f'encoder_layer_norm2_{i}')(Z)
        return Z

    def compute_mask(self, inputs, mask=None):
        return mask



# ## Encoder (Multi-Attention Head)

class DecoderTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, n_units=128, activation='relu', num_heads=8, dropout_rate=0.0, epsilon=1e-4, N=2, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.N = N
        self.n_units = n_units
        self.num_heads = num_heads
        self.epsilon = epsilon
        self.activation =activation
        self.dropout_rate = dropout_rate
        self.supports_masking = True

    def build(self, input_shape):
        _, _, embed_size = input_shape

        for i in range(self.N):
            layer = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=embed_size // self.num_heads,
                                               dropout=self.dropout_rate, name=f'decoder_attention_{i}')
            setattr(self, f"decoder_attention_{i}", layer)

            layer = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=embed_size // self.num_heads,
                                               dropout=self.dropout_rate, name=f'decoder_cross_attention_{i}')
            setattr(self, f"decoder_cross_attention_{i}", layer)
            
            layer = tf.keras.layers.Dense(self.n_units, activation=self.activation, kernel_initializer="he_normal", name=f'decoder_dense1_{i}')
            setattr(self, f"decoder_dense1_{i}", layer)

            layer = tf.keras.layers.Dense(embed_size, kernel_initializer="glorot_normal", name=f'decoder_dense2_{i}')
            setattr(self, f"decoder_dense2_{i}", layer)
        
            layer = tf.keras.layers.Dropout(self.dropout_rate, name=f'decoder_dropout_{i}')
            setattr(self, f"decoder_dropout_{i}", layer)
            
            layer = tf.keras.layers.LayerNormalization(epsilon=self.epsilon, name=f'decoder_layer_norm1_{i}')
            setattr(self, f"decoder_layer_norm1_{i}", layer)
            
            layer = tf.keras.layers.LayerNormalization(epsilon=self.epsilon, name=f'decoder_layer_norm2_{i}')
            setattr(self, f"decoder_layer_norm2_{i}", layer)

            layer = tf.keras.layers.LayerNormalization(epsilon=self.epsilon, name=f'decoder_layer_norm3_{i}')
            setattr(self, f"decoder_layer_norm3_{i}", layer)
            
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_units": self.n_units,
            'activation': self.activation,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "N": self.N,
            "epsilon": self.epsilon
        })
        return config
        
    def call(self, inputs, encoder_output, attention_mask1, attention_mask2, training=False):
        Z = inputs
        for i in range(self.N):
            skip = Z
            Z = getattr(self, f'decoder_attention_{i}')(query=Z, key=Z, value=Z, attention_mask=attention_mask1, training=training)
            Z = getattr(self, f'decoder_layer_norm1_{i}')(Z + skip)
            skip = Z
            Z = getattr(self, f'decoder_cross_attention_{i}')(query=Z, key=encoder_output, value=encoder_output, attention_mask=attention_mask2, training=training)
            Z = getattr(self, f'decoder_layer_norm2_{i}')(Z + skip)
            skip = Z
            Z = getattr(self, f'decoder_dense1_{i}')(Z)
            Z = getattr(self, f'decoder_dense2_{i}')(Z)
            Z = getattr(self, f'decoder_dropout_{i}')(Z, training=training)
            Z = Z + skip 
            Z = getattr(self, f'decoder_layer_norm3_{i}')(Z)
        return Z

    def compute_mask(self, inputs, mask=None):
        return mask
