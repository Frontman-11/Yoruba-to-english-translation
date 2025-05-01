# Clear all previously registered custom objects
keras.saving.get_custom_objects().clear()

@keras.saving.register_keras_serializable()
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        _, seq_length, embed_size = input_shape
        assert embed_size % 2 == 0, "Embedding size must be even"

        self.seq_length = seq_length
        self.embed_size = embed_size

        super().build(input_shape)

    def call(self, inputs):
        # Compute Positional Encoding dynamically
        p = tf.range(self.seq_length, dtype=self.dtype)[:, tf.newaxis]  # (seq_len, 1)
        i = tf.range(self.embed_size // 2, dtype=self.dtype)[tf.newaxis, :] * 2  # (1, embed_size//2)

        angle_rates = tf.pow(10_000.0, -i / tf.cast(self.embed_size, dtype=self.dtype))  # (1, embed_size//2)
        angle_rads = tf.einsum('ij, jk -> ik', p, angle_rates)  # (seq_len, embed_size//2)

        # Properly interleave sin and cos
        pos_encodings = tf.stack([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)  # (seq_len, embed_size//2, 2)
        pos_encodings = tf.reshape(pos_encodings, (self.seq_length, self.embed_size))  # (seq_len, embed_size)

        # Ensure shape matches the batch size dynamically
        pos_encodings = tf.broadcast_to(pos_encodings, tf.shape(inputs))  

        return inputs + pos_encodings

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return super().get_config()



@keras.saving.register_keras_serializable()
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


    def call(self, inputs, attention_mask, training=False):
        Z = inputs
        for i in range(self.N):
            skip = Z
            Z = getattr(self, f'encoder_attention_{i}')(query=Z, key=Z, value=Z, attention_mask=attention_mask, training=training)
            Z = getattr(self, f'encoder_layer_norm1_{i}')(tf.keras.layers.Add()([Z, skip]))
            skip = Z
            Z = getattr(self, f'encoder_dense1_{i}')(Z)
            Z = getattr(self, f'encoder_dense2_{i}')(Z)
            Z = getattr(self, f'encoder_dropout_{i}')(Z, training=training) 
            Z = getattr(self, f'encoder_layer_norm2_{i}')(tf.keras.layers.Add()([Z, skip]))
        return Z

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape
        
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



@keras.saving.register_keras_serializable()
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

        
    def call(self, inputs, encoder_output, attention_mask1, attention_mask2, training=False):
        Z = inputs
        for i in range(self.N):
            skip = Z
            Z = getattr(self, f'decoder_attention_{i}')(query=Z, key=Z, value=Z, attention_mask=attention_mask1, training=training)
            Z = getattr(self, f'decoder_layer_norm1_{i}')(tf.keras.layers.Add()([Z, skip]))
            skip = Z
            Z = getattr(self, f'decoder_cross_attention_{i}')(query=Z, key=encoder_output, value=encoder_output, attention_mask=attention_mask2, training=training)
            Z = getattr(self, f'decoder_layer_norm2_{i}')(tf.keras.layers.Add()([Z, skip])) 
            skip = Z
            Z = getattr(self, f'decoder_dense1_{i}')(Z)
            Z = getattr(self, f'decoder_dense2_{i}')(Z)
            Z = getattr(self, f'decoder_dropout_{i}')(Z, training=training)
            Z = getattr(self, f'decoder_layer_norm3_{i}')(tf.keras.layers.Add()([Z, skip]))
        return Z

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape
        
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



@keras.saving.register_keras_serializable()
class TransformerModel(tf.keras.Model):
    def __init__(self, 
                 positional_embedding,
                 encoder_block,
                 decoder_block,
                 max_seq_length, 
                 embed_kwargs,
                 **kwargs):
        super().__init__(**kwargs)
        self.embed_kwargs = embed_kwargs
        
        self.encoder_embed_layer = tf.keras.layers.Embedding(**self.embed_kwargs)
        self.decoder_embed_layer = tf.keras.layers.Embedding(**self.embed_kwargs)
        
        self.positional_embedding = positional_embedding
        
        self.encoder_block = encoder_block
        self.decoder_block = decoder_block
        
        self.linear_layer = tf.keras.layers.Dense(self.encoder_embed_layer.input_dim)
        
        self.max_seq_length = max_seq_length

 
    def call(self, inputs, training=False):
        encoder_input_ids = inputs["encoder_input_ids"] 
        decoder_input_ids = inputs["decoder_input_ids"]
        
        encoder_pad_mask = inputs["encoder_attention_mask"][:, tf.newaxis]
        decoder_pad_mask = inputs["decoder_attention_mask"][:, tf.newaxis]
        
        # Causal Mask for Decoder
        causal_mask = tf.linalg.band_part(
            tf.ones((self.max_seq_length, self.max_seq_length), tf.bool), -1, 0)
        
        attention_mask1 = causal_mask & tf.cast(decoder_pad_mask, tf.bool)
        
        # Encoder processing
        Z = self.encoder_embed_layer(encoder_input_ids)
        Z = self.positional_embedding(Z)
        Z = self.encoder_block(inputs=Z, attention_mask=encoder_pad_mask, training=training)
    
        # Decoder processing
        X = self.decoder_embed_layer(decoder_input_ids)
        X = self.positional_embedding(X)
        X = self.decoder_block(inputs=X, encoder_output=Z, attention_mask1=attention_mask1,
                               attention_mask2=encoder_pad_mask, training=training)
        # final layer
        logits = self.linear_layer(X)
        return logits

    def get_config(self):
        config = super().get_config()
        config.update({
            "positional_embedding": keras.saving.serialize_keras_object(self.positional_embedding),
            "encoder_block": keras.saving.serialize_keras_object(self.encoder_block),
            "decoder_block": keras.saving.serialize_keras_object(self.decoder_block),
            "embed_kwargs":self.embed_kwargs,
            "max_seq_length": self.max_seq_length
        })
        return config

    @classmethod
    def from_config(cls, config):
        config["positional_embedding"] = keras.saving.deserialize_keras_object(config["positional_embedding"])
        config["encoder_block"] = keras.saving.deserialize_keras_object(config["encoder_block"])
        config["decoder_block"] = keras.saving.deserialize_keras_object(config["decoder_block"])
        return cls(**config)
