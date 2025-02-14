import tensorflow as tf


# ## Creating Positional Encoding

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_seq_length, embed_size, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        assert embed_size % 2 == 0, "embed_size must be even"
        
        p, i = np.meshgrid(np.arange(max_seq_length), 2 * np.arange(embed_size // 2))
        pos_emb = np.empty((1, max_seq_length, embed_size))
        pos_emb[0, :, ::2] = np.sin(p / 10_000 ** (i / embed_size)).T
        pos_emb[0, :, 1::2] = np.cos(p / 10_000 ** (i / embed_size)).T
        self.pos_encodings = tf.constant(pos_emb.astype(self.dtype))
        self.supports_masking = True
        
    def call(self, inputs):
        batch_max_length = tf.shape(inputs)[1]
        return inputs + self.pos_encodings[:, :batch_max_length]



# ## Decoder (Multi-Attention Head)

class EncoderTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_size, n_units=128, num_heads=8, dropout_rate=0.0, N=2, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.N = N
        self.n_units = n_units
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.dropout_rate = dropout_rate

        self.self_attn_layer = [
            tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads,
                                               key_dim=self.embed_size,
                                               dropout=self.dropout_rate) for _ in range(N)]
        
        self.dense1 = [tf.keras.layers.Dense(self.n_units, activation="relu") for _ in range(N)]
        self.dense2 =  [tf.keras.layers.Dense(self.embed_size) for _ in range(N)]
        self.dropout = [tf.keras.layers.Dropout(self.dropout_rate) for _ in range(N)]
        self.layer_norm1 = [tf.keras.layers.LayerNormalization() for _ in range(N)]
        self.layer_norm2 = [tf.keras.layers.LayerNormalization() for _ in range(N)]

    def call(self, inputs, attention_mask):
        Z = inputs
        for i in range(self.N):
            skip = Z
            Z = self.self_attn_layer[i](query=Z, key=Z, value=Z, attention_mask=attention_mask)
            Z = self.layer_norm1[i](Z + skip)
            skip = Z
            Z = self.dense1[i](Z)
            Z = self.dense2[i](Z)
            Z = self.dropout[i](Z)
            Z = self.layer_norm2[i](Z + skip)
            
        return Z



# ## Encoder (Multi-Attention Head)

class DecoderTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_size, n_units=128, num_heads=8, dropout_rate=0.0, N=2, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.N = N
        self.n_units = n_units
        self.num_heads = num_heads
        self.embed_size= embed_size
        self.dropout_rate = dropout_rate
        
        self.self_attn_layer = [
            tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads,
                                               key_dim=self.embed_size,
                                               dropout=self.dropout_rate) for _ in range(N)]
        self.cross_attn_layer = [
            tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads,
                                               key_dim=self.embed_size,
                                               dropout=self.dropout_rate) for _ in range(N)]
        
        self.layer_norm1 = [tf.keras.layers.LayerNormalization() for _ in range(N)]
        self.layer_norm2 = [tf.keras.layers.LayerNormalization() for _ in range(N)]
        self.layer_norm3 = [tf.keras.layers.LayerNormalization() for _ in range(N)]
        self.dense1 = [tf.keras.layers.Dense(self.n_units, activation="relu") for _ in range(N)]
        self.dense2 = [tf.keras.layers.Dense(self.embed_size) for _ in range(N)]


    def call(self, inputs, encoder_output, attention_mask1, attention_mask2):
        Z = inputs
        for i in range(self.N):
            skip = Z
            Z = self.self_attn_layer[i](query=Z, key=Z, value=Z, attention_mask=attention_mask1)
            Z = self.layer_norm1[i](Z + skip)
            skip = Z
            Z = self.cross_attn_layer[i](query=Z, key=encoder_output, value=encoder_output, attention_mask=attention_mask2)
            Z = self.layer_norm2[i](Z + skip)
            skip = Z
            Z = self.dense1[i](Z)
            Z = self.dense2[i](Z)
            Z = self.layer_norm3[i](Z + skip)
            
        return Z