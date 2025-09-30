import keras
import tensorflow as tf
from transformer.encoder import Encoder
from transformer.decoder import Decoder


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

    def _translate_batch(self, sentence, tgt_tokenizer, max_seq_length):
        batch_size = tf.shape(sentence)[0].numpy()
        bos_id = tgt_tokenizer.piece_to_id('<BOS>')
        eos_id = tgt_tokenizer.piece_to_id('<EOS>')
    
        bos_tokens = tf.fill([batch_size, 1], tf.constant(bos_id, dtype=tf.int64))
        decoded = bos_tokens
    
        for _ in range(max_seq_length):
            logits = self.call((sentence, decoded), training=False)
            next_token = tf.argmax(logits[:, -1:, :], axis=-1, output_type=tf.int64)
            decoded = tf.concat([decoded, next_token], axis=-1)
    
        decoded_sentences = []
        for i in range(batch_size):
            seq = decoded[i].numpy().tolist()
            if eos_id in seq:
                seq = seq[:seq.index(eos_id)]
            decoded_sentences.append(seq)
    
        return decoded_sentences  
    
    # def translate(self, sentence, tgt_tokenizer, max_seq_length=128, batch_size=128):
    #     assert isinstance(sentence, tf.Tensor), 'Input sentence not instance of tf.Tensor'
        
    #     if len(sentence.shape) == 1:
    #         sentence = sentence[tf.newaxis, :]
    
    #     total_samples = tf.shape(sentence)[0].numpy()
    #     translations = []
    
    #     for i in range(0, total_samples, batch_size):
    #         batch_sentence = sentence[i:i+batch_size]
    #         translations.extend(self._translate_batch(batch_sentence, tgt_tokenizer, max_seq_length))
    
    #     return translations

    def _translate_batch_beam(
        self, sentence, tgt_tokenizer, max_seq_length, beam_width=5, length_penalty=0.7):
        """
        Beam search translation (vectorized).
    
        Args:
            sentence: [batch, src_len] int tensor
            tgt_tokenizer: tokenizer with BOS/EOS ids
            max_seq_length: maximum target length
            beam_width: number of beams to keep
            length_penalty: exponent for length normalization (0 = none, >0 = normalize)
        """
        batch_size = tf.shape(sentence)[0]
    
        bos_id = tgt_tokenizer.piece_to_id("<BOS>")
        eos_id = tgt_tokenizer.piece_to_id("<EOS>")
    
        # [batch, beam, seq_len] -> flatten to [batch*beam, seq_len]
        sequences = tf.fill([batch_size * beam_width, max_seq_length], eos_id)
        sequences = tf.tensor_scatter_nd_update(
            sequences,
            indices=tf.expand_dims(tf.range(batch_size * beam_width), 1),
            updates=tf.repeat(tf.constant([bos_id], dtype=tf.int64), batch_size * beam_width),
        )
    
        # [batch, beam] scores
        scores = tf.concat(
            [tf.zeros([batch_size, 1]), -1e9 * tf.ones([batch_size, beam_width - 1])],
            axis=1,
        )
        scores = tf.reshape(scores, [-1])  # [batch*beam]
    
        # finished flags
        finished = tf.zeros_like(scores, dtype=tf.bool)
    
        for step in range(1, max_seq_length):
            # Run decoder on current sequences
            logits = self.call(
                (tf.repeat(sentence, repeats=beam_width, axis=0), sequences[:, :step]),
                training=False,
            )  # [batch*beam, step, vocab]
            logits = logits[:, -1, :]  # last step -> [batch*beam, vocab]
    
            log_probs = tf.nn.log_softmax(logits, axis=-1)
    
            # Prevent updating finished beams
            log_probs = tf.where(
                finished[:, None], tf.constant(-1e9, dtype=log_probs.dtype), log_probs
            )
    
            # Add to running scores
            total_scores = tf.reshape(scores, [-1, 1]) + log_probs  # [batch*beam, vocab]
    
            # Reshape to [batch, beam*vocab]
            vocab_size = tf.shape(log_probs)[-1]
            total_scores = tf.reshape(total_scores, [batch_size, beam_width * vocab_size])
    
            # Select top-k
            topk_scores, topk_indices = tf.math.top_k(total_scores, k=beam_width)
    
            # Compute beam and token indices
            beam_indices = topk_indices // vocab_size
            token_indices = topk_indices % vocab_size
    
            # Gather previous sequences
            batch_offsets = tf.range(batch_size) * beam_width
            flat_beam_indices = tf.reshape(
                beam_indices + batch_offsets[:, None], [-1]
            )  # [batch*beam]
    
            gathered = tf.gather(sequences, flat_beam_indices)
    
            # Update with new tokens at `step`
            updates = tf.reshape(token_indices, [-1])
            indices = tf.stack(
                [tf.range(batch_size * beam_width, dtype=tf.int64), tf.fill([batch_size * beam_width], step)], axis=1
            )
            sequences = tf.tensor_scatter_nd_update(gathered, indices, updates)
    
            # Update scores
            scores = tf.reshape(topk_scores, [-1])
    
            # Update finished flags
            finished = tf.logical_or(finished[flat_beam_indices], updates == eos_id)
    
        # Reshape sequences and scores
        sequences = tf.reshape(sequences, [batch_size, beam_width, max_seq_length])
        scores = tf.reshape(scores, [batch_size, beam_width])
    
        # Length normalization
        if length_penalty > 0:
            lengths = tf.reduce_sum(
                tf.cast(tf.not_equal(sequences, eos_id), tf.float32), axis=-1
            )
            scores = scores / tf.pow((5.0 + lengths) / 6.0, length_penalty)
    
        # Select best beam
        best_indices = tf.argmax(scores, axis=1)
        best_sequences = []
        for b in range(batch_size):
            seq = sequences[b, best_indices[b]].numpy().tolist()
            if eos_id in seq:
                seq = seq[: seq.index(eos_id)]
            best_sequences.append(seq)
    
        return best_sequences

    def translate(self, sentence, tgt_tokenizer, max_seq_length=128, batch_size=128, method="greedy", beam_width=5):
        assert isinstance(sentence, tf.Tensor), 'Input sentence not instance of tf.Tensor'
        
        if len(sentence.shape) == 1:
            sentence = sentence[tf.newaxis, :]
    
        total_samples = tf.shape(sentence)[0].numpy()
        translations = []
    
        for i in range(0, total_samples, batch_size):
            batch_sentence = sentence[i:i+batch_size]
            if method == "greedy":
                translations.extend(self._translate_batch(batch_sentence, tgt_tokenizer, max_seq_length))
            elif method == "beam":
                translations.extend(self._translate_batch_beam(batch_sentence, tgt_tokenizer, max_seq_length, beam_width))
            else:
                raise ValueError(f"Unknown translation method: {method}")
    
        return translations
