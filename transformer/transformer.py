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
        self, sentence, tgt_tokenizer, max_seq_length, beam_width=5, length_penalty=0.7
    ):
        """
        Improved beam search translation with proper batch processing.
    
        Args:
            sentence: [batch, src_len] int tensor
            tgt_tokenizer: tokenizer with BOS/EOS ids
            max_seq_length: maximum target length
            beam_width: number of beams to keep
            length_penalty: exponent for length normalization (0 = none, >0 = normalize)
        """
        batch_size = tf.shape(sentence)[0]
        
        bos_id = tf.constant(tgt_tokenizer.piece_to_id("<BOS>"), dtype=tf.int64)
        eos_id = tf.constant(tgt_tokenizer.piece_to_id("<EOS>"), dtype=tf.int64)
        
        # Initialize sequences: [batch*beam, max_seq_length]
        # Start with BOS token only, rest will be filled during generation
        sequences = tf.fill([batch_size * beam_width, max_seq_length], tf.constant(0, dtype=tf.int64))
        sequences = tf.tensor_scatter_nd_update(
            sequences,
            tf.stack([tf.range(batch_size * beam_width, dtype=tf.int32), 
                      tf.zeros([batch_size * beam_width], dtype=tf.int32)], axis=1),
            tf.fill([batch_size * beam_width], bos_id)
        )
        
        # Initialize scores: [batch*beam]
        # First beam in each batch starts with 0, others with -inf
        scores = tf.concat([
            tf.zeros([batch_size, 1], dtype=tf.float32),
            tf.fill([batch_size, beam_width - 1], -1e9)
        ], axis=1)
        scores = tf.reshape(scores, [-1])
        
        # Track finished beams: [batch*beam]
        finished = tf.zeros([batch_size * beam_width], dtype=tf.bool)
        
        # Expand source sentence for beam search: [batch*beam, src_len]
        sentence_expanded = tf.repeat(sentence, repeats=beam_width, axis=0)
        
        for step in range(1, max_seq_length):
            # Early stopping if all beams are finished
            if tf.reduce_all(finished):
                break
            
            # Slice current sequence up to current step
            current_seq = sequences[:, :step]
            
            # Get logits from model: [batch*beam, step, vocab]
            logits = self.call((sentence_expanded, current_seq), training=False)
            logits = logits[:, -1, :]  # [batch*beam, vocab]
            
            # Convert to log probabilities
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            
            # Mask finished beams: set all their log_probs to -inf except for EOS
            # This ensures finished beams just keep generating EOS
            mask = tf.expand_dims(finished, axis=1)  # [batch*beam, 1]
            eos_mask = tf.one_hot(eos_id, depth=tf.shape(log_probs)[-1], dtype=log_probs.dtype)
            log_probs = tf.where(mask, eos_mask * 0.0 - 1e9 * (1.0 - eos_mask), log_probs)
            
            # Add current scores: [batch*beam, vocab]
            total_scores = tf.reshape(scores, [-1, 1]) + log_probs
            
            # Reshape to [batch, beam*vocab] for top-k selection
            vocab_size = tf.shape(log_probs)[-1]
            total_scores = tf.reshape(total_scores, [batch_size, beam_width * vocab_size])
            
            # Select top-k beams: [batch, beam]
            topk_scores, topk_indices = tf.math.top_k(total_scores, k=beam_width)
            
            # Decompose indices into beam and token
            beam_indices = topk_indices // vocab_size  # [batch, beam]
            token_indices = tf.cast(topk_indices % vocab_size, dtype=tf.int64)  # [batch, beam]
            
            # Gather sequences from selected beams
            batch_offsets = tf.range(batch_size, dtype=tf.int32) * beam_width
            flat_beam_indices = tf.reshape(beam_indices + batch_offsets[:, None], [-1])
            
            gathered_sequences = tf.gather(sequences, flat_beam_indices)
            gathered_finished = tf.gather(finished, flat_beam_indices)
            
            # Update sequences with new tokens at current step
            step_indices = tf.stack([
                tf.range(batch_size * beam_width, dtype=tf.int32),
                tf.fill([batch_size * beam_width], step)
            ], axis=1)
            sequences = tf.tensor_scatter_nd_update(
                gathered_sequences,
                step_indices,
                tf.cast(tf.reshape(token_indices, [-1]), dtype=tf.int64)
            )
            
            # Update scores
            scores = tf.reshape(topk_scores, [-1])
            
            # Update finished flags: beam is finished if it was already finished OR just generated EOS
            new_tokens = tf.reshape(token_indices, [-1])
            finished = tf.logical_or(gathered_finished, new_tokens == eos_id)
        
        # Reshape to [batch, beam, max_seq_length]
        sequences = tf.reshape(sequences, [batch_size, beam_width, max_seq_length])
        scores = tf.reshape(scores, [batch_size, beam_width])
        
        # Apply length normalization
        if length_penalty > 0:
            # Count actual tokens (exclude padding/BOS, count up to EOS)
            lengths = tf.reduce_sum(
                tf.cast(sequences != 0, tf.float32), axis=-1
            )  # [batch, beam]
            # Apply length penalty (Google NMT style)
            scores = scores / tf.pow((5.0 + lengths) / 6.0, length_penalty)
        
        # Select best beam for each batch
        best_indices = tf.argmax(scores, axis=1, output_type=tf.int32)
        
        # Extract best sequences
        best_sequences = []
        for b in range(batch_size):
            seq = sequences[b, best_indices[b]].numpy().tolist()
            # Remove BOS at start
            if seq[0] == bos_id.numpy():
                seq = seq[1:]
            # Truncate at EOS
            if eos_id.numpy() in seq:
                seq = seq[:seq.index(eos_id.numpy())]
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
