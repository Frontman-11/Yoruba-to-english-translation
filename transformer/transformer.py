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

    def _translate_batch_beam(self, sentence, tgt_tokenizer, max_seq_length, beam_width=5):
        batch_size = tf.shape(sentence)[0].numpy()
        bos_id = tgt_tokenizer.piece_to_id('<BOS>')
        eos_id = tgt_tokenizer.piece_to_id('<EOS>')
    
        # Start with BOS token for each item
        bos_tokens = tf.fill([batch_size, 1], tf.constant(bos_id, dtype=tf.int64))
    
        final_sequences = []
    
        for b in range(batch_size):
            # initialize beams: (sequence, score)
            beams = [(bos_tokens[b:b+1], 0.0)]  # log-prob score
            completed = []
    
            for _ in range(max_seq_length):
                all_candidates = []
    
                for seq, score in beams:
                    logits = self.call((sentence[b:b+1], seq), training=False)
                    logits = logits[:, -1, :]  # last timestep
                    log_probs = tf.nn.log_softmax(logits, axis=-1).numpy().flatten()
    
                    # get top-k next tokens
                    topk_ids = tf.math.top_k(log_probs, k=beam_width).indices.numpy()
                    topk_scores = tf.math.top_k(log_probs, k=beam_width).values.numpy()
    
                    for token, logp in zip(topk_ids, topk_scores):
                        new_seq = tf.concat([seq, [[token]]], axis=-1)
                        new_score = score + float(logp)
    
                        if token == eos_id:
                            completed.append((new_seq, new_score))
                        else:
                            all_candidates.append((new_seq, new_score))
    
                # prune to beam_width best
                ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
                beams = ordered[:beam_width]
    
                if not beams:
                    break
    
            # pick best completed if exists, else best unfinished
            if completed:
                best_seq = max(completed, key=lambda tup: tup[1])[0]
            else:
                best_seq = beams[0][0]
    
            final_sequences.append(best_seq.numpy().flatten().tolist())
    
        # cut off at EOS
        decoded_sentences = []
        for seq in final_sequences:
            if eos_id in seq:
                seq = seq[:seq.index(eos_id)]
            decoded_sentences.append(seq)
    
        return decoded_sentences

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

