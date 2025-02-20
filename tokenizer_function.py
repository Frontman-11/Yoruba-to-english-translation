import tensorflow as tf
import sentencepiece as spm

class FrontmanTokenizer(spm.SentencePieceProcessor):
    def __init__(self, model_path, max_length, truncation=True, pad_token_id=0, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.truncation = truncation
        self.spm = spm.SentencePieceProcessor(model_file=model_path)

    def encode(self, text, out_type='tf', exclude_token_ids=None, with_attention_mask=True, **kwargs):
        if out_type == 'tf':
            input_ids = self.spm.encode(text, out_type=int, **kwargs)
            input_ids = tf.ragged.constant(input_ids, dtype=tf.int32)            

            # Exclude unwanted token IDs
            if exclude_token_ids:
                mask = ~tf.reduce_any(tf.equal(input_ids[..., None], exclude_token_ids), axis=-1)
                input_ids = tf.ragged.boolean_mask(input_ids, mask)

            # Convert to regular tensor
            input_ids = input_ids.to_tensor(default_value=self.pad_token_id)

            # Apply truncation
            if  self.truncation:
                input_ids = input_ids[:, :self.max_length]  # Truncate if too long
                pad_length = tf.maximum(0, self.max_length - tf.shape(input_ids)[1])

                # Apply padding to ensure all sequences are max_length
                input_ids = tf.pad(input_ids, [[0, 0], [0, pad_length]], constant_values=self.pad_token_id)

        else:
            input_ids = self.spm.encode(text, out_type=out_type, **kwargs)

            # Exclude unwanted tokens
            if exclude_token_ids:
                input_ids = [token for token in input_ids if token not in exclude_token_ids]

            # Add padding tokens if necessary
            if self.pad_token_id:
                ids = []
                for input_id in input_ids:
                    input_id += [self.pad_token_id] * (self.max_length - len(input_id))  # Append padding tokens
                    ids.append(input_id)
                input_ids = ids

            # Handle truncation and padding
            if self.truncation and len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]

        # Create attention mask
        if with_attention_mask:
            attention_mask = tf.cast(tf.math.not_equal(input_ids, self.pad_token_id), dtype=tf.int32) if out_type == 'tf' else [int(token != self.pad_token_id) for token in input_ids]
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
        return input_ids

    def decode(self, input_ids, out_type=str, **kwargs):
        if isinstance(input_ids, tf.Tensor):
            input_ids = input_ids.numpy().tolist()
        return self.spm.decode(input_ids, out_type=out_type, **kwargs)
