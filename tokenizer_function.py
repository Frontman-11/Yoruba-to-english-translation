import tensorflow as tf
import sentencepiece as spm


class FrontmanTokenizer(spm.SentencePieceProcessor):
    def __init__(self, model_path, max_length, pad_token_id=0, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.spm = spm.SentencePieceProcessor(model_file=model_path)

    
    def encode(self, text, out_type='tf', exclude_token_ids=None, **kwargs):
        if out_type == 'tf':
            input_ids = self.spm.encode(text, out_type=int, **kwargs)
            input_ids = tf.ragged.constant(input_ids, dtype=tf.int32)            

            if exclude_token_ids:
                mask = ~tf.reduce_any(tf.equal(input_ids[..., None], exclude_token_ids), axis=-1)
                input_ids = tf.ragged.boolean_mask(input_ids, mask)

            if self.pad_token_id != None:
                input_ids = input_ids.to_tensor(default_value=self.pad_token_id)

        else:
            input_ids = self.spm.encode(text, out_type=out_type, **kwargs)

            if exclude_token_ids:
                input_ids = [token for token in input_ids if token not in exclude_token_ids]

            input_ids = input_ids[:self.max_length] + [self.pad_token_id] * max(0, self.max_length - len(input_ids))

        # Create attention mask
        attention_mask = [int(token != self.pad_token_id) for token in input_ids] if out_type != 'tf' else tf.cast(tf.math.not_equal(input_ids, self.pad_token_id), dtype=tf.int32)

        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    
    def decode(self, input_ids, out_type=str, **kwargs):
        if isinstance(input_ids, tf.Tensor):
            input_ids = input_ids.numpy().tolist()
        return self.spm.decode(input_ids, out_type=out_type, **kwargs)