import numpy as np
import tensorflow as tf
import sentencepiece as spm


class FrontmanTokenizer(spm.SentencePieceProcessor):
    def __init__(self,
                 model_path, 
                 **kwargs):
        super().__init__(model_file=model_path, **kwargs)
        
    def special_encode(self, 
               text,
               out_type='tf',
               with_attention_mask=False,
               max_length=None, 
               pad_token_id=0,
               padding=False,
               truncation=False,
               add_bos=False,
               add_eos=False
                      ):
        """Tokenizes input text and returns token IDs, with optional padding, truncation, and attention mask."""

        # ✅ Convert TensorFlow tensor input to Python list (if needed)
        if isinstance(text, tf.Tensor):
            text = text.numpy().tolist()
            text = [txt.decode("utf-8") for txt in text]  # Decode bytes to string

        # ✅ Batch tokenization for speed
        if isinstance(text, list):
            input_ids = super().encode_as_ids(text, add_bos=add_bos, add_eos=add_eos) 
        else:
            input_ids = [super().encode_as_ids(text, add_bos=add_bos, add_eos=add_eos)]

        # ✅ First, truncate sequences before converting to NumPy
        if truncation:
            input_ids = [seq[:max_length] for seq in input_ids]  
        
        # ✅ Efficient Padding with NumPy Broadcasting
        if padding:
            padded_array = np.full((len(input_ids), max_length), pad_token_id, dtype=np.int32)
            for i, seq in enumerate(input_ids):
                padded_array[i, :len(seq)] = seq 
            input_ids = padded_array

        # ✅ Convert to Tensor if needed
        if out_type == 'tf':
            input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int32)
        elif out_type == int:
            input_ids = input_ids.tolist()

        # ✅ Compute attention mask if requested
        if with_attention_mask:
            attention_mask = (input_ids != pad_token_id).astype(np.int32) if isinstance(input_ids, np.ndarray) \
                else tf.cast(tf.math.not_equal(input_ids, pad_token_id), dtype=tf.int32)
            return {'input_ids': input_ids, 'attention_mask': attention_mask}

        return input_ids

    def decode(self, input_ids, out_type=str, **kwargs):
        assert isinstance(input_ids, list), f'input_ids must be of instance List, got {type(input_ids)}'
        return super().decode(input_ids, out_type=out_type, **kwargs)
