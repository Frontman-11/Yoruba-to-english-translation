import tensorflow as tf
import sentencepiece as spm


class FrontmanTokenizer():
    def __init__(self, max_length=128, pad_token_id=0):
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        path = '/kaggle/input/tokenizer/yo_en_bpe.model'
        self.spm = spm.SentencePieceProcessor(model_file=path)
        

    def encode(self, text, out_type='tf', **kwargs):
        if out_type == 'tf':
            input_ids = tf.ragged.constant(self.spm.encode(text, out_type=int, **kwargs), dtype=tf.int32)
            input_ids = input_ids.to_tensor(default_value=self.pad_token_id, shape=[None, self.max_length])
        else:
            input_ids = self.spm.encode(text, out_type=out_type)
            
        attention_mask = tf.cast(tf.math.not_equal(input_ids, self.pad_token_id), dtype=tf.int32)

        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    
    def decode(self, input_ids, out_type=str, **kwargs):
        if isinstance(input_ids, tf.Tensor):
            input_ids = input_ids.numpy().tolist()
        if out_type == 'tf':
            return tf.constant(self.spm.decode(input_ids, out_type=int, **kwargs), dtype=tf.int32)
        else:
            return self.spm.decode(input_ids, out_type=out_type, **kwargs)


            
# A = ['Ní òwúrọ̀ ọjọ́, Ẹtì, ọjọ́ Kẹrìnlá, ọdún 2024 ni Adebanjo.',
#      'dágbére fáyé lẹ́ni ọdún mẹ́rìndínlọ́gọ́rùn-ún (96).',
#     'Agbẹnusọ ẹgbẹ́ Afenifere, Jare Ajayi ló fìdí ìròyìn náà múlẹ̀ fún BBC News Yorùbá.',
#     'Ilé rẹ̀ tó wà ní agbègbè Lekki, ìpínlẹ̀ Eko ni Adebanjo dákẹ́ sí']

# sentences = ['This is the first test sentence.',
#              'Here is the second sentence for testing.',
#              'sentence Here is the second for testing.', 
#              'Here is the nothing.', 
#              'Finally, this is the many sentence.']

# tokenizer = FrontmanTokenizer(max_length=20)
# input_ids = tokenizer.encode(A)
# input_ids

# from transformers import T5Tokenizer

# t5tokenizer = T5Tokenizer.from_pretrained('t5-small')

# text = sentences

# encoded_input = t5tokenizer(text, return_tensors='np', max_length=20, truncation=True, padding=True)  # Use 'tf' for TensorFlow

# print(encoded_input)

# print("Decoded Text:", decoded_text)