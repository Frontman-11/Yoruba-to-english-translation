# %% [code]
import tensorflow as tf
import sentencepiece as spm
import time

import time
import numpy as np
import tensorflow as tf
import sentencepiece as spm

class FrontmanTokenizer(spm.SentencePieceProcessor):
    def __init__(self, model_path, max_length, truncation=False, padding=False, pad_token_id=0, **kwargs):
        super().__init__(model_file=model_path, **kwargs)
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.truncation = truncation
        self.padding = padding  # Allow explicit padding control

    def encode(self, text, out_type='tf', with_attention_mask=False, **kwargs):
        """Tokenizes input text and returns token IDs, with optional padding, truncation, and attention mask."""
        start_time = time.time()

        # ✅ Convert TensorFlow tensor input to Python list (if needed)
        if isinstance(text, tf.Tensor):
            text = text.numpy().tolist()
            text = [txt.decode("utf-8") for txt in text]  # Decode bytes to string

        # ✅ Batch tokenization for speed
        input_ids = self.EncodeAsIds(text) if isinstance(text, list) else [self.EncodeAsIds(text)]
        print(f'Tokenization Time: {time.time() - start_time:.4f}s')

        # ✅ Convert to NumPy array for fast vectorized operations
        input_ids = np.array([np.array(seq[:self.max_length]) for seq in input_ids], dtype=object)

        # ✅ Efficient padding (if needed)
        if self.padding:
            pad_widths = [(0, self.max_length - len(seq)) for seq in input_ids]  # Compute padding lengths
            input_ids = np.array([np.pad(seq, pad_width, constant_values=self.pad_token_id) 
                                  for seq, pad_width in zip(input_ids, pad_widths)], dtype=np.int32)

        # ✅ Convert to Tensor if needed
        if out_type == 'tf':
            input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int32)

        # ✅ Compute attention mask if requested
        if with_attention_mask:
            attention_mask = (input_ids != self.pad_token_id).astype(np.int32) if isinstance(input_ids, np.ndarray) \
                else tf.cast(tf.math.not_equal(input_ids, self.pad_token_id), dtype=tf.int32)
            return {'input_ids': input_ids, 'attention_mask': attention_mask}

        return input_ids


def decode(self, input_ids, out_type=str, **kwargs):
    if isinstance(input_ids, tf.Tensor):
        input_ids = input_ids.numpy().tolist()
    return super().decode(input_ids, out_type=out_type, **kwargs)  # Use superclass decode method


# # Test Script
# import tensorflow as tf
# import sentencepiece as spm

# # Initialize tokenizer with a valid SentencePiece model file
# model_path = "your_sentencepiece_model.model"  # Change this to your actual model path
# max_length = 10

# tokenizer = FrontmanTokenizer(model_path, max_length)

# # Test different input conditions
# test_cases = [
#     "This is a test sentence.",
#     ["This is the first sentence.", "This is the second sentence."],
#     "",
#     "    ",  # String with only spaces
#     "😃🚀🔥",  # Emoji input
#     "ThisIsOneLongUnbrokenSentenceWithoutSpacesOrPunctuation",
#     "Short",
# ]

# exclude_token_ids = [0]  # Example token ID to exclude

# for i, text in enumerate(test_cases):
#     print(f"\nTest Case {i+1}: {text}")

#     # Test encoding with and without exclusion
#     encoded = tokenizer.encode(text, out_type='tf', exclude_token_ids=exclude_token_ids)
#     print("Encoded:", encoded)

#     # Test decoding
#     decoded = tokenizer.decode(encoded['input_ids'] if isinstance(encoded, dict) else encoded)
#     print("Decoded:", decoded)

# # Edge Case: List of empty strings
# empty_list_test = tokenizer.encode(["", "", ""], out_type='tf')
# print("\nTest Case: List of Empty Strings\nEncoded:", empty_list_test)

# # Edge Case: Tensor input
# tensor_test = tf.constant(["This is a test"], dtype=tf.string)
# encoded_tensor = tokenizer.encode(tensor_test, out_type='tf')
# print("\nTest Case: Tensor Input\nEncoded:", encoded_tensor)

# # Edge Case: Non-string input (should raise an error or handle gracefully)
# try:
#     invalid_input = tokenizer.encode(12345, out_type='tf')
#     print("\nTest Case: Integer Input\nEncoded:", invalid_input)
# except Exception as e:
#     print("\nTest Case: Integer Input\nError:", e)

# # Edge Case: Long text exceeding max_length
# long_text = " ".join(["word"] * 100)  # Very long text
# encoded_long = tokenizer.encode(long_text, out_type='tf')
# print("\nTest Case: Long Text\nEncoded:", encoded_long)

# # Edge Case: Padding and truncation check
# padded_text = "Short"
# encoded_padded = tokenizer.encode(padded_text, out_type='tf')
# print("\nTest Case: Padding Check\nEncoded:", encoded_padded)

# # Edge Case: Check access to native SentencePiece methods
# print("\nTest Case: Native SentencePiece Methods")
# try:
#     vocab_size = tokenizer.get_piece_size()
#     print("Vocab Size:", vocab_size)
#     print("First Token ID -> Piece:", tokenizer.id_to_piece(0))
# except Exception as e:
#     print("Error accessing native methods:", e)
