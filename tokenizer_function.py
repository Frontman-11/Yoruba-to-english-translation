import tensorflow as tf
import sentencepiece as spm

class FrontmanTokenizer(spm.SentencePieceProcessor):
    def __init__(self, model_path, max_length, truncation=True, pad_token_id=0, **kwargs):
        super().__init__(model_file=model_path, **kwargs)  # Initialize superclass properly
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.truncation = truncation

    def encode(self, text, out_type='tf', exclude_token_ids=None, with_attention_mask=True, **kwargs):
        # Ensure text is a regular Python string, not a Tensor
        if isinstance(text, tf.Tensor):
            text = text.numpy().decode("utf-8")  # Convert Tensor -> NumPy -> String
    
        if out_type == 'tf':
            input_ids = super().encode(text, out_type=int, **kwargs)  # Call SentencePiece encode
    
            input_ids = tf.ragged.constant(input_ids, dtype=tf.int32)  
    
            # (Rest of your method remains unchanged...)

            # def encode(self, text, out_type='tf', exclude_token_ids=None, with_attention_mask=True, **kwargs):
            #     if out_type == 'tf':
            #         input_ids = super().encode(text, out_type=int, **kwargs)  # Directly use superclass method
            #         input_ids = tf.ragged.constant(input_ids, dtype=tf.int32)            

            # Exclude unwanted token IDs
            if exclude_token_ids:
                mask = ~tf.reduce_any(tf.equal(input_ids[..., None], exclude_token_ids), axis=-1)
                input_ids = tf.ragged.boolean_mask(input_ids, mask)

            # Convert to regular tensor
            try:
                input_ids = input_ids.to_tensor(default_value=self.pad_token_id)
            except AttributeError:
                input_ids = [input_ids].to_tensor(default_value=self.pad_token_id)
                # Apply truncation
                if self.truncation:
                    input_ids = input_ids[:, :self.max_length]
                    pad_length = tf.maximum(0, self.max_length - tf.shape(input_ids)[1])
                    input_ids = tf.pad(input_ids, [[0, 0], [0, pad_length]], constant_values=self.pad_token_id)[0]
            
            # Apply truncation
            if self.truncation:
                input_ids = input_ids[:, :self.max_length]
                pad_length = tf.maximum(0, self.max_length - tf.shape(input_ids)[1])
                input_ids = tf.pad(input_ids, [[0, 0], [0, pad_length]], constant_values=self.pad_token_id)

        else:
            input_ids = super().encode(text, out_type=out_type, **kwargs)  # Use superclass encode method

            if exclude_token_ids:
                input_ids = [token for token in input_ids if token not in exclude_token_ids]

            ids = []
            for input_id in input_ids:
                input_id += [self.pad_token_id] * (self.max_length - len(input_id))
                ids.append(input_id)
            input_ids = ids

            if self.truncation:
                input_ids = input_ids[:self.max_length]

        # Create attention mask
        if with_attention_mask:
            attention_mask = tf.cast(tf.math.not_equal(input_ids, self.pad_token_id), dtype=tf.int32) if out_type == 'tf' else [int(token != self.pad_token_id) for token in input_ids[0]]
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
