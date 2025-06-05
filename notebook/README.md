## 📘 `training.ipynb`

This notebook contains the pipeline for the **pretraining phase** of the model. It:

- Utilizes the bulk of the dataset
- Saves weights after each epoch
- Saves the **best model** based on `val_masked_accuracy`
- Applies **early stopping**

To switch between **English → Yoruba** and **Yoruba → English** training, only **three lines of code** need to be changed — specifically the keys used in the `DataFrame`. These are annotated with comments:

```python
def create_dataset(df, tokenizer, max_length=128, batch_size=128, drop_remainder=False, shuffle_size=False, cache=False):
    
    encoder_input = tokenizer.special_encode(
        df['Yoruba'].values.tolist(),       # English for English to Yoruba training
        max_length=max_length,
        truncation=True,
        padding=True
    )
    
    decoder_input = tokenizer.special_encode(
        df['English'].values.tolist(),      # Yoruba for English to Yoruba training 
        max_length=max_length,
        truncation=True,
        padding=True,
        add_bos=True
    )

    decoder_target = tokenizer.special_encode(
        df['English'].values.tolist(),      # Yoruba for English to Yoruba training 
        max_length=max_length,
        truncation=True,
        padding=True,
        add_eos=True
    )
```

## 🛠️ `finetuning.ipynb`

This notebook provides the pipeline used to **finetune** the pretrained models by:

- Setting a **constant learning rate**
- Adjusting the **dropout rate** of the model

As with `training.ipynb`, switching between directions (**English → Yoruba** or **Yoruba → English**) is done by modifying the **same three lines** of code.

> ### 🔁 Reproducibility  
> No major changes are required to reproduce results or reverse the direction of translation.
