# 📁 Directory Structure

### `pretrained/`
Contains models saved directly from `training.ipynb`.

### `fineTuned/`
Contains models saved after further training in `finetuning.ipynb`.

---

# 📝 Naming Conventions

- `EnYo` → English to Yoruba  
- `YoEn` → Yoruba to English  
- `FineTuned` → Prefix for models saved during or after finetuning  
- `Base` → Suffix for models saved at final epoch or via early stopping  
- `Best` → Suffix for models saved by `ModelCheckpoint` based on `val_masked_accuracy`

> **Note:**  
> All saved models **exclude the optimizer’s state** to reduce file size. This keeps each model under GitHub’s 100 MB file limit (each is ~76 MB).

---

# 🚀 Model Usage

To translate a tokenized input:
```python
translated_tokens = model.translate(tokens)
```

- **`tokens`** should be a tensor containing tokenized input.
- The function returns **translated tokens**, which can then be decoded into a readable sequence.
- **Translation method:** Greedy Search (selects the most probable token at each step).

---

## ⚠️ Observed Limitations

- **EnYo** (English → Yoruba) models perform consistently well on both **CPU** and **GPU**.
- **YoEn** (Yoruba → English) models perform best when run **in batches on GPU**, similar to the training condition.
- **Translation quality may degrade** when using `YoEn` models on **CPU** or for **single-sample inference**.

---

## 📦 Model Size

Each model uploaded to this repository is approximately **76 MB** in size.  
This compact size is achieved by **excluding the optimizer's state** from the saved model files.

