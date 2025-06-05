# 🧠 Yoruba ↔ English Neural Machine Translation

This project focuses on building a Neural Machine Translation (NMT) system to translate between Yoruba and English using a Transformer-based architecture inspired by Google's [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper.

Yorùbá is a low-resource language predominantly spoken in Nigeria and across the diaspora. Existing parallel datasets are limited and largely religious in nature. Despite these challenges, this project successfully trains high-performing translation models using fewer than 600,000 sentence pairs.

---

## 🚀 Project Overview

- **Goal**: Build monolingual Transformer models for Yoruba ⇄ English translation.
- **Architecture**: Custom implementation of the Transformer model using TensorFlow/Keras.
- **Pipeline Phases/Directory Structures**:
  - `data/train/`: Pretraining on large datasets from scratch.
  - `data/dev/`: Fine-tuning and experimentation on smaller or domain-specific subsets.
  - `data/test/`: Final evaluation and inference.
  - `notebook/`: Contains high-level execution logic and experimental notebooks.
  - `models/`: Contains the models saved from the notebooks.
  - `utils/`, `transformer/`, `tokenizer/`: Custom utility modules, Transformer layers and model, and tokenization logic.

---

## 📊 Results

- **Training Accuracy**: Peaked around the 60% range (masked accuracy).
- **BLEU Score**: Ranged from 70%–78% across ~6,000+ evaluation samples.
- **Best Performance**: Achieved when evaluating Yoruba → English translations in batch mode on GPU.

---

## 🖥️ Platform

- All training and fine-tuning were done using the free GPU on [Kaggle](https://www.kaggle.com/).
  
---

## 📁 Dataset Sources

All data used were manually processed, cleaned, and renamed before training. The names seen in the codebase reflect the final cleaned versions.

### 🔗 Aggregated Public Datasets

- [Andrews2017/africanlp-public-datasets](https://github.com/Andrews2017/africanlp-public-datasets)  
  - Origin of many datasets including:
    - `train.tsv`, `dev.tsv`, `test.tsv` from [menyo-20k](https://github.com/uds-lsv/menyo-20k_MT)
    - `test_yo_en.tsv`, `GNOME_yo_en.tsv`, `opus_yo_en.tsv`, `raw_yo_en.tsv`

### 📖 Religious and Educational Sources

- [Yorùbá Yé Mi Textbook (PDF)](https://coerll.utexas.edu/yemi/pdfs/YorubaYeMi-textbook.pdf)  
  - Source of parallel educational Yoruba-English examples.

- [The Holy Bible – NKJV (PDF)](https://archive.org/download/labibliaeningles/The%20Holy%20Bible%20-%20New%20King%20James%20Version%20-%20NKJV%20(DOC).pdf)  
  - Used for aligning English biblical texts with Yoruba equivalents.

- [Biblica Yoruba Bible (YCB)](https://www.biblica.com/bible/ycb/)  
  - A high-quality source for Yoruba religious texts.

### 📦 Curated Datasets

- [JW300 Yoruba Dataset](https://github.com/Niger-Volta-LTI/yoruba-text/tree/master/JW300)  
  - File used: `JW300_en-yo.csv`

- [Yoruba-English ML Dataset on Kaggle](https://www.kaggle.com/datasets/areolajoshua/yoruba-to-english-with-machine-learning)  
  - Used: `Train.csv`, `Test.csv`

---

## 🧪 Reproducibility

Each major directory contains its own `README.md` with instructions, file descriptions, and usage notes. The training and fine-tuning pipelines are modular and can be reversed by changing only a few lines of code (e.g., switching `df["Yoruba"]` with `df["English"]`).

No major changes are required to reproduce results or switch translation directions.

---

## ⚖️ License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](./LICENSE) file for details.
