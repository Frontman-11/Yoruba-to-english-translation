# 📦 Dataset Overview

This project utilizes a variety of Yoruba–English bilingual datasets sourced from publicly available corpora, textbooks, religious texts, and curated resources. The file names used here reflect the **final versions after all preprocessing** and cleaning steps. Some datasets were split into train/dev/test, while others were used as-is for finetuning or evaluation purposes.

---

## 🔍 Primary Source Repository

Many of the datasets were initially discovered through the following excellent collection:

- [Andrews2017/africanlp-public-datasets](https://github.com/Andrews2017/africanlp-public-datasets)

This repo links to datasets spanning multiple African languages, including Yoruba. Please refer to it for original credits and raw dataset locations.

---

## 📁 Dataset List & Sources

### 📘 Menyo-20k MT Corpus
Sourced via [UDS-LSV Menyo-20k](https://github.com/uds-lsv/menyo-20k_MT) and referenced in the African NLP repo above.

**Files used:**
- `train.tsv`  
- `dev.tsv`  
- `test.tsv`  

These files provide high-quality Yoruba–English sentence pairs for finetuning and evaluation.

---

### 📂 Additional Parallel Datasets (Yo-En)

These parallel datasets were curated from multiple publicly available resources referenced in the African NLP repo above:

- `test_yo_en.tsv`  
- `GNOME_yo_en.tsv`  
- `opus_yo_en.tsv`  
- `raw_yo_en.tsv`  

Each of these files contains Yoruba–English aligned sentence pairs, gathered and standardized during preprocessing.

---

### 📖 Yorùbá Yé Mi Textbook

- [Yorùbá Yé Mi Textbook (PDF)](https://coerll.utexas.edu/yemi/pdfs/YorubaYeMi-textbook.pdf)  
  Provided by COERLL, University of Texas at Austin. This textbook was used to extract structured, formal Yoruba sentences and align them with English equivalents.

---

### 📜 JW300 Dataset (Yoruba-English)

- [JW300_en-yo.csv](https://github.com/Niger-Volta-LTI/yoruba-text/tree/master/JW300)  
  Sourced from the [Niger-Volta LTI project](https://github.com/Niger-Volta-LTI). This dataset offers multilingual content translated across several African languages, including Yoruba.

---

### 📖 Bible-Based Datasets

These were curated using religious texts in both Yoruba and English to create aligned pairs for translation and modeling:

- `bible_yo_en.csv`  
  A curated bilingual version built using the following two sources:

  - [📄 The Holy Bible – New King James Version (NKJV)](https://archive.org/download/labibliaeningles/The%20Holy%20Bible%20-%20New%20King%20James%20Version%20-%20NKJV%20(DOC).pdf)  
    Used as the English reference translation. Hosted on [Archive.org](https://archive.org).

  - [📖 Biblica Yoruba Bible (Yorùbá Bíbélì)](https://www.biblica.com/bible/ycb/)  
    Used as the Yoruba source. Provides access to the full Yorùbá Bíbélì in digital form.

---

### 📊 Kaggle Dataset: Yoruba to English

- [Yoruba to English with Machine Learning](https://www.kaggle.com/datasets/areolajoshua/yoruba-to-english-with-machine-learning)  
  Created by [areolajoshua](https://www.kaggle.com/areolajoshua). A clean and ready-to-use dataset for machine translation projects.

**Files used:**
- `Train.csv`  
- `Test.csv`

---

## 📌 Notes

- All filenames listed above reflect their **final form after preprocessing**, token cleanup, deduplication, and alignment.
- Tokenization, length normalization, and language direction adjustments were applied where appropriate.
- Some datasets were used directly, while others were split or merged into standardized formats suitable for Transformer-based NMT training.

---
