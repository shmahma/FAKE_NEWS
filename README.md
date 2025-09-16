# Fake News Detection

## Overview
This project detects fake news using NLP and transformer models. A `DistilBERT` classifier is trained to distinguish between true and fake news.

A Streamlit interface allows users to easily input news text for prediction and get instant classification results.

---

## Features

### NLP Model
- **Model used:** DistilBERT (`DistilBertForSequenceClassification`)  
- **Tokenizer:** DistilBertTokenizerFast  
- **Task:** Binary classification
  - `0` → Fake news  
  - `1` → True news  

### Preprocessing
- Tokenization with padding/truncation to 512 tokens  
- Converts text into input IDs and attention masks for the model  

### Evaluation
- Metrics used: Accuracy, Precision, Recall, F1-score  

---

## Installation
Clone the repository:

```bash
git clone https://github.com/shmahma/FAKE_NEWS
cd FAKE_NEWS
