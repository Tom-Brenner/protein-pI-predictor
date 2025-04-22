---
title: 'Protein pI Predictor'
emoji: 'clap'
colorFrom: 'pink'
colorTo: 'blue'
sdk: 'gradio'
sdk_version: ""
app_file: app.py
pinned: false
---


# Protein pI Predictor

![header](./header.png)

Predict protein isoelectric points (pI) using a state-of-the-art ensemble model that combines a bidirectional LSTM network and a globally-attentive RoPE-based transformer.

---

## 🔬 What It Does
- Accepts a single protein sequence, a FASTA file, or a CSV file of sequences.
- Computes pI as the sum of a baseline Henderson–Hasselbalch estimate and a learned correction from two deep learning models.

---

## 🚀 Try It Live
Hosted on Hugging Face Spaces:  
👉 https://huggingface.co/spaces/YOUR_USERNAME/YOUR_APP_NAME

---

## 🖥️ Run It Locally

```bash
git clone https://github.com/YOUR_USERNAME/pI-predictor.git
cd pI-predictor && pip install -r requirements.txt && python app.py
