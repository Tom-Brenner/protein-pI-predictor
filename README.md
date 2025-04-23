---
title: Protein pI Predictor
emoji: 🧪
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
license: apache-2.0
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

## Performance

| Model | Outliers <br>(Peptide) | Outliers <br>(Protein) | RMSE <br>(Peptide) | RMSE <br>(Protein) |
|-------|----------------------:|----------------------:|-------------------:|-------------------:|
| **Ours**              |  874 | 249 | 0.225 | 0.87 |
| Brenner GNN 2022✝     | 1 638 | 251 | 0.271 | 0.87 |
| IPC2.Conv2D✝✝         | 2 691 |  –  | 0.222 |  –  |
| IPC2.svr.1✝✝          | 2 490 | 247 | 0.23  | 0.85 |

✝ Graph neural networks for prediction of protein isoelectric points; Brenner 2022  
✝✝ Prediction of isoelectric point and pKa dissociation constants; Kozlowski 2021, *IPC 2.0*, *Nucleic Acids Research* 49, W285–W292

---
## 🚀 Try It Live
Hosted on Hugging Face Spaces:  
👉 https://huggingface.co/spaces/Tom-Brenner/protein-pI-predictor

---

## 🖥️ Run It Locally

```bash
git clone https://github.com/YOUR_USERNAME/pI-predictor.git
cd pI-predictor && pip install -r requirements.txt && python app.py
