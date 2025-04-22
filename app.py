import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gradio as gr
from Bio import SeqIO
import pandas as pd

# ---------- Model + Charge Logic ----------
LN10 = math.log(10.0)
AA_List     = ['C','D','E','H','K','R','Y']
AA_List_Full= ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
IsAcid      = [True,True,True,False,False,False,True,True,False]
pKa_List    = [8.3,3.9,4.3,6.0,10.5,12.5,10.1,2.4,9.6]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiLSTMNet(nn.Module):
    def __init__(self, input_size=21, hidden_size=256, num_layers=4, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        forward_last = h_n[-2]
        backward_last = h_n[-1]
        h_cat = torch.cat([forward_last, backward_last], dim=1)
        return self.fc(h_cat).squeeze()

class GlobalRoPETransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, ff_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim, padding_idx=0)
        self.encoder_layers = nn.TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.transformer_encoder(x, src_key_padding_mask=(mask == 0))
        return self.fc(x.mean(dim=1))

def Calc_Charge(pH, is_acid, pKa, counts):
    charge, dqdPH = 0.0, 0.0
    for n, pk, acid in zip(counts, pKa, is_acid):
        delta = pk - pH
        ratio = 10.0 ** delta
        frac = ratio / (1.0 + ratio)
        charge += n * (-acid + frac)
        dqdPH -= n * LN10 * ratio / (1.0 + ratio) ** 2
    return charge, dqdPH

def pI_HH(counts, is_acid, pKa, pH_low=0.0, pH_high=14.5, tol=1e-6):
    q_low, _ = Calc_Charge(pH_low, is_acid, pKa, counts)
    q_high, _ = Calc_Charge(pH_high, is_acid, pKa, counts)
    if q_low * q_high > 0:
        return pH_low if abs(q_low) < abs(q_high) else pH_high
    x = (pH_low + pH_high) / 2.0
    for _ in range(200):
        q, dq = Calc_Charge(x, is_acid, pKa, counts)
        if abs(q) < tol:
            return x
        if q * q_low > 0:
            pH_low, q_low = x, q
        else:
            pH_high, q_high = x, q
        if abs(dq) > 1e-6:
            x_new = x - q / dq
            if pH_low < x_new < pH_high:
                x = x_new
                continue
        x = 0.5 * (pH_low + pH_high)
    raise RuntimeError("pI solver did not converge")

# ---------- Load Models ----------
bilstm = BiLSTMNet().to(device)
bilstm.load_state_dict(torch.load("best_bilstm.pt", map_location=device))
bilstm.eval()

rope = GlobalRoPETransformer(21, 256, 8, 6, 512, 0.15).to(device)
rope.load_state_dict(torch.load("best_rope.pt", map_location=device))
rope.eval()

# ---------- Prediction Logic ----------
def predict(sequence):
    sequence = sequence.upper().replace("*", "")
    if not sequence or not all(a in AA_List_Full for a in sequence):
        return "Invalid sequence"
    idx = torch.tensor([[AA_List_Full.index(a) for a in sequence]], dtype=torch.long)
    mask = (idx != 0).float()
    oh = F.one_hot(idx, num_classes=21).float()
    lengths = torch.tensor([len(sequence)])

    counts = [sequence.count(aa) for aa in AA_List] + [1, 1]
    baseline = pI_HH(counts, IsAcid, pKa_List)

    with torch.no_grad():
        p1 = rope(idx.to(device), mask.to(device)).item()
        p2 = bilstm(oh.to(device), lengths.to(device)).item()
    return round(baseline + (p1 + p2) / 2.0, 4)

# ---------- Upload Handlers ----------
def handle_fasta(file):
    results = []
    for record in SeqIO.parse(file.name, "fasta"):
        pI = predict(str(record.seq))
        results.append(f">{record.id}\nPredicted pI: {pI}")
    return "\n\n".join(results)

def handle_csv(file):
    df = pd.read_csv(file.name)
    if 'sequence' not in df.columns:
        return "CSV must contain a 'sequence' column."
    results = [f"Row {i}: {predict(seq)}" for i, seq in enumerate(df['sequence'])]
    return "\n".join(results)

# ---------- Gradio UI ----------
gr.Interface(
    fn=lambda input_text, fasta_file, csv_file: (
        handle_fasta(fasta_file) if fasta_file else
        handle_csv(csv_file) if csv_file else
        predict(input_text)
    ),
    inputs=[
        gr.Textbox(label="Protein Sequence (optional)"),
        gr.File(label="FASTA File (optional)", file_types=[".fasta", ".fa"]),
        gr.File(label="CSV File (optional)", file_types=[".csv"])
    ],
    outputs="text",
    title="Protein Isoelectric Point Predictor",
    description="Predict pI from a single sequence, a FASTA file, or a CSV file containing sequences."
).launch()
