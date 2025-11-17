import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('BAAI/bge-base-en-v1.5')
df = pd.read_json("data/dev_track_a.jsonl", lines=True)

anchor_embeddings = model.encode(df["anchor_text"].tolist(), convert_to_tensor=True, show_progress_bar=True)
text_a_embeddings = model.encode(df["text_a"].tolist(), convert_to_tensor=True, show_progress_bar=True)
text_b_embeddings = model.encode(df["text_b"].tolist(), convert_to_tensor=True, show_progress_bar=True)

sim_a = torch.nn.functional.cosine_similarity(anchor_embeddings, text_a_embeddings, dim=1).cpu().numpy()
sim_b = torch.nn.functional.cosine_similarity(anchor_embeddings, text_b_embeddings, dim=1).cpu().numpy()

df["predicted_text_a_is_closer"] = sim_a > sim_b

if "text_a_is_closer" in df.columns:
    accuracy = (df["predicted_text_a_is_closer"] == df["text_a_is_closer"]).mean()
    print(f"Accuracy: {accuracy:.3f}")

df["text_a_is_closer"] = df["predicted_text_a_is_closer"]
del df["predicted_text_a_is_closer"]

with open("output/track_a.jsonl", "w") as f:
    f.write(df.to_json(orient='records', lines=True))
