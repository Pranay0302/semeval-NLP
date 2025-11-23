"""
Track A baseline system.

We use a naive prompt for chatGPT.
"""

import random
from enum import Enum
from openai import OpenAI
import pandas as pd
from pydantic import BaseModel

from cli import build_parser

import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import tensorflow_hub as hub


class ResponseEnum(str, Enum):
    A = "A"
    B = "B"


class SimilarityPrediction(BaseModel):
    explanation: str
    closer: ResponseEnum


def embeddings(model_name, texts, embedding_dim=None):

    # OpenAI embedding models
    if model_name in ["text-embedding-3-small", "text-embedding-3-large"]:
        client = OpenAI()

        resp = client.embeddings.create(
            model=model_name,
            input=texts,
            dimensions=embedding_dim  # None = model default
        )
        return np.array([e.embedding for e in resp.data])

    #Sentence-BERT
    if model_name.startswith("sbert:"):
        real_name = model_name.split("sbert:")[1]
        model = SentenceTransformer(real_name)
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    # Universal Sentence Encoder
    if model_name == "use":
        use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        # USE outputs TF tensors → convert to numpy
        return use_model(texts).numpy()

    raise ValueError(f"Unknown embedding model {model_name}")


def openai_func(row, client):
    """
    Uses the OpenAI API to determine which of two stories (A or B) is more narratively similar to an anchor story.

    Returns:
        bool: True if story A is predicted to be more similar to the anchor than story B; False otherwise.
    """
    anchor, text_a, text_b = row["anchor_text"], row["text_a"], row["text_b"]
    completion = client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role":
                "system",
                "content":
                "You are an expert on stories and narratives. Tell us which of two stories is narratively similar to the anchor story.",
            },
            {
                "role": "user",
                "content": f"Anchor story: {anchor}\n\nStory A: {text_a}\n\nStory B: {text_b}",
            },
        ],
        response_format=SimilarityPrediction,
    )
    return completion.choices[0].message.parsed == ResponseEnum.A


def cosine(df, model_name, embedding_dim):
    anchor_embeddings = embeddings(model_name, df["anchor_text"].tolist(), embedding_dim)
    text_a_embeddings = embeddings(model_name, df["text_a"].tolist(), embedding_dim)
    text_b_embeddings = embeddings(model_name, df["text_b"].tolist(), embedding_dim)

    anchor_norm = anchor_embeddings / np.linalg.norm(anchor_embeddings, axis=1, keepdims=True)
    a_norm = text_a_embeddings / np.linalg.norm(text_a_embeddings, axis=1, keepdims=True)
    b_norm = text_b_embeddings / np.linalg.norm(text_b_embeddings, axis=1, keepdims=True)

    # True cosine similarity = dot product of normalized vectors
    sim_a = np.sum(anchor_norm * a_norm, axis=1)
    sim_b = np.sum(anchor_norm * b_norm, axis=1)

    return sim_a > sim_b


def main():
    parser = build_parser("track_a")
    args = parser.parse_args()
    baseline = args.baseline
    df = pd.read_json(args.input, lines=True)

    match baseline:
        case "openai":
            client = OpenAI()
            df["predicted_text_a_is_closer"] = df.apply(lambda row: openai_func(row, client),
                                                        axis=1)
        case "random":
            df["predicted_text_a_is_closer"] = df.apply(lambda _: random.choice([True, False]),
                                                        axis=1)
        case "cosine":
            df["predicted_text_a_is_closer"] = cosine(df, args.model, args.embedding_dim)
        case _:
            print(f"Error: Unknown Baseline {baseline}")

    accuracy = (df["predicted_text_a_is_closer"] == df["text_a_is_closer"]).mean()
    print(f"Accuracy: {accuracy:.3f}")

    df["text_a_is_closer"] = df["predicted_text_a_is_closer"]
    del df["predicted_text_a_is_closer"]

    with open(args.output, "w") as f:
        f.write(df.to_json(orient='records', lines=True))


if __name__ == "__main__":
    main()
