"""
Baseline system for Track B.

Notice that we embed the texts from the Track B file but perform the actual evaluation using labels from Track A.
"""

import sys
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np
import logging, os
from cli import build_parser
from log import setup_logging


def valid_file(path, required_col):
    try:
        df = pd.read_json(path, lines=True)
        missing = [c for c in required_col if c not in df.columns]
        if missing:
            logging.error(f"Input file missing required columns: {missing}")
            sys.exit(4)  # Exit code 4: missing columns
        return df
    except ValueError as e:
        logging.error(f"Failed to parse JSON: {e}")
        sys.exit(3)  # Exit code 3: JSON parse error


def generate_story_embedding(story_text, baseline, model=None):
    match baseline:
        case "random":
            return torch.rand(512)
        case "sbert":
            if model is None:
                raise ValueError("Model must be provided for sbert baseline")
            # Encode single story - returns numpy array by default
            return model.encode(story_text, show_progress_bar=False)
        case _:
            raise ValueError(f"Unknown baseline: {baseline}")


def evaluate(labeled_data_path, embedding_lookup):
    df = valid_file(labeled_data_path, ["anchor_text", "text_a", "text_b", "text_a_is_closer"])

    # Map texts to embeddings
    df["anchor_embedding"] = df["anchor_text"].map(embedding_lookup)
    df["a_embedding"] = df["text_a"].map(embedding_lookup)
    df["b_embedding"] = df["text_b"].map(embedding_lookup)

    # Look up cosine similarities
    df["sim_a"] = df.apply(lambda row: cos_sim(row["anchor_embedding"], row["a_embedding"]), axis=1)
    df["sim_b"] = df.apply(lambda row: cos_sim(row["anchor_embedding"], row["b_embedding"]), axis=1)

    # Predict and calculate accuracy
    df["predicted_text_a_is_closer"] = df["sim_a"] > df["sim_b"]
    accuracy = (df["predicted_text_a_is_closer"] == df["text_a_is_closer"]).mean()
    return accuracy


def main():
    parser = build_parser("track_b")
    args = parser.parse_args()
    baseline = args.baseline
    if not os.path.exists(args.input):
        logging.error(f"Input file does not exist: {args.input}")
        sys.exit(2)  # Exit code 2: file not found
    df = valid_file(args.input, ["text"])
    _ = setup_logging(verbose=args.verbose, quiet=args.quiet, log_file=args.log_file)
    logging.info("Starting Track B...")
    logging.info(f"Baseline: {args.baseline}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Embedding Dim: {args.embedding_dim}")
    logging.info(f"Loaded {len(df)} rows from {args.input}")

    # Generate embeddings for each story individually
    embedding_lookup = {}
    model = None
    
    if baseline == "sbert":
        real_name = args.model.split("sbert:")[1]
        model = SentenceTransformer(real_name)
    
    logging.info("Generating embeddings for each story...")
    for idx, story_text in enumerate(df["text"]):
        embedding = generate_story_embedding(story_text, baseline, model)
        embedding_lookup[story_text] = embedding
        if (idx + 1) % 100 == 0:
            logging.info(f"Processed {idx + 1}/{len(df)} stories")
    
    # Convert to numpy array for saving (maintaining original format)
    # Handle both torch tensors (random) and numpy arrays (sbert)
    embeddings_list = []
    for text in df["text"]:
        emb = embedding_lookup[text]
        if isinstance(emb, torch.Tensor):
            embeddings_list.append(emb.numpy())
        else:
            embeddings_list.append(emb)
    embeddings = np.array(embeddings_list)
    accuracy = evaluate("data/dev_track_a.jsonl", embedding_lookup)
    logging.info(f"Accuracy: {accuracy:.3f}")

    np.save(args.output, embeddings)
    logging.info(f"Saved result to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(f"Unexpected error: {e}")
        sys.exit(1)
