import argparse
import logging
from logging.handlers import RotatingFileHandler


def build_parser(track: str):
    parser = argparse.ArgumentParser(
        description=f"Argument parser for {track.upper()}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--baseline",
        type=str,
        choices=["random", "openai", "cosine"] if track == "track_a" else ["random", "sbert"],
        default="random",
        help="Baseline method to use.")

    parser.add_argument("--input",
                        type=str,
                        default=f"data/dev_{track}.jsonl",
                        help="Path to input file.")

    parser.add_argument(
        "--output",
        type=str,
        default=f"output/{track}.jsonl" if track == "track_a" else f"output/{track}.npy",
        help="Path to output file.")

    parser.add_argument("--model",
                        type=str,
                        default="sbert:all-mpnet-base-v2",
                        help="Embedding model name.")

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help=
        "Optional dimension override for OpenAI embedding models(track A). Track B must be 10–8192."
    )

    parser.add_argument("--verbose",
                        "--v",
                        action="store_true",
                        help="Enable verbose logging (DEBUG level)")

    parser.add_argument("--quiet",
                        "--q",
                        action="store_true",
                        help="Quiet mode (warnings and errors only)")

    parser.add_argument("--log-file",
                        type=str,
                        default=f"{track}.log",
                        help="Path to log file (optional)")

    return parser
