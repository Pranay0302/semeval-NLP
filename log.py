# logging.py or log_utils.py
import logging
import sys
import time
import os
from tqdm import tqdm
import json
from logging.handlers import RotatingFileHandler


class TqdmLoggingHandler(logging.Handler):
    """Send logs to tqdm.write() instead of normal stdout to avoid breaking progress bars."""

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            pass


def setup_logging(verbose=False, quiet=False, log_file=None):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # root logger stores all events

    # Remove default handlers so we don’t duplicate logs
    logger.handlers = []

    # ───── Console Handler (tqdm-safe) ─────────────────────────────────

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    ch = TqdmLoggingHandler()
    if quiet:
        ch.setLevel(logging.WARNING)
    elif verbose:
        ch.setLevel(logging.DEBUG)
    else:
        ch.setLevel(logging.INFO)

    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        fh = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
