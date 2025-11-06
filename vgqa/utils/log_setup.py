import logging
import os
import sys


def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
    """Create a logger that logs to stdout and optionally to a file.

    Avoids duplicate handlers on repeated calls.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if distributed_rank > 0:
        return logger

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")

        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(save_dir, filename))
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


