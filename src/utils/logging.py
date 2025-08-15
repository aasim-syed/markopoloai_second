import logging, os

def get_logger(name: str = __name__):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        fmt = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)