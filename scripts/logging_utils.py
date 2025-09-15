import logging

logger = logging.getLogger("custom_hires_fix")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[Custom Hires Fix] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

__all__ = ["logger"]
