import logging

from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
