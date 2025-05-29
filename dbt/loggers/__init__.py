import logging

from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
