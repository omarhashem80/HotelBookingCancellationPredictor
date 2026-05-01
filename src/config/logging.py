import os
import sys
from loguru import logger


def configure_logging(level: str | int | None = None) -> None:
    """Clean, consistent, production-ready logging."""

    resolved_level = level or os.getenv("LOG_LEVEL", "INFO")

    logger.remove()

    # ─────────────────────────────────────────────
    # 🎨 Console (colored + structured + consistent)
    # ─────────────────────────────────────────────
    logger.add(
        sys.stderr,
        level=resolved_level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name: <25}</cyan> | "
            "<cyan>{function: <15}</cyan>:<cyan>{line: >4}</cyan> | "
            "<level>{message}</level>"
        ),
    )

    # ─────────────────────────────────────────────
    # 💾 File (clean + consistent)
    # ─────────────────────────────────────────────
    log_path = os.getenv("LOG_FILE", "logs/app.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger.add(
        log_path,
        level=resolved_level,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name: <25} | "
            "{function: <15}:{line: >4} | "
            "{message}"
        ),
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )