"""Operational logging setup. Master prompt section 37: log lifecycle events,
never secrets, never raw document content.
"""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger("legal_research_app")
    root.setLevel(level.upper())
    if root.handlers:
        return  # already configured (e.g. re-imported in tests)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"legal_research_app.{name}")
