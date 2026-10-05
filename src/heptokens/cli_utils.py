"""Shared CLI utilities."""

import sys
from functools import wraps
from typing import Callable


def with_help(text: str) -> Callable:
    """Decorator that intercepts --help/-h before Hydra processes sys.argv.

    Hydra composes the full config when rendering --help, which fails when required
    config groups (e.g. datamodule) are not supplied. Wrapping with this decorator
    short-circuits that and prints a human-readable help message instead.

    Usage::

        @with_help(_HELP)
        @hydra.main(version_base=None, config_path="...", config_name="...")
        def main(cfg: DictConfig) -> None:
            ...
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if "--help" in sys.argv or "-h" in sys.argv:
                print(text)
                sys.exit(0)
            return fn(*args, **kwargs)

        return wrapper

    return decorator
