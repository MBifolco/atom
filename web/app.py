"""Compatibility wrapper for the Atom Combat web application."""

from apps.web.app import app, main

__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
