"""Vercel entry point: exposes the FastAPI ASGI app (see vercel.json for the /api/* rewrite)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datacourt.main import app  # noqa: E402,F401
