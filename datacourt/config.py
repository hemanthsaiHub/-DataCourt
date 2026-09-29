"""Central configuration. Values come from environment variables (.env locally, Project Settings on Vercel)."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # python-dotenv is optional
    pass

# Vercel serverless functions accept request bodies up to ~4.5 MB, so keep the default below that.
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "4"))
MAX_ROWS = int(os.getenv("MAX_ROWS", "200000"))
