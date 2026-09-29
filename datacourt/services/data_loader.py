"""Reading uploaded files into clean DataFrames."""
from __future__ import annotations

import io
import json
import re
import warnings

import pandas as pd

from ..config import MAX_ROWS

_DATE_LIKE = re.compile(r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")


def clean_column_names(cols) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for c in cols:
        name = re.sub(r"\W+", "_", str(c).strip().lower()).strip("_") or "col"
        if name[0].isdigit():
            name = "c_" + name
        n = seen.get(name, 0)
        seen[name] = n + 1
        out.append(name if n == 0 else f"{name}_{n}")
    return out


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        sample = s.dropna().astype(str).head(200)
        if sample.empty or sample.map(lambda v: bool(_DATE_LIKE.match(v))).mean() < 0.8:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(s, errors="coerce", format="mixed")
        if parsed.notna().mean() >= 0.9:
            df[col] = parsed
    return df


def load_dataframe(raw: bytes, filename: str) -> pd.DataFrame:
    name = (filename or "").lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(raw))
        else:
            df = pd.read_csv(io.BytesIO(raw), encoding_errors="replace")
            if df.shape[1] == 1:  # maybe ; or tab separated
                df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding_errors="replace")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not read '{filename}': {exc}") from exc
    if df.empty:
        raise ValueError("The file has no rows.")
    df = df.head(MAX_ROWS)
    df.columns = clean_column_names(df.columns)
    return parse_dates(df)


def to_records(df: pd.DataFrame, limit: int | None = None) -> list[dict]:
    """JSON-safe list of dicts (NaN -> null, datetimes -> ISO strings)."""
    if limit:
        df = df.head(limit)
    return json.loads(df.to_json(orient="records", date_format="iso", date_unit="s"))


def for_sql(df: pd.DataFrame) -> pd.DataFrame:
    """Datetimes become ISO text so SQLite's date functions (strftime, date) work."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            has_time = (out[c].dropna().dt.normalize() != out[c].dropna()).any()
            out[c] = out[c].dt.strftime("%Y-%m-%d %H:%M:%S" if has_time else "%Y-%m-%d")
    return out
