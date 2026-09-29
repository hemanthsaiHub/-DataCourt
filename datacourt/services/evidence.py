"""Deterministic evidence engine. Every fact the AI lawyers may use is computed here with pandas, not by the LLM.

Each evidence item has an id (E1, E2, ...), a plain-English text, and the numbers that text contains. Later, the
verifier checks that every number an AI lawyer quotes really appears in the evidence it cites.
"""
from __future__ import annotations

import itertools
import re

import numpy as np
import pandas as pd

from .data_loader import to_records

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_EVID_ID = re.compile(r"\bE\d+\b")
_NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*\.?\d*")


def fmt(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:,.1f}"


def pct(x: float) -> str:
    return f"{x:+.1f}%"


def extract_numbers(text: str) -> list[tuple[float, bool]]:
    """Numbers in a sentence as (value, is_percent). Dates and evidence ids are ignored."""
    text = _EVID_ID.sub(" ", _ISO_DATE.sub(" ", text))
    out = []
    for m in _NUM.finditer(text):
        raw = m.group().replace(",", "").rstrip(".")
        try:
            val = float(raw)
        except ValueError:
            continue
        out.append((val, text[m.end() : m.end() + 1] == "%"))
    return out


def describe(df: pd.DataFrame) -> dict:
    """Column roles used by the UI to pre-fill the motion form."""
    date, numeric, cats = [], [], []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            date.append(c)
        elif pd.api.types.is_bool_dtype(s):
            cats.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            if not (pd.api.types.is_integer_dtype(s) and s.nunique() == len(s)):
                numeric.append(c)
        elif 2 <= s.nunique() <= 60 and s.nunique() / max(len(s), 1) < 0.5:
            cats.append(c)
    span = None
    if date:
        span = {"min": str(df[date[0]].min())[:10], "max": str(df[date[0]].max())[:10]}
    return {"rows": len(df), "date": date, "numeric": numeric, "categorical": cats, "span": span}


class _Ev:
    def __init__(self):
        self.items: list[dict] = []

    def add(self, kind: str, title: str, text: str, data: dict | None = None) -> str:
        eid = f"E{len(self.items) + 1}"
        self.items.append({"id": eid, "kind": kind, "title": title, "text": text,
                           "numbers": [v for v, _ in extract_numbers(text)], "data": data or {}})
        return eid


def _period_change(recent: pd.DataFrame, prior: pd.DataFrame, metric: str) -> tuple[float, float, float, float]:
    r, p = float(recent[metric].sum()), float(prior[metric].sum())
    return p, r, r - p, ((r - p) / abs(p) * 100 if p else float("nan"))


def build_evidence(df: pd.DataFrame, date_col: str, metric_col: str, dims: list[str] | None = None,
                   period_days: int | None = None) -> dict:
    for c in (date_col, metric_col):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found.")
    info = describe(df)
    dims = [d for d in (dims or info["categorical"]) if d in df.columns and d not in (date_col, metric_col)][:4]
    if not dims:
        raise ValueError("No usable category columns (like region or product) were found to explain changes.")

    d = df[[date_col, metric_col] + dims].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[metric_col] = pd.to_numeric(d[metric_col], errors="coerce")
    d = d.dropna(subset=[date_col, metric_col])
    if len(d) < 50:
        raise ValueError("Need at least 50 valid rows to build a case.")
    end, start = d[date_col].max().normalize(), d[date_col].min().normalize()
    span = (end - start).days + 1
    if span < 28:
        raise ValueError("Need at least 4 weeks of dates to compare two periods.")
    p = int(period_days) if period_days else int(np.clip(span // 4, 7, 90))
    p = max(7, min(p, span // 2))

    recent = d[d[date_col] > end - pd.Timedelta(days=p)]
    prior = d[(d[date_col] > end - pd.Timedelta(days=2 * p)) & (d[date_col] <= end - pd.Timedelta(days=p))]
    if len(recent) < 10 or len(prior) < 10:
        raise ValueError("One of the two comparison periods has too few rows. Try a different period length.")
    r_from = (end - pd.Timedelta(days=p - 1)).strftime("%Y-%m-%d")
    p_from = (end - pd.Timedelta(days=2 * p - 1)).strftime("%Y-%m-%d")
    p_to = (end - pd.Timedelta(days=p)).strftime("%Y-%m-%d")

    ev = _Ev()
    P, R, delta, chg = _period_change(recent, prior, metric_col)
    ev.add("headline", "Overall change",
           f"Total {metric_col} was {fmt(R)} in the recent {p} days ({r_from} to {end:%Y-%m-%d}) versus {fmt(P)} in the "
           f"{p} days before ({p_from} to {p_to}): a change of {fmt(delta)} ({pct(chg)}).",
           {"prior": P, "recent": R, "delta": delta, "pct": chg})

    weekly = d.set_index(date_col)[metric_col].resample("W").sum()
    if len(weekly) > 6:
        weekly = weekly.iloc[1:-1]
    if len(weekly) >= 6 and weekly.mean():
        slope = np.polyfit(np.arange(len(weekly)), weekly.values, 1)[0]
        slope_pct = slope / weekly.mean() * 100
        cv = weekly.std() / weekly.mean() * 100
        ev.add("trend", "Long-run trend",
               f"Across {len(weekly)} weeks, weekly {metric_col} trends {'up' if slope > 0 else 'down'} by about "
               f"{abs(slope_pct):.1f}% of its average per week, and weekly totals swing by roughly {cv:.1f}% around the average.",
               {"slope_pct_per_week": slope_pct, "weekly_cv_pct": cv})

    ev.add("data", "Sample size",
           f"The recent window has {len(recent):,} rows and the prior window has {len(prior):,} rows across "
           f"{', '.join(dims)}.")

    movers = []  # (delta, label, dim-or-pair, data)
    for dim in dims:
        g = pd.DataFrame({"prior": prior.groupby(dim)[metric_col].sum(), "recent": recent.groupby(dim)[metric_col].sum()}).fillna(0.0)
        g["delta"] = g["recent"] - g["prior"]
        gross = g["delta"].abs().sum() or 1.0
        for seg, row in g.sort_values("delta").iterrows():
            movers.append((row["delta"], f"{dim} = {seg}", dim, {"seg": seg, "prior": row["prior"], "recent": row["recent"], "gross": gross}))
        falls, rises = g[g["delta"] < 0].sort_values("delta").head(2), g[g["delta"] > 0].sort_values("delta", ascending=False).head(1)
        for seg, row in pd.concat([falls, rises]).iterrows():
            pc = (row["delta"] / abs(row["prior"]) * 100) if row["prior"] else float("nan")
            word = "fell" if row["delta"] < 0 else "rose"
            ev.add("driver", f"{dim}: {seg}",
                   f"In {dim}, '{seg}' {word} from {fmt(row['prior'])} to {fmt(row['recent'])} ({pct(pc)}), which is "
                   f"{abs(row['delta']) / gross * 100:.1f}% of all movement between {dim} groups.",
                   {"dim": dim, "seg": seg, "delta": float(row["delta"])})

    # Combination drivers across pairs of dimensions: finds e.g. "West + Furniture" hidden behind two single-dimension views.
    combos = []
    for a, b in itertools.combinations(dims, 2):
        g = pd.DataFrame({"prior": prior.groupby([a, b])[metric_col].sum(), "recent": recent.groupby([a, b])[metric_col].sum()}).fillna(0.0)
        g["delta"] = g["recent"] - g["prior"]
        for (sa, sb), row in g.iterrows():
            combos.append((row["delta"], a, sa, b, sb, row["prior"], row["recent"]))
    combos.sort(key=lambda t: t[0])
    picks = [c for c in combos[:2] if c[0] < 0] + [c for c in combos[::-1][:1] if c[0] > 0]
    for dv, a, sa, b, sb, pr, rc in picks:
        pc = (dv / abs(pr) * 100) if pr else float("nan")
        ev.add("combo", f"{sa} + {sb}",
               f"The combination {a} = '{sa}' and {b} = '{sb}' {'fell' if dv < 0 else 'rose'} from {fmt(pr)} to {fmt(rc)} "
               f"({pct(pc)}), a swing of {fmt(dv)} that equals {abs(dv) / abs(delta) * 100 if delta else 0:.0f}% of the overall change.",
               {"dims": [a, b], "segs": [str(sa), str(sb)], "delta": float(dv)})

    # Counterfactual: what if the biggest faller (or riser) had not moved?
    for label, pick in (("faller", min(movers, key=lambda m: m[0]) if movers else None),
                        ("riser", max(movers, key=lambda m: m[0]) if movers else None)):
        if not pick or pick[0] == 0 or (label == "faller" and pick[0] > 0) or (label == "riser" and pick[0] < 0):
            continue
        dim, seg = pick[2], pick[3]["seg"]
        _, _, d2, c2 = _period_change(recent[recent[dim] != seg], prior[prior[dim] != seg], metric_col)
        ev.add("counterfactual", f"Without {seg}",
               f"Excluding {dim} = '{seg}' (the biggest single-group {label}), the overall change would be {pct(c2)} instead of {pct(chg)}.",
               {"without": seg, "pct_without": c2})

    for dim in dims:
        share = recent.groupby(dim)[metric_col].sum().sort_values(ascending=False)
        if share.sum() > 0:
            cum = share.cumsum() / share.sum()
            k = int((cum < 0.8).sum() + 1)
            ev.add("concentration", f"{dim} concentration",
                   f"In {dim}, '{share.index[0]}' provides {share.iloc[0] / share.sum() * 100:.1f}% of recent {metric_col}; "
                   f"{k} of {len(share)} groups make up 80% of it.")

    daily = recent.groupby(recent[date_col].dt.normalize())[metric_col].sum()
    if len(daily) >= 7 and daily.sum() > 0:
        med = daily.median()
        mad = (daily - med).abs().median() or daily.std() or 1.0
        outliers = int((((daily - med).abs() * 0.6745 / mad) > 3.5).sum())
        top_day = daily.idxmax()
        ev.add("risk", "Single-day dependence",
               f"The single biggest day ({top_day:%Y-%m-%d}) provides {daily.max() / daily.sum() * 100:.1f}% of recent {metric_col}, "
               f"and {outliers} of {len(daily)} recent days are statistical outliers.")

    return {"metric": metric_col, "date_col": date_col, "dims": dims, "period_days": p,
            "windows": {"recent": [r_from, f"{end:%Y-%m-%d}"], "prior": [p_from, p_to]},
            "headline": {"prior": P, "recent": R, "delta": delta, "pct": chg}, "items": ev.items}


def format_for_prompt(items: list[dict]) -> str:
    return "\n".join(f"{i['id']} ({i['kind']}): {i['text']}" for i in items)
