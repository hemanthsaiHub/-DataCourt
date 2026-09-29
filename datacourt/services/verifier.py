"""Citation verifier: the guard against AI hallucination.

A claim is only allowed to influence the verdict if (a) every evidence id it cites exists and (b) every meaningful
number in the claim text matches a number in the evidence it cites (within rounding).
"""
from __future__ import annotations

from .evidence import extract_numbers


def _matches(x: float, allowed: list[float]) -> bool:
    ax = abs(x)
    integer = float(ax).is_integer()
    for a in allowed:
        a = abs(a)
        tol = 0.5 if integer else max(0.06, 0.006 * a)   # "12%" may stand for 12.3; "12.3" must be close to 12.3
        if abs(ax - a) <= tol:
            return True
    return False


def verify_claim(claim: dict, evidence_by_id: dict[str, dict]) -> dict:
    text = str(claim.get("text", "")).strip()
    ids = [str(i).strip().upper() for i in (claim.get("evidence_ids") or [])]
    valid = [i for i in ids if i in evidence_by_id]
    out = {"text": text, "evidence_ids": valid, "invalid_ids": [i for i in ids if i not in evidence_by_id],
           "checked": [], "status": "unsupported", "reason": ""}
    if not text:
        out["reason"] = "Empty claim."
        return out
    if not valid:
        out["reason"] = "Cites no valid evidence."
        return out

    allowed = [n for i in valid for n in evidence_by_id[i]["numbers"]]
    nums = [(v, is_pct) for v, is_pct in extract_numbers(text) if is_pct or not float(abs(v)).is_integer() or abs(v) >= 10]
    hits = 0
    for v, is_pct in nums:
        ok = _matches(v, allowed)
        hits += ok
        out["checked"].append({"value": v, "ok": ok})
    if not nums or hits == len(nums):
        out["status"] = "verified"
    elif hits:
        out["status"] = "partial"
        out["reason"] = "Some numbers are not in the cited evidence."
    else:
        out["reason"] = "Numbers do not match the cited evidence."
    return out


def verify_claims(claims: list[dict], evidence: list[dict]) -> list[dict]:
    by_id = {e["id"]: e for e in evidence}
    return [verify_claim(c, by_id) for c in claims if isinstance(c, dict)]


def grounding(claims: list[dict]) -> float:
    return round(sum(c["status"] == "verified" for c in claims) / len(claims), 3) if claims else 0.0
