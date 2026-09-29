"""The hearing: two AI counsels argue a business decision from the evidence, a verifier checks every citation,
and a judge rules using only verified claims. Confidence is capped by how well-grounded the arguments were."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import pandas as pd

from . import llm
from .evidence import build_evidence, format_for_prompt
from .verifier import grounding, verify_claims

Complete = Callable[..., str]

COUNSEL_SYSTEM = (
    "You are {side} counsel in a business-decision hearing. Argue ONLY from the numbered EVIDENCE. "
    "Rules: every claim must cite one or more evidence ids (like E3); any number you write must be copied exactly from the "
    "evidence you cite; never invent facts, causes or numbers; each claim is one sentence of at most 35 words. "
    'Reply with JSON only: {{"claims":[{{"text":"...","evidence_ids":["E1"]}}]}}'
)
JUDGE_SYSTEM = (
    "You are an impartial judge for a business-decision hearing. Decide using only the evidence and the VERIFIED claims. "
    "Be cautious: if evidence is thin or conflicting, choose PROCEED WITH CONDITIONS or DO NOT PROCEED. "
    'Reply with JSON only: {"verdict":"PROCEED"|"PROCEED WITH CONDITIONS"|"DO NOT PROCEED","confidence":0-100,'
    '"summary":"2-3 sentences","deciding_evidence":["E1"],"risks":["..."],"conditions":["..."],"what_would_change_my_mind":["..."]}'
)
SIDES = {"for": "FOR", "against": "AGAINST"}


def _ask_json(complete: Complete, prompt: str, system: str, tries: int = 2) -> dict:
    last: Exception | None = None
    for _ in range(tries):
        try:
            return llm.extract_json(complete(prompt, system=system, json_mode=True))
        except ValueError as exc:      # bad JSON: retry once
            last = exc
    raise ValueError(f"The AI returned an unreadable answer ({last}). Please run the hearing again.")


def _claims_text(claims: list[dict]) -> str:
    return "\n".join(f"- {c['text']} [{', '.join(c['evidence_ids'])}]" for c in claims) or "- (none)"


def _round(complete: Complete, decision: str, evidence: list[dict], prompts: dict[str, str]) -> dict[str, list[dict]]:
    def one(side: str) -> list[dict]:
        data = _ask_json(complete, prompts[side], COUNSEL_SYSTEM.format(side=SIDES[side]))
        return verify_claims(data.get("claims", []), evidence)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = {s: pool.submit(one, s) for s in ("for", "against")}
        return {s: f.result() for s, f in futs.items()}


def run_hearing(df: pd.DataFrame, decision: str, date_col: str, metric_col: str, dims: list[str] | None = None,
                period_days: int | None = None, complete: Complete = llm.complete) -> dict:
    decision = (decision or "").strip()
    if len(decision) < 8:
        raise ValueError("Describe the decision you want to test (for example: 'Should we drop Furniture in the West region?').")
    case = build_evidence(df, date_col, metric_col, dims, period_days)
    ev = case["items"]
    ev_text = format_for_prompt(ev)
    head = f"DECISION UNDER REVIEW: {decision}\n\nEVIDENCE:\n{ev_text}\n\n"

    opening = _round(complete, decision, ev, {
        "for": head + "Argue FOR taking this decision. Give 3 or 4 claims.",
        "against": head + "Argue AGAINST taking this decision. Give 3 or 4 claims."})

    ok = {s: [c for c in opening[s] if c["status"] == "verified"] for s in opening}
    rebuttal = _round(complete, decision, ev, {
        "for": head + f"The opposing counsel argued (verified claims):\n{_claims_text(ok['against'])}\n\nRebut these and reinforce your side. Give 2 or 3 claims.",
        "against": head + f"The opposing counsel argued (verified claims):\n{_claims_text(ok['for'])}\n\nRebut these and reinforce your side. Give 2 or 3 claims."})

    rounds = [{"name": "Opening statements", **opening}, {"name": "Rebuttals", **rebuttal}]
    all_for, all_against = opening["for"] + rebuttal["for"], opening["against"] + rebuttal["against"]
    g = {"for": grounding(all_for), "against": grounding(all_against)}
    g["overall"] = round((g["for"] + g["against"]) / 2, 3)

    verified = {s: [c for c in (all_for if s == "for" else all_against) if c["status"] == "verified"] for s in SIDES}
    if not verified["for"] and not verified["against"]:
        raise ValueError("None of the AI arguments could be verified against the data, so no ruling was made. Try again.")

    verdict = _ask_json(complete,
        head + f"VERIFIED CLAIMS FOR:\n{_claims_text(verified['for'])}\n\nVERIFIED CLAIMS AGAINST:\n{_claims_text(verified['against'])}\n\n"
        "Give your ruling. Do not cite numbers that are not in the evidence.", JUDGE_SYSTEM)

    allowed = {"PROCEED", "PROCEED WITH CONDITIONS", "DO NOT PROCEED"}
    v = str(verdict.get("verdict", "")).upper().strip()
    if v not in allowed:
        v = "PROCEED WITH CONDITIONS"
    try:
        raw_conf = max(0, min(100, int(float(verdict.get("confidence", 50)))))
    except (TypeError, ValueError):
        raw_conf = 50
    cap = round(40 + 60 * g["overall"])          # weak grounding => the court refuses to sound sure
    ids = {e["id"] for e in ev}
    listify = lambda k: [str(x) for x in (verdict.get(k) or [])][:5]
    return {
        "decision": decision, "case": {k: v_ for k, v_ in case.items() if k != "items"}, "evidence": ev,
        "rounds": rounds, "grounding": g,
        "verdict": {
            "ruling": v, "confidence": min(raw_conf, cap), "model_confidence": raw_conf, "confidence_cap": cap,
            "summary": str(verdict.get("summary", "")).strip(),
            "deciding_evidence": [i for i in (str(x).upper() for x in verdict.get("deciding_evidence") or []) if i in ids],
            "risks": listify("risks"), "conditions": listify("conditions"),
            "what_would_change_my_mind": listify("what_would_change_my_mind"),
        },
        "provider": llm.active_provider(),
    }
