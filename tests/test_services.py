"""Service-layer tests. Run with: pytest -q"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datacourt.services import evidence as ev_mod  # noqa: E402
from datacourt.services import verifier  # noqa: E402
from datacourt.services.data_loader import load_dataframe  # noqa: E402
from datacourt.services.hearing import run_hearing  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "sample_business.csv"


def _df():
    return load_dataframe(DATA.read_bytes(), "sample_business.csv")


def test_load_and_describe():
    df = _df()
    info = ev_mod.describe(df)
    assert info["rows"] > 5000
    assert "date" in info["date"][0] or info["date"]
    assert "region" in info["categorical"]


def test_build_evidence_finds_the_planted_story():
    df = _df()
    case = ev_mod.build_evidence(df, "date", "revenue", period_days=30)
    assert case["headline"]["pct"] is not None
    texts = " ".join(i["text"] for i in case["items"])
    # The West/Furniture collapse and East/Electronics ramp should surface somewhere in the evidence.
    assert "Furniture" in texts or "Electronics" in texts
    assert any(i["kind"] == "combo" for i in case["items"])
    assert any(i["kind"] == "counterfactual" for i in case["items"])


def test_evidence_numbers_are_extractable_and_self_consistent():
    df = _df()
    case = ev_mod.build_evidence(df, "date", "revenue", period_days=30)
    headline = next(i for i in case["items"] if i["kind"] == "headline")
    assert headline["numbers"], "headline evidence should contain parseable numbers"


def test_verifier_accepts_matching_numbers_and_rejects_invented_ones():
    evidence = [{"id": "E1", "text": "Revenue fell from 1,000 to 800 (-20.0%).", "numbers": [1000.0, 800.0, -20.0]}]
    good = verifier.verify_claim({"text": "Revenue dropped about 20%.", "evidence_ids": ["E1"]}, {"E1": evidence[0]})
    assert good["status"] == "verified"
    bad = verifier.verify_claim({"text": "Revenue dropped 87%.", "evidence_ids": ["E1"]}, {"E1": evidence[0]})
    assert bad["status"] == "unsupported"
    missing = verifier.verify_claim({"text": "Revenue dropped 20%.", "evidence_ids": ["E99"]}, {"E1": evidence[0]})
    assert missing["status"] == "unsupported"


def test_verifier_grounding_score():
    claims = [{"status": "verified"}, {"status": "verified"}, {"status": "unsupported"}]
    assert abs(verifier.grounding(claims) - 2 / 3) < 0.001


def test_hearing_runs_end_to_end_with_a_stub_llm():
    """No network in CI: a small stub model stands in for Gemini/Groq and must still produce a sound ruling."""
    df = _df()
    case_preview = ev_mod.build_evidence(df, "date", "revenue", period_days=30)
    ids = [i["id"] for i in case_preview["items"]]
    calls = {"n": 0}

    def stub_complete(prompt, system="", temperature=0.3, max_tokens=2048, json_mode=False):
        calls["n"] += 1
        if "JUDGE" in system or "judge" in system.lower():
            return ('{"verdict":"PROCEED WITH CONDITIONS","confidence":70,"summary":"Mixed evidence.",'
                    f'"deciding_evidence":{ids[:2]!r},"risks":["Some risk"],"conditions":["Monitor weekly"],'
                    '"what_would_change_my_mind":["A rebound next month"]}').replace("'", '"')
        # counsel round: cite the first evidence item with one of its real numbers
        first = case_preview["items"][0]
        num = first["numbers"][0] if first["numbers"] else 1
        return f'{{"claims":[{{"text":"The metric changed by about {num}.","evidence_ids":["{first["id"]}"]}}]}}'

    result = run_hearing(df, "Should we exit Furniture in the West region?", "date", "revenue",
                         period_days=30, complete=stub_complete)
    assert result["verdict"]["ruling"] in {"PROCEED", "PROCEED WITH CONDITIONS", "DO NOT PROCEED"}
    assert 0 <= result["verdict"]["confidence"] <= 100
    assert result["verdict"]["confidence"] <= result["verdict"]["confidence_cap"]
    assert result["grounding"]["overall"] > 0
    assert calls["n"] == 5  # 2 sides x (opening + rebuttal) + 1 judge


def test_hearing_rejects_short_decision():
    df = _df()
    try:
        run_hearing(df, "hi", "date", "revenue")
        raise AssertionError("should have rejected a too-short decision")
    except ValueError:
        pass
