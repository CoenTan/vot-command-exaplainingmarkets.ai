"""Offline tests for the agent pipeline (v02 rubric): no API key spent, no network.

A fake OpenAI client returns canned JSON for each agent, so we can check the
arithmetic, the close-call / disagreement paths, and every fallback.
Run:  uv run pytest -q
"""

from __future__ import annotations

import json
import types

import pytest

import predict as P

STATS = {
    "implied_absolute_earnings_move": {"value": 0.051, "status": "ok"},
    "implied_earnings_volatility": {"value": 0.064, "status": "ok"},
}
BUNDLE = {"items": [
    {"id": "earnings-call-facts", "kind": "facts", "content": ["Revenue $96B.", "Q3 guide $108B."]},
    {"id": "earnings-preview", "kind": "text", "content": "Consensus revenue $92B, whisper $95B."},
    {"id": "option-implied-stats", "kind": "stats", "content": STATS},
]}

# short names for the dimensions used in tests
G, F, RV = "guidance_revision", "forward_operating_outlook", "revenue_growth"


def sheet(**pts) -> dict:
    return {"components": {k: {"score": v, "evidence": "x"} for k, v in pts.items()},
            "score": 0.99, "weighted_total": 999}  # deliberately wrong maths: Python must override


class FakeClient:
    """Answers each agent from a queue; records what was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.sent.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        msg = types.SimpleNamespace(content=reply if isinstance(reply, str) else json.dumps(reply))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("ALWAYS_CALL_PM", raising=False)
    monkeypatch.delenv("CLOSE_CALL_GAP", raising=False)
    monkeypatch.delenv("OPENAI_REASONING_EFFORT", raising=False)
    # Pipeline tests use simple arithmetic: every weight 1, so
    # score = 0.5 + (sum of dimension scores) / 20, and no Quant sizing.
    # The real v02 weights and sizing are tested on their own below.
    monkeypatch.setenv("COMPONENT_WEIGHTS", ",".join(f"{k}=1" for k in P.DIMENSIONS))
    monkeypatch.setenv("QUANT_POWER", "0")
    monkeypatch.setenv("QUANT_REF_MOVE", "0.06")
    monkeypatch.setenv("QUANT_PIVOT", "0.5")

    def install(*replies):
        client = FakeClient(replies)
        monkeypatch.setattr(P, "_client", client)
        return client
    return install


def run(bundle=BUNDLE):
    return P.run_agents(bundle=bundle, ticker="NVDA", event_type="EARNINGS_RELEASE")


# ---------- arithmetic ----------

def test_v02_has_17_dimensions_and_weights_sum_to_one():
    assert len(P.DIMENSIONS) == 17
    assert abs(sum(P.DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9
    assert P.DEFAULT_WEIGHTS[G] == 0.31 and P.DEFAULT_WEIGHTS[F] == 0.2


def test_every_dimension_is_in_every_scoring_prompt():
    for prompt in (P.ANALYST, P.SCRUTINIZER, P.PORTFOLIO_MANAGER):
        for d in P.DIMENSIONS:
            assert f'"{d}"' in prompt
    assert "## Boundaries" in P.RUBRIC and "## Worked TRAIN examples" in P.RUBRIC
    # v02.1: guidance is scored against the highest investor bar
    assert "AMENDMENT v02.1" in P.RUBRIC and "HIGHEST investor bar" in P.RUBRIC


def test_scores_are_read_and_clamped():
    pts = P.points_from_sheet({G: {"score": 17}, "new_risks": {"points": -4.5}, "gross_margin": "3"})
    assert pts[G] == 10 and pts["new_risks"] == -4.5 and pts["gross_margin"] == 3 and pts[RV] == 0


@pytest.mark.parametrize("total, expected", [
    (0, 0.50), (4, 0.70), (-4, 0.30), (30, 0.999), (-30, 0.001), (1.3, 0.565),
])
def test_score_from_points(total, expected, monkeypatch):
    monkeypatch.setenv("COMPONENT_WEIGHTS", ",".join(f"{k}=1" for k in P.DIMENSIONS))
    assert P.score_from_points({G: total})["score"] == expected


def test_component_weights_from_env(monkeypatch):
    monkeypatch.setenv("COMPONENT_WEIGHTS", f"{G}=0.5, bogus=9, new_risks=abc, gross_margin=99")
    w = P.component_weights()
    assert w[G] == 0.5
    assert w["new_risks"] == P.DEFAULT_WEIGHTS["new_risks"]
    assert w["gross_margin"] == P.DEFAULT_WEIGHTS["gross_margin"]


def test_v02_default_weights(monkeypatch):
    monkeypatch.delenv("COMPONENT_WEIGHTS", raising=False)
    assert P.component_weights() == P.DEFAULT_WEIGHTS
    # guidance_revision +10 alone: 0.31 x 10 = 3.1 -> 0.5 + 3.1/20 = 0.655
    assert P.score_from_points({G: 10}) == {"weighted_total": 3.1, "total_s": 3.1, "score": 0.655}
    # +5 guidance, +5 forward outlook: 1.55 + 1.0 = 2.55 -> 0.6275
    assert P.score_from_points({G: 5, F: 5})["score"] == 0.6275


def test_quant_defaults(monkeypatch):
    for k in ("QUANT_POWER", "QUANT_REF_MOVE", "QUANT_PIVOT"):
        monkeypatch.delenv(k, raising=False)
    assert P.apply_quant(0.70, 0.08) == 0.70       # median move: unchanged
    assert P.apply_quant(0.70, 0.32) == 0.7828     # 4x move -> distance x 4^0.25 = 1.414
    assert P.apply_quant(0.50, 0.32) == 0.50       # the pivot never moves
    assert P.apply_quant(0.70, None) == 0.70


@pytest.mark.parametrize("score, move, expected", [
    (0.70, 0.06, 0.70),    # normal move: unchanged
    (0.70, 0.24, 0.90),    # 4x move -> distance x2: 0.5 + 0.2*2
    (0.70, 0.015, 0.60),   # quarter move -> distance x0.5
    (0.30, 0.24, 0.10),
    (0.70, None, 0.70),    # no options data: unchanged
    (0.50, 0.24, 0.50),    # the pivot never moves
    (0.90, 0.24, 0.999),   # capped
])
def test_quant_sizing(score, move, expected, monkeypatch):
    monkeypatch.setenv("QUANT_POWER", "0.5")
    monkeypatch.setenv("QUANT_REF_MOVE", "0.06")
    monkeypatch.setenv("QUANT_PIVOT", "0.5")
    assert P.apply_quant(score, move) == expected


def test_quant_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("QUANT_POWER", "0")
    assert P.apply_quant(0.70, 0.24) == 0.70


def test_quant_view_text():
    text = P.quant_view(BUNDLE)
    assert "Expected absolute move: 5.1%" in text and "No proven link" not in text  # BUNDLE has no skew
    assert "no usable options data" in P.quant_view({"items": []})


@pytest.mark.parametrize("a, r, expected", [
    (0.65, 0.60, 0.625), (0.40, 0.35, 0.375), (0.60, 0.70, 0.65),
])
def test_average_rule(a, r, expected):
    assert P.average_rule(a, r) == expected


# ---------- pipeline paths ----------

def test_close_call_averages_without_calling_pm(fake):
    client = fake(sheet(**{G: 4, RV: 2}), sheet(**{G: 4, RV: 1}))
    assert run() == 0.775         # analyst 6 -> 0.80, scrutinizer 5 -> 0.75, gap 0.05 -> average
    assert len(client.sent) == 2  # PM skipped


def test_close_call_gap_can_be_set(fake, monkeypatch):
    monkeypatch.setenv("CLOSE_CALL_GAP", "0.01")
    client = fake(sheet(**{G: 4, RV: 2}), sheet(**{G: 4, RV: 1}), {"final_score": 0.77})
    assert run() == 0.77
    assert len(client.sent) == 3  # gap 0.05 > 0.01 -> PM called


def test_disagreement_calls_pm(fake):
    client = fake(sheet(**{G: 6}), scrut({G: "FACT"}, **{G: -2}), {"final_score": 0.58})
    assert run() == 0.58          # no settled sheet: the PM's number, kept as is
    assert len(client.sent) == 3
    pm_msg = client.sent[2]["messages"][1]["content"]
    assert "analyst_score=0.8" in pm_msg and "scrutinizer_score=0.4" in pm_msg


def test_pm_settled_scores_are_rescored_in_python(fake):
    fake(sheet(**{G: 6}), scrut({G: "FACT"}, **{G: -2}),
         {"final_score": 0.80, "settled_components": {G: 2, RV: 1}})
    trace = {}
    final = P.run_agents(bundle=BUNDLE, ticker="NVDA", event_type="EARNINGS_RELEASE", trace=trace)
    assert trace["pm_score"] == 0.65   # total 3 -> 0.65, not the model's 0.80
    assert final == 0.65
    assert trace["pm_points"][G] == 2


def test_models_maths_is_overwritten_before_the_next_agent_sees_it(fake):
    client = fake(sheet(**{G: 2}), sheet(**{G: 2}))
    run()
    sent_to_scrutinizer = client.sent[1]["messages"][1]["content"]
    assert '"score": 0.6' in sent_to_scrutinizer and '"weighted_total": 2.0' in sent_to_scrutinizer


def test_json_inside_code_fences_is_accepted(fake):
    fake("```json\n" + json.dumps(sheet(**{RV: 4})) + "\n```", sheet(**{RV: 4}))
    assert run() == 0.70


def test_rubric_is_sent_with_every_role(fake):
    client = fake(sheet(), sheet())
    run()
    for call in client.sent:
        system = call["messages"][0]["content"]
        assert "UNIVERSAL SCORING RUBRIC (v02.1)" in system
        assert call["response_format"] == {"type": "json_object"}


# ---------- fallbacks ----------

def test_no_key_returns_half(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert run() == 0.5


def test_analyst_failure_returns_half(fake):
    fake(RuntimeError("down"), RuntimeError("still down"))
    assert run() == 0.5


def test_analyst_is_retried_once_at_default_effort(fake):
    client = fake(RuntimeError("too slow"), sheet(**{G: 2}), sheet(**{G: 2}))
    trace = {}
    assert P.run_agents(bundle=BUNDLE, ticker="NVDA", event_type="E", trace=trace) == 0.60
    assert client.sent[0]["reasoning_effort"] == "high"      # first try: high effort
    assert "reasoning_effort" not in client.sent[1]          # retry: model default
    assert trace["analyst_retried"] is True


def test_scoring_agents_use_high_effort_by_default(fake, monkeypatch):
    client = fake(sheet(**{G: 6}), scrut({G: "FACT"}, **{G: -2}), {"final_score": 0.5})
    run()
    assert [c["reasoning_effort"] for c in client.sent] == ["high", "high", "high"]  # Analyst, Scrutinizer, PM
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "xhigh")
    client = fake(sheet(**{G: 2}), sheet(**{G: 2}))
    run()
    assert all(c["reasoning_effort"] == "xhigh" for c in client.sent)


def test_preview_comes_before_the_call_facts(fake):
    client = fake(sheet(), sheet())
    run()
    user = client.sent[0]["messages"][1]["content"]
    assert user.index("Research note written BEFORE") < user.index("Facts from the earnings call")


def test_rule_zero_is_the_first_line_of_every_prompt():
    for prompt in (P.RUBRIC, P.ANALYST, P.SCRUTINIZER, P.PORTFOLIO_MANAGER):
        assert prompt.lstrip().startswith("RULE ZERO: Every fact you use must come from the materials")


def test_scrutinizer_failure_returns_analyst(fake):
    fake(sheet(**{G: 6}), RuntimeError("timeout"))
    assert run() == 0.80


def test_pm_failure_returns_average(fake):
    fake(sheet(**{G: 6}), scrut({G: "FACT"}, **{G: -2}), RuntimeError("timeout"))
    assert run() == 0.60          # (0.80 + 0.40) / 2


def test_pm_garbage_returns_average(fake):
    fake(sheet(**{G: 6}), scrut({G: "FACT"}, **{G: -2}), {"final_score": "high"})
    assert run() == 0.60


def test_out_of_time_skips_remaining_agents(fake, monkeypatch):
    client = fake(sheet(**{G: 6}))
    monkeypatch.setattr(P, "TOTAL_BUDGET_SECONDS", 10.0)  # less than MIN_CALL_SECONDS
    assert run() == 0.80
    assert len(client.sent) == 1


def test_implied_move_read_from_bundle():
    assert P.implied_move(BUNDLE) == 0.051
    assert P.implied_move({"items": []}) is None
    bad = {"items": [{"id": "option-implied-stats", "content": {
        "implied_absolute_earnings_move": {"value": None, "status": "illiquid"}}}]}
    assert P.implied_move(bad) is None


# ---------- judgment cap ----------

def scrut(audit, **pts) -> dict:
    out = sheet(**pts)
    out["audit"] = [{"component": c, "error": e} for c, e in audit.items()]
    return out


def test_big_judgment_change_is_cut_to_three_points():
    a = P.points_from_sheet(sheet()["components"])
    r_sheet = scrut({G: "JUDGMENT", "gross_margin": "JUDGMENT"}, **{G: -5, "gross_margin": -2})
    pts, notes = P.cap_judgment_changes(a, r_sheet)
    assert pts[G] == -3                     # 0 -> -5 under JUDGMENT is capped at -3
    assert pts["gross_margin"] == -2        # 0 -> -2 is allowed
    assert [n["component"] for n in notes] == [G]


def test_unlisted_big_change_is_capped_too():
    a = P.points_from_sheet(sheet(**{F: 6})["components"])
    pts, notes = P.cap_judgment_changes(a, sheet(**{F: -2}))   # no audit row at all
    assert pts[F] == 3 and len(notes) == 1


def test_hard_error_codes_may_move_more_than_three_points():
    a = P.points_from_sheet(sheet()["components"])
    r_sheet = scrut({G: "SHAVE", RV: "bar"}, **{G: 8, RV: 6})
    pts, notes = P.cap_judgment_changes(a, r_sheet)
    assert pts[G] == 8 and pts[RV] == 6 and notes == []


def test_cap_is_applied_before_scoring_and_shown_to_pm(fake):
    client = fake(sheet(**{G: 6}), scrut({G: "JUDGMENT"}, **{G: 0}), {"final_score": 0.7})
    trace = {}
    P.run_agents(bundle=BUNDLE, ticker="NVDA", event_type="EARNINGS_RELEASE", trace=trace)
    assert trace["scrutinizer_points"][G] == 3 and trace["judgment_caps"] == 1
    assert trace["scrutinizer_score"] == 0.65
    assert "judgment_caps_applied" in client.sent[2]["messages"][1]["content"]


# ---------- materials only: no outside research ----------

def test_no_agent_ever_fetches_outside_information(fake, monkeypatch):
    """Only the event's own materials: nothing may touch the network but OpenAI."""
    def blocked(*args, **kwargs):
        raise AssertionError("an agent tried to fetch outside information")
    monkeypatch.setattr(P.httpx, "get", blocked)
    monkeypatch.setenv("USE_CONTEXT_AGENT", "1")   # old switches must do nothing now
    monkeypatch.setenv("USE_SEC_FILINGS", "1")
    client = fake(sheet(**{G: 2}), sheet(**{G: 2}))
    assert run() == 0.60
    assert len(client.sent) == 2                    # Analyst + Scrutinizer only
    assert "ROLE: ANALYST" in client.sent[0]["messages"][0]["content"]


def test_every_agent_is_told_materials_only():
    assert "MATERIALS ONLY" in P.RUBRIC
    for prompt in (P.ANALYST, P.SCRUTINIZER, P.PORTFOLIO_MANAGER):
        assert "MATERIALS ONLY" in prompt
    assert not hasattr(P, "sec_dossier") and not hasattr(P, "CONTEXT")


def test_agents_see_only_the_event_materials(fake):
    client = fake(sheet(**{G: 2}), sheet(**{G: 2}))
    run()
    user = client.sent[0]["messages"][1]["content"]
    assert "Revenue $96B." in user and "whisper $95B" in user
    assert "BACKGROUND" not in user and "SEC FILINGS" not in user


# ---------- trace + quant inside the pipeline ----------

def test_trace_records_every_stage_and_quant_is_applied(fake, monkeypatch):
    monkeypatch.setenv("QUANT_POWER", "0.5")
    monkeypatch.setenv("QUANT_REF_MOVE", "0.0255")   # BUNDLE move 0.051 -> factor sqrt(2)=1.414
    fake(sheet(**{G: 2, RV: 1}), sheet(**{G: 2, RV: 1}))
    trace = {}
    final = P.run_agents(bundle=BUNDLE, ticker="NVDA", event_type="EARNINGS_RELEASE", trace=trace)
    assert trace["unsized"] == 0.65 and trace["path"] == "close_call"
    assert final == trace["final"] == 0.7121              # 0.5 + 0.15*1.414 = 0.7121
    assert trace["analyst_points"][G] == 2 and trace["scrutinizer_score"] == 0.65
    assert trace["errors"] == []


def test_pm_sees_quant_view(fake):
    client = fake(sheet(**{G: 6}), scrut({G: "FACT"}, **{G: -2}),
                  {"final_score": 0.6})
    run()
    assert "QUANT VIEW" in client.sent[2]["messages"][1]["content"]


def test_error_fallback_is_not_sized(fake, monkeypatch):
    monkeypatch.setenv("QUANT_PIVOT", "0.6")
    fake(RuntimeError("down"), RuntimeError("down"))
    trace = {}
    assert P.run_agents(bundle=BUNDLE, ticker="X", event_type="E", trace=trace) == 0.5
    assert trace["path"] == "analyst_failed"


def test_hard_deadline_cuts_off_slow_call():
    import time as _t
    import predict as P
    started = _t.monotonic()
    try:
        P._hard_deadline(lambda: _t.sleep(5), 0.5)
        raise AssertionError("should have timed out")
    except TimeoutError:
        pass
    assert _t.monotonic() - started < 2


def test_hard_deadline_passes_result_and_errors():
    import predict as P
    assert P._hard_deadline(lambda: 42, 2) == 42
    try:
        P._hard_deadline(lambda: (_ for _ in ()).throw(ValueError("x")), 2)
        raise AssertionError("should have raised")
    except ValueError:
        pass
