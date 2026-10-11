"""Offline tests for the 3-agent pipeline: no API key spent, no network.

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


def sheet(**pts) -> dict:
    return {"components": {k: {"points": v, "evidence": "x"} for k, v in pts.items()},
            "score": 0.99, "total_s": 999}  # deliberately wrong maths: Python must override


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

    def install(*replies):
        client = FakeClient(replies)
        monkeypatch.setattr(P, "_client", client)
        return client
    return install


def run(bundle=BUNDLE):
    return P.run_agents(bundle=bundle, ticker="NVDA", event_type="EARNINGS_RELEASE")


# ---------- arithmetic ----------

def test_points_are_clamped_and_summed():
    pts = P.points_from_sheet({"guidance": {"points": 7}, "eps": {"points": -4}, "tone": "1"})
    assert pts["guidance"] == 3 and pts["eps"] == -1 and pts["tone"] == 1 and pts["revenue"] == 0


@pytest.mark.parametrize("s, move, expected", [
    (0, 0.12, 0.50),     # 0.50 never moves
    (3, 0.05, 0.65),     # 4-8% move: no change
    (3, 0.09, 0.70),     # >=8%: one step further
    (-3, 0.09, 0.30),
    (3, 0.02, 0.60),     # <4%: one step toward 0.50
    (-1, 0.02, 0.50),
    (11, 0.20, 0.95),    # capped
    (-11, None, 0.05),
])
def test_score_from_points(s, move, expected):
    pts = {k: 0 for k in P.COMPONENT_LIMITS}
    pts["guidance"] = s  # bypass clamp on purpose to test the conversion only
    assert P.score_from_points(pts, move)["score"] == expected


@pytest.mark.parametrize("a, r, expected", [
    (0.65, 0.60, 0.65),  # 0.625 tie -> away from 0.50
    (0.40, 0.35, 0.35),  # 0.375 tie -> away from 0.50
    (0.60, 0.70, 0.65),
])
def test_average_rule(a, r, expected):
    assert P.average_rule(a, r) == expected


# ---------- pipeline paths ----------

def test_close_call_averages_without_calling_pm(fake):
    client = fake(sheet(guidance=2, revenue=1), sheet(guidance=2))
    assert run() == 0.65          # analyst S=3 -> 0.65, scrutinizer S=2 -> 0.60, tie -> 0.65
    assert len(client.sent) == 2  # PM skipped


def test_disagreement_calls_pm_and_snaps_its_answer(fake):
    client = fake(sheet(guidance=3, revenue=2), sheet(guidance=-1),
                  {"final_score": 0.58, "settled_components": {"guidance": 1}})
    assert run() == 0.60          # 0.58 snapped to grid
    assert len(client.sent) == 3
    pm_msg = client.sent[2]["messages"][1]["content"]
    assert "analyst_score=0.75" in pm_msg and "scrutinizer_score=0.45" in pm_msg


def test_models_maths_is_overwritten_before_the_next_agent_sees_it(fake):
    client = fake(sheet(guidance=2), sheet(guidance=2))
    run()
    sent_to_scrutinizer = client.sent[1]["messages"][1]["content"]
    assert '"score": 0.6' in sent_to_scrutinizer and '"total_s": 2' in sent_to_scrutinizer


def test_json_inside_code_fences_is_accepted(fake):
    fake("```json\n" + json.dumps(sheet(revenue=2)) + "\n```", sheet(revenue=2))
    assert run() == 0.60


def test_rubric_is_sent_with_every_role(fake):
    client = fake(sheet(), sheet())
    run()
    for call in client.sent:
        system = call["messages"][0]["content"]
        assert "UNIVERSAL SCORING RUBRIC" in system
        assert call["response_format"] == {"type": "json_object"}


# ---------- fallbacks ----------

def test_no_key_returns_half(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert run() == 0.5


def test_analyst_failure_returns_half(fake):
    fake(RuntimeError("down"))
    assert run() == 0.5


def test_scrutinizer_failure_returns_analyst(fake):
    fake(sheet(guidance=3, revenue=2), RuntimeError("timeout"))
    assert run() == 0.75


def test_pm_failure_returns_average(fake):
    fake(sheet(guidance=3, revenue=2), sheet(guidance=-1), RuntimeError("timeout"))
    assert run() == 0.60          # (0.75 + 0.45) / 2


def test_pm_garbage_returns_average(fake):
    fake(sheet(guidance=3, revenue=2), sheet(guidance=-1), {"final_score": "high"})
    assert run() == 0.60


def test_out_of_time_skips_remaining_agents(fake, monkeypatch):
    client = fake(sheet(guidance=3, revenue=2))
    monkeypatch.setattr(P, "TOTAL_BUDGET_SECONDS", 10.0)  # less than MIN_CALL_SECONDS
    assert run() == 0.75
    assert len(client.sent) == 1


def test_implied_move_read_from_bundle():
    assert P.implied_move(BUNDLE) == 0.051
    assert P.implied_move({"items": []}) is None
    bad = {"items": [{"id": "option-implied-stats", "content": {
        "implied_absolute_earnings_move": {"value": None, "status": "illiquid"}}}]}
    assert P.implied_move(bad) is None
