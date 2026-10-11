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
    monkeypatch.setenv("USE_CONTEXT_AGENT", "0")   # iteration-1 tests; context tests switch it on
    monkeypatch.setenv("USE_SEC_FILINGS", "0")     # never touch the network in tests
    for k in ("QUANT_POWER", "QUANT_REF_MOVE", "QUANT_PIVOT", "COMPONENT_WEIGHTS"):
        monkeypatch.delenv(k, raising=False)

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


@pytest.mark.parametrize("s, expected", [
    (0, 0.50), (3, 0.65), (-3, 0.35), (11, 0.95), (-11, 0.05),
])
def test_score_from_points(s, expected):
    pts = {k: 0 for k in P.COMPONENT_LIMITS}
    pts["guidance"] = s  # bypass clamp on purpose to test the conversion only
    assert P.score_from_points(pts)["score"] == expected


def test_component_weights_from_env(monkeypatch):
    monkeypatch.setenv("COMPONENT_WEIGHTS", "guidance=2, eps=0, bogus=9, tone=abc, revenue=99")
    w = P.component_weights()
    assert w["guidance"] == 2 and w["eps"] == 0 and w["tone"] == 1 and w["revenue"] == 1
    pts = {k: 0 for k in P.COMPONENT_LIMITS} | {"guidance": 2, "eps": 1}
    assert P.score_from_points(pts) == {"total_s": 4.0, "score": 0.70}


@pytest.mark.parametrize("score, move, expected", [
    (0.70, 0.06, 0.70),    # normal move: unchanged
    (0.70, 0.24, 0.90),    # 4x move -> distance x2: 0.5 + 0.2*2
    (0.70, 0.015, 0.60),   # quarter move -> distance x0.5
    (0.30, 0.24, 0.10),
    (0.70, None, 0.70),    # no options data: unchanged
    (0.50, 0.24, 0.50),    # the pivot never moves
    (0.90, 0.24, 0.95),    # capped
])
def test_quant_sizing(score, move, expected, monkeypatch):
    for k in ("QUANT_POWER", "QUANT_REF_MOVE", "QUANT_PIVOT"):
        monkeypatch.delenv(k, raising=False)
    assert P.apply_quant(score, move) == expected


def test_quant_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("QUANT_POWER", "0")
    assert P.apply_quant(0.70, 0.24) == 0.70


def test_quant_view_text():
    text = P.quant_view(BUNDLE)
    assert "Expected absolute move: 5.1%" in text and "No proven link" not in text  # BUNDLE has no skew
    assert "no usable options data" in P.quant_view({"items": []})


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
    client = fake(sheet(guidance=3, revenue=2), scrut({"guidance": "FACT", "revenue": "FACT"}, guidance=-1),
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
    fake(sheet(guidance=3, revenue=2), scrut({"guidance": "FACT", "revenue": "FACT"}, guidance=-1), RuntimeError("timeout"))
    assert run() == 0.60          # (0.75 + 0.45) / 2


def test_pm_garbage_returns_average(fake):
    fake(sheet(guidance=3, revenue=2), scrut({"guidance": "FACT", "revenue": "FACT"}, guidance=-1), {"final_score": "high"})
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


# ---------- judgment cap (from the real NVDA run, 2026-10-11) ----------

def scrut(audit, **pts) -> dict:
    out = sheet(**pts)
    out["audit"] = [{"component": c, "error": e} for c, e in audit.items()]
    return out


def test_big_judgment_change_is_cut_to_one_point():
    a = P.points_from_sheet(sheet(tone=-1, setup=1)["components"])
    r_sheet = scrut({"key_metric": "JUDGMENT", "quality": "JUDGMENT"},
                    key_metric=-2, quality=-1, tone=-1, setup=1)
    pts, notes = P.cap_judgment_changes(a, r_sheet)
    assert pts["key_metric"] == -1          # 0 -> -2 under JUDGMENT is capped at -1
    assert pts["quality"] == -1             # 0 -> -1 is allowed
    assert [n["component"] for n in notes] == ["key_metric"]


def test_hard_error_codes_may_move_more_than_one_point():
    a = P.points_from_sheet(sheet(guidance=0, revenue=0)["components"])
    r_sheet = scrut({"guidance": "SHAVE", "revenue": "bar"}, guidance=3, revenue=2)
    pts, notes = P.cap_judgment_changes(a, r_sheet)
    assert pts["guidance"] == 3 and pts["revenue"] == 2 and notes == []


def test_cap_is_applied_before_scoring_and_shown_to_pm(fake):
    client = fake(sheet(guidance=3, revenue=2),
                  scrut({"guidance": "JUDGMENT"}, guidance=-2, revenue=2),
                  {"final_score": 0.7})
    run()
    # scrutinizer's guidance -2 capped to +2: S = 4 -> 0.70; analyst S = 5 -> 0.75 -> close call
    assert len(client.sent) == 2


# ---------- iteration 2: the Context agent ----------

CONTEXT_REPLY = {
    "company": "Designs GPUs and networking for data centers.",
    "industry": "AI infrastructure; hyperscaler capex drives demand.",
    "what_moves_the_stock": "Next-quarter guide vs the buy-side bar matters most.",
    "reaction_history": "Has sold off on beats when guidance only met the whisper.",
    "storyline_into_print": "Debate over gross margin and China.",
    "key_terms": "ACIE: AI cloud, industrial and enterprise customers.",
    "reading_guide": "Compare the Q3 guide with the $105B bull line.",
}


@pytest.fixture
def with_context(monkeypatch):
    monkeypatch.setenv("USE_CONTEXT_AGENT", "1")


def test_context_runs_first_and_reaches_every_later_agent(fake, with_context):
    client = fake(CONTEXT_REPLY, sheet(guidance=3, revenue=2),
                  scrut({"guidance": "FACT", "revenue": "FACT"}, guidance=-1),
                  {"final_score": 0.6})
    assert run() == 0.60
    assert len(client.sent) == 4
    context_system = client.sent[0]["messages"][0]["content"]
    assert "ROLE: CONTEXT AGENT" in context_system
    assert "UNIVERSAL SCORING RUBRIC" not in context_system     # context never gets scoring rules
    for later in client.sent[1:]:
        user = later["messages"][1]["content"]
        assert "BACKGROUND (from the Context agent" in user
        assert "Compare the Q3 guide with the $105B bull line." in user
        assert "UNIVERSAL SCORING RUBRIC" in later["messages"][0]["content"]


def test_context_failure_does_not_stop_the_pipeline(fake, with_context):
    client = fake(RuntimeError("context timeout"), sheet(guidance=2), sheet(guidance=2))
    assert run() == 0.60
    assert "BACKGROUND" not in client.sent[1]["messages"][1]["content"]


def test_context_junk_is_ignored(fake, with_context):
    client = fake({"score": 0.9, "company": ""}, sheet(guidance=2), sheet(guidance=2))
    assert run() == 0.60
    assert "BACKGROUND" not in client.sent[1]["messages"][1]["content"]


def test_context_can_be_switched_off(fake):
    client = fake(sheet(guidance=2), sheet(guidance=2))   # USE_CONTEXT_AGENT=0 from the fixture
    assert run() == 0.60
    assert "ROLE: ANALYST" in client.sent[0]["messages"][0]["content"]


# ---------- SEC filings (mocked EDGAR; no network) ----------

from datetime import datetime, timezone  # noqa: E402

TICKERS = {"0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"}}
SUBMISSIONS = {"filings": {"recent": {
    "form":               ["8-K", "4", "8-K", "10-Q", "8-K", "8-K"],
    "accessionNumber":    ["0001045810-26-000200", "x", "0001045810-26-000150",
                           "0001045810-26-000120", "0001045810-26-000100", "0001045810-26-000090"],
    "filingDate":         ["2026-08-26", "2026-08-20", "2026-07-15", "2026-05-29", "2026-05-28", "2026-03-01"],
    "acceptanceDateTime": ["2026-08-26T20:20:00.000Z", "2026-08-20T20:00:00.000Z",
                           "2026-07-15T21:00:00.000Z", "2026-05-29T20:00:00.000Z",
                           "2026-05-28T20:20:00.000Z", "2026-03-01T20:00:00.000Z"],
    "reportDate":         ["2026-08-26", "", "2026-07-14", "2026-04-26", "2026-05-28", ""],
    "primaryDocument":    ["q2.htm", "f4.xml", "ceo.htm", "nvda-10q.htm", "q1.htm", "old.htm"],
    "items":              ["2.02,9.01", "", "5.02", "", "2.02,9.01", "8.01"],
}}}
INDEX_HTML = """<table><tr><td>1</td><td>8-K</td><td><a href="/Archives/edgar/data/1045810/000104581026000100/q1.htm">q1.htm</a></td><td>8-K</td></tr>
<tr><td>2</td><td>PRESS RELEASE</td><td><a href="/Archives/edgar/data/1045810/000104581026000100/q1fy27pr.htm">q1fy27pr.htm</a></td><td>EX-99.1</td></tr></table>"""
RELEASE_HTML = "<html><p>NVIDIA Announces Financial Results for First Quarter Fiscal 2027</p>" + \
    "<p>Revenue of $81.6 billion.</p>" + "<p>filler</p>" * 900 + \
    "<p>Outlook: Revenue is expected to be $91.0 billion, plus or minus 2%.</p></html>"
TENQ_HTML = ("<html><ix:header><hidden>XBRL junk</hidden></ix:header>"
             "<p>Item 2. Management's Discussion and Analysis of Financial Condition 25</p>"
             "<p>Item 3. Quantitative and Qualitative Disclosures 40</p>"
             + "<p>other text</p>" * 50 +
             "<p>Item 2. Management&#8217;s Discussion and Analysis of Financial Condition and Results</p>"
             "<p>Data Center revenue grew 21% sequentially driven by Blackwell.</p>" + "<p>more</p>" * 100 +
             "<p>Item 3. Quantitative and Qualitative Disclosures About Market Risk</p><p>after</p></html>")


class FakeResp:
    def __init__(self, body):
        self._body = body
        self.text = body if isinstance(body, str) else json.dumps(body)
        self.content = self.text.encode()

    def json(self):
        return self._body


@pytest.fixture
def edgar(monkeypatch):
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        if url.endswith("company_tickers.json"):
            return FakeResp(TICKERS)
        if "submissions" in url:
            return FakeResp(SUBMISSIONS)
        if url.endswith("-index.htm"):
            return FakeResp(INDEX_HTML)
        if url.endswith("q1fy27pr.htm"):
            return FakeResp(RELEASE_HTML)
        if url.endswith("nvda-10q.htm"):
            return FakeResp(TENQ_HTML)
        raise AssertionError("unexpected SEC url " + url)

    monkeypatch.setattr(P, "_sec_get", fake_get)
    monkeypatch.setattr(P, "_ticker_to_cik", None)
    return calls


CUTOFF = P.filings_cutoff("2026-08-26T21:00:00+00:00", "2026-08-26T20:00:00+00:00")


def test_cutoff_is_24h_before_event_or_the_official_cutoff():
    assert CUTOFF == datetime(2026, 8, 25, 21, 0, tzinfo=timezone.utc)
    wmt = P.filings_cutoff("2026-08-20T12:00:00+00:00", "2026-08-19T20:00:00+00:00")
    assert wmt == datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc)


def test_release_day_filings_are_never_used(edgar):
    used = P.eligible_filings(SUBMISSIONS, CUTOFF)
    assert "0001045810-26-000200" not in [f["accession"] for f in used]   # this quarter's results
    assert used[0]["accession"] == "x"                                     # newest eligible first


def test_dossier_has_previous_release_mdna_and_other_8ks(edgar):
    text = P.sec_dossier("NVDA", CUTOFF, 20)
    assert "PREVIOUS EARNINGS RELEASE (8-K filed 2026-05-28" in text
    assert "Revenue of $81.6 billion" in text
    assert "Revenue is expected to be $91.0 billion" in text          # the outlook paragraph survives
    assert "Data Center revenue grew 21% sequentially" in text        # real MD&A, not the contents page
    assert "XBRL junk" not in text and "after" not in text.split("MD&A excerpt")[1][-200:]
    assert "- 2026-07-15: 8-K - 5.02 Director or officer change" in text
    assert "2026-08-26" not in text.split("accepted before")[1][30:]  # nothing from release day
    assert not any("000104581026000200" in u for u in edgar)          # never even downloaded


def test_dossier_returns_empty_on_failure(monkeypatch):
    def boom(url, timeout):
        raise RuntimeError("sec down")
    monkeypatch.setattr(P, "_sec_get", boom)
    monkeypatch.setattr(P, "_ticker_to_cik", None)
    assert P.sec_dossier("NVDA", CUTOFF, 20) == ""


def test_dossier_respects_time_budget(edgar):
    assert P.sec_dossier("NVDA", CUTOFF, 1) == ""
    assert edgar == []


def test_unknown_ticker_gives_empty(edgar):
    assert P.sec_dossier("ZZZZ", CUTOFF, 20) == ""


def test_filings_reach_the_context_agent_only(fake, with_context, edgar, monkeypatch):
    monkeypatch.setenv("USE_SEC_FILINGS", "1")
    client = fake(CONTEXT_REPLY | {"filings_digest": "Q1 release guided $91.0B +/-2%."},
                  sheet(guidance=2), sheet(guidance=2))
    P.run_agents(bundle=BUNDLE, ticker="NVDA", event_type="EARNINGS_RELEASE",
                 event_datetime="2026-08-26T21:00:00+00:00")
    context_msg = client.sent[0]["messages"][1]["content"]
    assert "SEC FILINGS (from EDGAR" in context_msg
    analyst_msg = client.sent[1]["messages"][1]["content"]
    assert "From SEC filings:\nQ1 release guided $91.0B +/-2%." in analyst_msg
    assert "SEC FILINGS (from EDGAR" not in analyst_msg              # raw filings stay with Context


# ---------- trace + quant inside the pipeline ----------

def test_trace_records_every_stage_and_quant_is_applied(fake, monkeypatch):
    monkeypatch.setenv("QUANT_REF_MOVE", "0.0255")   # BUNDLE move 0.051 -> factor sqrt(2)=1.414
    fake(sheet(guidance=2, revenue=1), sheet(guidance=2, revenue=1))
    trace = {}
    final = P.run_agents(bundle=BUNDLE, ticker="NVDA", event_type="EARNINGS_RELEASE", trace=trace)
    assert trace["unsized"] == 0.65 and trace["path"] == "close_call"
    assert final == trace["final"] == 0.70                # 0.5 + 0.15*1.414 = 0.712 -> 0.70
    assert trace["analyst_points"]["guidance"] == 2 and trace["scrutinizer_score"] == 0.65
    assert trace["errors"] == []


def test_pm_sees_quant_view(fake):
    client = fake(sheet(guidance=3, revenue=2), scrut({"guidance": "FACT", "revenue": "FACT"}, guidance=-1),
                  {"final_score": 0.6})
    run()
    assert "QUANT VIEW" in client.sent[2]["messages"][1]["content"]


def test_error_fallback_is_not_sized(fake, monkeypatch):
    monkeypatch.setenv("QUANT_PIVOT", "0.6")
    fake(RuntimeError("down"))
    trace = {}
    assert P.run_agents(bundle=BUNDLE, ticker="X", event_type="E", trace=trace) == 0.5
    assert trace["path"] == "analyst_failed"
