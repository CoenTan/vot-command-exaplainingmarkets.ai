"""predict() returns the right shape on the no-OpenAI-key fallback path.

With OPENAI_API_KEY unset, predict() must still return one well-formed prediction
per focal asset (the 0.5 baseline) without making any network calls. We stub the
disclosure fetch so the test is fully offline.
"""

from __future__ import annotations

import predict as predict_module


SAMPLE_EVENT = {
    "id": "evt_test_1",
    "event_id": "11111111-1111-1111-1111-111111111111",
    "event_type": "EARNINGS_RELEASE",
    "timing_category": "SCHEDULED",
    "event_datetime": "2026-01-15T21:00:00Z",
    "focal_assets": [
        {"identifier_type": "TICKER", "identifier_value": "AAPL"},
        {"identifier_type": "TICKER", "identifier_value": "MSFT"},
    ],
    "information_url": "https://example.test/disclosure",
    "prediction_deadline": "2026-01-15T21:05:00Z",
}


PREVIEW = "# Acme (ACME) — Q3 FY2026 Earnings Preview\n\n## Consensus Estimates\nEPS $1.20\n"
STATS = {
    "as_of": "2026-10-05T19:36:12Z",
    "methodology": "v1",
    "implied_earnings_volatility": {"value": 0.0847, "status": "ok"},
    "implied_absolute_earnings_move": {"value": 0.0676, "status": "ok"},
    "skew_25_delta": {"value": -0.0412, "status": "ok"},
}
FACTS_ITEM = {
    "id": "earnings-call-facts",
    "kind": "facts",
    "source": "earnings_call",
    "media_type": "application/json",
    "content": ["Revenue rose 12% year over year.", "Full-year guidance was raised."],
}
PREVIEW_ITEM = {
    "id": "earnings-preview",
    "kind": "text",
    "source": "claude_code_web_research",
    "media_type": "text/markdown",
    "content": PREVIEW,
}
STATS_ITEM = {
    "id": "option-implied-stats",
    "kind": "stats",
    "source": "option_market",
    "media_type": "application/json",
    "content": STATS,
}
# The real shape behind `information_url`.
BUNDLE = {
    "schema_version": "1.0",
    "event_id": "11111111-1111-1111-1111-111111111111",
    "generated_at": "2026-10-05T21:30:00Z",
    "items": [FACTS_ITEM, PREVIEW_ITEM, STATS_ITEM],
}


def _bundle(*items) -> dict:
    return {**BUNDLE, "items": list(items)}


class _FakeResponse:
    def raise_for_status(self) -> None:  # noqa: D401 - stub
        return None

    def json(self) -> dict:
        return BUNDLE


def test_predict_fallback_shape(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(predict_module.httpx, "get", lambda *a, **k: _FakeResponse())

    preds = predict_module.predict(SAMPLE_EVENT)

    assert isinstance(preds, list)
    assert len(preds) == len(SAMPLE_EVENT["focal_assets"])
    returned = {p["identifier_value"] for p in preds}
    assert returned == {"AAPL", "MSFT"}
    for p in preds:
        assert set(p) == {"identifier_value", "predicted_percentile"}
        assert 0.0 <= p["predicted_percentile"] <= 1.0
        assert p["predicted_percentile"] == 0.5  # fallback baseline


def test_every_item_reaches_the_prompt_under_its_own_label() -> None:
    text = predict_module.format_materials(BUNDLE)
    assert "1. Revenue rose 12% year over year." in text
    assert "2. Full-year guidance was raised." in text
    assert "Implied earnings volatility: 8.5%" in text
    assert "Implied absolute move: 6.8%" in text
    assert "25-delta skew: -4.1 volatility points" in text
    assert "as of 2026-10-05T19:36:12Z" in text
    assert "## Consensus Estimates" in text
    # The raw envelope is not pasted in.
    for noise in ("schema_version", "media_type", "generated_at", '"kind"'):
        assert noise not in text


def test_items_are_found_by_id_not_by_position() -> None:
    shuffled = predict_module.format_materials(_bundle(STATS_ITEM, PREVIEW_ITEM, FACTS_ITEM))
    assert shuffled == predict_module.format_materials(BUNDLE)


def test_a_long_preview_never_costs_the_facts_or_the_statistics() -> None:
    """The old prompt took the first 8,000 characters of the raw JSON, so a long
    preview pushed every later item out of the model's view."""
    long_preview = {**PREVIEW_ITEM, "content": "# Preview\n" + "x" * 50_000}
    text = predict_module.format_materials(_bundle(FACTS_ITEM, long_preview, STATS_ITEM))
    assert "1. Revenue rose 12% year over year." in text
    assert "Implied earnings volatility: 8.5%" in text
    assert "[preview truncated]" in text
    assert text.startswith("Research note written BEFORE the release")   # preview goes first
    kept = predict_module.PREVIEW_MAX_CHARS - len("# Preview\n")
    assert "x" * kept in text and "x" * (kept + 1) not in text


def test_optional_items_may_be_absent() -> None:
    facts_only = predict_module.format_materials(_bundle(FACTS_ITEM))
    assert facts_only.startswith("Facts from the earnings call:")
    assert "Option-market" not in facts_only and "Research note" not in facts_only


def test_an_unavailable_skew_is_said_plainly() -> None:
    stats = {**STATS, "skew_25_delta": {"value": None, "status": "illiquid_wings"}}
    text = predict_module.format_materials(_bundle(FACTS_ITEM, {**STATS_ITEM, "content": stats}))
    assert "Implied earnings volatility: 8.5%" in text
    assert "25-delta skew: unavailable" in text


def test_statistics_without_a_usable_volatility_are_left_out() -> None:
    for broken in (
        {**STATS, "implied_earnings_volatility": {"value": None, "status": "illiquid_atm"}},
        {**STATS, "implied_earnings_volatility": {"value": "0.08", "status": "ok"}},
        {},
        {"as_of": "2026-10-05T19:36:12Z"},
    ):
        text = predict_module.format_materials(
            _bundle(FACTS_ITEM, {**STATS_ITEM, "content": broken})
        )
        assert "Option-market" not in text and "Facts from the earnings call" in text


def test_a_kind_this_code_has_never_seen_is_never_an_error() -> None:
    text = predict_module.format_materials(
        _bundle(
            FACTS_ITEM,
            {"id": "analyst-note", "kind": "text", "content": "Street is cautious."},
            {"id": "bullets", "kind": "list", "content": ["one", "two"]},
            {"id": "future-table", "kind": "table", "content": {"rows": [[1, 2]]}},
            {"id": "by-reference", "kind": "document", "url": "https://example.test/x.pdf"},
            "not even an object",
        )
    )
    assert "Additional material (analyst-note):\nStreet is cautious." in text
    assert "Additional material (bullets):\none\ntwo" in text
    assert "future-table" not in text and "by-reference" not in text


def test_anything_that_is_not_a_bundle_formats_to_nothing() -> None:
    for odd in (None, [], "text", {"summary": "legacy shape"}, {"items": "nope"}, {"items": []}):
        assert predict_module.format_materials(odd) == ""
