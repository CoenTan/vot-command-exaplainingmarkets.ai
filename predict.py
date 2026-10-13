"""VOT COMMAND, iteration 3: Context -> Analyst -> Scrutinizer -> Portfolio Manager,
scored with the v02.1 rubric: the v2 bot's v02 (17 dimensions, -10..+10) plus one
amendment that scores guidance against investor expectations (whisper > consensus >
the company's prior guidance).

THIS IS THE ONLY FILE YOU EDIT. Everything else in the repo is plumbing.

How one event flows:
  1. Fetch the event's materials (call facts, preview, option stats).
  1b. SEC FILINGS: fetch the previous earnings release, latest 10-Q/10-K MD&A
     and recent 8-Ks from EDGAR. Only filings accepted 24h+ before the event
     (knowledge-cutoff safe). Hard 20s cap; skipped on any problem.
  1c. CONTEXT writes background on the company, industry and filings
     (paragraphs, no number). Every later agent reads it; never evidence.
  2. ANALYST scores all 17 rubric dimensions from -10 to +10.
  3. SCRUTINIZER audits the Analyst and scores it again.
  4. PORTFOLIO MANAGER settles it (skipped on close calls: Python averages
     the two scores and saves a call).
  5. Python re-does every agent's arithmetic: weighted_total = sum(weight x score),
     percentile = 0.5 + weighted_total / 20, kept within 0.001..0.999.
  6. QUANT (Python, no AI) sizes the final score with the options market's
     expected move: pivot + (score - pivot) * (move / reference_move) ** power.
     Big expected move -> further from the middle. It never picks a direction.
     Scores are continuous (no 0.05 grid), so fewer ties.

Safety nets (a missed event counts against you, so something is ALWAYS sent):
  - No OPENAI_API_KEY             -> 0.50
  - SEC fetch fails / slow        -> context runs without filings
  - Context fails / slow          -> carry on without background
  - Analyst fails                 -> 0.50
  - Scrutinizer fails / no time   -> Analyst's score
  - PM fails / no time            -> average of Analyst and Scrutinizer

Optional settings in .env:
  OPENAI_MODEL               model for every agent (default: the starter's default)
  OPENAI_REASONING_EFFORT    e.g. low / medium / high (leave unset if unsure)
  ALWAYS_CALL_PM             set to 1 to call the PM even on close calls
  USE_CONTEXT_AGENT          set to 0 to switch the Context agent off (iteration 1)
  USE_SEC_FILINGS            set to 0 to stop fetching SEC filings
  SEC_USER_AGENT             "Your Name your@email.com" (the SEC asks for contact info)
  CONTEXT_REASONING_EFFORT   effort for the Context agent only (e.g. low)
  COMPONENT_WEIGHTS          e.g. "guidance_revision=0.4" (others keep the v02 weights)
  CLOSE_CALL_GAP             max Analyst/Scrutinizer gap that skips the PM (default 0.05)
  QUANT_POWER                sizing strength (default 0.25; 0 = sizing off)
  QUANT_REF_MOVE             a "normal" expected move (default 0.08 = 8%, the 2026Q3 median)
  QUANT_PIVOT                centre for sizing (default 0.50)

The 17 dimension weights are the v02 rubric's (rubric/weights.json in the v2 bot).
  BOT_QUIET                  set to 1 to hide the agents' full JSON in logs

To change how agents think, edit the prompt blocks below
(RUBRIC, CONTEXT, ANALYST, SCRUTINIZER, PORTFOLIO_MANAGER).
"""

from __future__ import annotations

import html
import json
import math
import os
import queue
import re
import threading
import time
from datetime import datetime, timedelta, timezone

import httpx
from openai import OpenAI

from explaining_markets.config import openai_model

# ----------------------------------------------------------------------
# Time budget. The platform gives 5 minutes from the ACK. We aim to finish
# all agents inside TOTAL_BUDGET_SECONDS, leaving room to submit.
# ----------------------------------------------------------------------
SUMMARY_TIMEOUT_SECONDS = 15.0
TOTAL_BUDGET_SECONDS = 240.0   # all agents together
CONTEXT_MAX_SECONDS = 75.0     # the Context agent gets at most this
SEC_MAX_SECONDS = 20.0         # all SEC fetching together, hard cap
MAX_CALL_SECONDS = 110.0       # one agent call at most
MIN_CALL_SECONDS = 25.0        # don't start a call with less time than this

_client: OpenAI | None = None  # lazy: importing this file must not need a key
_warned_no_key = False


def predict(event: dict) -> list[dict]:
    """Return one prediction per focal asset. Contract required by the plumbing:
    [{"identifier_value": "AAPL", "predicted_percentile": 0.65}, ...]
    """
    resp = httpx.get(event["information_url"], timeout=SUMMARY_TIMEOUT_SECONDS)
    resp.raise_for_status()
    bundle = resp.json()
    return [
        {
            "identifier_value": asset["identifier_value"],
            "predicted_percentile": run_agents(
                bundle=bundle,
                ticker=asset["identifier_value"],
                event_type=event.get("event_type", "EARNINGS_RELEASE"),
                event_datetime=event.get("event_datetime"),
                knowledge_cutoff=event.get("knowledge_cutoff"),
            ),
        }
        for asset in event["focal_assets"]
    ]


# ----------------------------------------------------------------------
# Reading the event's materials.
#
# The document behind `information_url` is a bundle: a list of `items`, each
# with an `id`, a `kind`, and a `content` whose JSON type follows the kind.
# Today an earnings event can carry three:
#
#   earnings-call-facts   kind "facts"  a list of strings     always present
#   earnings-preview      kind "text"   one markdown string   may be absent
#   option-implied-stats  kind "stats"  an OBJECT of numbers  may be absent
#
# Select items by `id`, never by position, and expect kinds you have not seen:
# new ones can be added at any time.
# ----------------------------------------------------------------------

FACTS_ID = "earnings-call-facts"
PREVIEW_ID = "earnings-preview"
OPTION_STATS_ID = "option-implied-stats"

# The preview is by far the largest item (often ~8,000 characters). It is the
# only one that gets cut, and it goes LAST in the prompt so a cut never costs
# you the facts or the option statistics.
PREVIEW_MAX_CHARS = 20000
OTHER_ITEM_MAX_CHARS = 4000


def _percent(block: object) -> str | None:
    """A `{value, status}` statistic as a percentage, or None if unavailable."""
    if not isinstance(block, dict) or block.get("status") != "ok":
        return None
    value = block.get("value")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return f"{value * 100:+.1f}" if value < 0 else f"{value * 100:.1f}"


def _format_option_stats(stats: dict) -> str | None:
    volatility = _percent(stats.get("implied_earnings_volatility"))
    if volatility is None:
        return None
    lines = [
        f"- Implied earnings volatility: {volatility}% (the standard deviation of the "
        "stock's move on this release that option prices imply)",
    ]
    move = _percent(stats.get("implied_absolute_earnings_move"))
    if move is not None:
        lines.append(
            f"- Implied absolute move: {move}% (the size of move options price in; "
            "it says nothing about direction)"
        )
    skew = _percent(stats.get("skew_25_delta"))
    if skew is not None:
        lines.append(
            f"- 25-delta skew: {skew} volatility points (call minus put implied "
            "volatility; negative means downside protection is priced richer)"
        )
    else:
        lines.append("- 25-delta skew: unavailable (options too thinly traded)")
    as_of = stats.get("as_of")
    header = "Option-market expectations, measured before the release"
    if isinstance(as_of, str) and as_of:
        header += f" (as of {as_of})"
    return header + ":\n" + "\n".join(lines)


def format_materials(bundle: object) -> str:
    """Turn the event's bundle into the text the model reads.

    Each item gets its own labelled section. Items are found by `id`, so the
    order they arrive in does not matter, and an item this code has never heard
    of is included if it is text and skipped otherwise, never an error.
    """
    if not isinstance(bundle, dict):
        return ""
    raw_items = bundle.get("items")
    items = [i for i in raw_items if isinstance(i, dict)] if isinstance(raw_items, list) else []
    by_id = {i.get("id"): i for i in items}
    sections: list[str] = []

    facts = (by_id.get(FACTS_ID) or {}).get("content")
    if isinstance(facts, list) and facts:
        sections.append(
            "Facts from the earnings call:\n"
            + "\n".join(f"{n}. {fact}" for n, fact in enumerate(facts, start=1))
        )

    stats = (by_id.get(OPTION_STATS_ID) or {}).get("content")
    if isinstance(stats, dict):
        formatted = _format_option_stats(stats)
        if formatted:
            sections.append(formatted)

    for item in items:
        if item.get("id") in (FACTS_ID, PREVIEW_ID, OPTION_STATS_ID):
            continue
        content = item.get("content")
        if isinstance(content, list) and all(isinstance(c, str) for c in content):
            content = "\n".join(content)
        if isinstance(content, str) and content.strip():
            label = item.get("id") or item.get("kind") or "item"
            sections.append(f"Additional material ({label}):\n{content[:OTHER_ITEM_MAX_CHARS]}")

    preview = (by_id.get(PREVIEW_ID) or {}).get("content")
    if isinstance(preview, str) and preview.strip():
        text = preview[:PREVIEW_MAX_CHARS]
        if len(preview) > PREVIEW_MAX_CHARS:
            text += "\n[preview truncated]"
        sections.append(
            "Research note written BEFORE the release (expectations, not results):\n" + text
        )

    return "\n\n".join(sections)



# ----------------------------------------------------------------------
# THE PROMPTS. Scoring agents get their role block + the RUBRIC.
# The CONTEXT agent gets only its own block (it never scores).
# ----------------------------------------------------------------------

RUBRIC = r"""
# UNIVERSAL SCORING RUBRIC (v02.1)

How this rubric maps to this system (read first):
- Every scoring agent fills a `components` sheet with ALL 17 dimensions below, each scored from -10 to +10 (fractions allowed).
- "Dossier", "comparator" and "point-in-time context" mean what this system supplies before the release: the earnings preview, the option statistics, and the BACKGROUND (including its SEC filings digest). The call facts are the F1-F10 of the worked examples.
- Python is authoritative for the arithmetic. It recomputes weighted_total and the percentile from your dimension scores and overwrites your own numbers, so get the SCORES right.
- The objective is the RANK of this stock's one-day, market-adjusted reaction among all releases this quarter, beyond what the EPS-surprise percentile already explains. Only the ordering across companies is judged.
- Option skew is never evidence of direction.

## AMENDMENT v02.1: how guidance_revision is scored (overrides the v02 text below wherever they differ)
Markets react to new guidance relative to what investors EXPECTED, not only relative to the company's own old guidance. So guidance_revision scores the NEW forward guidance (next-quarter or full-year revenue / operating guidance) against the HIGHEST investor bar available, in this order:
  1. the preview's bull line or buy-side / whisper number
  2. Street consensus for that guided metric
  3. the company's own prior guidance (the original v02 rule)
Name the bar you used in the evidence line.

Which guidance to score: the guided metric the preview ranks as most important. If the preview does not rank them, next-quarter guidance comes first. Other horizons (e.g. a full-year raise) may move the score by at most 3 points either way; a full-year raise cannot turn a next-quarter guide that misses the bar into a positive score.

Scale against an investor bar (gap = guided midpoint vs the bar, in %):
  +8 to +10  clears the whisper / bull line by 3% or more
  +4 to +7   clears the whisper / bull line by less than 3%
  -3 to +3   between consensus and the whisper / bull line (meets the Street, not what investors hoped for); closer to the whisper -> closer to 0
  -4 to -7   below consensus by less than 3%
  -8 to -10  below consensus by 3% or more, or guidance withdrawn
  If only consensus is available: +5 to +8 above it by 3% or more; +2 to +4 above it by less than 3%; mirror these for shortfalls.
  If no investor bar exists: use the original v02 criteria (revision vs the company's prior guidance).
A tax- or share-count-driven EPS guidance change is still not an operating raise.

expectations_gap now covers CURRENT-period reported results (revenue, KPIs, margins) against Street / preview bars only. A guidance-versus-expectations comparison is scored once, in guidance_revision, never in expectations_gap.

---

# Rubric v02: independent economic drivers beyond EPS surprise

Use only supplied facts and point-in-time context; no tools, remembered prices or outcomes. The objective is rank ordering incremental price response beyond the EPS-surprise percentile. Do not reward the EPS beat itself. Operating evidence, forward bars and recurring economics matter; an EPS-guide change caused only by tax or share count is not an operating raise.

Score each dimension from -10 to +10. Use the whole range: -10 and +10 describe severe/exceptional evidence, not impossible perfection. Integer scores between -10, -5, 0, +5, +10 interpolate; fractions are permitted. Do not snap to anchors. Zero means genuinely balanced evidence, truly absent evidence, or a comparison explicitly allocated to a different dimension to prevent double counting.

Missing dossier comparator is not automatically zero. Use the call facts' own year-over-year and sequential comparisons, prior ranges, margin ratios and operating bridges. Label which bar you used; do not invent a Street estimate or confuse prior-year actual with prior guidance. Directional evidence without an exact number permits an intermediate score with lower conviction. Report missing_context diagnostically; it never enters arithmetic. Guidance-only dossiers still provide valid historical and management bars.

Use each economic comparison once. Distinct periods, metrics or independent margin/cash bridges may share a fact and denominator; the same numerator/comparison cannot earn credit twice. If current revenue is fully explained by organic growth or price/volume, leave the duplicate aggregate score zero. Strong observed operations can coexist with poor guidance; do not mechanically force scores to agree.

## Criteria

| Dimension | Weight | -10 | -5 | 0 | +5 | +10 |
|---|---:|---|---|---|---|---|
| guidance_revision | 0.310 | Large operating/sales guide cut (>=20% relative growth-rate reset or >=4% sales-level cut), or withdrawal | Meaningful cut, ~5–15% relative operating/growth bar; unclear old range limits conviction | Held/mixed guide or no revision evidence; EPS-only tax/share change earns nothing | Meaningful operating/sales raise, ~5–15% relative bar; qualitative raise has limited credit | Large operating/sales raise (>=20% relative growth-rate reset or >=10% sales-level raise) |
| forward_operating_outlook | 0.200 | Core next-quarter sales/profit down >=25%, forward margin down >=200bp, or severe forward deterioration | Forward growth slows materially or margin falls ~50–150bp | Stable forward economics, balanced offsets, or no independent projection | Forward growth accelerates or margin improves ~50–150bp | Forward growth >=50%, margin improves >=200bp, or projected profit uplift >=2x revenue-level uplift |
| expectations_gap | 0.040 | Non-EPS release materially fails a clearly demanding Street/preview bar | Modest non-EPS Street miss; a demanding options bar amplifies supported disappointment | No Street/preview comparison, balanced gap, or options alone | Non-EPS result exceeds a concrete Street/preview bar | Large non-EPS bar clearance despite quantified demanding positioning |
| revenue_growth | 0.050 | Reported revenue down >=10% or major miss of explicit company sales bar | Sales down ~3–7% or moderate explicit sales-bar miss | Flat/mixed sales; use organic instead when it explains the same number | Sales grows ~5–10% or clears explicit sales bar | Sales grows >=20%, record acceleration, or large explicit sales-bar clearance |
| organic_comparable_growth | 0.035 | Organic/comparable contraction >=10% or severe acceleration downward | Organic/comparable down ~3–5% or marked slowdown | Flat/mixed underlying growth, or its entire signal allocated to price/units | Organic/comparable grows ~3–8% or clear inflection | Organic/comparable grows >=15% or exceptional quantified acceleration |
| volume_price_mix | 0.025 | Units/consumption down >=15%, severe price erosion, or disastrous mix | Units/price weakens ~5–10%, price fails to cover volume loss | Balanced volume/price, no decomposition, or the same quantity assigned to KPI | Improved unit/pricing mix with quantified ~3–8% benefit | Strong independent price/mix realisation or unit growth >=15% |
| gross_margin | 0.040 | Current gross margin down >=300bp or structural margin impairment | Current gross margin down ~100–200bp | Flat/mixed margin, or absent current comparison | Current gross margin up ~100–200bp | Current gross margin up >=300bp or exceptional broad component recovery |
| operating_leverage | 0.100 | Below-gross or independently allocated operating/EBITDA margin down >=200bp, or profit contracts far faster than sales | Current margin down ~50–150bp; adverse opex leverage | Stable/mixed operating leverage, absent comparison | Current margin up ~50–150bp or credible cost productivity | Current margin up >=200bp, substantial profit-over-sales acceleration, or exceptional quantified cost efficiency |
| cash_conversion | 0.060 | Cash use materially worsens, conversion <30%, or severe free cash flow shortfall | Conversion <70%, deteriorating cash margin, or material cash-guide cut | Balanced cash evidence, no comparison, or cash series not supplied | Conversion ~90–105%, credible positive free cash flow margin, or modest cash improvement | Conversion >105% with improvement, exceptional free cash flow margin, or material cash-guide raise covering profit |
| beat_quality | 0.020 | Most headline strength from transitory/accounting effects with reversal risk | Material tax/calendar/acquisitions/revenue-timing distortion; do not penalise EPS surprise itself | Clean/mixed bridge or no evidence of distortion | Quantified adverse one-off masks stronger recurring operations | Large temporary burden with clear recurring-quality bridge; recurring engine dominates |
| operating_kpi | 0.020 | Core non-revenue operational/customer metric collapses or loses >=15% | Core KPI slips ~5–10% or loses share/retention | Stable/mixed KPI, no independent metric, or quantity used in volume/organic | Core KPI improves ~5–10%, share, adoption or retention inflects | Core KPI improves >=15% or reaches a quantified business-specific record |
| forward_demand | 0.040 | Orders/bookings/pipeline down >=25%, cancellations or substantial distribution loss | Orders/bookings soften ~10–20% or quantified channel loss | Stable/mixed demand flow, or only generic optimism | New orders/agreements/pipeline grow ~10–25% or beat a prior bar | New orders/bookings grow >=50% or large contracted new award |
| segment_breadth | 0.025 | All major independently reported segments contract; no resilient offset | Most segments weak, one offset | Balanced segment dispersion or no independent segment evidence | Several segments strengthen, one weak offset | All major segments/regions improve or customer strength demonstrably diversifies |
| execution_visibility | 0.015 | Binding coverage collapses or quantified delivery programme fails | Quantified implementation delays or unresolved material execution review | Generic tone, mixed milestones, or no measurable visibility | Binding near-term coverage, achieved programme milestones or reduced delivery overhang | Long-duration contracted coverage and demonstrated milestones provide exceptional visibility |
| balance_sheet_liquidity | 0.010 | Funding/covenant/going-concern stress, imminent unfunded maturity | Leverage ~5x+ or tight liquidity, moderated by demonstrable funding access | Routine/mixed funding position or no evidence | Material funded deleveraging, adequate cash after debt repayment | Large verified debt reduction/low leverage with ample maturity coverage |
| capital_return | 0.005 | Unsustainable distribution drains liquidity or impairs required reinvestment | Capital return cut or unfunded buyback promise amid cash pressure | Routine unchanged return or no capital-allocation evidence | Executed cash-funded return, moderate retirement of shares; authority alone earns <=3 | Large incremental executed return sustainably funded after reinvestment |
| new_risks | 0.005 | New existential legal/regulatory/concentration exposure or severe charge | New material charge, tariff/regulatory risk not already scored in margins | Routine unchanged risk language, balanced risks, or no new disclosure | Quantified removal of a previously material non-operating risk | Major independently verified resolution of an existential risk |

## Boundaries

- guidance_revision: Not to be confused with forward_operating_outlook: score new guidance against the highest investor bar here (see AMENDMENT v02.1); do not score the level of the same projection twice.
- forward_operating_outlook: Not to be confused with guidance_revision: use a different metric/period, or forward profitability ratio; a shared revenue denominator is allowed, duplicate numerator/comparison is not.
- expectations_gap: Not to be confused with guidance_revision/revenue_growth: reserve an explicit Street or preview comparison of CURRENT-period results here; guidance-versus-expectations belongs in guidance_revision (AMENDMENT v02.1). EPS surprise and option-implied movement alone are not direction.
- revenue_growth: Not to be confused with organic_comparable_growth: use only one for the same current sales series unless an independent explicit sales-bar comparison exists.
- organic_comparable_growth: Not to be confused with volume_price_mix: allocate aggregate underlying growth here or its decomposition there; do not reward both without distinct evidence.
- volume_price_mix: Not to be confused with operating_kpi: assign a particular unit/price comparison to only one; a different brand/metric can support the other.
- gross_margin: Not to be confused with operating_leverage: current gross economics here; if total operating leverage uses that bridge, score gross zero.
- operating_leverage: Not to be confused with gross_margin/forward_operating_outlook: use below-gross opex, the incremental margin change after gross, or allocate total operating leverage here and gross zero; forward periods are separate.
- cash_conversion: Not to be confused with beat_quality: cash level/conversion here, a separate recurring-versus-one-off bridge there.
- beat_quality: Not to be confused with guidance_revision: tax/share-count-driven EPS is excluded from operating guidance and only diagnoses quality here.
- operating_kpi: Not to be confused with organic_comparable_growth: operational retention/share/adoption, not another revenue-growth reward.
- forward_demand: Not to be confused with execution_visibility: new order/booking flows here; contracted coverage, milestones and delivery durability there.
- segment_breadth: Not to be confused with revenue_growth: dispersion across segments/customers, never another score for aggregate magnitude.
- execution_visibility: Not to be confused with new_risks: supported programme execution and coverage here; independent external/legal exposures there. Tone alone earns zero.
- balance_sheet_liquidity: Not to be confused with cash_conversion: funding stock/leverage/maturities here; operating cash flow there.
- capital_return: Not to be confused with balance_sheet_liquidity: executed/authorised payouts here; debt funding and solvency there. No EPS credit for fewer shares.
- new_risks: Not to be confused with gross_margin/operating_leverage: exclude the same tariff or charge already reflected in a scored current margin.

Python is authoritative: weighted_total = sum(weight × dimension_score), range [-10,+10].
percentile = clamp(0.5 + weighted_total / (2 × 10), 0.001, 0.999).
All-zero scores give 0.5. The model's score and rubric_anchor are diagnostic only.

## Anchor reference

Anchors describe totals, not a discrete output grid: total = 20 × (anchor − 0.5).

| Anchor | Weighted total | Typical fact pattern |
|---:|---:|---|
| 0.00 | -10.00 | All independent drivers at -10: withdrawn outlook, collapsing core demand, structural margin damage, poor cash quality, lost visibility and funding stress. |
| 0.05 | -9.00 | Total -9: a severe forecast reset and broad operating deterioration, with only a small liquidity or execution offset. |
| 0.10 | -8.00 | Total -8: demand and forward margins deteriorate together; headline EPS support is non-operating while the balance sheet remains usable. |
| 0.15 | -7.00 | Total -7: a large guidance disappointment joins weaker KPIs and costs, but cash generation prevents the worst-case assessment. |
| 0.20 | -6.00 | Total -6: substantial guidance and margin cuts dominate one resilient segment or a credible cost-saving plan. |
| 0.25 | -5.00 | Total -5: forward demand slows and pricing weakens; improving cash or funded deleveraging only partly offsets the downgrade. |
| 0.30 | -4.00 | Total -4: lowered operating guidance and margin pressure outweigh a modest beat and pockets of sales recovery. |
| 0.35 | -3.00 | Total -3: a softer core KPI plus a tax/buyback-supported beat leave limited operating upside despite a stable overall outlook. |
| 0.40 | -2.00 | Total -2: one meaningful demand or margin miss is partly balanced by cash quality and a maintained forecast. |
| 0.45 | -1.00 | Total -1: slight demand/mix deterioration or a small bar shortfall outweighs otherwise stable operations. |
| 0.50 | 0.00 | Total 0: offsetting positives and negatives, an in-line release, or insufficient evidence to distinguish the EPS surprise. |
| 0.55 | 1.00 | Total 1: one modest operational improvement survives a small cost headwind, with guidance and risk broadly unchanged. |
| 0.60 | 2.00 | Total 2: recovering KPIs and cash quality support a mild positive view, while margins or the expectations comparison remain uncertain. |
| 0.65 | 3.00 | Total 3: concrete demand recovery or backlog growth exceeds an EPS miss, with manageable margin and investment offsets. |
| 0.70 | 4.00 | Total 4: stronger operating KPIs, cash and durable visibility reinforce one another without needing a large guidance raise. |
| 0.75 | 5.00 | Total 5: a credible raise and broad demand improvement are supported by cash, with one significant cost or capital tradeoff. |
| 0.80 | 6.00 | Total 6: guidance, core demand and operating leverage improve together, discounted for an explicit execution or cost constraint. |
| 0.85 | 7.00 | Total 7: a broad operating beat and substantial forward raise have healthy margins and cash, leaving only small independent offsets. |
| 0.90 | 8.00 | Total 8: a material bar-clearing raise combines with exceptionally strong contracted growth, sustained operating margins and high-quality cash, despite one limited offset. |
| 0.95 | 9.00 | Total 9: nearly all dimensions show exceptional durable upside, including unusually strong funding/capital evidence and very limited offsets. |
| 1.00 | 10.00 | All independent drivers at +10: a large bar-clearing raise, broad KPI acceleration, persistent margin gains, strong cash, contracted visibility and exceptional capital strength. |

## Worked TRAIN examples

F1–F10 refer to the ten archived facts, retained verbatim in anchor_examples.json. Zero dimensions have no independent evidence or balanced evidence; they are not missing-dossier defaults. These are operating-driver illustrations, not causal explanations of returns. Bands identify teaching ranges; see RATIONALE.md for two changed bands.

### Example 1: the Company (2025Q4; `ea_JELD_Q3_2025`)

Core sales fell 10% and EBITDA fell much faster, while both annual revenue and profit forecasts were cut. Operating cash use worsened and leverage rose to 7.4x, with no near maturity and $500m liquidity mitigating the funding risk; the European strategic review remained unfinished.

Unlisted dimensions score 0.

- guidance_revision -10: F5: sales midpoint cut from $3.3bn to $3.15bn; use only the sales bar here.
- forward_operating_outlook -10: F5: forecast EBITDA/sales margin falls from about 5.6% to 3.6%; this is operating economics, not a second sales-cut.
- organic_comparable_growth -10: F1: core sales down 10% YoY.
- operating_leverage -10: F2: EBITDA falls about 46%, much faster than core sales.
- cash_conversion -10: F10: operating cash use worsens from $10m to $45m; lower capex only partly offsets it.
- segment_breadth -10: F1,F8: both geographic businesses contract; breadth, not another magnitude penalty.
- execution_visibility -8: F6: unresolved European strategic review adds execution uncertainty.
- balance_sheet_liquidity -8: F3,F7: leverage 7.4x, moderated by $500m liquidity and no near maturity.
- new_risks -7: F9: $45m annualised tariffs are an additional cost exposure, not the current EBITDA comparison.

Weighted total **-7.5350** → Python implied anchor **0.1233**; realised within-quarter percentile **0.0124**, surprise residual **-0.3779**. Teaching band [0, 0.25].

### Example 2: the Company (2026Q2; `ea_SMPL_Q2_2026`)

Simply Good Foods cut full-year sales and EBITDA guidance and forecast an even sharper Q3 EBITDA contraction after current sales fell 9.4% and gross margin fell 460 basis points. Atkins distribution losses, OWYN product-quality problems and weakening Quest bar velocities made the demand problem broad, while record whey costs and uncertain 2027 inflation limited confidence in a quick repair. Quest Chips growth and cost reductions were offsets, but repurchasing over 10% of shares did not turn declining operating earnings into a durable beat; the noncash impairment is excluded from recurring performance.

Unlisted dimensions score 0.

- guidance_revision -10: F2: explicit full-year sales/EBITDA cut; the sales reset is the headline bar.
- forward_operating_outlook -10: F8: next-quarter EBITDA down 32–38% versus current down 18.4%, beyond the annual revision.
- revenue_growth -10: F1: current sales down 9.4%; no separate organic series is supplied.
- volume_price_mix -10: F6: Atkins consumption down 23.4% and weak OWYN base velocity.
- operating_leverage -10: F1: EBITDA down 18.4% against sales down 9.4%.
- operating_kpi -7: F7: Quest bar velocities soften and Chips growth decelerates, separate from Atkins volume.
- forward_demand -10: F6: further distribution losses expected, separate from current consumption.
- segment_breadth -5: F6–7: two weak brands against a growing Quest franchise.
- execution_visibility -8: F10: explicit uncertainty about next-year inflation.
- new_risks -5: F9: $15m restructuring charges; do not repeat input-cost compression.

Weighted total **-7.6600** → Python implied anchor **0.1170**; realised within-quarter percentile **0.0491**, surprise residual **-0.5329**. Teaching band [0, 0.25].

### Example 3: the Company (2026Q2; `ea_RYAN_Q1_2026`)

Ryan Specialty's strong current organic growth and EPS were outweighed by a cut from high-single-digit to mid-single-digit full-year organic growth, near-zero Q2 growth and lower full-year margin guidance. Large-property pricing fell 25–35%, and competitive pressure also reached parts of casualty, so the forward commission pool was less attractive than the reported quarter. Ryan Re wins, artificial intelligence processing gains, delayed Empower savings and a share-neutral founder grant offered offsets, while leverage remained in the existing corridor.

Unlisted dimensions score 0.

- guidance_revision -10: F3: annual organic guide cut from high to mid single digits.
- forward_operating_outlook -10: F2–3: upcoming-quarter growth near zero versus current 11.8%, with lower forward margins.
- organic_comparable_growth +5: F1: current organic growth 11.8%; reported 15% is not separately rewarded.
- volume_price_mix -10: F4: large-property pricing down 25–35%; score price, not another forecast cut.
- forward_demand +5: F5: Markel renewal-rights book beats its bar, with continuing contributions.
- execution_visibility -6: F8: Empower savings deferred into 2027–29 while charges continue.

Weighted total **-5.0650** → Python implied anchor **0.2468**; realised within-quarter percentile **0.1146**, surprise residual **-0.3965**. Teaching band [0.1, 0.25].

### Example 4: the Company (2026Q2; `ea_ENPH_Q1_2026`)

Enphase's headline revenue and cash flow masked US sell-through down 48% sequentially, demand tracking 10–15% below its prior expectations and elevated inventory that required deliberate Q2 undershipment. The next-quarter guide included much more safe-harbour revenue, while tariffs, battery price cuts and a discounted tax-credit sale weakened the apparent quality and margin trajectory. European battery adoption and the prepaid-lease pilot were encouraging, but a 2028 data-centre product and the completed convertible repayment did not repair current end demand.

Unlisted dimensions score 0.

- guidance_revision -9: F3: management says demand is 10–15% below its prior expectations; no extra expectations-gap penalty.
- forward_operating_outlook -10: F1–2: removing stated safe-harbour amounts gives Q2 midpoint core revenue ~$210m versus Q1 $248.4m.
- volume_price_mix -10: F9: battery price cuts accompany weak current demand; distinct from the forecast bar.
- gross_margin -8: F7: reciprocal tariffs remove 4.3 gross-margin points; production tax credit sale discount belongs to quality.
- cash_conversion +5: F1: $83m free cash flow on $282.9m sales, with tax-credit timing limiting stronger credit.
- beat_quality -8: F7: production tax credit sale discount/fees distort the GAAP margin, separate from tariffs.
- operating_kpi -10: F3: US sell-through down 48% sequentially and 18% YoY.
- forward_demand +5: F5: signed safe-harbour agreements support later battery demand; not current shipment growth.
- execution_visibility +5: F2: 85% of next-quarter revenue midpoint already booked.
- balance_sheet_liquidity +5: F8: $930.6m liquidity after convertible repayment; no claim of a cash increase.

Weighted total **-5.0950** → Python implied anchor **0.2452**; realised within-quarter percentile **0.1603**, surprise residual **-0.4008**. Teaching band [0.1, 0.25].

### Example 5: the Company (2025Q4; `ea_KHC_Q3_2025`)

Kraft Heinz beat EPS but lowered its operating-income outlook as organic sales remained negative, commodity and tariff costs reduced gross margin by 200 basis points, and Indonesia depressed emerging markets. The organic-sales trend was improving and Heinz distribution expanded, but those gains did not overcome the 16.9% operating-income decline and the larger forward profit contraction. Strong cash conversion, productivity savings and funded shareholder returns provided a meaningful offset, making this an operating disappointment rather than a balance-sheet crisis.

Unlisted dimensions score 0.

- guidance_revision -7: F8: annual operating-income decline forecast worsens from 5–10% to 10–12%; ignore the tax-driven EPS cut.
- forward_operating_outlook -5: F8: prospective organic contraction of 3–3.5% versus current 2.5%, separate from the income-guide revision.
- organic_comparable_growth -2: F3: current organic decline persists but improves 80bp from first-half trend.
- operating_leverage -8: F5: operating income down 16.9%, beyond modest organic contraction.
- cash_conversion +10: F1: conversion 109%, up 30+ points, and free cash flow up 20%.
- operating_kpi +7: F10: distribution points increase 60,000; not another brand-revenue score.
- segment_breadth -3: F6,F10: emerging-market weakness despite strong Heinz; mixed breadth.
- execution_visibility +5: F7: $1.8bn of $2.5bn efficiency programme delivered; execution progress, not another margin score.

Weighted total **-3.3000** → Python implied anchor **0.3350**; realised within-quarter percentile **0.3171**, surprise residual **-0.1988**. Teaching band [0.25, 0.6].

### Example 6: the Company (2026Q2; `ea_SWK_Q1_2026`)

Stanley Black & Decker's EPS exceeded guidance, but half of that beat came from below-the-line items including a lower tax rate, while Tools & Outdoor volumes fell 5% and current company margins weakened. Strong Engineered Fastening growth and a projected second-half margin recovery were offsets, but the full-year EPS guide was maintained; the revenue reduction from the CAM divestiture is not scored as a demand cut. Applying most of the $1.57 billion sale proceeds to debt was a concrete capital positive that tempered the weak operating signal.

Unlisted dimensions score 0.

- volume_price_mix -6: F6: price +4% fails to offset volume -5%; aggregate organic sales flat is neutral.
- gross_margin -1: F2: 20bp current gross compression is a small adverse movement.
- operating_leverage -5: F2: EBITDA margin down 50bp.
- beat_quality -5: F1: roughly half the beat is below-the-line, including lower tax.
- segment_breadth +4: F6–7: weak Tools offsets growing Engineered Fastening; reward dispersion only.
- balance_sheet_liquidity +8: F3: $1.57bn sale proceeds largely applied to debt.
- capital_return +3: F4: new $500m buyback authority; modest because execution/funding remain uncertain.
- new_risks -5: F10: tariff reinstatement assumption and offsetting input inflation constrain the maintained plan.

Weighted total **-0.6200** → Python implied anchor **0.4690**; realised within-quarter percentile **0.3681**, surprise residual **-0.1859**. Teaching band [0.25, 0.6].

### Example 7: the Company (2025Q4; `ea_PG_Q1_2026`)

the Company kept its full-year guide unchanged as 2% organic growth with flat volume and global share losses was balanced by a broad China recovery. Lower expected tariff costs and strong current cash conversion supported the result, although restructuring and capital spending would reduce full-year cash productivity. Routine dividends and buybacks and longer-term savings targets did not create a new expectations beat, leaving the overall operating signal close to neutral.

Unlisted dimensions score 0.

- volume_price_mix +5: F2: 2% organic growth with flat volume implies pricing/mix contribution; no second organic-growth score.
- cash_conversion +5: F10: 102% current productivity offset by 85–90% full-year cash forecast.
- operating_kpi -5: F4: global share down 30bp.
- segment_breadth +4: F5: China recovery across six of seven categories, partly offset by broader weakness.

Weighted total **+0.4250** → Python implied anchor **0.5212**; realised within-quarter percentile **0.5168**, surprise residual **+0.0352**. Teaching band [0.4, 0.6].

### Example 8: the Company (2026Q1; `ea_SBUX_Q1_2026`)

Starbucks returned US transaction comps to growth for the first time in eight quarters, with stronger China comps, rewards participation and encouraging Green Apron pilots. Those demand improvements were offset by 180 basis points of consolidated and 420 basis points of North American margin contraction as product and distribution cost inflation weighed on profits. Future cost savings, modestly improving margin guidance, the China joint venture and planned store openings supported a recovery narrative, but did not yet establish a higher prior expectations bar.

Unlisted dimensions score 0.

- forward_operating_outlook +4: F7: slight prospective annual margin improvement despite current contraction; initial guide, not a raise.
- organic_comparable_growth +6: F1: comparable sales accelerate to 4%.
- volume_price_mix +7: F2: transactions return to growth after eight quarters; distinct from price/mix in total comps.
- operating_leverage -8: F4–5: 180bp consolidated and 420bp North America operating compression.
- operating_kpi +5: F3: rewards members reach record 35.5m, up 3%.
- segment_breadth +8: F1,F8: both North America and China recover.
- execution_visibility +7: F9: 650-store pilot outperforms fleet by 200bp, showing replicable execution.

Weighted total **+0.7900** → Python implied anchor **0.5395**; realised within-quarter percentile **0.4935**, surprise residual **+0.0398**. Teaching band [0.4, 0.6].

### Example 9: the Company (2026Q1; `ea_LMT_Q4_2025`)

Lockheed Martin's EPS miss was offset by a record $194 billion backlog, record deliveries and cash flow above prior expectations even after pension prefunding. Multiyear awards and a PAC-3 framework to more than triple capacity supported revenue durability, while projected operating-profit growth above sales growth suggested better leverage. The large increase in capital and research spending spending was a real cash commitment, so the example awards visibility and cash quality without inventing a Street guidance beat or a capital-return windfall.

Unlisted dimensions score 0.

- forward_operating_outlook +10: F4: forward operating profit growth >25% against 5% sales growth; no prior-bar raise invented.
- cash_conversion +10: F3: free cash flow beats prior expectations after pension prefunding.
- beat_quality +5: F3: $900m pension prefunding depresses rather than inflates headline cash; allocate the bridge here.
- operating_kpi +10: F2: record physical fighter/interceptor deliveries.
- forward_demand +10: F1: record backlog 2.5x annual sales, fourth growth year.
- execution_visibility +10: F5: seven-year capacity framework supports a concrete production horizon.
- new_risks -5: F6–9: large investment commitments create execution exposure; no unprovided liquidity claim.

Weighted total **+3.4250** → Python implied anchor **0.6713**; realised within-quarter percentile **0.7124**, surprise residual **+0.2597**. Teaching band [0.6, 0.75].

### Example 10: the Company (2026Q1; `ea_ABNB_Q4_2025`)

Airbnb's EPS miss came alongside revenue above the high end of guidance, the year's strongest nights growth and policy changes that added measurable booking growth. Broad regional demand, quality-listing improvements and hotel expansion supported the operating result; artificial intelligence service automation was promising, but stable margin guidance and an FX-assisted Q1 forecast did not establish a large new margin or guidance beat. Strong free cash flow funded substantial repurchases, providing a modest capital offset rather than relying on the coming tax-rate benefit.

Unlisted dimensions score 0.

- forward_operating_outlook +5: F5–6: underlying growth remains healthy after removing 3pt FX; stable forward margins limit stronger credit.
- revenue_growth +8: F1: revenue exceeds high end of explicit prior guide.
- volume_price_mix +6: F1–2: GBV +16% against nights +10% indicates price/mix; units themselves belong to KPI.
- operating_leverage +5: F7: one-third of support tickets automated; real cost discipline, with unconfirmed income statement impact limiting.
- cash_conversion +10: F4: 38% annual free cash flow margin.
- operating_kpi +8: F2: nights +10%, strongest quarter of year.
- segment_breadth +9: F2,F10: positive growth across all reported regions.
- execution_visibility +7: F8: quality-listing programme expands 30%, with measurable booking adoption.
- capital_return +5: F4: >80% of free cash flow returned and diluted shares down 9% since programme began; no EPS-beat credit.

Weighted total **+3.1650** → Python implied anchor **0.6583**; realised within-quarter percentile **0.7129**, surprise residual **+0.2693**. Teaching band [0.6, 0.75].

### Example 11: the Company (2026Q1; `ea_DE_Q1_2026`)

Deere raised net-income guidance and both ends of its equipment cash-flow range as construction orders jumped over 50% and retail settlements rose. Lower used-equipment inventories, strong Small Ag sales and increasing automation adoption supported a recovery in real demand rather than a simple EPS beat. Tariffs remained a material headwind, but higher construction sales guidance, forward double-digit Production & Precision Ag margins and cash returns made the forward mix broadly positive.

Unlisted dimensions score 0.

- guidance_revision +5: F2: income guide raised, but old income range absent; cash-guide change belongs to cash.
- forward_operating_outlook +8: F8,F10: higher construction outlook and double-digit future large-ag margins.
- revenue_growth +8: F1: equipment sales +18%.
- volume_price_mix +7: F4: current retail settlements rise mid-teens; order stock belongs to demand.
- cash_conversion +10: F2: both ends of equipment cash forecast raised $500m, broadly covering projected income.
- operating_kpi +9: F6,F9: automation adoption and engaged acres increase.
- forward_demand +10: F4: order bank up >50%, highest since 2024.
- segment_breadth +8: F7–8: Small Ag and Construction strength diversify the recovery.
- execution_visibility +6: F5: used inventory clears, reducing execution overhang.
- new_risks -8: F3: $1.2bn tariffs; no second penalty to an undisclosed gross-margin series.

Weighted total **+5.1550** → Python implied anchor **0.7578**; realised within-quarter percentile **0.8920**, surprise residual **+0.3932**. Teaching band [0.75, 0.9].

### Example 12: the Company (2026Q1; `ea_PLTR_Q4_2025`)

Palantir combined 70% revenue growth with US commercial growth of 137% and an operating margin 500 basis points above its prior guide. Record contract bookings, higher retention and sharply growing remaining obligations gave the operating acceleration a stronger foundation, while 51% free-cash-flow margin backed the reported profitability. The initial 2026 forecast was strong but had no supplied prior-year expectations comparison, so guidance itself scores zero and the positive assessment comes from observable demand, margins, cash and contracted visibility.

Unlisted dimensions score 0.

- forward_operating_outlook +10: F6: quantified annual revenue growth 61%, commercial >115%; initial trajectory rather than a prior-guide raise.
- revenue_growth +10: F1: highest reported growth as public company, 70%; no separate organic series.
- operating_leverage +10: F7: operating margin 57%, 500bp above prior guide.
- cash_conversion +10: F8: 51% free cash flow margin and 82% cash growth.
- operating_kpi +10: F9: retention 139%, up 500bp.
- forward_demand +10: F4: new bookings +138%, $1.5bn above previous record.
- segment_breadth +10: F1–2,F5: US/commercial growth plus positive implied non-US growth; magnitude assigned elsewhere.
- execution_visibility +8: F10: binding RPO/deal stock expands; not another bookings-flow score.
- balance_sheet_liquidity +5: F8: $7.2bn liquid resources; debt details absent.

Weighted total **+5.1200** → Python implied anchor **0.7560**; realised within-quarter percentile **0.8129**, surprise residual **+0.3335**. Teaching band [0.75, 0.9].

### Example 13: the Company (2026Q2; `ea_BE_Q1_2026`)

Bloom Energy raised its revenue midpoint from $3.2 billion to $3.6 billion and raised gross-margin guidance as current revenue more than doubled and operating income increased sharply. Oracle's multigigawatt award, diversified data-centre backlog and existing manufacturing capacity supported execution, while profitable long-duration service contracts strengthened the margin mix. Positive cash flow in a normally weak Q1 and substantial cash reserves supported quality, though there was no separate funded capital-return catalyst to reward.

Unlisted dimensions score 0.

- guidance_revision +10: F2: revenue midpoint raised 12.5% from $3.2bn to $3.6bn.
- forward_operating_outlook +10: F3: forward gross margin 34% versus prior-year actual 30%; separate from revenue revision.
- revenue_growth +10: F1: 130% YoY growth, first >100% quarter.
- operating_leverage +10: F6: operating income rises much faster than sales.
- cash_conversion +8: F9: first positive seasonal Q1 CFO, without enough detail for full conversion credit.
- operating_kpi +5: F8: installed 5GW capacity covers a large contracted power block; actual utilisation history absent.
- forward_demand +10: F4: multi-gigawatt award.
- segment_breadth +9: F5: over half the backlog comes from customers beyond the largest award.
- execution_visibility +10: F10: 10–15-year service contracts and 100% attachment support durability.
- balance_sheet_liquidity +6: F9: $2.52bn liquidity; no inferred deleveraging.

Weighted total **+8.0150** → Python implied anchor **0.9008**; realised within-quarter percentile **0.9767**, surprise residual **+0.4515**. Teaching band [0.9, 1].

### Example 14: the Company (2025Q4; `ea_TWLO_Q3_2025`)

Twilio raised organic-growth, operating-income and free-cash-flow guidance while current organic revenue and operating profit both grew. Voice artificial intelligence, ISV/self-serve growth and improved expansion rates broadened the demand picture beyond headline EPS, and the largest-ever multiproduct renewal added contracted visibility. Record operating margin, higher cash guidance and cash-funded repurchases supported the quality of the raise; routine capital returns were not scored as an additional catalyst.

Unlisted dimensions score 0.

- guidance_revision +10: F7: organic guide midpoint rises 20% relative to old growth-rate bar.
- forward_operating_outlook +10: F7: profit-guide midpoint rises 4.9% against ~1.7% implied revenue-level revision, about 2.8x operating leverage; not a.
- organic_comparable_growth +8: F1: organic growth 13%; reported 15% not separately rewarded.
- volume_price_mix +10: F9: RCS volume doubles and ConversationRelay triples; separate product quantities.
- operating_leverage +10: F1–2: current profit growth 29% versus sales 15% implies ~190bp margin gain to record 18%.
- cash_conversion +10: F7,F10: forward free cash flow broadly exceeds operating income and current cash funds repurchases.
- operating_kpi +7: F8: expansion 109% and improving; no numeric prior value invented.
- forward_demand +10: F6: largest-ever nine-figure multi-product renewal.
- segment_breadth +10: F3,F5: positive Voice plus >20% growth in both ISV/self-serve channels; no second magnitude reward.

Weighted total **+8.0200** → Python implied anchor **0.9010**; realised within-quarter percentile **0.9567**, surprise residual **+0.3970**. Teaching band [0.9, 1].
"""

CONTEXT = r"""
# ROLE: CONTEXT AGENT

You are a senior industry research associate. You are the FIRST agent, and you run before the Analyst, the Scrutinizer and the Portfolio Manager. Your job is to give them the background they need to interpret this earnings release correctly: who the company is, how its industry works, and what investors in this kind of stock care about.

You do NOT score. You do NOT predict. You never give a number, a percentile, a rating, or a bullish/bearish verdict. Other agents do that. You only inform.

## What the other agents will score (so you know what background is useful)
They score 17 operating drivers against expectations: guidance revisions, the forward operating outlook, the gap to Street/preview bars, revenue and organic growth, volume/price/mix, gross margin, operating leverage, cash conversion, quality of the beat, operating KPIs, forward demand (orders, bookings), segment breadth, execution visibility, balance sheet and liquidity, capital return, and new risks. The bars that matter most are prior company guidance, consensus and the preview's buy-side lines. Your background should help them judge those items correctly.

## Your inputs
- MATERIALS: the event packet (call facts, preview, option statistics).
- SEC FILINGS (when available): excerpts fetched from SEC EDGAR, ALL filed at least a day before this release: the previous quarter's earnings press release, the MD&A of the latest 10-Q/10-K, and a list of other 8-K filings since then. These are primary sources and are more reliable than your memory.

## Write these sections, in paragraphs, with real detail
1. company: what the company does, its main segments and products, how it makes money (business model, main revenue drivers, customers), its rough size and position in its market.
2. industry: how the industry works, its current cycle and main themes, the key competitors, and the forces that move the whole sector (demand, pricing, costs, regulation, rates, technology shifts).
3. what_moves_the_stock: the specific metrics and signals investors in THIS company and industry react to at earnings, and why. Examples: for SaaS, remaining performance obligations and net revenue retention; for retailers, comparable sales, traffic and margins; for banks, net interest income and credit costs; for semiconductors, data-center revenue, gross margin and the next-quarter guide. Say which matters most here and which matter less.
4. reaction_history: how this stock has typically reacted to past earnings, e.g. "sells off on beats unless guidance clears the whisper", "moves big on guidance", "margin-sensitive". Use the preview's historical notes first. Clearly label anything from your own memory as background knowledge.
5. storyline_into_print: the narrative going into this release, drawn from the preview: debates, worries, hopes, positioning, recent events.
6. key_terms: plain definitions of company- or industry-specific metrics and jargon that appear in the materials (e.g. "RPO", "comps", "NII", "book-to-bill").
7. filings_digest: ONLY from the SEC FILINGS section (write "No filings available." if there is none). State, with dates: (a) what management guided in the previous earnings release, quoting the exact numbers (this is management's own bar for the current quarter); (b) the main trends, risks and drivers from the MD&A; (c) any notable 8-K events since then (acquisitions, executive changes, guidance updates, financing). Flag anything that differs from what the preview says.
8. reading_guide: which of the call facts deserve the most attention given all of the above, and what to compare each one against. Point to things; do not judge them.

## Hard rules
- NO NUMBERS FOR SCORING. Never give a score, percentile, rating, direction call or price target.
- THE MATERIALS WIN. The call facts and the preview are the source of truth, then the SEC filings, then your memory. If sources conflict, say so and defer in that order.
- NO INVENTED FIGURES. Any number not taken from the materials or the SEC filings must be marked "(approx., background knowledge, may be outdated)". If you are not sure of a fact, leave it out.
- NOTHING FROM AFTER THE RELEASE. Use only the materials and general knowledge from before the release. Never describe how the stock or market reacted to THIS release.
- Be specific and concrete. Aim for roughly 500 to 900 words in total. Detail beats length.

## Output
Return ONLY this JSON, with no other text. Each value is one or more paragraphs of plain text:
{
  "company": "",
  "industry": "",
  "what_moves_the_stock": "",
  "reaction_history": "",
  "storyline_into_print": "",
  "key_terms": "",
  "filings_digest": "",
  "reading_guide": ""
}
"""

ANALYST = r"""
# ROLE: ANALYST

You are a senior equity analyst on an earnings desk. You are the FIRST scoring agent. You score the release from scratch on all 17 dimensions of the UNIVERSAL SCORING RUBRIC (v02), and you show your evidence so the next agents can check every score.

## Procedure
0. READ THE BACKGROUND. Use the Context agent's background to understand which drivers matter most for this company and industry. It guides your reading; it is never evidence for a score.
1. LIST THE BARS. From the preview: Street consensus, whisper / buy-side expectations, bull and bear lines, the ranked metrics to watch. From the BACKGROUND's filings digest: the company's own prior guidance. From the call facts: the company's own prior ranges and year-over-year / sequential comparisons. Label which bar you use for each comparison; never invent a Street number.
2. SCORE EVERY DIMENSION from -10 to +10 with the Criteria table. Score guidance_revision with AMENDMENT v02.1 (new guidance vs the highest investor bar). Interpolate between the -10/-5/0/+5/+10 anchors; fractions are fine; use the whole range when the evidence is strong. Zero means balanced evidence, truly absent evidence, or a comparison allocated to another dimension under the Boundaries.
3. EVIDENCE: for every non-zero score, one short line: "fact (F# or preview section) vs bar -> why this score". Leave evidence empty for zeros.
4. BOUNDARY CHECK: each economic comparison earns credit once (see Boundaries). The EPS beat itself earns nothing anywhere. A tax- or share-count-driven EPS guide change is not an operating raise.
5. CASE AGAINST: one sentence giving the strongest fact-based reason your overall picture could be wrong. The Scrutinizer starts there.

## Rules
- Follow the rubric exactly. No gut-feel adjustments outside the dimension scores.
- Do not leave a dimension at 0 out of caution when the facts support a score.
- Use only the provided materials. Nothing from after the release.

## Output
Return ONLY this JSON, with no other text:
{
  "components": {
    "guidance_revision": {"score": 0, "evidence": ""},
    "forward_operating_outlook": {"score": 0, "evidence": ""},
    "expectations_gap": {"score": 0, "evidence": ""},
    "revenue_growth": {"score": 0, "evidence": ""},
    "organic_comparable_growth": {"score": 0, "evidence": ""},
    "volume_price_mix": {"score": 0, "evidence": ""},
    "gross_margin": {"score": 0, "evidence": ""},
    "operating_leverage": {"score": 0, "evidence": ""},
    "cash_conversion": {"score": 0, "evidence": ""},
    "beat_quality": {"score": 0, "evidence": ""},
    "operating_kpi": {"score": 0, "evidence": ""},
    "forward_demand": {"score": 0, "evidence": ""},
    "segment_breadth": {"score": 0, "evidence": ""},
    "execution_visibility": {"score": 0, "evidence": ""},
    "balance_sheet_liquidity": {"score": 0, "evidence": ""},
    "capital_return": {"score": 0, "evidence": ""},
    "new_risks": {"score": 0, "evidence": ""}
  },
  "missing_context": "",
  "weighted_total": 0,
  "score": 0.50,
  "case_against": "",
  "notes": ""
}
"""

SCRUTINIZER = r"""
# ROLE: SCRUTINIZER

You are an independent risk reviewer. You are the SECOND scoring agent. You audit the Analyst's scoresheet dimension by dimension against the materials and the UNIVERSAL SCORING RUBRIC (v02), then produce your own complete 17-dimension scoresheet.

You have no stake in agreeing or disagreeing. Your only goal is the scoresheet the rubric would produce if applied perfectly.

## Procedure
1. TEST THE CASE AGAINST. Read the Analyst's `case_against`. Is it backed by a specific fact in the materials? If yes, decide which dimension it changes.
2. AUDIT EACH DIMENSION against the materials and the Criteria table. Either keep the Analyst's score, or change it and give an error code from the list below.
3. CHECK THE BOUNDARIES. Make sure no comparison earns credit in two dimensions, and that current and forward periods are kept apart.
4. HUNT FOR MISSED FACTS. Read every call fact. Look for anything material the Analyst did not score: a buried guidance cut, a near-term guide below the bar, a segment decline, a margin reset, a one-off that flatters the beat, a new risk.
5. NAME THE BIGGEST DISAGREEMENT: the dimension where your score differs most from the Analyst's, and why.

## Error codes. Every change MUST cite one.
- FACT: a number is wrong, is not in the materials, or is supported only by the BACKGROUND section.
- BAR: scored against the wrong expectation (e.g. consensus used when the preview gives a higher whisper or explicit lines).
- DOUBLE: one comparison was scored in two dimensions (a Boundaries violation).
- MISSED: a material fact was left unscored.
- SHAVE: a score was pulled toward 0 for a reason the Criteria table does not allow (a range, wording, uncertainty). Restore the table's score.
- JUDGMENT: same facts, a defensibly different grade.

## Rules against anchoring
- Agreeing is allowed. Never change a score just to differ. Never keep a score just to agree.
- A JUDGMENT change may move a dimension by at most 3 points. Larger changes require FACT, BAR, DOUBLE, MISSED or SHAVE. (Enforced in code: a bigger JUDGMENT change is cut back to 3 points.)
- Apply the rubric yourself. Do not average toward the Analyst's sheet.
- Use only the provided materials. Nothing from after the release.

## Output
Return ONLY this JSON, with no other text:
{
  "case_against_verdict": "supported | not supported",
  "audit": [
    {"component": "", "analyst_score": 0, "my_score": 0, "error": "FACT | BAR | DOUBLE | MISSED | SHAVE | JUDGMENT", "reason": ""}
  ],
  "components": {
    "guidance_revision": {"score": 0, "evidence": ""},
    "forward_operating_outlook": {"score": 0, "evidence": ""},
    "expectations_gap": {"score": 0, "evidence": ""},
    "revenue_growth": {"score": 0, "evidence": ""},
    "organic_comparable_growth": {"score": 0, "evidence": ""},
    "volume_price_mix": {"score": 0, "evidence": ""},
    "gross_margin": {"score": 0, "evidence": ""},
    "operating_leverage": {"score": 0, "evidence": ""},
    "cash_conversion": {"score": 0, "evidence": ""},
    "beat_quality": {"score": 0, "evidence": ""},
    "operating_kpi": {"score": 0, "evidence": ""},
    "forward_demand": {"score": 0, "evidence": ""},
    "segment_breadth": {"score": 0, "evidence": ""},
    "execution_visibility": {"score": 0, "evidence": ""},
    "balance_sheet_liquidity": {"score": 0, "evidence": ""},
    "capital_return": {"score": 0, "evidence": ""},
    "new_risks": {"score": 0, "evidence": ""}
  },
  "weighted_total": 0,
  "score": 0.50,
  "biggest_disagreement": ""
}
List in `audit` ONLY the dimensions whose score you changed (use the dimension key as "component"). `components` must contain all 17 dimensions.
"""

PORTFOLIO_MANAGER = r"""
# ROLE: PORTFOLIO WEIGHING AGENT

You are the portfolio manager. You are the THIRD and FINAL agent. You are called when the Analyst and the Scrutinizer disagree. You settle every disputed dimension using the UNIVERSAL SCORING RUBRIC (v02) and the materials, and produce the settled 17-dimension sheet that becomes the submitted score.

## Procedure
1. COMPARE. Let A = the Analyst's score and R = the Scrutinizer's score (both computed by Python).
2. SETTLE DIMENSION BY DIMENSION. Where the two sheets differ, pick the side with the stronger evidence, using this ladder (strongest first):
   a. A reported number compared against an explicit line or whisper in the preview, or against prior company guidance
   b. A reported number compared against consensus only, or the company's own year-over-year / sequential comparison
   c. A qualitative fact stated in the call facts
   d. Tone, opinion or inference
   - If the evidence is equally strong, take the score closer to 0.
   - Where the sheets agree, keep that score.
   - Check the Scrutinizer's FACT, MISSED and SHAVE claims against the materials yourself. A verified one always wins.
   - Apply the Boundaries to your settled sheet: one comparison earns credit once.
3. SANITY CHECKS on the settled sheet, in order.
   a. SCENARIO CHECK: if the preview has bear/base/bull scenarios, decide which one the results match on the HEADLINE numbers (reported revenue and the guidance midpoint against each scenario's ranges; if they point to different scenarios, guidance wins). The weighted total should fit that scenario: bear <= -3 (score <= 0.35), base -2..+2 (0.40-0.60), bull >= +3 (score >= 0.65). If it does not, re-check the settled scores against the materials. Change a dimension only for a fact-based reason, and write that reason in `scenario_override_reason`. Never change the result without changing scores.
   b. NO LAZY ZEROS: do not set a dimension to 0 out of caution when the evidence supports a score.
4. Python recomputes the final score from your `settled_components` (weighted_total = sum of weight x score, percentile = 0.5 + weighted_total / 20). The SCORES are what count.

## The QUANT VIEW
You will see a QUANT VIEW block computed from the options market: the expected size of the move and the skew.
- It tells you how BIG a move the market expects, not which way. Skew has shown no relationship with the direction of the reaction, so never move a score because of skew.
- Your sheet is the UNSIZED view. The Quant step sizes it automatically after you.

## Rules
- The BACKGROUND section may help you weigh which disputed dimension matters more for this company. It is never evidence for a score.
- Do not introduce new facts the agents did not cite, except to verify a FACT or MISSED claim against the materials.
- Do not default to 0 out of caution. Only the ordering of scores across companies is judged.
- Use only the provided materials. Nothing from after the release.

## Output
Return ONLY this JSON, with no other text:
{
  "analyst_score": 0.50,
  "scrutinizer_score": 0.50,
  "settled_components": {
    "guidance_revision": 0,
    "forward_operating_outlook": 0,
    "expectations_gap": 0,
    "revenue_growth": 0,
    "organic_comparable_growth": 0,
    "volume_price_mix": 0,
    "gross_margin": 0,
    "operating_leverage": 0,
    "cash_conversion": 0,
    "beat_quality": 0,
    "operating_kpi": 0,
    "forward_demand": 0,
    "segment_breadth": 0,
    "execution_visibility": 0,
    "balance_sheet_liquidity": 0,
    "capital_return": 0,
    "new_risks": 0
  },
  "rulings": [
    {"component": "", "chose": "analyst | scrutinizer | own", "evidence_level": "a | b | c | d", "reason": ""}
  ],
  "weighted_total": 0,
  "scenario_match": "bear | base | bull | none",
  "scenario_evidence": "revenue X vs ranges..., guidance midpoint Y vs ranges...",
  "scenario_override_reason": "",
  "final_score": 0.50,
  "thesis": "one sentence: why this stock lands here in the quarter's ranking"
}
`settled_components` must contain all 17 dimensions. List in `rulings` only the dimensions where the sheets differed.
"""

# ----------------------------------------------------------------------
# Scoring arithmetic: done in Python, never trusted to the model.
# ----------------------------------------------------------------------

SCALE = 10.0                       # every dimension is scored from -10 to +10
# The v02 rubric's weights (they sum to 1, so weighted_total is also -10..+10).
DEFAULT_WEIGHTS = {
    "guidance_revision": 0.31,
    "forward_operating_outlook": 0.2,
    "expectations_gap": 0.04,
    "revenue_growth": 0.05,
    "organic_comparable_growth": 0.035,
    "volume_price_mix": 0.025,
    "gross_margin": 0.04,
    "operating_leverage": 0.1,
    "cash_conversion": 0.06,
    "beat_quality": 0.02,
    "operating_kpi": 0.02,
    "forward_demand": 0.04,
    "segment_breadth": 0.025,
    "execution_visibility": 0.015,
    "balance_sheet_liquidity": 0.01,
    "capital_return": 0.005,
    "new_risks": 0.005,
}
DIMENSIONS = list(DEFAULT_WEIGHTS)
COMPONENT_LIMITS = {d: SCALE for d in DIMENSIONS}   # kept for older scripts
LOWEST, HIGHEST = 0.001, 0.999
CLOSE_CALL_GAP_DEFAULT = 0.05      # Analyst/Scrutinizer gap that still skips the PM
JUDGMENT_MAX_MOVE = 3.0            # a JUDGMENT change may move a dimension this far
# QUANT sizing defaults (2026Q3 median move; pivot at the middle of the v02 scale).
QUANT_POWER_DEFAULT = 0.25
QUANT_REF_MOVE_DEFAULT = 0.08
QUANT_PIVOT_DEFAULT = 0.50


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _tidy(x: float) -> float:
    """Final scores are continuous; 4 decimals keeps logs readable."""
    return round(_clamp(x, LOWEST, HIGHEST), 4)


def _as_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    if isinstance(value, str):
        try:
            v = float(value.strip())
            return v if math.isfinite(v) else None
        except ValueError:
            return None
    return None


def implied_move(bundle: object) -> float | None:
    """The implied absolute earnings move from the bundle, or None if unusable."""
    if not isinstance(bundle, dict) or not isinstance(bundle.get("items"), list):
        return None
    for item in bundle["items"]:
        if isinstance(item, dict) and item.get("id") == OPTION_STATS_ID:
            block = (item.get("content") or {}).get("implied_absolute_earnings_move")
            if isinstance(block, dict) and block.get("status") == "ok":
                return _as_number(block.get("value"))
    return None


def points_from_sheet(components: object) -> dict[str, float]:
    """Read each dimension's score ({"score": x}, {"points": x} or a bare x),
    clamp to -10..+10. Missing or garbage -> 0."""
    out = {}
    comps = components if isinstance(components, dict) else {}
    for key in DIMENSIONS:
        raw = comps.get(key)
        if isinstance(raw, dict):
            raw = raw.get("score", raw.get("points"))
        n = _as_number(raw)
        out[key] = round(_clamp(n, -SCALE, SCALE), 2) if n is not None else 0.0
    return out


def component_weights() -> dict[str, float]:
    """Weights per dimension (default: the v02 weights). Override in .env, e.g.
    COMPONENT_WEIGHTS="guidance_revision=0.4,new_risks=0". Bad entries are ignored."""
    weights = dict(DEFAULT_WEIGHTS)
    for part in os.environ.get("COMPONENT_WEIGHTS", "").split(","):
        if "=" in part:
            key, _, value = part.partition("=")
            number = _as_number(value)
            if key.strip() in weights and number is not None and 0 <= number <= 5:
                weights[key.strip()] = number
    return weights


def score_from_points(points: dict[str, float]) -> dict:
    """weighted_total = sum(weight x score); percentile = 0.5 + weighted_total / 20."""
    weights = component_weights()
    total = sum(weights[k] * points.get(k, 0) for k in DIMENSIONS)
    return {"weighted_total": round(total, 4), "total_s": round(total, 4),
            "score": _tidy(0.5 + total / (2 * SCALE))}


def _env_number(name: str, default: float) -> float:
    value = _as_number(os.environ.get(name, ""))
    return default if value is None else value


def quant_factor(move: float | None) -> float:
    """How much to stretch the distance from the pivot: (move / reference) ** power."""
    power = _env_number("QUANT_POWER", QUANT_POWER_DEFAULT)
    ref = _env_number("QUANT_REF_MOVE", QUANT_REF_MOVE_DEFAULT)
    if move is None or move <= 0 or power <= 0 or ref <= 0:
        return 1.0
    return (move / ref) ** power


def apply_quant(score: float, move: float | None) -> float:
    """QUANT sizing: pivot + (score - pivot) * factor, kept within 0.001..0.999."""
    pivot = _env_number("QUANT_PIVOT", QUANT_PIVOT_DEFAULT)
    return _tidy(pivot + (score - pivot) * quant_factor(move))


def quant_view(bundle: object) -> str:
    """The QUANT VIEW text the Portfolio Manager sees."""
    move = implied_move(bundle)
    if move is None:
        return "QUANT VIEW: no usable options data for this stock, so no sizing will be applied."
    stats = {}
    for item in (bundle.get("items") or []) if isinstance(bundle, dict) else []:
        if isinstance(item, dict) and item.get("id") == OPTION_STATS_ID:
            stats = item.get("content") or {}
    def value(key: str) -> float | None:
        block = stats.get(key) if isinstance(stats, dict) else None
        return _as_number(block.get("value")) if isinstance(block, dict) and block.get("status") == "ok" else None
    ref = _env_number("QUANT_REF_MOVE", QUANT_REF_MOVE_DEFAULT)
    factor = quant_factor(move)
    size = "large" if move >= ref * 1.33 else "small" if move <= ref * 0.67 else "normal"
    lines = [
        "QUANT VIEW (computed in Python from the options market; no AI):",
        f"- Expected absolute move: {move:.1%} ({size}; a normal move is about {ref:.1%})",
    ]
    vol = value("implied_earnings_volatility")
    if vol is not None:
        lines.append(f"- Implied earnings volatility: {vol:.1%}")
    skew = value("skew_25_delta")
    if skew is not None:
        lean = "puts priced richer" if skew < 0 else "calls priced richer"
        lines.append(f"- 25-delta skew: {skew * 100:+.1f} vol points ({lean}). No proven link to direction.")
    lines.append(f"- After you decide, your score's distance from the middle is multiplied by {factor:.2f}.")
    return "\n".join(lines)


HARD_ERRORS = {"FACT", "BAR", "DOUBLE", "MISSED", "SHAVE"}


def cap_judgment_changes(analyst_pts: dict[str, float], scrutinizer: dict) -> tuple[dict[str, float], list]:
    """The Scrutinizer may move a dimension by more than 3 points ONLY with a hard
    error code (FACT, BAR, DOUBLE, MISSED, SHAVE). A bigger move labelled
    JUDGMENT (or unlabelled) is cut back to 3 points. Returns (scores, notes)."""
    r_pts = points_from_sheet(scrutinizer.get("components"))
    errors = {}
    audit = scrutinizer.get("audit")
    if isinstance(audit, list):
        for row in audit:
            if isinstance(row, dict) and isinstance(row.get("component"), str):
                errors[row["component"]] = str(row.get("error", "")).upper().strip()
    notes = []
    for key in DIMENSIONS:
        a, r = analyst_pts[key], r_pts[key]
        if abs(r - a) > JUDGMENT_MAX_MOVE + 1e-9 and errors.get(key) not in HARD_ERRORS:
            capped = round(a + (JUDGMENT_MAX_MOVE if r > a else -JUDGMENT_MAX_MOVE), 2)
            notes.append({"component": key, "scrutinizer": r, "capped_to": capped,
                          "error_given": errors.get(key, "none")})
            r_pts[key] = capped
    return r_pts, notes


def average_rule(a: float, r: float) -> float:
    """Close call: the plain average of the two scores."""
    return _tidy((a + r) / 2)


def close_call_gap() -> float:
    return _env_number("CLOSE_CALL_GAP", CLOSE_CALL_GAP_DEFAULT)


# ----------------------------------------------------------------------
# SEC EDGAR filings (free, public). Everything here is best-effort: any
# problem returns "" and the pipeline carries on without filings.
#
# KNOWLEDGE-CUTOFF SAFETY: a filing is used only if the SEC accepted it at
# least 24 hours before the event time (and before the official knowledge
# cutoff when the event carries one). The current quarter's results 8-K is
# filed on release day, so it can never get in.
# ----------------------------------------------------------------------

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/"
SEC_CALL_TIMEOUT = 8.0
SEC_MAX_BYTES = 15_000_000
RELEASE_EXCERPT_CHARS = 9000
MDNA_EXCERPT_CHARS = 9000
OTHER_8K_LOOKBACK_DAYS = 120

ITEM_NAMES = {
    "1.01": "Entered a material agreement", "1.02": "Ended a material agreement",
    "1.03": "Bankruptcy or receivership", "2.01": "Completed an acquisition or sale of assets",
    "2.02": "Results of operations (earnings release)", "2.03": "New debt or financial obligation",
    "2.05": "Restructuring / exit costs", "2.06": "Material impairment",
    "3.01": "Delisting notice", "3.02": "Unregistered sale of shares",
    "4.01": "Auditor change", "4.02": "Prior financials no longer reliable (restatement)",
    "5.02": "Director or officer change", "5.03": "Bylaw or fiscal-year change",
    "5.07": "Shareholder vote results", "7.01": "Regulation FD disclosure",
    "8.01": "Other events", "9.01": "Financial statements and exhibits",
}

_ticker_to_cik: dict[str, int] | None = None


def _sec_headers() -> dict:
    agent = os.environ.get("SEC_USER_AGENT", "").strip() or "VotCommand research bot contact@votcommand.example"
    return {"User-Agent": agent, "Accept-Encoding": "gzip, deflate"}


def _sec_get(url: str, timeout: float) -> httpx.Response:
    """One GET to the SEC. Separate function so tests can replace it."""
    resp = httpx.get(url, headers=_sec_headers(), timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    if len(resp.content) > SEC_MAX_BYTES:
        raise ValueError(f"SEC document too large: {len(resp.content)} bytes")
    return resp


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def filings_cutoff(event_datetime: str | None, knowledge_cutoff: str | None) -> datetime:
    """Latest acceptance time a filing may have: 24h before the event, and before
    the knowledge cutoff if known. With no dates at all: 24h before now."""
    event = _parse_time(event_datetime) or datetime.now(timezone.utc)
    cutoff = event - timedelta(hours=24)
    official = _parse_time(knowledge_cutoff)
    if official is not None and official < cutoff:
        cutoff = official
    return cutoff


def html_to_text(raw: str) -> str:
    """Rough HTML -> readable text, dropping scripts, styles and hidden iXBRL data."""
    raw = re.sub(r"(?is)<(script|style|ix:header)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</(p|div|tr|li|h[1-6]|table)>", "\n", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    text = html.unescape(raw).replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _cik_for(ticker: str, deadline: float) -> int | None:
    global _ticker_to_cik
    if _ticker_to_cik is None:
        data = _sec_get(SEC_TICKERS_URL, _call_timeout(deadline)).json()
        rows = data.values() if isinstance(data, dict) else data
        _ticker_to_cik = {
            str(r.get("ticker", "")).upper(): int(r["cik_str"])
            for r in rows if isinstance(r, dict) and "cik_str" in r
        }
    t = ticker.upper()
    return _ticker_to_cik.get(t) or _ticker_to_cik.get(t.replace(".", "-"))


def _call_timeout(deadline: float) -> float:
    left = deadline - time.monotonic()
    if left < 1.5:
        raise TimeoutError("SEC time budget used up")
    return min(SEC_CALL_TIMEOUT, left)


def eligible_filings(submissions: dict, cutoff: datetime) -> list[dict]:
    """Recent filings accepted before `cutoff`, newest first."""
    recent = ((submissions or {}).get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    out = []
    for i, form in enumerate(forms):
        def col(name: str) -> str:
            values = recent.get(name) or []
            return values[i] if i < len(values) and values[i] is not None else ""
        accepted = _parse_time(col("acceptanceDateTime"))
        if accepted is None:
            filed = _parse_time(col("filingDate"))
            accepted = filed + timedelta(hours=23, minutes=59) if filed else None
        if accepted is None or accepted >= cutoff:
            continue
        out.append({
            "form": str(form), "accession": col("accessionNumber"),
            "filing_date": col("filingDate"), "report_date": col("reportDate"),
            "primary_document": col("primaryDocument"), "items": str(col("items")),
            "accepted": accepted,
        })
    out.sort(key=lambda f: f["accepted"], reverse=True)
    return out


def _filing_url(cik: int, accession: str, name: str = "") -> str:
    return SEC_ARCHIVE_URL.format(cik=cik, acc_nodash=accession.replace("-", "")) + name


def find_exhibit_991(index_html: str) -> str | None:
    """From a filing's -index.htm page, the file name of EX-99.1 (else any EX-99)."""
    best = None
    for row in re.findall(r"(?is)<tr[^>]*>(.*?)</tr>", index_html):
        type_match = re.search(r"(?i)>\s*(EX-99[.\d]*)\s*<", row)
        href_match = re.search(r'(?i)href="([^"]+)"', row)
        if not type_match or not href_match:
            continue
        name = href_match.group(1).rsplit("/", 1)[-1]
        if type_match.group(1).upper() == "EX-99.1":
            return name
        best = best or name
    return best


def release_excerpt(text: str, limit: int = RELEASE_EXCERPT_CHARS) -> str:
    """The opening of a press release plus its outlook / guidance section."""
    half = limit // 2
    match = re.search(r"(?i)\b(outlook|guidance)\b", text[half:])
    if not match:
        return text[:limit]
    at = half + match.start()
    return text[:half] + "\n[...]\n" + text[max(half, at - 200): at - 200 + half]


def mdna_excerpt(text: str, annual: bool, limit: int = MDNA_EXCERPT_CHARS) -> str:
    """Management's Discussion and Analysis, skipping the table of contents."""
    item = r"item\s*7" if annual else r"item\s*2"
    stop = r"item\s*7a" if annual else r"item\s*3"
    pattern = re.compile(item + r"\.?\s*[\-–—:.]?\s*management.{0,3}s\s+discussion\s+and\s+analysis", re.I)
    for match in pattern.finditer(text):
        after = text[match.start(): match.start() + 400]
        if re.search(stop, after[60:], re.I):
            continue                                  # a table-of-contents line
        section = text[match.start(): match.start() + limit]
        end = re.search(stop + r"\.?\s*[\-–—:.]?\s*quantitative", section[200:], re.I)
        return section[: 200 + end.start()] if end else section
    return ""


def sec_dossier(ticker: str, cutoff: datetime, seconds: float) -> str:
    """Build the SEC FILINGS text for the Context agent. Never raises; "" on failure."""
    if seconds < 3:
        return ""
    deadline = time.monotonic() + seconds
    parts: list[str] = []
    try:
        cik = _cik_for(ticker, deadline)
        if cik is None:
            print(f"[SEC] {ticker}: no CIK found (foreign filer or unknown ticker)")
            return ""
        subs = _sec_get(SEC_SUBMISSIONS_URL.format(cik=cik), _call_timeout(deadline)).json()
        filings = eligible_filings(subs, cutoff)
        eightks = [f for f in filings if f["form"] in ("8-K", "8-K/A")]
        earnings = next((f for f in eightks if "2.02" in f["items"]), None)
        periodic = next((f for f in filings if f["form"] in ("10-Q", "10-K")), None)

        if earnings:
            try:
                index_html = _sec_get(
                    _filing_url(cik, earnings["accession"], earnings["accession"] + "-index.htm"),
                    _call_timeout(deadline)).text
                name = find_exhibit_991(index_html) or earnings["primary_document"]
                text = html_to_text(_sec_get(_filing_url(cik, earnings["accession"], name),
                                             _call_timeout(deadline)).text)
                if text:
                    parts.append(f"PREVIOUS EARNINGS RELEASE (8-K filed {earnings['filing_date']}, "
                                 f"exhibit {name}), excerpt:\n{release_excerpt(text)}")
            except Exception as exc:
                print(f"[SEC] {ticker}: earnings release skipped: {exc!r}")

        if periodic and periodic["primary_document"]:
            try:
                text = html_to_text(_sec_get(
                    _filing_url(cik, periodic["accession"], periodic["primary_document"]),
                    _call_timeout(deadline)).text)
                mdna = mdna_excerpt(text, annual=periodic["form"] == "10-K")
                if mdna:
                    parts.append(f"LATEST {periodic['form']} (filed {periodic['filing_date']}, period "
                                 f"{periodic['report_date'] or 'n/a'}), MD&A excerpt:\n{mdna}")
            except Exception as exc:
                print(f"[SEC] {ticker}: {periodic['form']} skipped: {exc!r}")

        since = (earnings["accepted"] if earnings
                 else cutoff - timedelta(days=OTHER_8K_LOOKBACK_DAYS))
        others = [f for f in eightks if f["accepted"] > since and f is not earnings]
        lines = []
        for f in others[:12]:
            codes = [c.strip() for c in f["items"].split(",") if c.strip() and c.strip() != "9.01"]
            names = "; ".join(f"{c} {ITEM_NAMES.get(c, '')}".strip() for c in codes) or "no items listed"
            lines.append(f"- {f['filing_date']}: {f['form']} - {names}")
        if lines:
            parts.append("OTHER 8-K FILINGS SINCE THE PREVIOUS EARNINGS RELEASE:\n" + "\n".join(lines))
    except Exception as exc:
        print(f"[SEC] {ticker}: filings skipped: {exc!r}")
    if not parts:
        return ""
    dossier = ("SEC FILINGS (from EDGAR; every filing here was accepted before "
               f"{cutoff.strftime('%Y-%m-%d %H:%M UTC')}, i.e. before this release):\n\n"
               + "\n\n".join(parts))
    print(f"[SEC] {ticker}: {len(dossier)} chars of filings")
    return dossier


# ----------------------------------------------------------------------
# Calling one agent.
# ----------------------------------------------------------------------

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(timeout=MAX_CALL_SECONDS, max_retries=0)
    return _client


def _parse_json(text: str) -> dict:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)  # tolerate ```json fences
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("agent did not return a JSON object")
    return data


def _hard_deadline(fn, seconds: float):
    """Run fn() but give up after `seconds` of WALL-CLOCK time, no matter what.
    The HTTP timeout alone is not a hard limit (it only fires when the connection
    goes quiet), so a slow model could otherwise blow the 5-minute deadline.
    A call that runs over keeps going in a background thread and is ignored."""
    box: queue.Queue = queue.Queue(maxsize=1)

    def work():
        try:
            box.put((True, fn()))
        except BaseException as exc:          # hand any error back to the caller
            box.put((False, exc))

    threading.Thread(target=work, daemon=True).start()
    try:
        ok, value = box.get(timeout=max(0.1, seconds))
    except queue.Empty:
        raise TimeoutError(f"agent call exceeded {seconds:.0f}s") from None
    if not ok:
        raise value
    return value


def call_agent(role_prompt: str, user_message: str, seconds: float,
               with_rubric: bool = True, effort: str | None = None) -> dict:
    """One chat call in JSON mode. Raises on any failure (callers fall back).
    `effort` overrides OPENAI_REASONING_EFFORT for this one call."""
    kwargs = {}
    if effort is None:
        effort = os.environ.get("OPENAI_REASONING_EFFORT", "")
    effort = effort.strip()
    if effort:
        kwargs["reasoning_effort"] = effort
    limit = max(5.0, min(MAX_CALL_SECONDS, seconds))
    client = _get_client()

    def _request():
        return client.chat.completions.create(
            model=openai_model(),
            messages=[
                {"role": "system",
                 "content": role_prompt + ("\n\n" + RUBRIC if with_rubric else "")},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            timeout=limit,
            **kwargs,
        )

    resp = _hard_deadline(_request, limit + 2.0)
    content = resp.choices[0].message.content or ""
    return _parse_json(content)


def _log(tag: str, payload: object) -> None:
    if os.environ.get("BOT_QUIET") == "1":
        return
    try:
        print(f"[{tag}] " + json.dumps(payload, ensure_ascii=False)[:4000])
    except Exception:
        print(f"[{tag}] (unprintable)")


BACKGROUND_SECTIONS = [
    ("company", "Company"),
    ("industry", "Industry"),
    ("what_moves_the_stock", "What moves the stock"),
    ("reaction_history", "Reaction history"),
    ("storyline_into_print", "Storyline into the print"),
    ("key_terms", "Key terms"),
    ("filings_digest", "From SEC filings"),
    ("reading_guide", "Reading guide"),
]
BACKGROUND_MAX_CHARS = 8000


def get_background(context: str, ticker: str, seconds: float, filings: str = "") -> str:
    """Run the Context agent and turn its JSON into labelled paragraphs.
    Returns "" on any failure, so the pipeline simply carries on without it."""
    if seconds < MIN_CALL_SECONDS:
        return ""
    try:
        data = call_agent(
            CONTEXT,
            context
            + ("\n\n" + filings if filings else "\n\nSEC FILINGS: none available.")
            + "\n\nWrite the background. Return JSON only.",
            seconds,
            with_rubric=False,
            # The Context agent only writes background, so it can think less hard
            # and finish faster. Unset = same effort as the other agents.
            effort=os.environ.get("CONTEXT_REASONING_EFFORT") or None,
        )
    except Exception as exc:
        print(f"[WARN] {ticker} context agent failed: {exc!r} -> continuing without it")
        return ""
    parts = []
    for key, label in BACKGROUND_SECTIONS:
        text = data.get(key)
        if isinstance(text, list):
            text = "\n".join(str(t) for t in text)
        if isinstance(text, str) and text.strip():
            parts.append(f"{label}:\n{text.strip()}")
    background = "\n\n".join(parts)[:BACKGROUND_MAX_CHARS]
    _log("CONTEXT", {"ticker": ticker, "background": background})
    return background


# ----------------------------------------------------------------------
# The pipeline: Context -> Analyst -> Scrutinizer -> Portfolio Manager.
# ----------------------------------------------------------------------

def run_agents(*, bundle: object, ticker: str, event_type: str,
               event_datetime: str | None = None,
               knowledge_cutoff: str | None = None,
               trace: dict | None = None) -> float:
    """Run the whole pipeline and return the submitted number.
    `trace` (optional dict) is filled with every intermediate result, for training."""
    t = trace if trace is not None else {}
    started = time.monotonic()
    t.update({"errors": [], "path": None})
    unsized = _decide(bundle=bundle, ticker=ticker, event_type=event_type,
                      event_datetime=event_datetime, knowledge_cutoff=knowledge_cutoff, t=t)
    move = implied_move(bundle)
    if t["path"] in ("no_key", "analyst_failed"):
        final = unsized                       # never size an error fallback
    else:
        final = apply_quant(unsized, move)
    t.update({"unsized": unsized, "final": final, "move": move,
              "quant_factor": round(quant_factor(move), 4),
              "seconds": round(time.monotonic() - started, 1)})
    if final != unsized:
        print(f"[QUANT] {ticker} {unsized} -> {final} (expected move {move:.1%})")
    return final


def _decide(*, bundle: object, ticker: str, event_type: str,
            event_datetime: str | None, knowledge_cutoff: str | None, t: dict) -> float:
    """Context -> Analyst -> Scrutinizer -> PM. Returns the UNSIZED score."""
    global _warned_no_key
    if not os.environ.get("OPENAI_API_KEY"):
        if not _warned_no_key:
            print("[WARN] OPENAI_API_KEY not set - submitting 0.5. Add it to .env and redeploy.")
            _warned_no_key = True
        t["path"] = "no_key"
        return 0.5

    start = time.monotonic()

    def seconds_left() -> float:
        return TOTAL_BUDGET_SECONDS - (time.monotonic() - start)

    t["stage_seconds"] = {}
    lap = [start]

    def mark(stage: str) -> None:
        """Log how long the stage that just ended took, so slow stages are visible."""
        now = time.monotonic()
        t["stage_seconds"][stage] = round(now - lap[0], 1)
        lap[0] = now
        print(f"[TIME] {ticker} {stage}: {t['stage_seconds'][stage]:.0f}s "
              f"(total {now - start:.0f}s of {TOTAL_BUDGET_SECONDS:.0f}s)")

    materials = format_materials(bundle) or json.dumps(bundle)[:PREVIEW_MAX_CHARS]
    move = implied_move(bundle)
    move_text = f"{move:.4f}" if move is not None else "missing"
    context = (
        f"Event type: {event_type}\nTicker: {ticker}\n"
        f"implied_absolute_earnings_move: {move_text}\n\n"
        f"MATERIALS:\n{materials}"
    )

    # 0. CONTEXT (background only, never a number; skipped on failure)
    t["filings_chars"] = t["background_chars"] = 0
    if os.environ.get("USE_CONTEXT_AGENT", "1") != "0":
        filings = ""
        if os.environ.get("USE_SEC_FILINGS", "1") != "0":
            filings = sec_dossier(ticker, filings_cutoff(event_datetime, knowledge_cutoff),
                                  min(SEC_MAX_SECONDS, max(0.0, seconds_left() - 200.0)))
            mark("sec_filings")
        background = get_background(context, ticker,
                                    min(CONTEXT_MAX_SECONDS, seconds_left()), filings)
        mark("context")
        t["filings_chars"], t["background_chars"] = len(filings), len(background)
        if background:
            context += (
                "\n\nBACKGROUND (from the Context agent: for interpretation only, "
                "NOT evidence; the materials win any conflict):\n" + background
            )

    # 1. ANALYST
    try:
        analyst = call_agent(ANALYST, context + "\n\nScore this release. Return JSON only.",
                             seconds_left())
    except Exception as exc:
        print(f"[ERROR] {ticker} analyst failed: {exc!r} -> 0.5")
        t["errors"].append(f"analyst: {exc!r}")
        t["path"] = "analyst_failed"
        return 0.5
    finally:
        mark("analyst")
    a_pts = points_from_sheet(analyst.get("components"))
    a_calc = score_from_points(a_pts)
    analyst.update(a_calc)                 # overwrite the model's arithmetic with ours
    a = a_calc["score"]
    t.update({"analyst_points": a_pts, "analyst_score": a})
    _log("ANALYST", {"ticker": ticker, **analyst})

    # 2. SCRUTINIZER
    if seconds_left() < MIN_CALL_SECONDS:
        print(f"[TIME] {ticker} no time for scrutinizer -> analyst {a}")
        t["path"] = "analyst_only"
        return a
    try:
        scrutinizer = call_agent(
            SCRUTINIZER,
            context + "\n\nANALYST SCORESHEET (arithmetic verified):\n"
            + json.dumps(analyst, ensure_ascii=False)
            + "\n\nAudit it and produce your own scoresheet. Return JSON only.",
            seconds_left(),
        )
    except Exception as exc:
        print(f"[ERROR] {ticker} scrutinizer failed: {exc!r} -> analyst {a}")
        t["errors"].append(f"scrutinizer: {exc!r}")
        t["path"] = "analyst_only"
        return a
    finally:
        mark("scrutinizer")
    r_pts, cap_notes = cap_judgment_changes(a_pts, scrutinizer)
    if cap_notes:
        scrutinizer["judgment_caps_applied"] = cap_notes
        comps = scrutinizer.get("components")
        if isinstance(comps, dict):
            for note in cap_notes:
                entry = comps.get(note["component"])
                if isinstance(entry, dict):
                    entry["score"] = note["capped_to"]
                    entry.pop("points", None)
                else:
                    comps[note["component"]] = {"score": note["capped_to"], "evidence": ""}
    r_calc = score_from_points(r_pts)
    scrutinizer.update(r_calc)
    r = r_calc["score"]
    t.update({"scrutinizer_points": r_pts, "scrutinizer_score": r, "judgment_caps": len(cap_notes)})
    _log("SCRUTINIZER", {"ticker": ticker, **scrutinizer})

    # 3. PORTFOLIO MANAGER
    close_call = abs(a - r) <= close_call_gap() + 1e-9 and (a - 0.5) * (r - 0.5) >= 0
    fallback = average_rule(a, r)
    if close_call and os.environ.get("ALWAYS_CALL_PM", "") != "1":
        print(f"[FINAL] {ticker} close call: analyst {a}, scrutinizer {r} -> {fallback}")
        t["path"] = "close_call"
        return fallback
    if seconds_left() < MIN_CALL_SECONDS:
        print(f"[TIME] {ticker} no time for PM -> average {fallback}")
        t["path"] = "pm_skipped_time"
        return fallback
    try:
        pm = call_agent(
            PORTFOLIO_MANAGER,
            context
            + "\n\nANALYST SCORESHEET:\n" + json.dumps(analyst, ensure_ascii=False)
            + "\n\nSCRUTINIZER SCORESHEET:\n" + json.dumps(scrutinizer, ensure_ascii=False)
            + "\n\n" + quant_view(bundle)
            + f"\n\nVerified scores: analyst_score={a}, scrutinizer_score={r}, "
              f"gap={abs(a - r):.2f}.\n\nReturn the final JSON only.",
            seconds_left(),
        )
    except Exception as exc:
        print(f"[ERROR] {ticker} PM failed: {exc!r} -> average {fallback}")
        t["errors"].append(f"pm: {exc!r}")
        t["path"] = "pm_failed"
        return fallback
    finally:
        mark("pm")
    final_raw = _as_number(pm.get("final_score"))
    final = _tidy(final_raw) if final_raw is not None else fallback
    settled = pm.get("settled_components")
    if isinstance(settled, dict) and settled:
        # The settled scores are what count: re-score them with the v02
        # weights, so the model's own arithmetic never reaches the submission.
        settled_pts = points_from_sheet(settled)
        pm["python_check"] = score_from_points(settled_pts)
        pm["model_final_score"] = final
        final = pm["python_check"]["score"]
        t["pm_points"] = settled_pts
    t.update({"pm_score": final, "path": "pm"})
    _log("PM", {"ticker": ticker, **pm})
    print(f"[FINAL] {ticker} analyst {a}, scrutinizer {r} -> PM {final}")
    return final
