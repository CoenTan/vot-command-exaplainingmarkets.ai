"""VOT COMMAND, iteration 2: Context -> Analyst -> Scrutinizer -> Portfolio Manager.

THIS IS THE ONLY FILE YOU EDIT. Everything else in the repo is plumbing.

How one event flows:
  1. Fetch the event's materials (call facts, preview, option stats).
  1b. SEC FILINGS: fetch the previous earnings release, latest 10-Q/10-K MD&A
     and recent 8-Ks from EDGAR. Only filings accepted 24h+ before the event
     (knowledge-cutoff safe). Hard 20s cap; skipped on any problem.
  1c. CONTEXT writes background on the company, industry and filings
     (paragraphs, no number). Every later agent reads it; never evidence.
  2. ANALYST scores the release with the rubric.
  3. SCRUTINIZER audits the Analyst and scores it again.
  4. PORTFOLIO MANAGER settles it (skipped on close calls: the rubric says
     "average them", so Python does that and saves a call).
  5. Python re-does every agent's arithmetic, so a model's maths mistake can
     never reach the submission.
  6. QUANT (Python, no AI) sizes the final score with the options market's
     expected move: pivot + (score - pivot) * (move / reference_move) ** power.
     Big expected move -> further from the middle. It never picks a direction.
     The answer is always on the 0.05 grid.

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
  COMPONENT_WEIGHTS          e.g. "guidance=1.5,eps=0.5" (others keep DEFAULT_WEIGHTS); see train/tune.py
  QUANT_POWER                sizing strength (default 0.25; 0 = sizing off)
  QUANT_REF_MOVE             a "normal" expected move (default 0.08 = 8%, the 2026Q3 median)
  QUANT_PIVOT                centre for sizing (default 0.55, about this bot's average score)

The defaults for the weights and the QUANT settings were fitted on the
historical archive (2025Q4-2026Q3, ~8,400 scored releases) by train/tune.py.
  BOT_QUIET                  set to 1 to hide the agents' full JSON in logs

To change how agents think, edit the prompt blocks below
(RUBRIC, CONTEXT, ANALYST, SCRUTINIZER, PORTFOLIO_MANAGER).
"""

from __future__ import annotations

import html
import json
import math
import os
import re
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
PREVIEW_MAX_CHARS = 8000
OTHER_ITEM_MAX_CHARS = 2000


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
# UNIVERSAL SCORING RUBRIC

You score one company's earnings release. Your output is a single number on a 0.05 grid from 0.05 to 0.95.

## What the number means
- The number is a RANK, not a probability. It predicts where this stock's one-day, market-adjusted price reaction will rank among ALL earnings reactions this quarter.
  - 0.05 = among the worst reactions of the quarter
  - 0.50 = a median reaction
  - 0.95 = among the best reactions of the quarter
- Only the ORDERING of scores across companies matters. Calibration earns nothing. Do not shrink toward 0.50 out of caution. Use 0.50 only when the evidence is genuinely balanced or absent.
- Score news RELATIVE TO EXPECTATIONS, not in absolute terms. A great quarter the market already expected is a 0.50.
- You are judged only on what you explain BEYOND the EPS surprise, which the scorer already controls for. So the EPS beat/miss earns nothing: its weight is 0.
- What does carry signal, measured on ~8,400 past releases (2025Q4-2026Q3), strongest first:
  1. Guidance direction (raised / cut / withdrawn vs the bar). By far the strongest, every quarter. Cuts move stocks more than raises.
  2. Management tone and risk flags. Second strongest per point.
  3. Beats or misses against a stated bar (the key metric), and the quality of the quarter (margins, one-offs).
  4. Revenue on its own adds little once guidance is known.

## Inputs you may receive
- `earnings-call-facts`: facts extracted from the earnings call. These are the RESULTS.
- `earnings-preview`: a research note written BEFORE the release. These are the EXPECTATIONS: consensus, whisper/buy-side numbers, bull/base/bear scenarios, metrics to watch, setup.
- `option-implied-stats`: implied earnings volatility, implied absolute move, 25-delta skew, all measured before the release. Any of these may be missing. They describe how BIG a move the market expects, not its direction. Skew has NO measurable link to the direction of the reaction (2026Q3: correlation 0.01 over 672 releases), so never score it, not even in `setup`.

- `BACKGROUND`: paragraphs from the Context agent about the company and its industry. Use it to INTERPRET the materials (what matters for this company, what a metric means). It is NOT evidence: every point you score must cite the call facts or the preview, never the background. If the background conflicts with the materials, the materials win.

Use ONLY these inputs plus general knowledge from before the release. Never use, search for or assume news, prices or reactions from after the release. If an input is missing, treat that item as missing. Do not invent numbers.

## Step 1: Find "the bar"
For each item, the bar is the HIGHEST credible expectation in the preview:
1. If the preview names explicit lines (e.g. "$105B is the bull line, $103B the disappointment line"), use those lines.
2. Otherwise use the whisper / buy-side number if one is given.
3. Otherwise use sell-side consensus.
4. If no expectation exists for an item, score that item 0 and mark it "N/A".

## Step 2: Score the seven components

| # | key | Component | Range | Weight |
|---|---|---|---|---|
| 1 | guidance | Next-period / full-year guidance vs the bar | -3 to +3 | 1.0 |
| 2 | revenue | Reported revenue vs the bar | -2 to +2 | 0.6 |
| 3 | key_metric | The preview's #1 "metric to watch" that is NOT guidance or revenue (e.g. gross margin, segment revenue, subscribers). With no preview: the result the company itself compares to a stated bar (its own prior guidance, "above/below expectations") | -2 to +2 | 0.7 |
| 4 | quality | Quality of the beat or miss | -1 to +1 | 0.6 |
| 5 | eps | EPS vs consensus | -1 to +1 | 0 |
| 6 | tone | Management tone and risk flags | -1 to +1 | 1.0 |
| 7 | setup | Positioning going into the print | -1 to +1 | 1.0 |

Score every component on its points scale as below. The weights are applied in Step 3 (in code); they never change how you award points.

How to score guidance, revenue and key_metric: compare NUMBERS, mechanically.
- If a result or expectation is a range (e.g. "$108B ±2%", "$94-95B"), use its MIDPOINT.
- For margins and other % metrics, count 1 percentage point as "3%".
- Go down this table and use the FIRST row that is true:

| Points | Rule |
|---|---|
| +3 | (guidance only) Result is at or above the preview's bull line or bull-scenario level, OR beats the whisper by 3% or more |
| +2 | Result beats the whisper / buy-side number (by any amount), OR beats consensus by 3% or more when no whisper is given |
| +1 | Result beats consensus but does not beat the whisper |
| 0 | Result within ±0.5% of consensus, OR no number available (mark "N/A") |
| -1 | Result below consensus by less than 3%, OR meets consensus but misses the whisper |
| -2 | Result below consensus by 3% or more |
| -3 | (guidance only) Result at or below the disappointment line or bear-scenario level, OR guidance cut or withdrawn |

revenue and key_metric can never go beyond ±2. If a +3 or -3 row applies to them, use ±2.

NO SHAVING. Apply the table exactly. Never lower a score toward 0 because:
- the guidance is a range,
- the call did not repeat the preview's lines,
- you feel unsure.
Uncertainty is handled by the Scrutinizer and the Portfolio Manager, not by you.

Components 4-7:
- quality: +1 = margin expansion, beat driven by core demand. -1 = beat flattered by one-offs, tax, buybacks or accounting, or margins compressing. 0 = neither. If key_metric is already a margin, quality may NOT score margins again.
- eps: +1 = beat consensus. -1 = missed. 0 = in line or N/A. Weight 0: record it (it keeps the EPS fact out of other components) but it does not move the score. Never use an EPS beat or miss as evidence anywhere else.
- tone: +1 = specific, confident commentary backed by numbers (record results, accelerating growth, demand clearly strengthening). -1 = new risk flags: investigation, lawsuit, restatement, executive departure, impairment or write-down, layoffs, delays, demand softness or weakness, new headwinds, rising uncertainty. 0 = neither. Tone may only use facts NOT already scored in another component. "Demand exceeds supply" is not a negative. Tone is the second-strongest signal per point: when the facts show a clear risk flag or clear confidence, score it; do not leave it at 0 out of caution.
- setup: +1 = expectations were LOW going in (stock fell into the print, bearish sentiment, per the preview) AND results were at least in line. -1 = expectations were STRETCHED (big run-up, euphoric sentiment, crowded) AND results were not clearly above the bar. 0 = otherwise, or no preview. Never use option skew.

Hard rules:
- Every non-zero point must cite evidence: the result, the bar, and the source (facts or preview).
- NO DOUBLE COUNTING. Each fact scores in ONE component only. Before you finish, check every fact you used. If one fact appears in two or more components, keep it ONLY in the first of these and set it to 0 in the others: guidance > revenue > key_metric > quality > eps > tone > setup. Example: a gross-margin reset scored in key_metric cannot also lower quality or tone.
- eps and tone can never exceed ±1.
- No adjustments outside the rubric. If something important fits no component, write it in `notes`. It does not change the score.

## Step 3: Convert points to a score
1. S = the WEIGHTED sum of the seven components:
   S = 1.0×guidance + 0.6×revenue + 0.7×key_metric + 0.6×quality + 0×eps + 1.0×tone + 1.0×setup
   (range about -8.2 to +8.2).
2. score = 0.50 + 0.05 × S, rounded to the 0.05 grid, capped to the range 0.05 to 0.95.
3. Python redoes this arithmetic from your points and overwrites yours. Your job is to get the POINTS right.
4. Do NOT adjust for the options data yourself. A separate Quant step sizes the final score using the implied move, automatically, after all agents finish.

## Step 4: Sanity check against the bands
| Score | weighted S | What the print looks like |
|---|---|---|
| 0.85-0.90 | +7 or more | Blowout: guidance, revenue and key metric all clear the whisper, confident tone |
| 0.75-0.80 | +5 to +6 | Clear beat-and-raise above the real bar |
| 0.65-0.70 | +3 to +4 | Good: guidance up or beats the bar on most items, nothing alarming |
| 0.55-0.60 | +1 to +2 | Modestly positive, or mixed leaning good |
| 0.50 | 0 | In line, or genuinely balanced |
| 0.40-0.45 | -1 to -2 | Modestly negative, or mixed leaning bad |
| 0.30-0.35 | -3 to -4 | Guidance below the bar, or misses the bar on a key item |
| 0.20-0.25 | -5 to -6 | Clear miss plus weak guidance |
| 0.10-0.15 | -7 or less | Disaster: guidance cut, big miss, major new risk flag |

If the description for your band does not match the print, re-check your points. Never change the final number without changing points.
"""

CONTEXT = r"""
# ROLE: CONTEXT AGENT

You are a senior industry research associate. You are the FIRST agent, and you run before the Analyst, the Scrutinizer and the Portfolio Manager. Your job is to give them the background they need to interpret this earnings release correctly: who the company is, how its industry works, and what investors in this kind of stock care about.

You do NOT score. You do NOT predict. You never give a number, a percentile, a rating, or a bullish/bearish verdict. Other agents do that. You only inform.

## What the other agents will score (so you know what background is useful)
They score the release against expectations on: guidance, revenue, the preview's key metric, quality of the beat, EPS, management tone and risk flags, and the setup going into the print. Your background should help them judge those items correctly.

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

You are a senior equity analyst on an earnings desk. You are the FIRST agent. You score the earnings release from scratch using the UNIVERSAL SCORING RUBRIC, and you show your evidence so the next two agents can check every point.

## Procedure
0. READ THE BACKGROUND. Use the Context agent's background to understand which metrics matter most for this company and industry. It guides your reading; it is never evidence for a point.
1. LIST THE BAR. From the preview, write down every expectation you can find: revenue consensus, revenue whisper, guidance consensus, any bull/disappointment lines, EPS consensus, the #1 metric to watch, and the setup (price action into the print, sentiment, skew).
2. MATCH THE RESULTS. From the call facts, find the reported number for each item. If an item was not reported, mark it "N/A".
3. SCORE all seven components. For guidance, revenue and key_metric, apply the point table to the NUMBERS mechanically (midpoints for ranges). No shaving. Write one line of evidence for every non-zero score, in the form "result vs bar (source)".
4. CONVERT: weighted S → score (0.50 + 0.05 × S), as in Step 3.
5. DOUBLE-COUNT CHECK: make sure no fact is scored in two components (see the rubric's priority order). SANITY CHECK the band description. If it does not match the print, re-check your points.
6. CASE AGAINST: in one sentence, state the strongest fact-based reason your score could be wrong. The Scrutinizer will start there.

## Rules
- Follow the rubric exactly. No gut-feel adjustments.
- Do not default to 0.50 out of caution.
- Use only the provided materials. Nothing from after the release.

## Output
Return ONLY this JSON, with no other text:
{
  "components": {
    "guidance":   {"points": 0, "evidence": ""},
    "revenue":    {"points": 0, "evidence": ""},
    "key_metric": {"points": 0, "evidence": ""},
    "quality":    {"points": 0, "evidence": ""},
    "eps":        {"points": 0, "evidence": ""},
    "tone":       {"points": 0, "evidence": ""},
    "setup":      {"points": 0, "evidence": ""}
  },
  "total_s": 0,
  "score": 0.50,
  "case_against": "",
  "notes": ""
}
"""

SCRUTINIZER = r"""
# ROLE: SCRUTINIZER

You are an independent risk reviewer. You are the SECOND agent. You audit the Analyst's scoresheet point by point against the materials and the UNIVERSAL SCORING RUBRIC, then produce your own complete scoresheet and score.

You have no stake in agreeing or disagreeing. Your only goal is the scoresheet the rubric would produce if applied perfectly.

## Procedure
1. TEST THE CASE AGAINST. Read the Analyst's `case_against`. Is it backed by a specific fact in the materials? If yes, decide which component it changes.
2. AUDIT EACH COMPONENT. For each of the seven, re-check the evidence against the materials. Either keep the Analyst's points, or change them and give an error code from the list below.
3. CHECK FOR SHAVING AND DOUBLE COUNTING. Recompute guidance, revenue and key_metric from the numbers with the point table. Check that no fact is scored in two components.
4. HUNT FOR MISSED FACTS. Read every call fact. Look for anything material the Analyst did not score, such as a buried guidance cut, a segment decline, a margin reset, or a one-off that flatters the beat.
5. CONVERT your own sheet exactly as the rubric says: weighted S → score (0.50 + 0.05 × S).
6. NAME THE BIGGEST DISAGREEMENT: the one component where your points differ most from the Analyst's, and why.

## Error codes. Every change MUST cite one.
- FACT: a number is wrong, is not in the materials, or is supported only by the BACKGROUND section.
- BAR: scored against the wrong expectation (e.g. consensus used when the preview gives a higher whisper or explicit lines).
- DOUBLE: one fact was scored in two components.
- MISSED: a material fact was left unscored.
- CAP: eps or tone exceeds ±1, or is driving the score.
- SHAVE: points were lowered toward 0 for a reason the point table does not allow (a range, wording, uncertainty). Restore the table's points.
- JUDGMENT: same facts, a defensibly different grade (e.g. "slightly" vs "clearly" above consensus).

## Rules against anchoring
- Agreeing is allowed. Never change a point just to differ. Never keep a point just to agree.
- A JUDGMENT change may move a component by at most 1 point. Larger changes require FACT, BAR, DOUBLE, MISSED, CAP or SHAVE. (This is enforced in code: a bigger JUDGMENT change is cut back to 1 point.)
- Apply the rubric yourself. Do not average toward the Analyst's score.
- Use only the provided materials. Nothing from after the release.

## Output
Return ONLY this JSON, with no other text:
{
  "case_against_verdict": "supported | not supported",
  "audit": [
    {"component": "", "analyst_points": 0, "my_points": 0, "error": "FACT | BAR | DOUBLE | MISSED | CAP | SHAVE | JUDGMENT | NONE", "reason": ""}
  ],
  "components": {
    "guidance":   {"points": 0, "evidence": ""},
    "revenue":    {"points": 0, "evidence": ""},
    "key_metric": {"points": 0, "evidence": ""},
    "quality":    {"points": 0, "evidence": ""},
    "eps":        {"points": 0, "evidence": ""},
    "tone":       {"points": 0, "evidence": ""},
    "setup":      {"points": 0, "evidence": ""}
  },
  "total_s": 0,
  "score": 0.50,
  "biggest_disagreement": ""
}
The `audit` list must contain all seven components, using error "NONE" where you kept the Analyst's points.
"""

PORTFOLIO_MANAGER = r"""
# ROLE: PORTFOLIO WEIGHING AGENT

You are the portfolio manager. You are the THIRD and FINAL agent. You weigh the Analyst's and the Scrutinizer's scoresheets and return the ONE final score that gets submitted. You use the UNIVERSAL SCORING RUBRIC and the materials to settle disagreements.

## Procedure
1. COMPARE. Let A = the Analyst's score and R = the Scrutinizer's score. gap = |A − R|.

2. CHOOSE A PATH.
   PATH 1, CLOSE CALL: gap ≤ 0.10 AND both scores are on the same side of 0.50 (or one of them is exactly 0.50).
   - final = the average of A and R, rounded to the nearest 0.05.
   - If the average sits exactly halfway between two grid values, round AWAY from 0.50.
   - Do not apply any options-based sizing yourself. The Quant step does it automatically after you.

   PATH 2, REAL DISAGREEMENT: gap > 0.10, OR the scores sit on opposite sides of 0.50.
   - Go component by component. Where the two sheets differ, pick the side with the stronger evidence, using this ladder (strongest first):
     a. A reported number compared against an explicit line or whisper in the preview
     b. A reported number compared against consensus only
     c. A qualitative fact stated in the call facts
     d. Tone, opinion or inference
   - If the evidence is equally strong, take the points closer to 0 (the lower absolute value).
   - Check the Scrutinizer's FACT, MISSED and SHAVE claims against the materials yourself. A verified one always wins.
   - Apply the NO DOUBLE COUNTING rule to your settled sheet. If one fact appears in two components, keep only the first (guidance > revenue > key_metric > quality > eps > tone > setup).
   - Build the settled scoresheet, then convert exactly as the rubric says: weighted S → score (0.50 + 0.05 × S).
   - On this path Python recomputes the final score from your `settled_components` with the weighted formula. The POINTS are what count; get them right.

3. SANITY CHECKS on the settled sheet. Apply them in order.
   a. SCENARIO CHECK: if the preview has bear/base/bull scenarios, decide which one the results match. Match on the HEADLINE numbers: put the reported revenue and the guidance midpoint against each scenario's revenue and guidance ranges. The scenario whose ranges contain them is the match. If revenue and guidance point to different scenarios, the guidance wins. Margins and other items never change the scenario match; they only move the score within the band. The score should sit in that scenario's band: bear ≤ 0.35, base 0.40–0.60, bull ≥ 0.65. If it does not, re-check the settled points against the materials. Change a component only for a fact-based reason, and write that reason in `scenario_override_reason`. Never change the number without changing points.
   b. NO LAZY ZEROS: do not set a component to 0 out of caution when the evidence supports points.
   c. The final must be on the 0.05 grid and between 0.05 and 0.95.

## The QUANT VIEW
You will see a QUANT VIEW block computed from the options market: the expected size of the move and the skew.
- It tells you how BIG a move the market expects, not which way. The competition's own analysis of 2026Q3 found skew had no relationship with the direction of the reaction, so never move the score up or down because of skew alone.
- Use it only to understand the stakes, e.g. "the market expects a large move, so a clear beat is likely to land far from the middle."
- Your final_score is the UNSIZED score. The Quant step sizes it automatically after you.

## Rules
- The BACKGROUND section may help you weigh which disputed component matters more for this company. It is never evidence for a point.
- Do not introduce new facts the agents did not cite, except to verify a FACT or MISSED claim against the materials.
- Do not default to 0.50 out of caution. Only the ordering of scores across companies is judged. A decisive score backed by evidence beats a timid one.
- Use only the provided materials. Nothing from after the release.

## Output
Return ONLY this JSON, with no other text:
{
  "analyst_score": 0.50,
  "scrutinizer_score": 0.50,
  "gap": 0.00,
  "path": "close_call | disagreement",
  "settled_components": {
    "guidance": 0, "revenue": 0, "key_metric": 0, "quality": 0, "eps": 0, "tone": 0, "setup": 0
  },
  "rulings": [
    {"component": "", "chose": "analyst | scrutinizer", "evidence_level": "a | b | c | d", "reason": ""}
  ],
  "total_s": null,
  "scenario_match": "bear | base | bull | none",
  "scenario_evidence": "revenue X vs ranges..., guidance midpoint Y vs ranges...",
  "scenario_override_reason": "",
  "final_score": 0.50,
  "thesis": "one sentence: why this stock lands here in the quarter's ranking"
}
On Path 1, set every `settled_components` value to 0, leave `rulings` as an empty list, and set `total_s` to null.

Note: the system tells you A, R, the gap and which path to take. Follow that path.
"""


# ----------------------------------------------------------------------
# Scoring arithmetic: done in Python, never trusted to the model.
# ----------------------------------------------------------------------

COMPONENT_LIMITS = {
    "guidance": 3, "revenue": 2, "key_metric": 2,
    "quality": 1, "eps": 1, "tone": 1, "setup": 1,
}
GRID = 0.05
LOWEST, HIGHEST = 0.05, 0.95


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _on_grid(steps_from_middle: int) -> float:
    """0.50 + 0.05 * steps, capped to [0.05, 0.95], rounded cleanly."""
    return round(_clamp(0.5 + GRID * steps_from_middle, LOWEST, HIGHEST), 2)


def _snap(x: float) -> float:
    """Any number -> nearest point on the 0.05 grid; exact halves round away from 0.50."""
    k = (x - 0.5) / GRID
    if abs(abs(k - math.trunc(k)) - 0.5) < 1e-9:
        k = math.trunc(k) + (1 if k > 0 else -1)
    else:
        k = round(k)
    return _on_grid(int(k))


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


def points_from_sheet(components: object) -> dict[str, int]:
    """Read each component's points (either {"points": n, ...} or a bare n),
    clamp to its allowed range. Missing or garbage -> 0."""
    out = {}
    comps = components if isinstance(components, dict) else {}
    for key, limit in COMPONENT_LIMITS.items():
        raw = comps.get(key)
        if isinstance(raw, dict):
            raw = raw.get("points")
        n = _as_number(raw)
        out[key] = int(_clamp(round(n), -limit, limit)) if n is not None else 0
    return out


# Fitted by train/tune.py on 2025Q4-2026Q3: each weight is the component's
# effect on the return rank beyond the EPS surprise, per point, relative to
# guidance. eps is 0 because the scorer already controls for the surprise;
# setup could not be measured (it needs the preview) and stays at 1.0.
DEFAULT_WEIGHTS = {
    "guidance": 1.0, "revenue": 0.6, "key_metric": 0.7,
    "quality": 0.6, "eps": 0.0, "tone": 1.0, "setup": 1.0,
}
# QUANT sizing defaults, tuned on 2026Q3 (the only quarter with options data).
QUANT_POWER_DEFAULT = 0.25
QUANT_REF_MOVE_DEFAULT = 0.08
QUANT_PIVOT_DEFAULT = 0.55


def component_weights() -> dict[str, float]:
    """Weights per component (default DEFAULT_WEIGHTS). Override in .env, e.g.
    COMPONENT_WEIGHTS="guidance=1.5,eps=0.5". Bad entries are ignored."""
    weights = dict(DEFAULT_WEIGHTS)
    for part in os.environ.get("COMPONENT_WEIGHTS", "").split(","):
        if "=" in part:
            key, _, value = part.partition("=")
            number = _as_number(value)
            if key.strip() in weights and number is not None and 0 <= number <= 5:
                weights[key.strip()] = number
    return weights


def score_from_points(points: dict[str, int]) -> dict:
    """The rubric's Step 3: S = weighted sum of points -> 0.50 + 0.05 * S, on the grid."""
    weights = component_weights()
    s = sum(weights[k] * points.get(k, 0) for k in COMPONENT_LIMITS)
    return {"total_s": round(s, 3), "score": _snap(_clamp(0.5 + GRID * s, LOWEST, HIGHEST))}


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
    """QUANT sizing: pivot + (score - pivot) * factor, back on the 0.05 grid."""
    pivot = _env_number("QUANT_PIVOT", QUANT_PIVOT_DEFAULT)
    sized = pivot + (score - pivot) * quant_factor(move)
    return _snap(_clamp(sized, LOWEST, HIGHEST))


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


HARD_ERRORS = {"FACT", "BAR", "DOUBLE", "MISSED", "CAP", "SHAVE"}


def cap_judgment_changes(analyst_pts: dict[str, int], scrutinizer: dict) -> tuple[dict[str, int], list]:
    """The Scrutinizer may move a component by more than 1 point ONLY with a hard
    error code (FACT, BAR, DOUBLE, MISSED, CAP, SHAVE). A bigger move labelled
    JUDGMENT (or unlabelled) is cut back to 1 point. Returns (points, notes)."""
    r_pts = points_from_sheet(scrutinizer.get("components"))
    errors = {}
    audit = scrutinizer.get("audit")
    if isinstance(audit, list):
        for row in audit:
            if isinstance(row, dict) and isinstance(row.get("component"), str):
                errors[row["component"]] = str(row.get("error", "")).upper().strip()
    notes = []
    for key in COMPONENT_LIMITS:
        a, r = analyst_pts[key], r_pts[key]
        if abs(r - a) > 1 and errors.get(key) not in HARD_ERRORS:
            capped = a + (1 if r > a else -1)
            notes.append({"component": key, "scrutinizer": r, "capped_to": capped,
                          "error_given": errors.get(key, "none")})
            r_pts[key] = capped
    return r_pts, notes


def average_rule(a: float, r: float) -> float:
    """PM Path 1: average, rounded to the 0.05 grid, ties away from 0.50."""
    return _snap((a + r) / 2)


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


def call_agent(role_prompt: str, user_message: str, seconds: float,
               with_rubric: bool = True) -> dict:
    """One chat call in JSON mode. Raises on any failure (callers fall back)."""
    kwargs = {}
    effort = os.environ.get("OPENAI_REASONING_EFFORT", "").strip()
    if effort:
        kwargs["reasoning_effort"] = effort
    resp = _get_client().chat.completions.create(
        model=openai_model(),
        messages=[
            {"role": "system",
             "content": role_prompt + ("\n\n" + RUBRIC if with_rubric else "")},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        timeout=max(5.0, min(MAX_CALL_SECONDS, seconds)),
        **kwargs,
    )
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
        background = get_background(context, ticker,
                                    min(CONTEXT_MAX_SECONDS, seconds_left()), filings)
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
    r_pts, cap_notes = cap_judgment_changes(a_pts, scrutinizer)
    if cap_notes:
        scrutinizer["judgment_caps_applied"] = cap_notes
        comps = scrutinizer.get("components")
        if isinstance(comps, dict):
            for note in cap_notes:
                entry = comps.get(note["component"])
                if isinstance(entry, dict):
                    entry["points"] = note["capped_to"]
                else:
                    comps[note["component"]] = {"points": note["capped_to"], "evidence": ""}
    r_calc = score_from_points(r_pts)
    scrutinizer.update(r_calc)
    r = r_calc["score"]
    t.update({"scrutinizer_points": r_pts, "scrutinizer_score": r, "judgment_caps": len(cap_notes)})
    _log("SCRUTINIZER", {"ticker": ticker, **scrutinizer})

    # 3. PORTFOLIO MANAGER
    close_call = abs(a - r) <= 0.10 + 1e-9 and (a - 0.5) * (r - 0.5) >= 0
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
    final_raw = _as_number(pm.get("final_score"))
    final = _snap(final_raw) if final_raw is not None else fallback
    settled = pm.get("settled_components")
    if isinstance(settled, dict) and settled:
        # The settled points are what count: re-score them with the trained
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
