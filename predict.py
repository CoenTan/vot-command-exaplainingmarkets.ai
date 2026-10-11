"""VOT COMMAND: 3-agent predictor (Analyst -> Scrutinizer -> Portfolio Manager).

THIS IS THE ONLY FILE YOU EDIT. Everything else in the repo is plumbing.

How one event flows:
  1. Fetch the event's materials (call facts, preview, option stats).
  2. ANALYST scores the release with the rubric.
  3. SCRUTINIZER audits the Analyst and scores it again.
  4. PORTFOLIO MANAGER settles it (skipped on close calls: the rubric says
     "average them", so Python does that and saves a call).
  5. Python re-does every agent's arithmetic, so a model's maths mistake can
     never reach the submission. The answer is always on the 0.05 grid.

Safety nets (a missed event counts against you, so something is ALWAYS sent):
  - No OPENAI_API_KEY             -> 0.50
  - Analyst fails                 -> 0.50
  - Scrutinizer fails / no time   -> Analyst's score
  - PM fails / no time            -> average of Analyst and Scrutinizer

Optional settings in .env:
  OPENAI_MODEL               model for every agent (default: the starter's default)
  OPENAI_REASONING_EFFORT    e.g. low / medium / high (leave unset if unsure)
  ALWAYS_CALL_PM             set to 1 to call the PM even on close calls

To change how agents think, edit the four prompt blocks below
(RUBRIC, ANALYST, SCRUTINIZER, PORTFOLIO_MANAGER).
"""

from __future__ import annotations

import json
import math
import os
import re
import time

import httpx
from openai import OpenAI

from explaining_markets.config import openai_model

# ----------------------------------------------------------------------
# Time budget. The platform gives 5 minutes from the ACK. We aim to finish
# all agents inside TOTAL_BUDGET_SECONDS, leaving room to submit.
# ----------------------------------------------------------------------
SUMMARY_TIMEOUT_SECONDS = 15.0
TOTAL_BUDGET_SECONDS = 230.0   # all three agents together
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
# THE PROMPTS. Each agent's instructions = its role block + the RUBRIC.
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
- The EPS beat/miss is the least useful signal, because everyone already sees it. Guidance, revenue and the key metric matter more.

## Inputs you may receive
- `earnings-call-facts`: facts extracted from the earnings call. These are the RESULTS.
- `earnings-preview`: a research note written BEFORE the release. These are the EXPECTATIONS: consensus, whisper/buy-side numbers, bull/base/bear scenarios, metrics to watch, setup.
- `option-implied-stats`: implied earnings volatility, implied absolute move, 25-delta skew, all measured before the release. Any of these may be missing.

Use ONLY these inputs plus general knowledge from before the release. Never use, search for or assume news, prices or reactions from after the release. If an input is missing, treat that item as missing. Do not invent numbers.

## Step 1: Find "the bar"
For each item, the bar is the HIGHEST credible expectation in the preview:
1. If the preview names explicit lines (e.g. "$105B is the bull line, $103B the disappointment line"), use those lines.
2. Otherwise use the whisper / buy-side number if one is given.
3. Otherwise use sell-side consensus.
4. If no expectation exists for an item, score that item 0 and mark it "N/A".

## Step 2: Score the seven components

| # | key | Component | Range |
|---|---|---|---|
| 1 | guidance | Next-period / full-year guidance vs the bar | -3 to +3 |
| 2 | revenue | Reported revenue vs the bar | -2 to +2 |
| 3 | key_metric | The preview's #1 "metric to watch" that is NOT guidance or revenue (e.g. gross margin, segment revenue, subscribers) | -2 to +2 |
| 4 | quality | Quality of the beat or miss | -1 to +1 |
| 5 | eps | EPS vs consensus | -1 to +1 |
| 6 | tone | Management tone and risk flags | -1 to +1 |
| 7 | setup | Positioning going into the print | -1 to +1 |

Point scale for components 1-3. Use only the part of the scale inside each component's range.
- +3: clears the bull line, or beats the whisper by a wide margin
- +2: beats the whisper, or clearly beats consensus
- +1: slightly above consensus
-  0: in line, mixed, or no information (mark "N/A" if no information)
- -1: slightly below consensus
- -2: clearly below consensus
- -3: below the disappointment line, or guidance cut sharply or withdrawn

Components 4-7:
- quality: +1 = margin expansion, beat driven by core demand. -1 = beat flattered by one-offs, tax, buybacks or accounting, or margins compressing. 0 = neither.
- eps: +1 = beat consensus. -1 = missed. 0 = in line or N/A.
- tone: +1 = specific, confident commentary backed by numbers. -1 = new risks, hedging, executive departure, investigation, demand warnings. 0 = neither.
- setup: +1 = expectations were LOW going in (stock fell into the print, negative skew, bearish sentiment) AND results were at least in line. -1 = expectations were STRETCHED (big run-up, euphoric sentiment, crowded) AND results were not clearly above the bar. 0 = otherwise.

Hard rules:
- Every non-zero point must cite evidence: the result, the bar, and the source (facts or preview).
- No double counting. Each fact scores in ONE component only. If a revenue beat comes from the key segment, score it under revenue OR key_metric, not both.
- eps and tone can never exceed ±1. They may tip a score but never drive it.
- No adjustments outside the rubric. If something important fits no component, write it in `notes`. It does not change the score.

## Step 3: Convert points to a score
1. S = the sum of all seven components (range -11 to +11).
2. base_score = 0.50 + 0.05 × S, capped to the range 0.05 to 0.95.
3. Implied-move adjustment, using `implied_absolute_earnings_move`:
   - 0.08 or more: move ONE step (0.05) further from 0.50
   - 0.04 to 0.08: no change
   - under 0.04: move ONE step (0.05) toward 0.50, but never onto 0.50 (so 0.55 and 0.45 stay put)
   - missing or status not "ok": no change
   - A base_score of exactly 0.50 never moves. Keep the final score within 0.05 to 0.95.

## Step 4: Sanity check against the bands
| Score | S | What the print looks like |
|---|---|---|
| 0.95 | +9 or more | Blowout: guidance, revenue and key metric all clear the whisper |
| 0.80-0.90 | +6 to +8 | Clear beat-and-raise above the real bar |
| 0.65-0.75 | +3 to +5 | Good: beats the bar on most items, nothing alarming |
| 0.55-0.60 | +1 to +2 | Modestly positive, or mixed leaning good |
| 0.50 | 0 | In line, or genuinely balanced |
| 0.40-0.45 | -1 to -2 | Modestly negative, or mixed leaning bad |
| 0.25-0.35 | -3 to -5 | Misses the bar on a key item, or guidance below expectations |
| 0.10-0.20 | -6 to -8 | Clear miss plus weak guidance |
| 0.05 | -9 or less | Disaster: big miss, guidance cut, major new negative |

If the description for your band does not match the print, re-check your points. Never change the final number without changing points.
"""

ANALYST = r"""
# ROLE: ANALYST

You are a senior equity analyst on an earnings desk. You are the FIRST agent. You score the earnings release from scratch using the UNIVERSAL SCORING RUBRIC, and you show your evidence so the next two agents can check every point.

## Procedure
1. LIST THE BAR. From the preview, write down every expectation you can find: revenue consensus, revenue whisper, guidance consensus, any bull/disappointment lines, EPS consensus, the #1 metric to watch, and the setup (price action into the print, sentiment, skew).
2. MATCH THE RESULTS. From the call facts, find the reported number for each item. If an item was not reported, mark it "N/A".
3. SCORE all seven components with the rubric's point scales. Write one line of evidence for every non-zero score, in the form "result vs bar (source)".
4. CONVERT: S → base_score → implied-move adjustment → score.
5. SANITY CHECK the band description. If it does not match the print, re-check your points.
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
  "base_score": 0.50,
  "implied_move": null,
  "adjustment": "none | further | toward",
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
3. HUNT FOR MISSED FACTS. Read every call fact. Look for anything material the Analyst did not score, such as a buried guidance cut, a segment decline, a margin reset, or a one-off that flatters the beat.
4. CONVERT your own sheet exactly as the rubric says: S → base_score → implied-move adjustment → score.
5. NAME THE BIGGEST DISAGREEMENT: the one component where your points differ most from the Analyst's, and why.

## Error codes. Every change MUST cite one.
- FACT: a number is wrong or is not in the materials.
- BAR: scored against the wrong expectation (e.g. consensus used when the preview gives a higher whisper or explicit lines).
- DOUBLE: one fact was scored in two components.
- MISSED: a material fact was left unscored.
- CAP: eps or tone exceeds ±1, or is driving the score.
- JUDGMENT: same facts, a defensibly different grade (e.g. "slightly" vs "clearly" above consensus).

## Rules against anchoring
- Agreeing is allowed. Never change a point just to differ. Never keep a point just to agree.
- A JUDGMENT change may move a component by at most 1 point. Larger changes require FACT, BAR, DOUBLE, MISSED or CAP.
- Apply the rubric yourself. Do not average toward the Analyst's score.
- Use only the provided materials. Nothing from after the release.

## Output
Return ONLY this JSON, with no other text:
{
  "case_against_verdict": "supported | not supported",
  "audit": [
    {"component": "", "analyst_points": 0, "my_points": 0, "error": "FACT | BAR | DOUBLE | MISSED | CAP | JUDGMENT | NONE", "reason": ""}
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
  "base_score": 0.50,
  "implied_move": null,
  "adjustment": "none | further | toward",
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
   - Both inputs already include the implied-move adjustment, so do not apply it again.

   PATH 2, REAL DISAGREEMENT: gap > 0.10, OR the scores sit on opposite sides of 0.50.
   - Go component by component. Where the two sheets differ, pick the side with the stronger evidence, using this ladder (strongest first):
     a. A reported number compared against an explicit line or whisper in the preview
     b. A reported number compared against consensus only
     c. A qualitative fact stated in the call facts
     d. Tone, opinion or inference
   - If the evidence is equally strong, take the points closer to 0 (the lower absolute value).
   - Check the Scrutinizer's FACT and MISSED claims against the materials yourself. A verified FACT or MISSED claim always wins.
   - Build the settled scoresheet, then convert exactly as the rubric says: S → base_score → implied-move adjustment → score.

3. SANITY CHECKS on the final score. Apply them in order.
   a. SCENARIO CHECK: if the preview has bear/base/bull scenarios, decide which one the results match. The final score should sit in that scenario's band: bear ≤ 0.35, base 0.40–0.60, bull ≥ 0.65. If it does not, either give a one-sentence, fact-based reason in `scenario_override_reason`, or move the score to the nearest edge of the band.
   b. NO LAZY 0.50: the final may be 0.50 only if the settled S is 0 (Path 2), or the average is 0.50 (Path 1).
   c. EPS/TONE-ONLY: if the only non-zero components in both sheets are eps and/or tone, the final must be between 0.40 and 0.60.
   d. The final must be on the 0.05 grid and between 0.05 and 0.95.

## Rules
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


def score_from_points(points: dict[str, int], move: float | None) -> dict:
    """The rubric's Step 3, exactly: S -> base_score -> implied-move adjustment."""
    s = sum(points.values())
    base = _on_grid(s)
    steps = round((base - 0.5) / GRID)
    adjustment = "none"
    if move is not None and steps != 0:
        if move >= 0.08:
            steps += 1 if steps > 0 else -1
            adjustment = "further"
        elif move < 0.04:
            steps += -1 if steps > 0 else 1
            adjustment = "toward"
    return {"total_s": s, "base_score": base, "adjustment": adjustment,
            "score": _on_grid(steps)}


def average_rule(a: float, r: float) -> float:
    """PM Path 1: average, rounded to the 0.05 grid, ties away from 0.50."""
    return _snap((a + r) / 2)


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


def call_agent(role_prompt: str, user_message: str, seconds: float) -> dict:
    """One chat call in JSON mode. Raises on any failure (callers fall back)."""
    kwargs = {}
    effort = os.environ.get("OPENAI_REASONING_EFFORT", "").strip()
    if effort:
        kwargs["reasoning_effort"] = effort
    resp = _get_client().chat.completions.create(
        model=openai_model(),
        messages=[
            {"role": "system", "content": role_prompt + "\n\n" + RUBRIC},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        timeout=max(5.0, min(MAX_CALL_SECONDS, seconds)),
        **kwargs,
    )
    content = resp.choices[0].message.content or ""
    return _parse_json(content)


def _log(tag: str, payload: object) -> None:
    try:
        print(f"[{tag}] " + json.dumps(payload, ensure_ascii=False)[:4000])
    except Exception:
        print(f"[{tag}] (unprintable)")


# ----------------------------------------------------------------------
# The pipeline: Analyst -> Scrutinizer -> Portfolio Manager.
# ----------------------------------------------------------------------

def run_agents(*, bundle: object, ticker: str, event_type: str) -> float:
    global _warned_no_key
    if not os.environ.get("OPENAI_API_KEY"):
        if not _warned_no_key:
            print("[WARN] OPENAI_API_KEY not set - submitting 0.5. Add it to .env and redeploy.")
            _warned_no_key = True
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

    # 1. ANALYST
    try:
        analyst = call_agent(ANALYST, context + "\n\nScore this release. Return JSON only.",
                             seconds_left())
    except Exception as exc:
        print(f"[ERROR] {ticker} analyst failed: {exc!r} -> 0.5")
        return 0.5
    a_calc = score_from_points(points_from_sheet(analyst.get("components")), move)
    analyst.update(a_calc)                 # overwrite the model's arithmetic with ours
    a = a_calc["score"]
    _log("ANALYST", {"ticker": ticker, **analyst})

    # 2. SCRUTINIZER
    if seconds_left() < MIN_CALL_SECONDS:
        print(f"[TIME] {ticker} no time for scrutinizer -> analyst {a}")
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
        return a
    r_calc = score_from_points(points_from_sheet(scrutinizer.get("components")), move)
    scrutinizer.update(r_calc)
    r = r_calc["score"]
    _log("SCRUTINIZER", {"ticker": ticker, **scrutinizer})

    # 3. PORTFOLIO MANAGER
    close_call = abs(a - r) <= 0.10 + 1e-9 and (a - 0.5) * (r - 0.5) >= 0
    fallback = average_rule(a, r)
    if close_call and os.environ.get("ALWAYS_CALL_PM", "") != "1":
        print(f"[FINAL] {ticker} close call: analyst {a}, scrutinizer {r} -> {fallback}")
        return fallback
    if seconds_left() < MIN_CALL_SECONDS:
        print(f"[TIME] {ticker} no time for PM -> average {fallback}")
        return fallback
    try:
        pm = call_agent(
            PORTFOLIO_MANAGER,
            context
            + "\n\nANALYST SCORESHEET:\n" + json.dumps(analyst, ensure_ascii=False)
            + "\n\nSCRUTINIZER SCORESHEET:\n" + json.dumps(scrutinizer, ensure_ascii=False)
            + f"\n\nVerified scores: analyst_score={a}, scrutinizer_score={r}, "
              f"gap={abs(a - r):.2f}.\n\nReturn the final JSON only.",
            seconds_left(),
        )
    except Exception as exc:
        print(f"[ERROR] {ticker} PM failed: {exc!r} -> average {fallback}")
        return fallback
    final_raw = _as_number(pm.get("final_score"))
    final = _snap(final_raw) if final_raw is not None else fallback
    settled = pm.get("settled_components")
    if isinstance(settled, dict) and settled:
        pm["python_check"] = score_from_points(points_from_sheet(settled), move)
    _log("PM", {"ticker": ticker, **pm})
    print(f"[FINAL] {ticker} analyst {a}, scrutinizer {r} -> PM {final}")
    return final
