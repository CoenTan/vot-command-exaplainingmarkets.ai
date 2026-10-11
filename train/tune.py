"""Refit the rubric's COMPONENT_WEIGHTS and QUANT settings on the historical archive.

No AI calls. Each release's call facts are read with keyword rules that mirror the
rubric's components (guidance raised/cut, revenue up/down, beat/miss vs a stated bar,
margins and one-offs, tone and risk flags). The EPS component is the sign of the
earnings surprise, which is what the rubric's "EPS vs consensus" amounts to.

The target is the competition's own: the percentile rank of the one-day abnormal
return (car1) within each quarter, explained BEYOND the percentile-ranked earnings
surprise (delta R^2). Weights are the per-point effect of each component, relative
to guidance. QUANT settings are grid-searched on the quarters that carry option
statistics, using the reference baselines' predictions as stand-in scores.

Download the archive first (see the explaining-markets/examples repo), then:

    python train/tune.py path/to/data/archive

Needs numpy and pandas. Prints suggested defaults for predict.py / .env.
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Typical |points| an agent gives when a component clearly points one way.
# Used to turn a +1 directional effect into an effect per rubric point.
TYPICAL_POINTS = {"guidance": 2.0, "revenue": 1.5, "key_metric": 1.5,
                  "quality": 1.0, "eps": 1.0, "tone": 1.0}
# Keyword proxies for revenue and key_metric are the weakest (they cannot see the
# preview's bar), so their fitted weights are shrunk halfway toward 1.0.
SHRINK_TO_ONE = {"revenue", "key_metric"}

G = r"(guidance|outlook|forecast|guide|target|range)"
UP = r"(rais(e|ed|ing)|increas(e|ed|ing)|lift(ed|ing)?|boost(ed|ing)?|upgrad(e|ed)|hik(e|ed))"
DN = r"(lower(ed|ing)?|cut(ting)?|reduc(e|ed|ing)|trimm(ed|ing)|downgrad(e|ed)|withdr(ew|awn|aw)|suspend(ed)?|pull(ed)?)"
PATTERNS = {
    "guid_up": [rf"{UP}\b[^.;]{{0,60}}\b{G}", rf"\b{G}\b[^.;]{{0,40}}\b(was|were|is|has been|been)?\s*{UP}"],
    "guid_dn": [rf"{DN}\b[^.;]{{0,60}}\b{G}", rf"\b{G}\b[^.;]{{0,40}}\b(was|were|is|has been|been)?\s*{DN}"],
    "beat_bar": [r"(exceed|surpass|top|beat|outperform|ahead of|above)\w*\b[^.;]{0,40}\b"
                 r"(guidance|outlook|consensus|expectations|estimates|high end)"],
    "miss_bar": [r"(miss|below|short of|underperform|disappoint)\w*\b[^.;]{0,40}\b"
                 r"(guidance|outlook|consensus|expectations|estimates|low end)"],
    "rev_up": [r"\b(revenue|sales)\b[^.;]{0,50}\b(rose|grew|increased|up|growth of|jumped|climbed|surged)\b"],
    "rev_dn": [r"\b(revenue|sales)\b[^.;]{0,50}\b(fell|declined|decreased|down|dropped|decline of|contracted)\b"],
    "mgn_up": [r"margin[^.;]{0,40}\b(expan\w+|improv\w+|increas\w+|rose|widen\w*|up)\b",
               r"\b(expan\w+|improv\w+)\b[^.;]{0,25}margin"],
    "mgn_dn": [r"margin[^.;]{0,40}\b(contract\w*|compress\w*|declin\w+|decreas\w+|fell|narrow\w*|down|pressure\w*)\b",
               r"\b(compress\w+|contract\w+)\b[^.;]{0,25}margin"],
    "tone_pos": [r"\brecord\b", r"\baccelerat\w+", r"\bstrong(er)? demand\b", r"\bmomentum\b", r"\bconfiden\w+"],
    "tone_neg": [r"\b(investigation|subpoena|lawsuit|restatement|material weakness|going concern)\b",
                 r"\b(resign\w*|depart\w*|step(ping)? down)\b", r"\bheadwind\w*", r"\b(softness|soft|weak\w*)\b",
                 r"\buncertain\w*", r"\bdelay\w*", r"\b(impairment|write-?down)\b", r"\b(layoff|workforce reduction)\w*"],
    "oneoff": [r"\bone-?time\b", r"\btax benefit\b", r"\bnon-?recurring\b", r"\bfavorable (timing|tax)\b",
               r"\bpull-?forward\b"],
}
REGEX = {k: [re.compile(p, re.I) for p in v] for k, v in PATTERNS.items()}
COMPONENTS = ["guidance", "revenue", "key_metric", "quality", "eps", "tone"]


def percentile_ranks(values: list[float]) -> list[float]:
    """The competition's percentile rank: min -> 0, max -> 1, ties share the average."""
    n = len(values)
    if n < 2:
        return [0.5] * n
    return list((pd.Series(values).rank(method="average") - 1) / (n - 1))


def components(facts: list[str], surprise: float) -> dict[str, float]:
    c = {k: sum(1 for f in facts if any(r.search(f) for r in rxs)) for k, rxs in REGEX.items()}
    sign = lambda a, b: float(np.sign(c[a] - c[b]))  # noqa: E731
    return {
        "guidance": sign("guid_up", "guid_dn"),
        "revenue": sign("rev_up", "rev_dn"),
        "key_metric": sign("beat_bar", "miss_bar"),
        "quality": float(np.clip(np.sign(c["mgn_up"] - c["mgn_dn"]) - (1 if c["oneoff"] else 0), -1, 1)),
        "eps": float(np.sign(surprise)),
        "tone": float(np.clip(c["tone_pos"] - c["tone_neg"], -1, 1)),
    }


def load(archive: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(archive.glob("*.jsonl.gz")):
        quarter = path.name.removesuffix(".jsonl.gz").rpartition("_")[2]
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                es = (r.get("metrics") or {}).get("earnings_surprise") or {}
                if r.get("status") != "scored" or es.get("surprise_status") != "ok":
                    continue
                items = {i.get("id"): i.get("content") for i in (r.get("disclosure") or {}).get("items") or []}
                facts = items.get("earnings-call-facts") or []
                stats = items.get("option-implied-stats") or {}
                mv = (stats.get("implied_absolute_earnings_move") or {}) if isinstance(stats, dict) else {}
                move = mv.get("value") if mv.get("status") == "ok" else None
                for a in r.get("focal_assets") or []:
                    t = a.get("identifier_value")
                    car1 = ((r.get("event_returns") or {}).get(t) or {}).get("car1")
                    if car1 is None:
                        continue
                    row = {"quarter": quarter, "event_id": r["event_id"], "car1": float(car1),
                           "surprise": float(es["surprise"]), "move": move,
                           **components(facts if isinstance(facts, list) else [], es["surprise"])}
                    for key, preds in (r.get("baseline_predictions") or {}).items():
                        if t in (preds or {}):
                            row["bl_" + key.split("/")[0]] = float(preds[t])
                    rows.append(row)
    df = pd.DataFrame(rows)
    df = df[df.groupby("quarter").car1.transform("size") >= 200]   # skip quarters still filling in
    df["y"] = df.groupby("quarter").car1.transform(lambda s: percentile_ranks(s.tolist()))
    df["s"] = df.groupby("quarter").surprise.transform(lambda s: percentile_ranks(s.tolist()))
    return df.reset_index(drop=True)


def r2(y: np.ndarray, xs: list) -> float:
    X = np.column_stack([np.ones(len(y))] + [np.asarray(x, float) for x in xs])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    return 1 - e @ e / ((y - y.mean()) @ (y - y.mean()))


def delta_r2(g: pd.DataFrame, pred) -> float:
    return r2(g.y.values, [g.s.values, pred]) - r2(g.y.values, [g.s.values])


def snap(x: np.ndarray) -> np.ndarray:
    """predict.py's _snap, vectorised: 0.05 grid, halves away from 0.50, capped."""
    k = (x - 0.5) / 0.05
    k = np.where(np.abs(np.abs(k - np.trunc(k)) - 0.5) < 1e-9, np.trunc(k) + np.sign(k), np.round(k))
    return np.round(np.clip(0.5 + 0.05 * k, 0.05, 0.95), 2)


def sized(p, move, power, ref, pivot):
    m = np.asarray(move, float)
    f = np.where(np.isnan(m) | (m <= 0) | (power <= 0), 1.0, (np.nan_to_num(m, nan=ref) / ref) ** power)
    return snap(np.clip(pivot + (np.asarray(p) - pivot) * f, 0.05, 0.95))


def fit_weights(df: pd.DataFrame) -> dict[str, float]:
    d = df.copy()
    for c in ("y", "s"):
        d[c] -= d.groupby("quarter")[c].transform("mean")
    X = np.column_stack([np.ones(len(d)), d.s] + [d[c] for c in COMPONENTS])
    b, *_ = np.linalg.lstsq(X, d.y.values, rcond=None)
    effect = dict(zip(COMPONENTS, b[2:], strict=True))
    print("Effect of a +1 directional signal on the return rank (pooled):")
    for c in COMPONENTS:
        print(f"  {c:11s} {effect[c]:+.4f}")
    per_point = {c: max(0.0, effect[c]) / TYPICAL_POINTS[c] for c in COMPONENTS}
    ref = per_point["guidance"]
    weights = {}
    for c in COMPONENTS:
        w = per_point[c] / ref
        if c in SHRINK_TO_ONE:
            w = (w + 1.0) / 2
        weights[c] = round(min(w, 1.0) if c != "guidance" else 1.0, 1)
    if effect["eps"] < 0.01:   # indistinguishable from zero: the scorer already has the surprise
        weights["eps"] = 0.0
    return weights


def tune_quant(df: pd.DataFrame, weights: dict[str, float]) -> tuple[float, float, float]:
    q = df[df.move.notna()].quarter.unique()
    if len(q) == 0:
        print("No options data in the archive: leaving QUANT settings alone.")
        return 0.25, 0.08, 0.55
    d = df[df.quarter.isin(q)]
    pts = {"guidance": 2 * d.guidance, "revenue": d.revenue, "key_metric": d.key_metric,
           "quality": d.quality, "eps": d.eps, "tone": d.tone}
    proxy = snap(0.5 + 0.05 * sum(weights[c] * pts[c] for c in COMPONENTS))
    preds = {"rubric": proxy} | {c: snap(d[c].fillna(d[c].mean()).values)
                                 for c in d.columns if c.startswith("bl_") and d[c].notna().mean() > 0.9}
    ref_move = round(float(d.move.median()), 2)
    pivot = round(float(np.mean(proxy)) * 20) / 20
    best = None
    print(f"\nQUANT grid on {', '.join(q)} (ref move = median {ref_move:.2f}; each predictor pivots at its own mean):")
    for power in (0.0, 0.25, 0.5, 0.75, 1.0):
        # each predictor is sized around its OWN average, as a live bot would be
        gains = [1000 * (delta_r2(d, sized(p, d.move, power, ref_move, float(np.mean(p)))) - delta_r2(d, p))
                 for p in preds.values()]
        print(f"  power {power:.2f}: mean gain {np.mean(gains):+.2f}, worst {min(gains):+.2f}  (x1e-3 delta R^2)")
        if best is None or np.mean(gains) > best[0]:
            best = (np.mean(gains), power)
    return best[1], ref_move, pivot


def main() -> None:
    archive = Path(sys.argv[1] if len(sys.argv) > 1 else "data/archive")
    df = load(archive)
    print(f"{len(df)} scored releases with a surprise, quarters: {sorted(df.quarter.unique())}\n")
    weights = fit_weights(df)
    weights["setup"] = 1.0  # needs the preview; not measurable with keyword rules
    power, ref, pivot = tune_quant(df, weights)

    print("\nPer-quarter delta R^2 x1e-3 (rubric proxy, unsized; weights fitted on all quarters):")
    pts = {"guidance": 2 * df.guidance, "revenue": df.revenue, "key_metric": df.key_metric,
           "quality": df.quality, "eps": df.eps, "tone": df.tone}
    for name, w in (("all weights 1", {c: 1.0 for c in COMPONENTS}), ("fitted", weights)):
        p = snap(0.5 + 0.05 * sum(w[c] * pts[c] for c in COMPONENTS))
        res = {q: round(1000 * delta_r2(g, p[g.index]), 1) for q, g in df.groupby("quarter")}
        print(f"  {name:14s} {res}")

    print("\nSuggested settings:")
    print("  COMPONENT_WEIGHTS=" + ",".join(f"{k}={v}" for k, v in weights.items()))
    print(f"  QUANT_POWER={power}\n  QUANT_REF_MOVE={ref}\n  QUANT_PIVOT={pivot}")


if __name__ == "__main__":
    main()
