"""Try the 3 agents on 6 real past events, using YOUR OpenAI key from .env.

    uv run python try_agents.py               # all 6 events
    uv run python try_agents.py 1             # just the first one (cheapest)
    uv run python try_agents.py CRM AVGO      # only these tickers

Costs a few cents. It checks that your key and model work and shows what
the agents say, before you deploy. It is a smoke test, NOT a performance
test: 6 events is far too few, and the AI may already know how these
stocks reacted.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

import predict  # noqa: E402  (must come after load_dotenv)

events = json.loads(Path(__file__).with_name("sample_events.json").read_text())
args = sys.argv[1:]
if args and args[0].isdigit():
    events = events[:int(args[0])]
elif args:
    wanted = {a.upper() for a in args}
    events = [ev for ev in events if ev["ticker"].upper() in wanted]
    missing = wanted - {ev["ticker"].upper() for ev in events}
    if missing:
        sys.exit(f"Not in sample_events.json: {', '.join(sorted(missing))}")

print(f"Model: {predict.openai_model()}\n")
rows = []
for ev in events:
    t0 = time.monotonic()
    score = predict.run_agents(bundle=ev["bundle"], ticker=ev["ticker"],
                               event_type="EARNINGS_RELEASE",
                               event_datetime=ev.get("event_datetime"),
                               knowledge_cutoff=ev.get("knowledge_cutoff"))
    secs = time.monotonic() - t0
    rows.append((ev["ticker"], score, ev["car1"], secs))
    print(f"--> {ev['ticker']}: predicted {score:.4f} | actual move {ev['car1']:+.1%} | {secs:.0f}s\n")

print("\nSUMMARY (higher prediction should mean a better actual move)")
print(f"{'ticker':<8}{'predicted':>10}{'actual':>10}{'seconds':>9}")
for ticker, score, car1, secs in sorted(rows, key=lambda r: -r[1]):
    print(f"{ticker:<8}{score:>10.4f}{car1:>+10.1%}{secs:>9.0f}")
slowest = max(r[3] for r in rows)
print(f"\nSlowest event: {slowest:.0f}s (must stay well under 300s)")
