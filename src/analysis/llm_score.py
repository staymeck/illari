"""Optional, out-of-band numeric context from a local LLM (Ollama, running
on this machine's own GPU — no Anthropic API cost) — strictly advisory.
Never used by any strategy piece or entry decision; only recorded
alongside a trade's already-known outcome, to check afterward whether it
correlates with anything. The project's founding decision to keep the
actual trading logic deterministic and backtestable stands unchanged —
this is a separate, clearly-labeled annotation layer, not a new
confirmation piece.

Deliberately a number, not narrative text (per the user's request): a
0-100 estimate is more useful for a correlation check against R-multiple
outcomes than a 0-10 scale (more resolution) or free text (not comparable
across trades at all).

Requires a local Ollama server (see docs/PLAN.md's live-lab notes) — not a
dependency of the lab itself, so nothing here is exercised by a live HTTP
call in the test suite; see tests/test_llm_score.py for the offline,
mocked-response tests.
"""
from __future__ import annotations

import json
import re
import urllib.request

import pandas as pd

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:7b-instruct"

_PROMPT_TEMPLATE = """You are looking at a sequence of 5-minute price candles (open, high, low, close) leading up to a possible trade entry on a higher timeframe. Estimate, from 0 to 100, how much this looks like a genuine, sustained bullish continuation about to happen versus noise or exhaustion. Respond with ONLY the integer, nothing else — no words, no explanation.

Candles (oldest to newest):
{candles}
"""


def format_candles(window: pd.DataFrame) -> str:
    """Plain numeric OHLC listing, oldest to newest — no chart, no
    indicators, just the same raw price data the deterministic engine
    itself works from."""
    lines = [f"O={row['open']:.4f} H={row['high']:.4f} L={row['low']:.4f} C={row['close']:.4f}" for _, row in window.iterrows()]
    return "\n".join(lines)


def parse_score(text: str) -> int | None:
    """Extracts the first integer in `text` and checks it's a valid 0-100
    score. Returns None (fails safe) rather than fabricating a
    plausible-looking number when the response doesn't parse."""
    match = re.search(r"-?\d+", text)
    if match is None:
        return None
    value = int(match.group())
    return value if 0 <= value <= 100 else None


def score_candles(
    window: pd.DataFrame,
    host: str = DEFAULT_HOST,
    model: str = DEFAULT_MODEL,
    timeout: float = 30.0,
) -> int | None:
    """Calls a local Ollama server with `window`'s candles, returns its
    0-100 score, or None if the server errored or the response wasn't
    parseable."""
    prompt = _PROMPT_TEMPLATE.format(candles=format_candles(window))
    payload = json.dumps(
        {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}}
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{host}/api/generate", data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read())
    except (OSError, json.JSONDecodeError):
        # OSError covers urllib.error.URLError and a plain connection
        # refused/timeout alike (the server isn't running, network hiccup,
        # etc.) - fails safe rather than raising into the caller's loop.
        return None
    return parse_score(body.get("response", ""))
