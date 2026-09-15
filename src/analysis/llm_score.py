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

v2, after the first version's result (a null correlation, and a
distribution clustered into ~14 repeated values across 125 trades)
prompted a redesign per the user's own diagnosis: a bare "give me a
number" prompt with only 24 candles and no framing didn't give the model
enough to differentiate on. Now: a trader-persona prompt, the same
analytical context our own deterministic pieces already computed for that
setup (pattern, confirmed trend, Fibonacci ratio, volume bias — not
raw candles in a vacuum), and a much longer lookback (300 candles,
matching the engine's own `lookback_bars` convention).

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
DEFAULT_NUM_CTX = 8192  # 300 candles + context + instructions comfortably exceeds Ollama's 4096 default

_PROMPT_TEMPLATE = """You are an experienced technical trader evaluating a potential LONG entry in a cryptocurrency market. A rule-based system has already identified this setup using the following analysis:

{context}

Below are the {n_candles} most recent 5-minute candles (open, high, low, close) leading up to this entry, oldest to newest:

{candles}

Rate how recommendable this entry looks, from 0 (not recommendable at all) to 100 (highly recommendable), considering both the analysis above and the recent price action. Respond with ONLY the integer, nothing else — no words, no explanation.
"""

_CONTEXT_LABELS = {
    "pattern": "Candlestick pattern that triggered the setup",
    "context_trend": "Confirmed trend (execution timeframe)",
    "higher_tf_trend": "Confirmed trend (higher timeframe)",
    "fib_ratio": "Fibonacci retracement ratio at entry",
    "volume_bias": "Volume bias (positive = buyer-dominant)",
}


def format_context(context: dict | None) -> str:
    """Plain-text listing of whichever of our own already-computed
    analysis fields are available for this trade — the same information
    the deterministic engine itself used to take the setup, not a summary
    written for the occasion."""
    if not context:
        return "(no additional analysis fields available)"
    lines = [f"- {label}: {context[key]}" for key, label in _CONTEXT_LABELS.items() if context.get(key) not in (None, "")]
    return "\n".join(lines) if lines else "(no additional analysis fields available)"


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
    context: dict | None = None,
    host: str = DEFAULT_HOST,
    model: str = DEFAULT_MODEL,
    num_ctx: int = DEFAULT_NUM_CTX,
    timeout: float = 60.0,
) -> int | None:
    """Calls a local Ollama server with `window`'s candles plus whatever
    `context` fields are available (see format_context), returns its
    0-100 score, or None if the server errored or the response wasn't
    parseable."""
    prompt = _PROMPT_TEMPLATE.format(
        context=format_context(context), n_candles=len(window), candles=format_candles(window)
    )
    payload = json.dumps(
        {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0, "num_ctx": num_ctx}}
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
