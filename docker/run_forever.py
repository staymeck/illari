"""Entry point for the live paper-trading lab's Docker container: runs
scripts/run_live_check.py once per hour, aligned to the hour boundary plus
a small buffer (so Binance has definitely published the candle), forever.

No cron daemon inside the container — a plain sleep loop is simpler and
more robust for a single-purpose container than getting cron to behave as
PID 1 (signal handling, logging, etc.). Runs on the machine's own network,
so it doesn't hit the egress restrictions a cloud sandbox does (see
docs/PLAN.md / Bitácora Illari — that path was tried and is a dead end for
this project specifically because it needs to reach api.binance.com).
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import sleep

BUFFER_SECONDS = 5 * 60  # run 5 minutes after each hour closes
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_live_check.py"


def seconds_until_next_run(now: datetime | None = None) -> float:
    """Seconds from `now` (UTC) until the next check, which lands
    BUFFER_SECONDS after an hour boundary. A pure function of `now` so
    it's testable without mocking real time.

    This hour's own target (:00 + BUFFER_SECONDS) is used directly if it's
    still ahead of `now` — only rolls over to the following hour once that
    target has already passed. A naive "always add a full hour" version
    would wait up to an extra hour on top of the correct answer whenever
    called before the current hour's target time."""
    now = now or datetime.now(timezone.utc)
    target = now.replace(minute=0, second=0, microsecond=0) + timedelta(seconds=BUFFER_SECONDS)
    if target <= now:
        target += timedelta(hours=1)
    return (target - now).total_seconds()


def main() -> None:
    while True:
        wait = seconds_until_next_run()
        print(f"[run_forever] sleeping {wait:.0f}s until next check ({datetime.now(timezone.utc).isoformat()})", flush=True)
        sleep(wait)
        print(f"[run_forever] running check at {datetime.now(timezone.utc).isoformat()}", flush=True)
        result = subprocess.run([sys.executable, str(SCRIPT_PATH)])
        if result.returncode != 0:
            print(f"[run_forever] check exited with code {result.returncode}", flush=True)


if __name__ == "__main__":
    main()
