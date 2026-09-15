"""Tests for docker/run_forever.py's pure scheduling logic.

Imported as a bare top-level module (not `docker.run_forever`) — the
`docker/` directory holds deployment files, not a Python package, and a
package-style import risks colliding with the unrelated `docker` PyPI
package (the Docker SDK) if it's ever installed in this environment."""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docker"))

from run_forever import BUFFER_SECONDS, seconds_until_next_run


def test_seconds_until_next_run_shortly_after_the_hour():
    # 04:00:01 -> next target is THIS hour's 04:05:00, not next hour's.
    now = datetime(2026, 9, 15, 4, 0, 1, tzinfo=timezone.utc)

    wait = seconds_until_next_run(now)

    assert wait == BUFFER_SECONDS - 1


def test_seconds_until_next_run_right_before_the_target():
    now = datetime(2026, 9, 15, 4, 4, 59, tzinfo=timezone.utc)  # 1 second before 04:05:00

    wait = seconds_until_next_run(now)

    assert wait == 1


def test_seconds_until_next_run_exactly_on_the_hour():
    now = datetime(2026, 9, 15, 4, 0, 0, tzinfo=timezone.utc)

    wait = seconds_until_next_run(now)

    assert wait == BUFFER_SECONDS


def test_seconds_until_next_run_rolls_over_once_the_target_has_passed():
    # 1 second after this hour's 04:05:00 target -> next check is 05:05:00.
    now = datetime(2026, 9, 15, 4, 5, 1, tzinfo=timezone.utc)

    wait = seconds_until_next_run(now)

    assert wait == 3600 - 1
