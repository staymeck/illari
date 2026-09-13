"""Central place for where lab reports live on disk.

`./reports` holds generated output only (never source-controlled — see
.gitignore); this module is the single source of truth for its layout so the
rest of the report core doesn't hardcode paths.
"""
from __future__ import annotations

import re
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


def _slug(text: str) -> str:
    """Filesystem-safe slug: lowercase, non-alphanumerics collapsed to '-'."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "unnamed"


def run_report_path(
    symbol: str, timeframe: str, scenario: str = "default", reports_dir: Path = REPORTS_DIR
) -> Path:
    """Standard path for a single run's report: reports/<symbol>_<timeframe>_<scenario>.md"""
    filename = f"{_slug(symbol)}_{_slug(timeframe)}_{_slug(scenario)}.md"
    return reports_dir / filename


def summary_report_path(reports_dir: Path = REPORTS_DIR) -> Path:
    """Path for the cross-run comparison report."""
    return reports_dir / "summary.md"
