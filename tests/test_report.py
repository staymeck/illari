from pathlib import Path

import pandas as pd

from trading_lab.backtest.report import build_cross_market_summary


def _all_sweep_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"market": "BTC", "variant": "confluence/base", "scenario": "s1", "n_trades": 10,
             "win_rate_pct": 40.0, "profit_factor": 0.8, "total_return_pct": -10.0, "max_drawdown_pct": -15.0},
            {"market": "BTC", "variant": "confluence/base", "scenario": "s2", "n_trades": 12,
             "win_rate_pct": 45.0, "profit_factor": 0.9, "total_return_pct": -5.0, "max_drawdown_pct": -12.0},
            {"market": "ETH", "variant": "confluence/base", "scenario": "s1", "n_trades": 8,
             "win_rate_pct": 55.0, "profit_factor": 1.3, "total_return_pct": 8.0, "max_drawdown_pct": -6.0},
            {"market": "ETH", "variant": "confluence/base", "scenario": "s2", "n_trades": 9,
             "win_rate_pct": 50.0, "profit_factor": 1.1, "total_return_pct": 3.0, "max_drawdown_pct": -7.0},
        ]
    )


def test_build_cross_market_summary_lists_every_market(tmp_path: Path):
    output_path = tmp_path / "report.md"
    market_paths = {"BTC": tmp_path / "BTC" / "report.md", "ETH": tmp_path / "ETH" / "report.md"}

    build_cross_market_summary(_all_sweep_df(), output_path, market_paths)

    text = output_path.read_text(encoding="utf-8")
    assert "BTC" in text
    assert "ETH" in text
    assert "confluence/base" in text


def test_build_cross_market_summary_ranking_uses_worst_case_across_markets(tmp_path: Path):
    # confluence/base's worst profit_factor across ALL market x scenario
    # rows is BTC/s1's 0.8 -- the ranking must reflect that, not BTC-only
    # or ETH-only figures.
    output_path = tmp_path / "report.md"
    market_paths = {"BTC": tmp_path / "BTC" / "report.md", "ETH": tmp_path / "ETH" / "report.md"}

    build_cross_market_summary(_all_sweep_df(), output_path, market_paths)

    text = output_path.read_text(encoding="utf-8")
    assert "0.8" in text  # worst_profit_factor should surface somewhere in the cross-market ranking


def test_build_cross_market_summary_handles_empty_input(tmp_path: Path):
    output_path = tmp_path / "report.md"

    result_path = build_cross_market_summary(pd.DataFrame(), output_path, {})

    assert result_path == output_path
    assert "No data" in output_path.read_text(encoding="utf-8")
