"""Lab CLI. Usage:

    python -m trading_lab.cli backtest [--config config/settings.yaml]
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import pandas as pd
import yaml

from trading_lab.analysis.probability import build_probability_table, signal_hit_probability
from trading_lab.backtest import metrics, report
from trading_lab.backtest import sweep as sweep_module
from trading_lab.backtest.engine import BacktestResult
from trading_lab.data.cache import get_ohlcv
from trading_lab.strategy import breakout_strategy, confluence_strategy
from trading_lab.strategy.base import Signal
from trading_lab.strategy.registry import get_strategy_class

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "settings.yaml"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"

# Extra history to download before the scenario's start, so structure (1h)
# and bias (1D) already have enough candles by the first day.
STRUCTURE_BUFFER_DAYS = 45
BIAS_BUFFER_DAYS = 400

ScenarioData = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]  # entry_df, structure_df, bias_df


def _parse_ts(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_scenario_window(scenario: dict, market: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    """A market can override a scenario's start/end date
    (`market["scenario_overrides"][scenario_name]`) for assets without
    full history on Binance for that period (e.g. PAXG, listed 2020-08-28,
    doesn't cover mixed_volatility_2019_2022's default 2019-07-01 start) —
    runs with its own real, shorter window instead of requesting data that
    doesn't exist."""
    override = (market.get("scenario_overrides") or {}).get(scenario["name"], {})
    start = _parse_ts(override.get("start", scenario["start"]))
    end = _parse_ts(override.get("end", scenario["end"]))
    return start, end


def fetch_scenario_data(cfg: dict, scenario: dict, market: dict) -> ScenarioData:
    """Downloads (or reads from cache) the candles for the 3 timeframes for
    a scenario on one market/pair, with enough extra history before the
    start so structure/bias already have confirmed swings from day one."""
    exchange_id, pair, tf = cfg["exchange"], market["pair"], cfg["timeframes"]
    name = scenario["name"]
    start, end = resolve_scenario_window(scenario, market)
    start, end = start.to_pydatetime(), end.to_pydatetime()

    print(f"[{market['name']}][{name}] downloading data ({pair}, {start.date()} -> {end.date()})...")

    entry_df = get_ohlcv(exchange_id, pair, tf["entry"], start, end, f"{name}_entry")
    structure_df = get_ohlcv(
        exchange_id, pair, tf["structure"], start - timedelta(days=STRUCTURE_BUFFER_DAYS), end, f"{name}_structure"
    )
    bias_df = get_ohlcv(exchange_id, pair, tf["bias"], start - timedelta(days=BIAS_BUFFER_DAYS), end, f"{name}_bias")

    if entry_df.empty or structure_df.empty or bias_df.empty:
        raise RuntimeError(f"[{market['name']}][{name}] not enough data was obtained — check the date range and connection.")

    print(f"[{market['name']}][{name}] candles: entry={len(entry_df)} structure={len(structure_df)} bias={len(bias_df)}")
    return entry_df, structure_df, bias_df


def _prepare_chart_and_probability_inputs(
    strategy_class: str,
    entry_df: pd.DataFrame,
    structure_df: pd.DataFrame,
    bias_df: pd.DataFrame,
    window_start: pd.Timestamp,
    cfg: dict,
    variant_params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (entry_for_chart, structure_for_chart, merged) for one
    concrete variant of a strategy, based on its `strategy_class`. Each
    strategy computes different columns, so it builds these inputs its own
    way — but the output (candlestick + probability table) is generated
    the same way for any of them."""
    timeframes = cfg["timeframes"]

    if strategy_class == "confluence":
        entry_for_chart = confluence_strategy.prepare_entry_df(
            entry_df, variant_params["volume_lookback"], variant_params["volume_min_ratio"], variant_params["atr_period"],
            variant_params.get("atr_expansion_lookback", 20),
        )
        structure_for_chart = confluence_strategy.prepare_structure_df(
            structure_df, variant_params["swing_lookback"], variant_params["fib_zone_min"], variant_params["fib_zone_max"],
            variant_params.get("trendline_min_points", 3), variant_params.get("trendline_max_points", 6),
        )
        structure_for_chart = structure_for_chart[structure_for_chart["timestamp"] >= window_start]
        merged = confluence_strategy.prepare_merged(
            entry_df, structure_df, bias_df, {"timeframes": timeframes, "strategy": variant_params}
        )
        return entry_for_chart, structure_for_chart, merged

    if strategy_class == "breakout":
        entry_for_chart = breakout_strategy.prepare_entry_df(
            entry_df, variant_params.get("donchian_period_bars", 96), variant_params["atr_period"],
            variant_params["volume_lookback"], variant_params.get("atr_expansion_lookback", 20),
        )
        # Reuses the chart's "zone" mechanism (designed for the Fibonacci
        # zone) to draw the Donchian channel.
        structure_for_chart = entry_for_chart[["timestamp"]].copy()
        structure_for_chart["zone_lo"] = entry_for_chart["donchian_low"]
        structure_for_chart["zone_hi"] = entry_for_chart["donchian_high"]
        merged = breakout_strategy.prepare_merged(entry_df, bias_df, {"timeframes": timeframes, "strategy": variant_params})
        return entry_for_chart, structure_for_chart, merged

    raise ValueError(f"Unknown strategy_class: '{strategy_class}'")


def _run_market_backtest(cfg: dict, market: dict, reports_dir: Path) -> tuple[pd.DataFrame, Path]:
    """Runs every strategy × variant × scenario for ONE market, writes that
    market's own detailed report (unchanged shape from the single-market
    version: metrics, hour/session breakdown, probability tables, charts),
    and returns its sweep rows (tagged with `market`) plus the report path,
    so `cmd_backtest` can build a cross-market summary on top."""
    from trading_lab.viz.charts import build_equity_curve_chart, build_scenario_chart

    market_name = market["name"]
    market_reports_dir = reports_dir / market_name

    data_by_scenario: dict[str, ScenarioData] = {
        scenario["name"]: fetch_scenario_data(cfg, scenario, market) for scenario in cfg["scenarios"]
    }
    window_start_by_scenario = {
        scenario["name"]: resolve_scenario_window(scenario, market)[0] for scenario in cfg["scenarios"]
    }

    sweep_rows: list[dict] = []
    # Any variant marked `detailed: true` in settings.yaml (not just
    # "base") gets a chart + probability table, indexed by
    # display_name = "<strategy>/<variant>".
    detailed_results: dict[str, dict[str, BacktestResult]] = {}
    detailed_signals: dict[str, dict[str, list[Signal]]] = {}
    detailed_meta: dict[str, dict] = {}  # display_name -> {"strategy_class":.., "params":..}

    for strategy_cfg in cfg["strategies"]:
        strategy_name = strategy_cfg["name"]
        strategy_cls = get_strategy_class(strategy_cfg["strategy_class"])
        base_params = strategy_cfg.get("base_params", {})
        variants = strategy_cfg.get("variants") or [{"name": "base", "overrides": {}}]

        for variant in variants:
            variant_name = variant["name"]
            variant_params = sweep_module.apply_overrides(base_params, variant.get("overrides", {}))
            display_name = f"{strategy_name}/{variant_name}"
            is_detailed = bool(variant.get("detailed", False))

            for scenario_name, (entry_df, structure_df, bias_df) in data_by_scenario.items():
                signals, result = sweep_module.run_variant(
                    strategy_cls, variant_params, cfg["timeframes"], cfg["initial_capital"], entry_df, structure_df, bias_df
                )
                summary = metrics.summarize(result, cfg["initial_capital"])
                sweep_rows.append({"variant": display_name, "scenario": scenario_name, **summary})
                print(
                    f"[{market_name}][{scenario_name}][{display_name}] trades={summary['n_trades']} "
                    f"win_rate={summary['win_rate_pct']}% pf={summary['profit_factor']} "
                    f"return={summary['total_return_pct']}%"
                )
                if is_detailed:
                    detailed_results.setdefault(display_name, {})[scenario_name] = result
                    detailed_signals.setdefault(display_name, {})[scenario_name] = signals

            if is_detailed:
                detailed_meta[display_name] = {"strategy_class": strategy_cfg["strategy_class"], "params": variant_params}

    sweep_df = pd.DataFrame(sweep_rows)

    chart_paths: dict[str, dict[str, dict[str, Path]]] = {}
    probability_tables: dict[str, dict[str, pd.DataFrame]] = {}
    signal_hit_tables: dict[str, dict[str, pd.DataFrame]] = {}

    for display_name, meta in detailed_meta.items():
        variant_params = meta["params"]
        chart_paths[display_name] = {}
        probability_tables[display_name] = {}
        signal_hit_tables[display_name] = {}

        for scenario in cfg["scenarios"]:
            name = scenario["name"]
            entry_df, structure_df, bias_df = data_by_scenario[name]

            entry_for_chart, structure_for_chart, merged = _prepare_chart_and_probability_inputs(
                meta["strategy_class"], entry_df, structure_df, bias_df, window_start_by_scenario[name], cfg, variant_params
            )

            scenario_dir = market_reports_dir / name / display_name.replace("/", "_")
            trades_df = metrics.trades_to_df(detailed_results[display_name][name].trades)
            candles_path = build_scenario_chart(
                entry_for_chart, structure_for_chart, trades_df, f"{market_name} — {name} — {display_name}", market["pair"],
                scenario_dir / "chart_5m.html",
            )
            equity_path = build_equity_curve_chart(
                detailed_results[display_name][name].equity_curve,
                f"{market_name} — {name} — {display_name}", scenario_dir / "equity.html",
            )
            chart_paths[display_name][name] = {"candles": candles_path, "equity": equity_path}

            horizon = variant_params["probability_horizon_bars"]
            probability_tables[display_name][name] = build_probability_table(
                merged, horizon, variant_params.get("trendline_proximity_atr_mult", 1.0)
            )
            signal_hit_tables[display_name][name] = signal_hit_probability(
                detailed_signals[display_name][name], entry_for_chart, horizon
            )

            print(f"[{market_name}][{name}][{display_name}] probability computed, charts generated.")

    report_path = report.build_report(
        detailed_results,
        cfg["initial_capital"],
        chart_paths,
        market_reports_dir / "report.md",
        probability_tables=probability_tables,
        signal_hit_tables=signal_hit_tables,
        sweep_df=sweep_df,
        title=f"Backtest report — {market_name} ({market['pair']})",
    )
    print(f"[{market_name}] report generated: {report_path}")

    sweep_df.insert(0, "market", market_name)
    return sweep_df, report_path


def cmd_backtest(args: argparse.Namespace) -> None:
    cfg = load_config(Path(args.config))
    reports_dir = Path(args.reports_dir)

    all_sweep_rows: list[pd.DataFrame] = []
    market_report_paths: dict[str, Path] = {}

    for market in cfg["markets"]:
        print(f"\n=== Market: {market['name']} ({market['pair']}) ===")
        market_sweep_df, market_report_path = _run_market_backtest(cfg, market, reports_dir)
        all_sweep_rows.append(market_sweep_df)
        market_report_paths[market["name"]] = market_report_path

    all_sweep_df = pd.concat(all_sweep_rows, ignore_index=True) if all_sweep_rows else pd.DataFrame()
    summary_path = report.build_cross_market_summary(all_sweep_df, reports_dir / "report.md", market_report_paths)
    print(f"\nCross-market summary generated: {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trading_lab", description="Trading strategy backtesting lab")
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="Runs the backtest (all strategies/variants) over the scenarios defined in config/settings.yaml")
    bt.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to the settings.yaml file")
    bt.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR), help="Output directory for reports")
    bt.set_defaults(func=cmd_backtest)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
