"""Motor de backtest mínimo, para validar el pipeline completo
(datos -> señal -> simulación -> métricas) antes de escalar a 5 mercados x 3
timeframes con todos los componentes del motor de análisis.

Estrategia usada en esta validación (deliberadamente simple, solo con
estructura de mercado — sin Fibonacci/velas/volumen todavía, ver docs/PLAN.md
"Próximos pasos inmediatos"): comprar cerca de un soporte confirmado mientras
la tendencia (Dow) es alcista, con stop bajo el soporte y objetivo a un
múltiplo R del riesgo.

Importante sobre look-ahead bias: en cada paso `t` solo se le pasa a
`estructura.py` la porción de la serie `df.iloc[:t+1]` (nada del futuro). Los
pivotes de `find_swing_points` usan una ventana centrada, así que los últimos
`order` bares de cualquier slice quedan automáticamente sin confirmar
(NaN -> False) hasta que existan suficientes barras "futuras" dentro del propio
slice — es decir, la función ya es segura contra look-ahead siempre que se la
alimente con datos hasta `t`, sin adelantar nada.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config.markets import session_for_hour
from src.analysis.estructura import classify_trend, support_resistance_levels


@dataclass(frozen=True)
class BacktestConfig:
    lookback_bars: int = 200          # ventana de historia para calcular estructura
    swing_order: int = 3              # ver estructura.find_swing_points
    support_tolerance_pct: float = 1.0  # qué tan "cerca" del soporte cuenta como toque
    stop_pct_below_support: float = 0.5  # stop = soporte * (1 - este%)
    reward_risk_ratio: float = 2.0    # objetivo = entrada + R * riesgo
    max_holding_bars: int = 24        # cierre forzado si no toca stop/target antes
    fee_pct: float = 0.1              # comisión de Binance por lado (~0.1% spot)
    initial_equity: float = 10_000.0


def _find_entry_signal(history: pd.DataFrame, cfg: BacktestConfig) -> float | None:
    """Devuelve el nivel de soporte que gatilla la entrada, o None si no hay señal,
    evaluando solo con datos hasta la última fila de `history` (sin futuro)."""
    if len(history) < cfg.lookback_bars:
        return None

    trend = classify_trend(history, order=cfg.swing_order)
    if trend != "alcista":
        return None

    levels = support_resistance_levels(history, order=cfg.swing_order)
    supports = levels["support"]
    if not supports:
        return None

    last_close = history["close"].iloc[-1]
    for level in supports:
        distance_pct = abs(last_close - level) / level * 100
        if last_close >= level and distance_pct <= cfg.support_tolerance_pct:
            return level
    return None


def run_backtest(df: pd.DataFrame, cfg: BacktestConfig | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Simula la estrategia sobre `df` (columnas: timestamp, open, high, low,
    close, volume) y devuelve (trades, equity_curve)."""
    cfg = cfg or BacktestConfig()
    df = df.reset_index(drop=True)

    trades: list[dict] = []
    equity = cfg.initial_equity
    equity_curve: list[float] = [equity]

    i = cfg.lookback_bars
    n = len(df)
    while i < n - 1:
        history = df.iloc[: i + 1]
        support = _find_entry_signal(history, cfg)

        if support is None:
            equity_curve.append(equity)
            i += 1
            continue

        entry_idx = i + 1  # se entra a la apertura de la próxima vela (sin lookahead)
        if entry_idx >= n:
            break
        entry_price = df["open"].iloc[entry_idx]
        stop_price = support * (1 - cfg.stop_pct_below_support / 100)
        risk = entry_price - stop_price
        if risk <= 0:
            equity_curve.append(equity)
            i += 1
            continue
        target_price = entry_price + cfg.reward_risk_ratio * risk

        exit_idx = None
        exit_price = None
        exit_reason = "timeout"
        last_possible = min(entry_idx + cfg.max_holding_bars, n - 1)
        for j in range(entry_idx, last_possible + 1):
            bar = df.iloc[j]
            if bar["low"] <= stop_price:
                exit_idx, exit_price, exit_reason = j, stop_price, "stop"
                break
            if bar["high"] >= target_price:
                exit_idx, exit_price, exit_reason = j, target_price, "target"
                break
        if exit_idx is None:
            exit_idx = last_possible
            exit_price = df["close"].iloc[exit_idx]

        gross_pnl_pct = (exit_price - entry_price) / entry_price * 100
        net_pnl_pct = gross_pnl_pct - 2 * cfg.fee_pct  # comisión de entrada + salida
        trade_equity_before = equity
        equity *= 1 + net_pnl_pct / 100
        pnl_abs = equity - trade_equity_before

        entry_ts = df["timestamp"].iloc[entry_idx]
        trades.append(
            {
                "entry_time": entry_ts,
                "exit_time": df["timestamp"].iloc[exit_idx],
                "hour_utc": entry_ts.hour,
                "session": session_for_hour(entry_ts.hour),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "pnl_pct": net_pnl_pct,
                "pnl_abs": pnl_abs,
                "equity_after": equity,
            }
        )
        equity_curve.append(equity)

        i = exit_idx + 1  # sin solapar operaciones

    trades_df = pd.DataFrame(trades)
    equity_series = pd.Series(equity_curve, name="equity")
    return trades_df, equity_series


def compute_metrics(trades: pd.DataFrame, equity_curve: pd.Series, initial_equity: float) -> dict:
    """Métricas estándar del laboratorio, mismo formato que resources/report.md."""
    if trades.empty:
        return {
            "n_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "max_drawdown_pct": 0.0,
            "total_return_pct": 0.0,
            "final_equity": initial_equity,
        }

    wins = trades[trades["pnl_abs"] > 0]
    losses = trades[trades["pnl_abs"] <= 0]
    gross_win = wins["pnl_abs"].sum()
    gross_loss = -losses["pnl_abs"].sum()

    running_max = equity_curve.cummax()
    drawdown_pct = (equity_curve - running_max) / running_max * 100
    max_drawdown_pct = drawdown_pct.min()

    final_equity = equity_curve.iloc[-1]
    total_return_pct = (final_equity - initial_equity) / initial_equity * 100

    return {
        "n_trades": len(trades),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 2),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "expectancy": round(trades["pnl_abs"].mean(), 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "total_return_pct": round(total_return_pct, 2),
        "final_equity": round(final_equity, 2),
    }
