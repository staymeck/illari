# Reporte de backtest — Laboratorio de trading

## Resumen comparativo

| escenario              |   n_trades |   win_rate_pct |   profit_factor |   expectancy |   max_drawdown_pct |   total_return_pct |   final_equity |
|:-----------------------|-----------:|---------------:|----------------:|-------------:|-------------------:|-------------------:|---------------:|
| tendencia_alcista_2023 |         18 |          50    |            1.19 |        12.35 |              -5.78 |               2.22 |       10222.3  |
| rango_lateral_2024     |          7 |          28.57 |            0.58 |       -33.48 |              -3.3  |              -2.34 |        9765.64 |

## Escenario: tendencia_alcista_2023

- Operaciones: **18** · Win rate: **50.0%** · Profit factor: **1.19** · Retorno total: **2.22%** · Drawdown máximo: **-5.78%**

[Ver gráfico de velas interactivo](chart_5m.html)

[Ver curva de equity interactiva](equity.html)

### Desglose por hora de entrada (UTC)

|   hour_utc |   n_trades |   win_rate_pct |   profit_factor |   avg_pnl |
|-----------:|-----------:|---------------:|----------------:|----------:|
|          1 |          1 |            100 |             inf |    176.25 |
|          2 |          1 |            100 |             inf |    167.57 |
|          3 |          2 |              0 |               0 |   -117.91 |
|          6 |          1 |              0 |               0 |   -127.34 |
|          7 |          2 |              0 |               0 |   -123.55 |
|          9 |          1 |              0 |               0 |   -131.62 |
|         13 |          1 |            100 |             inf |    136.57 |
|         14 |          1 |              0 |               0 |   -129.08 |
|         15 |          1 |            100 |             inf |    136.03 |
|         16 |          1 |              0 |               0 |   -139.86 |
|         17 |          1 |            100 |             inf |    131.58 |
|         18 |          1 |            100 |             inf |    149.03 |
|         20 |          1 |              0 |               0 |   -135.36 |
|         21 |          1 |            100 |             inf |    197.44 |
|         22 |          1 |            100 |             inf |    133.73 |
|         23 |          1 |            100 |             inf |    140.24 |

### Desglose por sesión de mercado

| session           |   n_trades |   win_rate_pct |   profit_factor |   avg_pnl |
|:------------------|-----------:|---------------:|----------------:|----------:|
| asia              |          7 |          28.57 |            0.56 |    -38.06 |
| new_york          |          4 |          50    |            1.02 |      1.35 |
| off_hours         |          3 |         100    |          inf    |    157.14 |
| overlap_london_ny |          3 |          66.67 |            2.11 |     47.84 |
| london            |          1 |           0    |            0    |   -131.62 |

## Escenario: rango_lateral_2024

- Operaciones: **7** · Win rate: **28.57%** · Profit factor: **0.58** · Retorno total: **-2.34%** · Drawdown máximo: **-3.3%**

[Ver gráfico de velas interactivo](chart_5m.html)

[Ver curva de equity interactiva](equity.html)

### Desglose por hora de entrada (UTC)

|   hour_utc |   n_trades |   win_rate_pct |   profit_factor |   avg_pnl |
|-----------:|-----------:|---------------:|----------------:|----------:|
|          2 |          1 |              0 |               0 |   -107.1  |
|          4 |          1 |              0 |               0 |   -113.26 |
|          7 |          1 |            100 |             inf |    163.43 |
|         14 |          2 |              0 |               0 |   -107.41 |
|         15 |          1 |              0 |               0 |   -121.17 |
|         17 |          1 |            100 |             inf |    158.57 |

### Desglose por sesión de mercado

| session           |   n_trades |   win_rate_pct |   profit_factor |   avg_pnl |
|:------------------|-----------:|---------------:|----------------:|----------:|
| asia              |          3 |          33.33 |            0.74 |    -18.98 |
| overlap_london_ny |          3 |           0    |            0    |   -112    |
| new_york          |          1 |         100    |          inf    |    158.57 |
