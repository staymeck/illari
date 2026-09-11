import pandas as pd

from trading_lab.data.mtf_align import align_higher_timeframe, align_multi_timeframe


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def test_align_higher_timeframe_never_uses_unclosed_candle():
    entry_df = pd.DataFrame(
        {
            "timestamp": [
                _ts("2024-01-01T00:55:00"),
                _ts("2024-01-01T01:00:00"),
                _ts("2024-01-01T01:05:00"),
                _ts("2024-01-01T01:55:00"),
            ],
            "close": [1, 2, 3, 4],
        }
    )
    structure_df = pd.DataFrame(
        {
            "timestamp": [_ts("2024-01-01T00:00:00"), _ts("2024-01-01T01:00:00")],
            "high": [100.0, 200.0],
        }
    )

    merged = align_higher_timeframe(entry_df, "5m", structure_df, "1h", prefix="struct")
    by_ts = merged.set_index("timestamp")["struct_high"]

    # 00:55 (closes 01:00): only the 00:00-01:00 candle (closes exactly at 01:00) is available.
    assert by_ts[_ts("2024-01-01T00:55:00")] == 100.0

    # 01:00 (closes 01:05): the candle that OPENED at 01:00 only closes at
    # 02:00 -> would be looking ahead if used. Should still see the 00:00 candle.
    assert by_ts[_ts("2024-01-01T01:00:00")] == 100.0

    # 01:05: same situation, the 01:00 candle still hasn't closed.
    assert by_ts[_ts("2024-01-01T01:05:00")] == 100.0

    # 01:55 (closes 02:00): only now is the 01:00 candle (closes at 02:00) available.
    assert by_ts[_ts("2024-01-01T01:55:00")] == 200.0


def test_align_multi_timeframe_combines_structure_and_bias():
    entry_df = pd.DataFrame({"timestamp": [_ts("2024-01-02T00:30:00")], "close": [1]})
    structure_df = pd.DataFrame({"timestamp": [_ts("2024-01-01T00:00:00")], "high": [100.0]})
    bias_df = pd.DataFrame({"timestamp": [_ts("2024-01-01T00:00:00")], "high": [1000.0]})

    merged = align_multi_timeframe(entry_df, "5m", structure_df, "1h", bias_df, "1d")

    assert merged.loc[0, "struct_high"] == 100.0
    assert merged.loc[0, "bias_high"] == 1000.0


def test_align_multi_timeframe_no_match_before_any_higher_candle_closes():
    entry_df = pd.DataFrame({"timestamp": [_ts("2024-01-01T00:10:00")], "close": [1]})
    structure_df = pd.DataFrame({"timestamp": [_ts("2024-01-01T00:00:00")], "high": [100.0]})
    bias_df = pd.DataFrame({"timestamp": [_ts("2024-01-01T00:00:00")], "high": [1000.0]})

    merged = align_multi_timeframe(entry_df, "5m", structure_df, "1h", bias_df, "1d")

    # The 1h candle (closes 01:00) and the 1D one (closes the next day) still
    # haven't closed by 00:10 -> there should be no match (NaN), never a "future" value.
    assert pd.isna(merged.loc[0, "struct_high"])
    assert pd.isna(merged.loc[0, "bias_high"])
