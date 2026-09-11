import pandas as pd
import pytest

from trading_lab.indicators.session import add_session_columns, classify_session


@pytest.mark.parametrize(
    "hour,expected",
    [
        (0, "asia"),
        (7, "asia"),
        (8, "london"),
        (12, "london"),
        (13, "overlap_london_ny"),
        (15, "overlap_london_ny"),
        (16, "new_york"),
        (20, "new_york"),
        (21, "off_hours"),
        (23, "off_hours"),
    ],
)
def test_classify_session(hour, expected):
    assert classify_session(hour) == expected


def test_classify_session_invalid_hour_raises():
    with pytest.raises(ValueError):
        classify_session(24)
    with pytest.raises(ValueError):
        classify_session(-1)


def test_add_session_columns_utc():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01T00:00:00Z", "2024-01-01T14:00:00Z", "2024-01-01T21:30:00Z"]
            )
        }
    )
    out = add_session_columns(df)
    assert list(out["hour_utc"]) == [0, 14, 21]
    assert list(out["session"]) == ["asia", "overlap_london_ny", "off_hours"]


def test_add_session_columns_converts_non_utc_timezone():
    # 09:00 in UTC-3 is equivalent to 12:00 UTC -> London session.
    ts = pd.Timestamp("2024-01-01T09:00:00").tz_localize("America/Argentina/Buenos_Aires")
    df = pd.DataFrame({"timestamp": [ts]})
    out = add_session_columns(df)
    assert out.loc[0, "hour_utc"] == 12
    assert out.loc[0, "session"] == "london"
