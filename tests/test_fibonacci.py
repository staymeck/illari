import pytest

from trading_lab.indicators.fibonacci import compute_fib_levels, confluence_zone, price_in_zone


def test_retracement_up_move():
    # Swing de 100 a 200 (movimiento alcista): el retroceso 0.5 debe caer en 150.
    levels = compute_fib_levels(swing_low=100, swing_high=200, direction="up")
    assert levels.retracements[0.0] == 200
    assert levels.retracements[1.0] == 100
    assert levels.retracements[0.5] == 150
    assert levels.retracements[0.618] == pytest.approx(200 - 0.618 * 100)


def test_retracement_down_move():
    # Swing de 200 a 100 (movimiento bajista): el retroceso mide subiendo desde el low.
    levels = compute_fib_levels(swing_low=100, swing_high=200, direction="down")
    assert levels.retracements[0.0] == 100
    assert levels.retracements[1.0] == 200
    assert levels.retracements[0.5] == 150


def test_confluence_zone_is_between_ratios():
    levels = compute_fib_levels(swing_low=100, swing_high=200, direction="up")
    zone = confluence_zone(levels, 0.5, 0.618)
    assert zone[0] < zone[1]
    assert zone == (pytest.approx(200 - 0.618 * 100), pytest.approx(150))


def test_price_in_zone():
    zone = (140.0, 160.0)
    assert price_in_zone(150.0, zone)
    assert not price_in_zone(139.9, zone)
    assert not price_in_zone(160.1, zone)


def test_invalid_swing_raises():
    with pytest.raises(ValueError):
        compute_fib_levels(swing_low=200, swing_high=100, direction="up")


def test_invalid_direction_raises():
    with pytest.raises(ValueError):
        compute_fib_levels(swing_low=100, swing_high=200, direction="sideways")
