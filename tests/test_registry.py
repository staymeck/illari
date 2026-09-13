"""Tests for src/strategies/registry.py."""
import pytest

from src.strategies.registry import Registry


def test_register_and_get():
    reg = Registry("test")

    @reg.register("piece_a")
    def piece_a():
        return "a"

    assert reg.get("piece_a") is piece_a
    assert reg.get("piece_a")() == "a"


def test_register_duplicate_name_raises():
    reg = Registry("test")

    @reg.register("dup")
    def first():
        pass

    with pytest.raises(ValueError, match="already registered"):
        @reg.register("dup")
        def second():
            pass


def test_get_unknown_name_raises_with_available_list():
    reg = Registry("test")

    @reg.register("known")
    def known():
        pass

    with pytest.raises(KeyError, match="Unknown test piece 'missing'"):
        reg.get("missing")


def test_names_returns_sorted_list():
    reg = Registry("test")
    reg.register("zeta")(lambda: None)
    reg.register("alpha")(lambda: None)

    assert reg.names() == ["alpha", "zeta"]
