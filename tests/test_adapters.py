"""Adapter SDK contract tests — every adapter must satisfy these."""
from __future__ import annotations

import pytest

from aoi_sentinel.adapters import (
    VendorAdapter,
    available_adapters,
    make_adapter,
)


def test_registry_has_all_three():
    names = available_adapters()
    assert "saki" in names
    assert "koh_young" in names
    assert "generic_csv" in names


@pytest.mark.parametrize("name", ["saki", "koh_young", "generic_csv"])
def test_adapter_constructs_and_satisfies_protocol(name):
    a = make_adapter(name)
    assert isinstance(a, VendorAdapter)
    assert a.name == name


def test_unknown_adapter_raises():
    with pytest.raises(KeyError):
        make_adapter("does_not_exist")
