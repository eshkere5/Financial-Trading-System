import pytest

from finvex.web.adapters import OrderRequest, Registry, default_registry


def test_execution_disabled_by_default():
    reg = default_registry()
    assert reg.describe()["execution_enabled"] is False
    with pytest.raises(PermissionError):
        reg.submit(OrderRequest("SBER", "buy", 1))


def test_enabled_adapter_validates_order():
    class Fake:
        id, name, enabled = "fake", "Fake", True
        def place_order(self, o): return {"ok": True, "symbol": o.symbol}
        def positions(self): return []
    reg = Registry(execution=Fake())
    assert reg.submit(OrderRequest("SBER", "buy", 2))["ok"]
    with pytest.raises(ValueError):
        reg.submit(OrderRequest("SBER", "buy", 0))
    with pytest.raises(ValueError):
        reg.submit(OrderRequest("SBER", "sell", 1, "limit"))
