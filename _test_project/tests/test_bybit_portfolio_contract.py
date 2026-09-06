from src.executors.bybit_client import BybitClient


class FakeBybitClient(BybitClient):
    def __init__(self, category: str):
        self.category = category

    def get_wallet_balance(self):
        return {
            "result": {
                "list": [
                    {
                        "totalEquity": "1018.20",
                        "totalWalletBalance": "778.44",
                        "totalPerpUPL": "2.97",
                        "totalAvailableBalance": "776.42",
                        "totalInitialMargin": "4.98",
                    }
                ]
            }
        }

    def _call(self, fn, *, write: bool, **kwargs):
        return {
            "result": {
                "list": [
                    {
                        "symbol": "BTCUSDT",
                        "size": "0.001",
                        "side": "Buy",
                        "avgPrice": "77000",
                        "unrealisedPnl": "1.25",
                    },
                    {
                        "symbol": "ETHUSDT",
                        "size": "0.02",
                        "side": "Sell",
                        "avgPrice": "2400",
                        "unrealisedPnl": "-0.30",
                    },
                    {
                        "symbol": "XRPUSDT",
                        "size": "0",
                        "side": "Buy",
                        "avgPrice": "1.0",
                        "unrealisedPnl": "0",
                    },
                ]
            }
        }

    class session:
        @staticmethod
        def get_positions(**kwargs):
            return {}


def test_linear_portfolio_uses_broker_equity():
    client = FakeBybitClient(category="linear")

    portfolio = client.get_portfolio()

    assert portfolio["accounting_mode"] == "broker_equity"
    assert portfolio["broker"] == "bybit"
    assert portfolio["broker_equity"] == 1018.20
    assert portfolio["cash"] == 1018.20
    assert portfolio["wallet_balance"] == 778.44
    assert portfolio["unrealized_pnl"] == 2.97
    assert portfolio["positions_count"] == 2

    positions = {
        item["symbol"]: item
        for item in portfolio["positions"]
    }

    assert positions["BTCUSDT"]["qty"] == 0.001
    assert positions["ETHUSDT"]["qty"] == -0.02


def test_spot_portfolio_has_no_broker_equity_override():
    client = FakeBybitClient(category="spot")

    portfolio = client.get_portfolio()

    assert portfolio["accounting_mode"] == "spot"
    assert portfolio["broker_equity"] is None
    assert portfolio["cash"] == 1018.20
    assert portfolio["positions"] == []
