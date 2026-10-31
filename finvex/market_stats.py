"""Статистика для чат-агента: тренды по списку бумаг, кластеры по корреляции, коинтеграция пар.

Только расчёты на дневных свечах (колонки close/high/low). Без сети, ничего не торгует. Результаты описательные.
"""
from __future__ import annotations

import itertools
import math
from typing import Optional

import numpy as np
import pandas as pd

MIN_BARS = 60


def _adx(df: pd.DataFrame, n: int = 14) -> Optional[float]:
    if len(df) < n * 3:
        return None
    h, l, c = (df[k].astype(float).reset_index(drop=True) for k in ("high", "low", "close"))
    up, dn = h.diff(), -l.diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0))
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0))
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    a = 1.0 / n
    atr = tr.ewm(alpha=a, adjust=False).mean()
    pdi, mdi = 100 * plus.ewm(alpha=a, adjust=False).mean() / atr, 100 * minus.ewm(alpha=a, adjust=False).mean() / atr
    v = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)).ewm(alpha=a, adjust=False).mean().iloc[-1]
    return None if pd.isna(v) else float(v)


def _ret(close: pd.Series, n: int) -> Optional[float]:
    return float(close.iloc[-1] / close.iloc[-1 - n] - 1) if len(close) > n else None


def trend_snapshot(df: pd.DataFrame, symbol: str) -> dict:
    if df is None or len(df) < MIN_BARS:
        raise ValueError(f"мало данных: {0 if df is None else len(df)} свечей, нужно от {MIN_BARS}")
    close = df["close"].astype(float).reset_index(drop=True)
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean() if len(close) >= 200 else pd.Series([np.nan] * len(close))
    last, adx = float(close.iloc[-1]), _adx(df)
    s50, s200 = sma50.iloc[-1], sma200.iloc[-1]
    a50 = bool(last > s50) if not pd.isna(s50) else None
    a200 = bool(last > s200) if not pd.isna(s200) else None
    slope = float(sma50.iloc[-1] / sma50.iloc[-11] - 1) if len(close) >= 61 and sma50.iloc[-11] else None
    lr = np.log(close).diff().dropna()
    trending = adx is not None and adx >= 20
    if trending and a50 and a200 is not False and (slope or 0) > 0:
        regime = "бычий тренд"
    elif trending and a50 is False and a200 is not True and (slope or 0) < 0:
        regime = "медвежий тренд"
    else:
        regime = "боковик или переход"
    return {"symbol": symbol, "last": last, "bars": int(len(close)), "ret_5d": _ret(close, 5), "ret_1m": _ret(close, 21),
            "ret_3m": _ret(close, 63), "vol_ann": float(lr.tail(63).std() * math.sqrt(252)) if len(lr) >= 20 else None,
            "dd_from_high": float(last / close.tail(252).max() - 1), "adx": adx,
            "sma50": None if pd.isna(s50) else float(s50), "sma200": None if pd.isna(s200) else float(s200),
            "sma50_slope_10d": slope, "above_sma50": a50, "above_sma200": a200, "regime": regime,
            "is_bull_regime": regime == "бычий тренд"}


def rank_trends(rows: "list[dict]") -> pd.DataFrame:
    """bull_score 0-100: выше SMA50/200, наклон SMA50, ранги доходности 1м и 3м."""
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    pct = lambda s: s.astype(float).rank(pct=True).fillna(0.5)  # noqa: E731
    flag = lambda s: s.map(lambda x: 1.0 if x is True else (0.0 if x is False else 0.5))  # noqa: E731
    up = d["sma50_slope_10d"].map(lambda x: 0.5 if x is None or pd.isna(x) else float(x > 0))
    d["bull_score"] = (100 * (0.25 * flag(d["above_sma50"]) + 0.25 * flag(d["above_sma200"]) + 0.20 * up
                              + 0.15 * pct(d["ret_1m"]) + 0.15 * pct(d["ret_3m"]))).round(1)
    d["rel_3m_vs_median"] = d["ret_3m"].astype(float) - d["ret_3m"].astype(float).median()
    return d.sort_values("bull_score", ascending=False).reset_index(drop=True)


def cluster_by_corr(returns: pd.DataFrame, threshold: float = 0.5) -> "list[list[str]]":
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    r = returns.dropna(axis=1, how="any")
    if r.shape[1] < 2 or len(r) < 30:
        return []
    dist = np.sqrt(0.5 * (1 - r.corr().clip(-1, 1).values))
    np.fill_diagonal(dist, 0)
    labels = fcluster(linkage(squareform(dist, checks=False), method="average"), t=math.sqrt(0.5 * (1 - threshold)), criterion="distance")
    groups: dict = {}
    for sym, lab in zip(r.columns, labels):
        groups.setdefault(int(lab), []).append(str(sym))
    return sorted((g for g in groups.values() if len(g) >= 2), key=len, reverse=True)


def engle_granger(log_y: pd.Series, log_x: pd.Series) -> dict:
    """y = a + b*x + e. Малый p-value: спред возвращается к среднему."""
    from statsmodels.tsa.stattools import adfuller, coint
    y, x = pd.concat([log_y, log_x], axis=1).dropna().T.values
    n = len(y)
    if n < 60:
        raise ValueError(f"мало общих наблюдений: {n}")
    b, a = np.polyfit(x, y, 1)
    spread = y - (a + b * x)
    tau, pval, _ = coint(y, x, trend="c", autolag="aic")
    lag, d = spread[:-1] - spread.mean(), np.diff(spread)
    phi = float(np.dot(lag, d) / np.dot(lag, lag)) if np.dot(lag, lag) else 0.0
    sd = spread.std(ddof=1)
    return {"beta": float(b), "alpha": float(a), "tau": float(tau), "pvalue": float(pval),
            "half_life": float(-math.log(2) / math.log(1 + phi)) if -1 < phi < 0 else None,
            "z_now": float((spread[-1] - spread.mean()) / sd) if sd else None, "n_obs": int(n),
            "adf_resid_p": float(adfuller(spread, autolag="AIC")[1])}


def _beta_stable(ly: pd.Series, lx: pd.Series, tol: float = 0.3) -> bool:
    d = pd.concat([ly, lx], axis=1).dropna()
    h = len(d) // 2
    b1 = np.polyfit(d.iloc[:h, 1], d.iloc[:h, 0], 1)[0]
    b2 = np.polyfit(d.iloc[h:, 1], d.iloc[h:, 0], 1)[0]
    return bool(abs(b1 - b2) <= tol * max(abs(b1), abs(b2), 1e-9))


def scan_cointegration(logp: pd.DataFrame, groups: "dict[str, list[str]]") -> dict:
    """Пары внутри групп. p_bonferroni учитывает число проверок."""
    pairs: list = []
    ret = logp.diff().dropna()
    for g, syms in groups.items():
        for a, b in itertools.combinations([s for s in syms if s in logp.columns], 2):
            try:
                r = engle_granger(logp[a], logp[b])
            except Exception:
                continue
            r.update({"pair": (a, b), "group": g, "corr_returns": float(ret[a].corr(ret[b])), "beta_stable": _beta_stable(logp[a], logp[b])})
            pairs.append(r)
    for r in pairs:
        r["p_bonferroni"] = float(min(1.0, r["pvalue"] * len(pairs)))
    pairs.sort(key=lambda r: r["pvalue"])
    ranking = []
    for g in groups:
        gp = [r for r in pairs if r["group"] == g]
        if gp:
            ranking.append({"group": g, "pairs_tested": len(gp), "best_pvalue": gp[0]["pvalue"],
                            "significant_bonferroni": sum(r["p_bonferroni"] < 0.05 for r in gp)})
    ranking.sort(key=lambda x: x["best_pvalue"])
    return {"pairs": pairs, "groups": ranking, "tested": len(pairs)}
