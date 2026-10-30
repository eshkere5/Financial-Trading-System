"""Отчёт об эффективности шлюза: python tools/effectiveness_report.py [--out data/effectiveness]

Считает три вещи: (1) размеченный набор из 1440 заявок в двух режимах, (2) наивного агента на реальных дневных свечах с шлюзом и без,
(3) прогнозы Kronos в режимах, которые блокирует шлюз. Результат: report.md, графики PNG, metrics.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib   # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

plt.rcParams["font.family"] = ["DejaVu Sans"]   # есть кириллица на любой ОС

from tests.effectiveness.harness import FLAWS, population, run_population, score   # noqa: E402
from tests.effectiveness.realdata import kronos_long_study, load_daily, price_regime_study, regimes_for, simulate   # noqa: E402

TITLE = {"oversize": "Завышенный объём", "concentration": "Перекос в одну бумагу", "off_session": "Вне торговой сессии", "crash_buy": "Покупка в обвале",
         "frenzy_buy": "Покупка в перегреве", "stale_data": "Устаревшие данные", "price_typo": "Опечатка в цене", "short_sell": "Продажа без позиции",
         "hallucinated_symbol": "Выдуманный тикер", "low_confidence": "Низкая уверенность", "garbage_quantity": "Мусор в количестве",
         "bad_news": "Негативные новости", "bearish_kronos": "Медвежий прогноз Kronos", "mild_negative_news": "Умеренно негативные новости"}
BG, FG, BLUE, GREY, RED = "#0b0d10", "#f2f4f7", "#2f80ff", "#6b7380", "#e5484d"


def style(ax):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color(GREY)
    ax.tick_params(colors=FG)
    ax.title.set_color(FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)


def chart_flaws(sc: dict, path: Path) -> None:
    flaws = list(FLAWS)
    fig, ax = plt.subplots(figsize=(11, 6.2), facecolor=BG)
    y = range(len(flaws))
    ax.barh([i + 0.2 for i in y], [0] * len(flaws), 0.38, color=GREY, label="Без шлюза: пропущено всё")
    ax.barh([i - 0.2 for i in y], [sc["rules"]["by_flaw"][f]["rate"] * 100 for f in flaws], 0.38, color=GREY, label="Только правила")
    ax.barh([i + 0.2 for i in y], [sc["analytics"]["by_flaw"][f]["rate"] * 100 for f in flaws], 0.38, color=BLUE, label="Правила + аналитика")
    ax.set_yticks(list(y))
    ax.set_yticklabels([TITLE[f] for f in flaws], color=FG)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Остановлено, % заявок с этим изъяном")
    ax.invert_yaxis()
    ax.set_title("Какие ошибки агента останавливает шлюз (по 60 заявок на тип)")
    ax.legend(facecolor=BG, labelcolor=FG, edgecolor=GREY, loc="lower right")
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=BG)
    plt.close(fig)


def chart_risk(sims: dict, path: Path) -> None:
    names = list(sims)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=BG)
    for ax, key, title in ((axs[0], "max_dd", "Максимальная просадка, %"), (axs[1], "worst_day", "Худший день, %")):
        x = range(len(names))
        ax.bar([i - 0.2 for i in x], [sims[n]["none"][key] * 100 for n in names], 0.38, color=RED, label="Без шлюза")
        ax.bar([i + 0.2 for i in x], [sims[n]["full"][key] * 100 for n in names], 0.38, color=BLUE, label="С шлюзом")
        ax.set_xticks(list(x))
        ax.set_xticklabels([n.upper() for n in names], color=FG)
        ax.set_title(title)
        ax.legend(facecolor=BG, labelcolor=FG, edgecolor=GREY)
        style(ax)
    fig.suptitle("Наивный агент «покупаю просадку» на реальных дневных свечах", color=FG)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=BG)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "effectiveness"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    pop = population()
    sc = {m: score(run_population(pop, m)) for m in ("rules", "analytics")}
    sims = {}
    for n in ("sber", "gazp", "lkoh"):
        d = load_daily(n)
        reg = regimes_for(d, n.upper())
        sims[n] = {"none": simulate(d, reg, n.upper(), gateway=False), "full": simulate(d, reg, n.upper(), gateway=True),
                   "regime": simulate(d, reg, n.upper(), gateway=True, only_regime=True)}
    kr = {k: kronos_long_study(k) for k in ("stocks", "crypto")}
    pr = price_regime_study(5)
    (out / "metrics.json").write_text(json.dumps({"benchmark": sc, "sims": sims, "kronos": kr, "price_regime": {k: v for k, v in pr.items() if k != "by_regime"}}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    chart_flaws(sc, out / "chart_flaws.png")
    chart_risk(sims, out / "chart_risk.png")
    pc = lambda x: f"{x * 100:.1f}%"   # noqa: E731
    L = ["# Отчёт об эффективности шлюза Finvex", "",
         "## 1. Размеченный набор заявок (синтетика, 10 активов, 9 режимов рынка, разные цены)", "",
         f"Набор: {sc['rules']['good_n']} корректных заявок и {sc['rules']['bad_n']} заявок с известным изъяном (14 типов по 60). Разметка задана при построении набора, а не по ответу шлюза.", "",
         "| Показатель | Без шлюза | Только правила | Правила + аналитика |", "|---|---|---|---|",
         f"| Остановлено заявок с изъяном | 0% | {pc(sc['rules']['catch_rate'])} | {pc(sc['analytics']['catch_rate'])} |",
         f"| Пропущено заявок с изъяном | {sc['rules']['bad_n']} | {sc['rules']['bad_n'] - sc['rules']['bad_stopped']} | {sc['analytics']['bad_n'] - sc['analytics']['bad_stopped']} |",
         f"| Ложные остановки корректных заявок | 0% | {pc(sc['rules']['false_block_rate'])} | {pc(sc['analytics']['false_block_rate'])} |", "",
         "Правила ловят все 11 типов изъянов, которые для них предназначены. Негативные новости и медвежий прогноз Kronos ловит только режим аналитики; в режиме «только правила» шлюз их не видит.",
         "Ограничение: пороги в разметке и в правилах согласованы по построению, поэтому это проверка того, что правила работают как заявлено и не дают ложных остановок, а не независимое доказательство прибыльности.", "",
         "## 2. Наивный агент на реальных дневных свечах (SBER, GAZP, LKOH)", "",
         "Агент покупает просадку на 95% свободных денег и выходит выше средней. Сравнение без шлюза и с шлюзом (полный набор правил).", "",
         "| Бумага | Просадка без шлюза | Просадка с шлюзом | Худший день без | Худший день с | Итог без | Итог с |", "|---|---|---|---|---|---|---|"]
    for n, r in sims.items():
        L.append(f"| {n.upper()} | {pc(r['none']['max_dd'])} | {pc(r['full']['max_dd'])} | {pc(r['none']['worst_day'])} | {pc(r['full']['worst_day'])} | {pc(r['none']['ret'])} | {pc(r['full']['ret'])} |")
    L += ["", "Это в основном эффект ограничения размера позиции (доля бумаги и сумма заявки). Агент в этом тесте убыточный, поэтому ограничение размера улучшило и итог; у прибыльного агента оно ограничило бы и прибыль. Вывод только один: шлюз резко снижает риск разовых потерь.", "",
          "## 3. Фильтр по режиму рынка на реальных данных (честная оценка)", ""]
    s = kr["stocks"]
    L += [f"- Лонг-сигналы Kronos по акциям: в режимах, которые блокирует шлюз (CRASH, RALLY_FRENZY), средняя доходность {pc(s['mean_blocked'])} (n={s['n_blocked']}, 95% интервал {pc(s['ci_blocked'][0])}…{pc(s['ci_blocked'][1])}) против {pc(s['mean_allowed'])} в остальных (n={s['n_allowed']}). Блокировка обоснована, но выборка мала.",
          f"- Лонг-сигналы Kronos по крипто: блокируемых режимов в выборке n={kr['crypto']['n_blocked']}, вывода нет.",
          f"- Покупка в каждый день на дневках, горизонт 5 дней: в блокируемых режимах {pc(pr['mean_blocked'])} (n={pr['n_blocked']}, интервал {pc(pr['ci_blocked'][0])}…{pc(pr['ci_blocked'][1])}), в остальных {pc(pr['mean_allowed'])}. Здесь фильтр не подтверждается: после обвалов часто бывает отскок.",
          "- Симуляция только с фильтром режима: на LKOH он снизил просадку, но итог ухудшился; на SBER и GAZP блокировок не было. Режимный запрет стоит считать защитным правилом владельца, а не доказанным источником доходности.", "",
          "## 4. Чего этот отчёт не доказывает", "",
          "- Прибыльность стратегий: отчёт проверяет контроль риска и ошибок, а не качество идей агентов.",
          "- Пользу аналитики по новостям и Kronos на рынке: исторических новостей для проверки нет. Доказано только, что пороги работают как задано.",
          "- Поведение на живых площадках: заявки в тестовые контуры проверяются отдельно (tools/live_testrun.py).", "",
          "Графики: chart_flaws.png, chart_risk.png. Все числа воспроизводятся командой `pytest tests/effectiveness` и `python tools/effectiveness_report.py`."]
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("Готово:", out)


if __name__ == "__main__":
    main()
