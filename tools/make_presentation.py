"""Презентация по итогам тестового прогона: python tools/make_presentation.py data/testruns/<папка> [--out файл.pptx]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

BG, PANEL, TX, MU, BLUE = RGBColor(0x0B, 0x0D, 0x10), RGBColor(0x14, 0x18, 0x1E), RGBColor(0xF2, 0xF4, 0xF7), RGBColor(0x9A, 0xA2, 0xAE), RGBColor(0x2F, 0x80, 0xFF)
RU = {"allow": "Разрешено", "revise": "На доработку", "deny": "Отклонено", "hold": "Не торговать", "model_error": "Ошибка модели",
      "no_context": "Нет данных", "rejected_format": "Неверный формат", "error": "Сбой"}
RULES = {"max_order_value": "Сумма заявки", "max_position_pct": "Доля бумаги", "trading_hours": "Торговая сессия", "cooldown": "Пауза по бумаге",
         "no_short": "Короткие продажи", "regime_block_buy": "Режим рынка", "min_confidence": "Уверенность", "price_deviation": "Цена",
         "data_freshness": "Свежесть данных", "max_revisions": "Лимит доработок", "blocked_symbols": "Запрещённые бумаги",
         "allowed_symbols": "Разрешённые бумаги", "max_rationale": "Длина обоснования", "kill_switch": "Аварийная остановка",
         "news_sentiment": "Новостной фон", "kronos_view": "Прогноз Kronos", "max_orders_per_day": "Заявок в сутки"}


def text(slide, x, y, w, h, s, size=16, color=TX, bold=False, align=None):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    lines = s if isinstance(s, list) else [s]
    for k, line in enumerate(lines):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        p.text = line
        p.font.size, p.font.color.rgb, p.font.bold = Pt(size), color, bold
        p.space_after = Pt(6)
        if align:
            p.alignment = align
    return tb


def slide(prs, title, notes="", watermark=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    text(s, 0.6, 0.35, 12, 0.8, title, 30, TX, True)
    if watermark:
        text(s, 8.2, 7.05, 5, 0.35, watermark, 11, MU, align=PP_ALIGN.RIGHT)
    s.notes_slide.notes_text_frame.text = notes
    return s


def box(s, x, y, w, h, title, sub, accent=False):
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = BLUE if accent else PANEL
    shp.line.color.rgb = BLUE
    shp.shadow.inherit = False
    tf = shp.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].text = title
    tf.paragraphs[0].font.size, tf.paragraphs[0].font.bold, tf.paragraphs[0].font.color.rgb = Pt(18), True, TX
    p = tf.add_paragraph()
    p.text = sub
    p.font.size, p.font.color.rgb = Pt(13), TX if accent else MU
    return shp


def chart(s, kind, cats, series, x, y, w, h, name="Количество"):
    cd = CategoryChartData()
    cd.categories = cats
    cd.add_series(name, series)
    gf = s.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    c = gf.chart
    c.has_legend = False
    c.font.size, c.font.color.rgb = Pt(13), TX
    pl = c.plots[0]
    pl.has_data_labels = True
    pl.data_labels.font.size, pl.data_labels.font.color.rgb = Pt(13), TX
    pl.series[0].format.fill.solid()
    pl.series[0].format.fill.fore_color.rgb = BLUE
    try:
        c.value_axis.visible = False
        c.value_axis.has_major_gridlines = False
        c.category_axis.format.line.color.rgb = MU
    except Exception:
        pass
    return c


def table(s, rows, cols, x, y, w, widths, size=12):
    t = s.shapes.add_table(len(rows) + 1, len(cols), Inches(x), Inches(y), Inches(w), Inches(0.4 * (len(rows) + 1))).table
    for j, c in enumerate(cols):
        t.columns[j].width = Inches(widths[j])
        cell = t.cell(0, j)
        cell.text = c
        cell.fill.solid()
        cell.fill.fore_color.rgb = BLUE
        cell.text_frame.paragraphs[0].font.size, cell.text_frame.paragraphs[0].font.bold = Pt(size), True
        cell.text_frame.paragraphs[0].font.color.rgb = TX
    for i, r in enumerate(rows, 1):
        for j, v in enumerate(r):
            cell = t.cell(i, j)
            cell.text = str(v)
            cell.fill.solid()
            cell.fill.fore_color.rgb = PANEL
            cell.text_frame.paragraphs[0].font.size = Pt(size)
            cell.text_frame.paragraphs[0].font.color.rgb = TX
    return t


def build(run_dir: Path, out: Path) -> Path:
    summ = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    ev = [json.loads(x) for x in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    fake = summ.get("fake")
    wm = "ПРИМЕР: синтетические данные, не результат" if fake else None
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

    s = slide(prs, "Finvex: тестовый прогон ИИ-агентов", "Вступление: что показываем и почему это безопасно (песочницы, виртуальные деньги).", wm)
    text(s, 0.6, 1.6, 11.5, 2, ["Агенты на модели DeepSeek принимают торговые решения.",
                                "Каждое решение проходит шлюз контроля и только потом попадает в тестовый контур:",
                                "Bybit Testnet и песочницу T-Invest. Деньги виртуальные."], 22)
    text(s, 0.6, 5.6, 11, 1, [f"Прогон: {summ['started']}, длительность {summ['duration_s']:.0f} с",
                              "Режим: заявки в тестовые контуры" if summ["send"] else "Режим: без отправки заявок"], 16, MU)

    s = slide(prs, "Как это устроено", "Идея: агент не касается денег напрямую. Между решением и заявкой стоит код, который нельзя уговорить.", wm)
    box(s, 0.6, 2.2, 3.4, 2.2, "Агент", "Модель DeepSeek с ролью: осторожный, трендовик, возврат к средней, агрессивный")
    box(s, 4.95, 2.2, 3.4, 2.2, "Шлюз контроля", "Проверяет заявку по правилам: размер, сессия, режим рынка, пауза. Решает: разрешить, доработать, отклонить", True)
    box(s, 9.3, 2.2, 3.4, 2.2, "Тестовый контур", "Bybit Testnet для крипто, песочница T-Invest для акций MOEX")
    text(s, 4.05, 2.9, 0.9, 0.6, "→", 40, BLUE, True)
    text(s, 8.4, 2.9, 0.9, 0.6, "→", 40, BLUE, True)
    text(s, 0.6, 5.0, 12, 1.5, ["Все решения пишутся в журнал с цепочкой хешей.", "На боевой счёт система не подключена: тестовые адаптеры отказываются работать с боевым режимом."], 16, MU)

    s = slide(prs, "Что произошло за прогон", "Главное число: сколько решений модели дошло до тестового контура и сколько остановил шлюз.", wm)
    oc = summ["outcomes"]
    cats = [RU.get(k, k) for k in oc]
    chart(s, XL_CHART_TYPE.BAR_CLUSTERED, cats, list(oc.values()), 0.6, 1.4, 7.6, 5.4)
    text(s, 8.6, 1.6, 4.3, 4.6, [f"Решений агентов: {summ['decisions']}", f"Заявок принято тестовыми контурами: {summ['orders_sent']}",
                                 f"Средняя задержка модели: {summ['avg_latency_ms']} мс", f"Токены: {summ['tokens_prompt']:,} + {summ['tokens_completion']:,}".replace(",", " ")], 18)

    s = slide(prs, "Что остановил шлюз", "Показываем, что правила работают: без них часть заявок ушла бы в рынок.", wm)
    tr = summ.get("top_rules") or []
    if tr:
        chart(s, XL_CHART_TYPE.BAR_CLUSTERED, [RULES.get(r, r) for r, _ in tr][::-1], [n for _, n in tr][::-1], 0.6, 1.4, 7.6, 5.4)
    else:
        text(s, 0.6, 2, 8, 1, "Шлюз не останавливал заявки в этом прогоне.", 20, MU)
    ex = next((e for e in ev if e.get("outcome") == "deny"), None) or next((e for e in ev if e.get("outcome") == "revise"), None)
    if ex:
        text(s, 8.6, 1.6, 4.3, 5, [f"Пример: {ex['agent']}, {ex['symbol']}", f"Решение: {RU.get(ex['outcome'])}", "Сообщение агенту:", (ex.get("agent_message") or "")[:260]], 15, TX)

    sent = [e for e in ev if e.get("sent")]
    s = slide(prs, "Заявки в тестовых контурах", "Это реальные API-вызовы к песочницам. Номер заявки можно сверить в кабинете площадки.", wm)
    if sent:
        table(s, [(e["ts"][11:19], e["agent"], e["symbol"], e.get("action"), e.get("quantity"), e.get("venue"), e.get("order_status"), (e.get("order_id") or "")[:14]) for e in sent[:10]],
              ["Время", "Агент", "Тикер", "Сторона", "Объём", "Контур", "Статус", "Номер заявки"], 0.6, 1.5, 12.1, [1.2, 1.7, 1.4, 1.2, 1.2, 1.3, 1.5, 2.6])
    else:
        text(s, 0.6, 2, 11, 1, "В этом прогоне заявки в контуры не отправлялись.", 20, MU)

    s = slide(prs, "Ошибки модели: тоже результат", "Показываем, что модель ошибается, и система это переживает: мусор вместо JSON, выдуманный объём.", wm)
    bad = [e for e in ev if e.get("outcome") in ("model_error", "rejected_format", "no_context")][:4]
    text(s, 0.6, 1.5, 12, 5, [f"{RU.get(e['outcome'])}: {e['agent']}, {e['symbol']}. {e.get('note', '')[:160]}" for e in bad] or ["Ошибок формата не было."], 16)

    bal = summ.get("balances_before") and summ.get("balances_after")
    s = slide(prs, "Балансы тестовых счетов", "Баланс до и после: подтверждение, что заявки действительно исполнялись.", wm)
    if bal:
        rows = [(v, str(summ["balances_before"].get(v)), str(summ["balances_after"].get(v))) for v in summ["balances_after"]]
        table(s, rows, ["Контур", "До", "После"], 0.6, 1.6, 12.1, [2.5, 4.8, 4.8], 12)
    else:
        text(s, 0.6, 2, 11, 1, "Балансы не снимались (прогон без отправки заявок).", 20, MU)

    s = slide(prs, "Выводы и ограничения", "Честно называем ограничения: малая выборка, виртуальные деньги, нет оценки доходности.", wm)
    text(s, 0.6, 1.5, 12, 5.2, ["Шлюз останавливает заведомо неверные и избыточные заявки, а допустимые доводит до контура.",
                                "Агенты разные по стилю, поэтому и решения разные: это видно в таблице исходов.",
                                "Ограничения: короткий прогон, виртуальные деньги, нет оценки доходности. Это проверка механики, а не торговая статистика.",
                                "Следующий шаг: длинный прогон на тех же контурах и добавление новостной аналитики в правила."], 18)

    s = slide(prs, "Сценарий записи экрана (3–4 минуты)", "Для записи используйте OBS или встроенную запись Windows (Win+Alt+R).", wm)
    text(s, 0.6, 1.4, 12, 5.5, ["1. Окно «Правила»: покажите три эффекта правил (отклоняет, на доработку, предупреждает).",
                                "2. Вкладка «Песочница»: выберите агента, тикер, запустите, покажите решение модели и вердикт.",
                                "3. Терминал: python tools/live_testrun.py --send --rounds 2, покажите строки с принятыми заявками.",
                                "4. Кабинет Bybit Testnet или T-Invest: сверьте номер заявки.",
                                "5. Вкладка «Вердикты»: фильтр по агенту, окно деталей с цепочкой проверок.",
                                "6. Итоговый слайд: что остановил шлюз и какие ограничения у прогона."], 17)
    prs.save(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = Path(a.run_dir)
    print("Готово:", build(d, Path(a.out) if a.out else d / "presentation.pptx"))


if __name__ == "__main__":
    main()
