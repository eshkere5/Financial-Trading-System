"""
rss_collector.py — сборщик новостей из открытых бесплатных RSS-источников.

Заменяет платные API — использует публичные RSS-фиды Interfax, РБК, Finam,
раскрытие информации МосБиржи. Для крипты (позже) — CoinDesk/CoinTelegraph.

Работает событийно: опрашивает RSS с высокой частотой (30-60 сек), но сам
факт опроса RSS почти бесплатен (обычный HTTP GET) — дорогая часть (DeepSeek)
подключается только через RiskFilterPipeline после keyword/embedding-фильтра.
"""
from __future__ import annotations

import hashlib
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, List, Optional, Tuple

import feedparser

logger = logging.getLogger(__name__)

DEFAULT_FEEDS: dict[str, str] = {
    "interfax_business": "https://www.interfax.ru/rss.asp",
    "rbc_economics": "https://rssexport.rbc.ru/rbcnews/news/30/full.rss",
    "finam_news": "https://www.finam.ru/analysis/conews/rsspoint/",
    "moex_disclosure": "https://www.e-disclosure.ru/rss.aspx",
    "cointelegraph": "https://cointelegraph.com//rss"
}
        
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
    "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
}

@dataclass
class RawNewsItem:
    """Необработанная новость из RSS до keyword-фильтрации и DeepSeek-анализа."""
    item_id: str
    source: str
    title: str
    summary: str
    link: str
    published_at: str


@dataclass
class RSSCollectorConfig:
    feeds: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_FEEDS))
    poll_interval_sec: int = 30  # событийный опрос, дешёвый HTTP GET
    request_timeout: int = 10


class RSSCollector:
    """
    Опрашивает несколько RSS-фидов и выдаёт новые записи через callback.
    Дедупликация — по хэшу title+link, чтобы не обрабатывать одну новость дважды.
    """

    def __init__(self, cfg: RSSCollectorConfig) -> None:
        self.cfg = cfg
        self._seen_hashes: set[str] = set()
        self.running = False

    @staticmethod
    def _hash_item(title: str, link: str) -> str:
        return hashlib.sha256(f"{title}|{link}".encode("utf-8")).hexdigest()

    ### FIXED 2026-08-01 (С-1, С-2) ###
    # Было: feedparser.parse(url) — сам ходил в сеть, request_timeout из
    # конфига не применялся никогда. Зависший фид блокировал поток сборщика
    # на socket-таймаут по умолчанию (в худшем случае — навсегда), и весь
    # новостной контур вставал молча.
    # Плюс: feedparser не бросает исключений на HTTP 404/500 — он возвращает
    # объект с пустым .entries. `except Exception` не срабатывал, а "фид
    # отдал ошибку" было неотличимо от "новостей нет".
    # Стало: сеть — через urllib с явным timeout, feedparser парсит уже
    # полученные байты; статус и bozo-флаг проверяются и логируются отдельно.
    def _load_feed(self, source: str, url: str):
        """Возвращает (parsed_feed | None, ok). ok=False — сетевая/HTTP-ошибка."""
        request = urllib.request.Request(url, headers=_HEADERS)
        try:
            with urllib.request.urlopen(request, timeout=self.cfg.request_timeout) as resp:
                status = getattr(resp, "status", 200) or 200
                if status >= 400:
                    logger.warning("RSSCollector: %s (%s) вернул HTTP %s", source, url, status)
                    return None, False
                content = resp.read()
        except urllib.error.HTTPError as exc:
            logger.warning("RSSCollector: %s (%s) вернул HTTP %s", source, url, exc.code)
            return None, False
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning(
                "RSSCollector: сетевая ошибка на %s (%s, timeout=%ss): %s",
                source, url, self.cfg.request_timeout, exc,
            )
            return None, False

        try:
            parsed = feedparser.parse(content)
        except Exception as exc:
            logger.warning("RSSCollector: не удалось разобрать %s (%s): %s", source, url, exc)
            return None, False

        if getattr(parsed, "bozo", 0) and not parsed.entries:
            logger.warning(
                "RSSCollector: %s (%s) отдал нечитаемый XML: %s",
                source, url, getattr(parsed, "bozo_exception", "unknown"),
            )
            return None, False

        return parsed, True

    @staticmethod
    def _entry_published(entry) -> Tuple[str, datetime]:
        """Нормализует дату записи к (исходная строка, aware datetime UTC)."""
        raw = getattr(entry, "published", "") or getattr(entry, "updated", "")
        struct = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
        if struct is not None:
            try:
                dt = datetime(*struct[:6], tzinfo=timezone.utc)
                return raw or dt.isoformat(), dt
            except (TypeError, ValueError):
                pass
        now = datetime.now(timezone.utc)
        return raw or now.isoformat(), now

    def _fetch_feed(self, source: str, url: str) -> List[RawNewsItem]:
        items: List[RawNewsItem] = []
        parsed, ok = self._load_feed(source, url)
        if not ok or parsed is None:
            return items

        if not parsed.entries:
            logger.debug("RSSCollector: %s — свежих записей нет", source)
            return items

        for entry in parsed.entries:
            title = (getattr(entry, "title", "") or "").strip()
            link = (getattr(entry, "link", "") or "").strip()
            if not title or not link:
                continue
            h = self._hash_item(title, link)
            if h in self._seen_hashes:
                continue
            self._seen_hashes.add(h)

            summary = getattr(entry, "summary", "") or getattr(entry, "description", "")
            published, _dt = self._entry_published(entry)

            items.append(RawNewsItem(
                item_id=h[:16],
                source=source,
                title=title,
                summary=summary,
                link=link,
                published_at=published,
            ))
        return items

    def poll_once(self) -> List[RawNewsItem]:
        """Один проход по всем фидам — возвращает только новые (не виденные) записи."""
        all_new: List[RawNewsItem] = []
        for source, url in self.cfg.feeds.items():
            all_new.extend(self._fetch_feed(source, url))
        if all_new:
            logger.info("RSSCollector: получено %d новых новостей за проход", len(all_new))
        else:
            logger.debug("RSSCollector: за проход новых новостей нет")
        return all_new

    def fetch_backlog(self, days: int = 3) -> List[RawNewsItem]:
        """
        Разовая выгрузка бэклога за последние `days` дней — для холодного старта
        перед первым price-тиком (см. LiveEngine._run_async). RSS-фиды обычно
        отдают только последние записи (не полноценный архив за недели), поэтому
        это best-effort: берём всё, что фид отдаёт, и фильтруем по published_at.
        Не помечает записи как "виденные" в _seen_hashes — это делает отдельный
        NewsMemoryStore.is_duplicate() на уровне пайплайна.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        backlog: List[RawNewsItem] = []
        failed_sources: List[str] = []

        for source, url in self.cfg.feeds.items():
            ### FIXED 2026-08-01 (С-1) ###
            # Тот же путь загрузки с таймаутом, что и в poll_once —
            # раньше здесь была вторая копия голого feedparser.parse(url).
            parsed, ok = self._load_feed(source, url)
            if not ok or parsed is None:
                failed_sources.append(source)
                continue

            for entry in parsed.entries:
                title = (getattr(entry, "title", "") or "").strip()
                link = (getattr(entry, "link", "") or "").strip()
                if not title or not link:
                    continue

                ### FIXED 2026-08-01 (М-9) ###
                # Было: python-dateutil импортировался только здесь и
                # отсутствовал в зависимостях — холодный старт падал с
                # ImportError. feedparser уже отдаёт разобранную дату
                # (published_parsed), внешний парсер не нужен.
                published_raw, published_dt = self._entry_published(entry)
                if published_dt < cutoff:
                    continue

                h = self._hash_item(title, link)
                summary = getattr(entry, "summary", "") or getattr(entry, "description", "")
                backlog.append(RawNewsItem(
                    item_id=h[:16],
                    source=source,
                    title=title,
                    summary=summary,
                    link=link,
                    published_at=published_raw,
                ))

        ### FIXED 2026-08-01 (С-2) ###
        # Явно отличаем «бэклог пуст» от «источники не ответили»: на холодном
        # старте это разница между «новостей правда не было» и «мы стартуем
        # вслепую».
        if failed_sources:
            logger.error(
                "RSSCollector.fetch_backlog: %d из %d источников недоступны (%s) — "
                "бэклог неполный, стартуем с частичным новостным контекстом",
                len(failed_sources), len(self.cfg.feeds), ", ".join(failed_sources),
            )
        logger.info("RSSCollector.fetch_backlog: собрано %d новостей за последние %d дн.", len(backlog), days)
        return backlog

    def run_forever(self, on_new_items: Callable[[List[RawNewsItem]], None]) -> None:
        """Блокирующий цикл — для отдельного потока/процесса сборщика новостей."""
        self.running = True
        while self.running:
            try:
                new_items = self.poll_once()
                if new_items:
                    on_new_items(new_items)
            except Exception as exc:
                logger.exception("RSSCollector: ошибка в цикле опроса: %s", exc)
            time.sleep(self.cfg.poll_interval_sec)

    def stop(self) -> None:
        self.running = False
