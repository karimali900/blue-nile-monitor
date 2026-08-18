"""OSINT layer: pull public reporting (news RSS + custom feeds) about the dams.

Uses only public, keyless sources (Google News RSS, configurable RSS/Atom feeds,
e.g. the Nile Basin Initiative, academic outlets). Results are merged into the
satellite-derived report so that quantitative change is paired with the public
narrative.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote_plus

import feedparser
import requests

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en&gl=US&ceid=US:en"


@dataclass
class NewsItem:
    title: str
    source: str
    url: str
    published: dt.datetime | None
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "source": self.source,
            "url": self.url,
            "published": self.published.isoformat() if self.published else None,
            "summary": self.summary,
        }


@dataclass
class OsintResult:
    items: list[NewsItem] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    fetched_at: dt.datetime = field(default_factory=dt.datetime.now)

    def to_json(self) -> str:
        return json.dumps(
            {"fetched_at": self.fetched_at.isoformat(), "items": [i.to_dict() for i in self.items]},
            indent=2,
        )


def _matches_keywords(text: str, required: list[str], exclude: list[str]) -> bool:
    low = text.lower()
    if required and not any(k.lower() in low for k in required):
        return False
    if any(k.lower() in low for k in exclude):
        return False
    return True


def _parse_feed(url: str, timeout: int = 25) -> list[NewsItem]:
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (nilemonitor)"})
    resp.raise_for_status()
    feed = feedparser.parse(resp.content)
    items = []
    for entry in feed.entries:
        published = None
        for key in ("published_parsed", "updated_parsed"):
            if entry.get(key):
                published = dt.datetime(*entry[key][:6])
                break
        items.append(
            NewsItem(
                title=getattr(entry, "title", ""),
                source=getattr(entry, "source", {}).get("title", "") or getattr(entry, "feed", {}).get("title", ""),
                url=getattr(entry, "link", ""),
                published=published,
                summary=getattr(entry, "summary", "")[:400],
            )
        )
    return items


def gather_news(cfg) -> OsintResult:
    """Fetch news items from configured queries + custom feeds, deduplicated."""
    osint_cfg = cfg.osint_config
    result = OsintResult()
    required = osint_cfg.get("required_keywords", [])
    exclude = osint_cfg.get("exclude_keywords", [])

    seen_urls: set[str] = set()

    def add(item: NewsItem) -> None:
        if not item.title or item.url in seen_urls:
            return
        if not _matches_keywords(item.title + " " + item.summary, required, exclude):
            return
        seen_urls.add(item.url)
        result.items.append(item)

    for query in osint_cfg.get("news_queries", []):
        url = GOOGLE_NEWS_RSS.format(query=quote_plus(query))
        try:
            for item in _parse_feed(url):
                add(item)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"query {query!r}: {exc}")
        time.sleep(0.4)

    for feed_url in osint_cfg.get("custom_feeds", []):
        try:
            for item in _parse_feed(feed_url):
                add(item)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"feed {feed_url!r}: {exc}")

    result.items.sort(key=lambda i: i.published or dt.datetime.min, reverse=True)
    max_items = int(osint_cfg.get("max_items_per_query", 10)) * len(osint_cfg.get("news_queries", [1])) * 2
    result.items = result.items[:max_items]
    return result


def save_osint(result: OsintResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "osint_news.json"
    path.write_text(result.to_json())
    return path
