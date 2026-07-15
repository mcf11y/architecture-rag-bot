#!/usr/bin/env python3
"""Воспроизводимый скачиватель чистого текста со страниц вики (MediaWiki/Fandom).

Использует публичный MediaWiki API (action=parse) и достаёт lead-секцию
(section=0) страницы, затем удаляет HTML-разметку средствами стандартной
библиотеки (без сторонних парсеров). Результат сохраняется в --out как .md.

Назначение: продемонстрировать этап «скачайте и очистите тексты» из задания
скриптом. В самом проекте итоговый корпус (source_raw/) написан вручную ради
чистоты и читаемости, но этот скрипт позволяет собрать сырьё автоматически.

Пример:
    python scripts/scrape_wiki.py \
        --api https://starwars.fandom.com/api.php \
        --out source_scraped \
        --titles "Luke Skywalker" "Darth Vader" "Tatooine"

    # либо список заголовков из файла (по одному на строку):
    python scripts/scrape_wiki.py --titles-file titles.txt
"""
from __future__ import annotations

import argparse
import html
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path

try:
    import requests
except ImportError:  # pragma: no cover
    print("Нужен пакет requests: pip install requests", file=sys.stderr)
    raise

USER_AGENT = "architecture-rag-bot/1.0 (knowledge-base builder; educational use)"

# Набор ключевых сущностей вселенной Star Wars по умолчанию (30+).
DEFAULT_TITLES = [
    "Luke Skywalker", "Anakin Skywalker", "Leia Organa", "Han Solo",
    "Obi-Wan Kenobi", "Yoda", "Palpatine", "Padmé Amidala", "Qui-Gon Jinn",
    "Mace Windu", "Darth Maul", "Boba Fett", "Chewbacca", "Lando Calrissian",
    "Jabba Desilijic Tiure", "Wilhuff Tarkin", "R2-D2", "C-3PO",
    "Tatooine", "Coruscant", "Alderaan", "Hoth", "Dagobah", "Endor",
    "Naboo", "Kamino", "Mustafar", "Kashyyyk",
    "The Force", "Jedi", "Sith", "Galactic Empire", "Galactic Republic",
    "Alliance to Restore the Republic", "Clone Wars", "Order 66",
    "Death Star", "Millennium Falcon", "Lightsaber", "X-wing starfighter",
    "Star Destroyer", "Wookiee", "Hutt",
]


class _Stripper(HTMLParser):
    """Простой сборщик текста из HTML с пропуском служебных тегов."""

    _SKIP = {"style", "script", "table", "sup", "ref", "figure", "figcaption"}
    _BREAK = {"p", "br", "li", "h1", "h2", "h3", "h4"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BREAK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


def clean(raw_html: str) -> str:
    stripper = _Stripper()
    stripper.feed(raw_html)
    text = html.unescape(stripper.text())
    text = re.sub(r"\[\d+\]", "", text)          # сноски [1]
    text = re.sub(r"\[edit\]", "", text, flags=re.I)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    # выкидываем совсем короткие строки-обрывки инфобоксов
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if len(ln) > 30 or not ln]
    return "\n".join(lines).strip()


def fetch_lead(api: str, title: str, session: requests.Session) -> str:
    resp = session.get(api, params={
        "action": "parse", "page": title, "prop": "text",
        "section": "0", "redirects": "1", "format": "json",
        "disabletoc": "1",
    }, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(data["error"].get("info", "unknown API error"))
    return data["parse"]["text"]["*"]


def slugify(title: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", title.lower()).strip()
    return re.sub(r"[\s_]+", "-", slug) or "page"


def main() -> int:
    ap = argparse.ArgumentParser(description="Скачивание и очистка страниц вики.")
    ap.add_argument("--api", default="https://starwars.fandom.com/api.php")
    ap.add_argument("--out", default="source_scraped")
    ap.add_argument("--titles", nargs="*", help="список заголовков страниц")
    ap.add_argument("--titles-file", help="файл со списком заголовков (по строкам)")
    ap.add_argument("--delay", type=float, default=0.5, help="пауза между запросами, сек")
    args = ap.parse_args()

    titles = list(args.titles) if args.titles else []
    if args.titles_file:
        titles += [ln.strip() for ln in Path(args.titles_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not titles:
        titles = DEFAULT_TITLES

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    ok, failed = 0, 0
    for title in titles:
        try:
            body = clean(fetch_lead(args.api, title, session))
            if not body:
                raise RuntimeError("пустой текст после очистки")
            doc = f"# {title}\n\n{body}\n"
            (out / f"{slugify(title)}.md").write_text(doc, encoding="utf-8")
            print(f"  [ok]   {title} -> {slugify(title)}.md ({len(body)} симв.)")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  [fail] {title}: {exc}", file=sys.stderr)
            failed += 1
        time.sleep(args.delay)

    print(f"\nГотово: {ok} страниц сохранено, {failed} ошибок. Папка: {out}")
    print("Дальше примените замену терминов:")
    print(f"  python scripts/replace_terms.py --src {out} --dst knowledge_base --clean")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
