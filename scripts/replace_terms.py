#!/usr/bin/env python3
"""Обфускация корпуса: замена канонических терминов на вымышленные.

Читает исходные документы из --src, применяет словарь замен из --map
(terms_map.json) и сохраняет обфусцированные документы в --dst.

Логика замены:
  * регистрозависимая, но с автоматическими вариантами (Title Case и UPPER),
    чтобы корректно обрабатывать начала предложений и заголовки;
  * по границам «слова» (с учётом апострофов и дефисов), чтобы не задевать
    подстроки внутри других слов;
  * за один проход, самые длинные ключи применяются первыми
    (иначе «Empire» сработал бы раньше «Galactic Empire»);
  * имя выходного файла берётся из обфусцированного заголовка H1.

Пример:
    python scripts/replace_terms.py \
        --src source_raw --dst knowledge_base --map terms_map.json
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


def load_replacements(map_path: Path) -> dict[str, str]:
    data = json.loads(map_path.read_text(encoding="utf-8"))
    repl = data.get("replacements", data)
    return {str(k): str(v) for k, v in repl.items() if not str(k).startswith("_")}


def expand_case_variants(repl: dict[str, str]) -> dict[str, str]:
    """Добавляет Title- и UPPER-варианты ключей.

    Явно заданные ключи имеют приоритет (setdefault не перезаписывает их).
    """
    expanded: dict[str, str] = {}
    for key, val in repl.items():
        expanded[key] = val
    for key, val in repl.items():
        cap_key = key[:1].upper() + key[1:]
        cap_val = val[:1].upper() + val[1:]
        expanded.setdefault(cap_key, cap_val)
        expanded.setdefault(key.upper(), val.upper())
    return expanded


def build_pattern(keys: list[str]) -> re.Pattern[str]:
    # Длинные ключи — первыми, чтобы жадно матчить многословные термины.
    ordered = sorted(keys, key=len, reverse=True)
    alternation = "|".join(re.escape(k) for k in ordered)
    # Границы: слева/справа не должно быть буквенно-цифрового символа.
    return re.compile(r"(?<![\w])(?:" + alternation + r")(?![\w])")


def slugify(title: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", title.lower()).strip()
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug)
    return slug or "document"


def obfuscate_text(text: str, pattern: re.Pattern[str], table: dict[str, str]) -> tuple[str, int]:
    count = 0

    def _sub(m: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return table[m.group(0)]

    return pattern.sub(_sub, text), count


def fix_articles(text: str) -> str:
    """Согласует артикли a/an после замены (эвристика по первой букве).

    После обфускации возникают сочетания вроде «a Aegis Warden»; приводим
    их к «an Aegis Warden». Слова-исключения (hour, honest) в корпусе не
    встречаются, поэтому простой эвристики достаточно.
    """
    text = re.sub(r"\b([Aa])n (?=[bcdfghjklmnpqrstvwxyz])",
                  lambda m: m.group(1) + " ", text)
    text = re.sub(r"\b([Aa]) (?=[aeiouAEIOU])",
                  lambda m: m.group(1) + "n ", text)
    return text


def output_name(obf_text: str, fallback_stem: str, suffix: str, used: set[str]) -> str:
    first = obf_text.lstrip().splitlines()[0] if obf_text.strip() else ""
    title = first.lstrip("# ").strip() if first.startswith("#") else fallback_stem
    base = slugify(title)
    name = f"{base}{suffix}"
    i = 2
    while name in used:
        name = f"{base}-{i}{suffix}"
        i += 1
    used.add(name)
    return name


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="Замена терминов для обфускации базы знаний.")
    ap.add_argument("--src", default=str(root / "source_raw"), help="папка с исходными документами")
    ap.add_argument("--dst", default=str(root / "knowledge_base"), help="папка для обфусцированных документов")
    ap.add_argument("--map", default=str(root / "terms_map.json"), help="путь к terms_map.json")
    ap.add_argument("--ext", default=".md,.txt", help="расширения исходных файлов через запятую")
    ap.add_argument("--clean", action="store_true", help="очистить dst перед записью")
    args = ap.parse_args()

    src, dst, map_path = Path(args.src), Path(args.dst), Path(args.map)
    exts = {e if e.startswith(".") else f".{e}" for e in args.ext.split(",")}

    if not src.is_dir():
        print(f"[ошибка] нет папки с источниками: {src}", file=sys.stderr)
        return 1
    if not map_path.is_file():
        print(f"[ошибка] нет словаря: {map_path}", file=sys.stderr)
        return 1

    repl = load_replacements(map_path)
    table = expand_case_variants(repl)
    pattern = build_pattern(list(table.keys()))

    if args.clean and dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in src.iterdir() if p.suffix.lower() in exts and p.is_file())
    if not files:
        print(f"[предупреждение] в {src} нет файлов с расширениями {sorted(exts)}", file=sys.stderr)

    used_names: set[str] = set()
    total_repl = 0
    for path in files:
        text = path.read_text(encoding="utf-8")
        obf, n = obfuscate_text(text, pattern, table)
        obf = fix_articles(obf)
        total_repl += n
        name = output_name(obf, path.stem, path.suffix.lower(), used_names)
        (dst / name).write_text(obf, encoding="utf-8")
        print(f"  {path.name:32s} -> {name:32s} ({n} замен)")

    # Проверка обфускации: не осталось ли исходных терминов.
    # Берём канонические ключи длиннее 3 символов (короткие вроде "Han"
    # слишком общие) и их регистровые варианты.
    check_map = {k: repl[k] for k in repl if len(k) > 3}
    check_pattern = build_pattern(list(expand_case_variants(check_map).keys()))
    leaks: dict[str, list[str]] = {}
    for out_file in dst.iterdir():
        if out_file.suffix.lower() not in exts:
            continue
        found = sorted(set(check_pattern.findall(out_file.read_text(encoding="utf-8"))))
        if found:
            leaks[out_file.name] = found

    print("\n" + "=" * 60)
    print(f"Файлов обработано : {len(files)}")
    print(f"Всего замен       : {total_repl}")
    print(f"Итоговых документов: {len(list(dst.glob('*')))}")
    if leaks:
        print("\n[ВНИМАНИЕ] возможные незамаскированные термины:")
        for fname, terms in leaks.items():
            print(f"  {fname}: {', '.join(terms)}")
        return 2
    print("Проверка обфускации: исходных терминов не найдено ✔")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
