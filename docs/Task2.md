# Задание 2. Подготовка базы знаний

- **Исходная вселенная:** **Star Wars** (`starwars.fandom.com`) 
- **Вымышленная вселенная (после замены):** условно **«Aethkeron Saga»**

Пример превращения:


| Star Wars         | Aethkeron Saga             |
| ----------------- | -------------------------- |
| Luke Skywalker    | Kael Dawnrider             |
| Darth Vader       | Xarn Velgor                |
| The Force         | The Synth Flux             |
| Jedi / Sith       | Aegis Warden / Void Reaver |
| Death Star        | Void Core                  |
| Tatooine          | Tashiin                    |
| Millennium Falcon | Void Kestrel               |


Текст остаётся связным и читаемым, но больше **не распознаётся** как Star Wars.

---

## Cструктура и артефакты)

```
architecture-rag-bot/
├── source_raw/            # 43 исходные статьи (канонические термины), одна сущность = один файл
├── knowledge_base/        # 43 ОБФУСЦИРОВАННЫХ документа (.md) — итоговая база знаний
├── terms_map.json         # словарь замен: исходный термин → вымышленный
└── scripts/
    ├── replace_terms.py    # ключевой скрипт: применяет словарь замен
    └── scrape_wiki.py      # бонус: воспроизводимый скачиватель текста с вики
```

---

## Словарь замен `terms_map.json`

Структура файла:

```json
{
  "_meta": { "source_universe": "Star Wars", "invented_universe": "Aethkeron Saga", "...": "..." },
  "replacements": {
    "Darth Vader": "Xarn Velgor",
    "The Force": "The Synth Flux",
    "lightsaber": "arc glaive"
  }
}
```

Принципы составления словаря:

- **Полнота покрытия.** Заменены не только имена, но и названия рас (`Wookiee → Grovak`, `Hutt → Vorr`), оружия (`lightsaber → arc glaive`, `blaster → pulse-caster`), планет, фракций, технологий (`hyperdrive → riftdrive`, `carbonite → stasis resin`) и абстрактных концепций (`the Force → the Synth Flux`, `dark side → Umbral Current`).
- **Согласованность стиля.** Вымышленные имена подобраны в едином sci-fi-ключе, чтобы мир звучал цельно.
- **Сохранение связей.** Родовые имена согласованы: `Skywalker → Dawnrider`, поэтому `Luke`/`Anakin Skywalker` → `Kael`/`Anixar Dawnrider` — родственная связь остаётся.
- **Одиночные алиасы.** Добавлены короткие формы (`Luke`, `Vader`, `Han`), чтобы в тексте не осталось незамаскированных упоминаний.

---

## Скрипт замены `replace_terms.py`

```bash
# из корня репозитория
python scripts/replace_terms.py --clean

# или с явными путями
python scripts/replace_terms.py --src source_raw --dst knowledge_base --map terms_map.json --clean
```

