# Задание 3. Векторный индекс

- **Какая модель использовалась:** `BAAI/bge-small-en-v1.5` — локальная модель эмбеддингов семейства BGE (Sentence-Transformers), размерность вектора **384**, косинусная близость. Репозиторий: [https://huggingface.co/BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **Какая база знаний:** `knowledge_base/` — **43** обфусцированных документа `.md` из Задания 2 (вселенная «Aethkeron Saga», переименованный Star Wars), один файл = одна сущность.
- **Сколько чанков в индексе:** **118** чанков (нарезка `RecursiveCharacterTextSplitter`, 600 символов, overlap 100). Хранилище — **ChromaDB** (`vector_store/`).
- **Сколько времени заняла генерация:** эмбеддинги — **≈1.25 c** (~94 чанка/с, CPU); первичная загрузка модели — ≈13 c.

Артефакты: индекс `vector_store/` (`chroma.sqlite3` + `index_manifest.json`), код `scripts/build_index.py` (построение) и `scripts/query_index.py` (пример запроса к индексу).