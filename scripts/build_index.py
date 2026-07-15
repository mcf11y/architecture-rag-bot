#!/usr/bin/env python3
"""Построение векторного индекса базы знаний в ChromaDB.

Пайплайн:
  1. читаем документы из knowledge_base/*.md;
  2. режем каждый документ на логические чанки (RecursiveCharacterTextSplitter);
  3. считаем эмбеддинги локальной моделью BGE (Sentence-Transformers);
  4. складываем векторы + метаданные (источник, заголовок, id, позиция) в ChromaDB;
  5. сохраняем манифест со статистикой.

ВАЖНО: индексация использует ТОЛЬКО модель эмбеддингов, не LLM.

Запуск:
    python scripts/build_index.py --clean
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Модель эмбеддингов из семейства BGE (см. Задание 1). Лёгкий вариант для
# локального тестового индекса; в проде — BAAI/bge-m3 (1024-мерный, мультиязычный).
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION = "quantumforge_kb"


def read_documents(src: Path) -> list[dict]:
    docs = []
    for path in sorted(src.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        first = text.splitlines()[0] if text else ""
        title = first.lstrip("# ").strip() if first.startswith("#") else path.stem
        docs.append({"path": str(path.relative_to(src.parent)), "file": path.name, "title": title, "text": text})
    return docs


def chunk_documents(docs: list[dict], chunk_size: int, chunk_overlap: int) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks: list[dict] = []
    gid = 0
    for doc in docs:
        pieces = splitter.split_text(doc["text"])
        for local_idx, piece in enumerate(pieces):
            chunks.append({
                "id": f"chunk-{gid:04d}",
                "text": piece,
                "metadata": {
                    "source": doc["file"],
                    "path": doc["path"],
                    "title": doc["title"],
                    "chunk_index": local_idx,
                    "chunk_id": gid,
                },
            })
            gid += 1
    return chunks


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="Строит векторный индекс базы знаний в ChromaDB.")
    ap.add_argument("--src", default=str(root / "knowledge_base"))
    ap.add_argument("--store", default=str(root / "vector_store"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--chunk-size", type=int, default=600, help="размер чанка в символах (~100-150 слов)")
    ap.add_argument("--chunk-overlap", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--clean", action="store_true", help="пересоздать коллекцию с нуля")
    args = ap.parse_args()

    src = Path(args.src)
    store = Path(args.store)
    store.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Чтение документов из {src} ...")
    docs = read_documents(src)
    print(f"      документов: {len(docs)}")

    print(f"[2/4] Чанкинг (size={args.chunk_size}, overlap={args.chunk_overlap}) ...")
    chunks = chunk_documents(docs, args.chunk_size, args.chunk_overlap)
    print(f"      чанков: {len(chunks)}")

    print(f"[3/4] Загрузка модели эмбеддингов: {args.model} ...")
    t_model = time.perf_counter()
    model = SentenceTransformer(args.model)
    dim = (model.get_embedding_dimension() if hasattr(model, "get_embedding_dimension")
           else model.get_sentence_embedding_dimension())
    print(f"      модель загружена за {time.perf_counter() - t_model:.1f} c, размерность вектора: {dim}")

    print("[3/4] Генерация эмбеддингов ...")
    t_embed = time.perf_counter()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,   # косинусная близость
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embed_time = time.perf_counter() - t_embed
    print(f"      эмбеддинги готовы за {embed_time:.2f} c "
          f"({len(chunks) / embed_time:.1f} чанков/с)")

    print(f"[4/4] Запись в ChromaDB: {store} ...")
    client = chromadb.PersistentClient(path=str(store))
    if args.clean:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine", "embedding_model": args.model},
    )
    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=[e.tolist() for e in embeddings],
        documents=texts,
        metadatas=[c["metadata"] for c in chunks],
    )

    manifest = {
        "embedding_model": args.model,
        "embedding_dim": dim,
        "vector_db": "ChromaDB (PersistentClient)",
        "collection": COLLECTION,
        "distance": "cosine",
        "documents": len(docs),
        "chunks": len(chunks),
        "chunk_size_chars": args.chunk_size,
        "chunk_overlap_chars": args.chunk_overlap,
        "embed_seconds": round(embed_time, 2),
        "chunks_per_second": round(len(chunks) / embed_time, 1),
        "store_path": str(store),
    }
    (store / "index_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Индекс готов. Коллекция '{COLLECTION}' содержит {collection.count()} векторов.")
    print(f"Модель: {args.model} (dim={dim}) | БД: ChromaDB (cosine)")
    print(f"Время генерации эмбеддингов: {embed_time:.2f} c")
    print(f"Манифест: {store / 'index_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
