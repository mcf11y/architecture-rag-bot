#!/usr/bin/env python3
"""Поиск по векторному индексу базы знаний (проверка качества retrieval).

Загружает коллекцию ChromaDB и ту же модель эмбеддингов, что и при индексации,
кодирует запрос и возвращает top-k релевантных чанков с метаданными и оценкой
близости.

Запуск:
    # встроенные демо-запросы (по обфусцированному миру):
    python scripts/query_index.py

    # свой запрос:
    python scripts/query_index.py -k 3 "Who is the father of Kael Dawnrider?"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

COLLECTION = "quantumforge_kb"
# BGE-модели рекомендуют добавлять инструкцию к ЗАПРОСУ (не к документам).
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

DEMO_QUERIES = [
    "Who is the father of Kael Dawnrider?",
    "What is the Synth Flux?",
    "Which planet did the Void Core destroy?",
    "What weapon do the Aegis Wardens and Void Reavers use?",
]


def load(store: Path):
    manifest = json.loads((store / "index_manifest.json").read_text(encoding="utf-8"))
    model = SentenceTransformer(manifest["embedding_model"])
    client = chromadb.PersistentClient(path=str(store))
    collection = client.get_collection(COLLECTION)
    return model, collection, manifest


def search(model, collection, query: str, k: int):
    vec = model.encode([QUERY_INSTRUCTION + query], normalize_embeddings=True,
                       convert_to_numpy=True)[0].tolist()
    res = collection.query(query_embeddings=[vec], n_results=k,
                           include=["documents", "metadatas", "distances"])
    out = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        out.append({"doc": doc, "meta": meta, "similarity": 1.0 - dist})
    return out


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="Поиск по векторному индексу.")
    ap.add_argument("query", nargs="*", help="текст запроса (если пусто — демо-запросы)")
    ap.add_argument("--store", default=str(root / "vector_store"))
    ap.add_argument("-k", "--top-k", type=int, default=3)
    args = ap.parse_args()

    model, collection, manifest = load(Path(args.store))
    print(f"Модель: {manifest['embedding_model']} (dim={manifest['embedding_dim']}) | "
          f"векторов в индексе: {collection.count()}\n")

    queries = [" ".join(args.query)] if args.query else DEMO_QUERIES
    for q in queries:
        print("=" * 70)
        print(f"Запрос: {q}")
        for i, hit in enumerate(search(model, collection, q, args.top_k), 1):
            snippet = " ".join(hit["doc"].split())
            snippet = snippet[:220] + ("…" if len(snippet) > 220 else "")
            print(f"  #{i} [sim={hit['similarity']:.3f}] "
                  f"{hit['meta']['title']} ({hit['meta']['source']}, чанк {hit['meta']['chunk_index']})")
            print(f"      {snippet}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
