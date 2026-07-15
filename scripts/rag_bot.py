#!/usr/bin/env python3
"""RAG-бот QuantumForge: retrieval из ChromaDB + промптинг (few-shot + CoT) + LLM.

Цепочка (реализована вручную, чтобы было видно, что происходит «под капотом»,
а не через готовый RetrievalQA):

    запрос -> эмбеддинг (тем же энкодером, что и индекс)
           -> поиск ближайших чанков в ChromaDB
           -> сборка промпта (system с Chain-of-Thought + few-shot примеры + контекст)
           -> вызов LLM (современная модель 2026: Claude Opus 4.8 / Sonnet 5)
           -> ответ пользователю (+ источники), либо «Я не знаю».

Интерфейсы:
    python scripts/rag_bot.py                      # интерактивный REPL
    python scripts/rag_bot.py "What is the Synth Flux?"   # одиночный запрос
    python scripts/rag_bot.py --demo               # прогон демо-диалогов

LLM берётся через переменные окружения (OpenAI-совместимо/Anthropic):
    RAG_LLM_MODEL      (по умолчанию claude-sonnet-5)
    ANTHROPIC_BASE_URL, ANTHROPIC_AUTH_TOKEN
    NODE_EXTRA_CA_CERTS (CA-бандл, если требуется)
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import chromadb
import requests
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "vector_store"
COLLECTION = "quantumforge_kb"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

DEFAULT_MODEL = os.environ.get("RAG_LLM_MODEL", "claude-sonnet-5")
TOP_K = 4
MIN_SIM = 0.30  # ниже этого порога считаем, что релевантного контекста нет

# --- System-промпт с Chain-of-Thought ---
SYSTEM_PROMPT = textwrap.dedent("""\
    Ты — корпоративный ассистент QuantumForge, который отвечает на вопросы
    сотрудников СТРОГО по внутренней базе знаний.

    Правила:
    - Используй ТОЛЬКО факты из предоставленного контекста. Не добавляй знания «из головы».
    - Если в контексте нет ответа — честно ответь «Я не знаю» и коротко поясни, чего не хватает.
    - Сначала рассуждай пошагово (Chain-of-Thought), затем дай финальный ответ.
    - Ссылайся на источники (имена файлов из контекста).
    - Отвечай на языке вопроса пользователя.

    Формат ответа строго такой:
    Рассуждение:
    1. ...
    2. ...
    Ответ: <краткий точный ответ>
    Источники: <файлы через запятую, либо «—»>
    """)

# --- Few-shot примеры (сформированы из реальных фрагментов базы знаний) ---
# Пример 1 — успешный ответ; Пример 2 — демонстрация «Я не знаю».
FEWSHOT = [
    {
        "role": "user",
        "content": (
            "Контекст (фрагменты базы знаний):\n"
            "[1] (source: the-synth-flux.md) The Synth Flux was a mysterious energy field that "
            "bound the galaxy together. It had two aspects: the Radiant Current, associated with "
            "peace and knowledge, and the Umbral Current, fueled by fear and hatred.\n\n"
            "Вопрос: What are the two aspects of the Synth Flux?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Рассуждение:\n"
            "1. В контексте сказано, что Synth Flux имеет два аспекта.\n"
            "2. Явно названы Radiant Current и Umbral Current.\n"
            "Ответ: Два аспекта Synth Flux — Radiant Current и Umbral Current.\n"
            "Источники: the-synth-flux.md"
        ),
    },
    {
        "role": "user",
        "content": (
            "Контекст (фрагменты базы знаний):\n"
            "[1] (source: tashiin.md) Tashiin was a harsh desert world in the Outer Rim, home to "
            "moisture farmers and the crime lord Jabomo the Vorr.\n\n"
            "Вопрос: What is the exact population of Tashiin?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Рассуждение:\n"
            "1. Вопрос о точной численности населения Tashiin.\n"
            "2. В контексте описаны климат и обитатели, но числа населения нет.\n"
            "Ответ: Я не знаю — в базе знаний нет данных о численности населения Tashiin.\n"
            "Источники: —"
        ),
    },
]


@dataclass
class Hit:
    text: str
    source: str
    title: str
    chunk_index: int
    similarity: float


class RagBot:
    def __init__(self, store: Path = STORE, model: str = DEFAULT_MODEL, top_k: int = TOP_K):
        import json
        manifest = json.loads((store / "index_manifest.json").read_text(encoding="utf-8"))
        self.embed_model_name = manifest["embedding_model"]
        self.encoder = SentenceTransformer(self.embed_model_name)
        client = chromadb.PersistentClient(path=str(store))
        self.collection = client.get_collection(COLLECTION)
        self.llm_model = model
        self.top_k = top_k
        # Конфигурация LLM-шлюза
        self.base = os.environ.get("ANTHROPIC_BASE_URL", "").rstrip("/")
        self.token = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
        self.verify = os.environ.get("NODE_EXTRA_CA_CERTS") or True

    # 1) retrieval
    def retrieve(self, query: str) -> list[Hit]:
        vec = self.encoder.encode([QUERY_INSTRUCTION + query], normalize_embeddings=True,
                                  convert_to_numpy=True)[0].tolist()
        res = self.collection.query(query_embeddings=[vec], n_results=self.top_k,
                                    include=["documents", "metadatas", "distances"])
        hits = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append(Hit(text=doc, source=meta["source"], title=meta["title"],
                            chunk_index=meta["chunk_index"], similarity=1.0 - dist))
        return hits

    # 2) сборка промпта
    @staticmethod
    def build_context_block(hits: list[Hit]) -> str:
        lines = []
        for i, h in enumerate(hits, 1):
            snippet = " ".join(h.text.split())
            lines.append(f"[{i}] (source: {h.source}) {snippet}")
        return "\n".join(lines)

    def build_messages(self, query: str, hits: list[Hit]) -> list[dict]:
        context = self.build_context_block(hits)
        user_turn = (
            f"Контекст (фрагменты базы знаний):\n{context}\n\n"
            f"Вопрос: {query}"
        )
        return FEWSHOT + [{"role": "user", "content": user_turn}]

    # 3) вызов LLM
    def call_llm(self, messages: list[dict], max_tokens: int = 700) -> str:
        if not self.base or not self.token:
            raise RuntimeError("LLM не настроена: задайте ANTHROPIC_BASE_URL и ANTHROPIC_AUTH_TOKEN")
        resp = requests.post(
            self.base + "/v1/messages",
            headers={"x-api-key": self.token, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": self.llm_model, "max_tokens": max_tokens,
                  "system": SYSTEM_PROMPT, "messages": messages},
            timeout=90, verify=self.verify,
        )
        resp.raise_for_status()
        data = resp.json()
        return "".join(part.get("text", "") for part in data.get("content", [])).strip()

    # 4) полный ответ
    def answer(self, query: str) -> dict:
        hits = self.retrieve(query)
        best = hits[0].similarity if hits else 0.0
        # Мягкий барьер: совсем нерелевантный запрос — сразу «Я не знаю» (без вызова LLM).
        if best < MIN_SIM:
            return {
                "query": query,
                "answer": ("Рассуждение:\n1. Ближайшие фрагменты базы знаний слабо связаны с запросом "
                           f"(максимальная близость {best:.2f} < порога {MIN_SIM}).\n"
                           "Ответ: Я не знаю — в базе знаний нет релевантной информации по этому вопросу.\n"
                           "Источники: —"),
                "hits": hits,
                "used_llm": False,
            }
        messages = self.build_messages(query, hits)
        text = self.call_llm(messages)
        return {"query": query, "answer": text, "hits": hits, "used_llm": True}


def print_answer(result: dict, show_retrieval: bool = True) -> None:
    print("─" * 74)
    print(f"❓ {result['query']}")
    if show_retrieval:
        print("  найденные чанки:")
        for i, h in enumerate(result["hits"], 1):
            print(f"    [{i}] sim={h.similarity:.3f}  {h.title} ({h.source}, чанк {h.chunk_index})")
    print()
    print(result["answer"])
    print()


DEMO_SUCCESS = [
    "What is the Synth Flux and what are its two sides?",
    "Кто отец Кэла Доунрайдера (Kael Dawnrider)?",
    "Which planet did the Void Core destroy, and who ordered it?",
    "Какое оружие используют Aegis Wardens?",
    "Where did Kael Dawnrider train with master Vodaan?",
]
DEMO_DONTKNOW = [
    "Who is Luke Skywalker?",                 # исходное имя из Star Wars — в базе его нет
    "What is the annual revenue of QuantumForge Software?",  # вне предметной области базы
]


def run_demo(bot: RagBot) -> None:
    print("\n########## УСПЕШНЫЕ ДИАЛОГИ ##########\n")
    for q in DEMO_SUCCESS:
        print_answer(bot.answer(q))
    print("\n########## СЛУЧАИ «Я НЕ ЗНАЮ» ##########\n")
    for q in DEMO_DONTKNOW:
        print_answer(bot.answer(q))


def repl(bot: RagBot) -> None:
    print(f"RAG-бот QuantumForge готов. Модель: {bot.llm_model}, энкодер: {bot.embed_model_name}.")
    print("Введите вопрос (пустая строка или 'exit' — выход).")
    while True:
        try:
            q = input("\n❓ > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in {"exit", "quit"}:
            break
        try:
            print_answer(bot.answer(q))
        except Exception as exc:  # noqa: BLE001
            print(f"[ошибка] {exc}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description="RAG-бот QuantumForge (few-shot + CoT).")
    ap.add_argument("query", nargs="*", help="одиночный запрос (если пусто — REPL)")
    ap.add_argument("--demo", action="store_true", help="прогнать демонстрационные диалоги")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="модель LLM (по умолчанию claude-sonnet-5)")
    ap.add_argument("-k", "--top-k", type=int, default=TOP_K)
    args = ap.parse_args()

    bot = RagBot(model=args.model, top_k=args.top_k)
    if args.demo:
        run_demo(bot)
    elif args.query:
        print_answer(bot.answer(" ".join(args.query)))
    else:
        repl(bot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
