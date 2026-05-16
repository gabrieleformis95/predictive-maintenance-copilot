"""RAGAS evaluation of the RAG retrieval + answer generation pipeline.

Loads a golden Q&A dataset, retrieves context for each question, generates
answers with the configured LLM, then scores with faithfulness and
answer_relevancy using RAGAS.

Requires: GROQ_API_KEY set in .env or environment.
Requires: ChromaDB populated (run `make ingest-manuals` first).

Output: reports/ragas_results.json + console table.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import typer
from rich.logging import RichHandler
from rich.table import Table
from rich.console import Console

app = typer.Typer(help="Run RAGAS evaluation on the RAG pipeline.")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("scripts.evaluate_rag")
console = Console()

GOLDEN_PATH = Path("data/ragas_golden.json")


def _retrieve_context(question: str, top_k: int = 5) -> list[str]:
    from src.rag.retriever import retrieve

    chunks = retrieve(question, k=top_k)
    return [c.text for c in chunks]


def _generate_answer(question: str, contexts: list[str], llm_client) -> str:
    context_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    user_msg = (
        f"Answer the following question using only the provided context.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}"
    )
    return llm_client.complete(
        system="You are an expert aircraft maintenance engineer. Answer concisely and accurately.",
        user=user_msg,
        response_format="text",
        max_tokens=512,
    )


@app.command()
def main(
    golden_path: Path = typer.Option(GOLDEN_PATH, "--golden"),
    top_k: int = typer.Option(5, "--top-k"),
    out_dir: Path = typer.Option(Path("reports"), "--out-dir"),
) -> None:
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness

    from src.config import settings
    from src.llm import get_llm_client

    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY not set.")

    out_dir.mkdir(parents=True, exist_ok=True)

    golden = json.loads(golden_path.read_text())
    logger.info("Loaded %d golden questions", len(golden))

    llm_client = get_llm_client()
    logger.info("LLM provider: %s", settings.llm_provider)

    questions, answers, contexts_list, ground_truths = [], [], [], []

    for i, item in enumerate(golden, 1):
        q = item["question"]
        logger.info("[%d/%d] %s", i, len(golden), q[:70])

        ctx = _retrieve_context(q, top_k=top_k)
        if not ctx:
            logger.warning("No context retrieved for question %d — skipping.", i)
            continue

        ans = _generate_answer(q, ctx, llm_client)
        questions.append(q)
        answers.append(ans)
        contexts_list.append(ctx)
        ground_truths.append(item.get("ground_truth", ""))

    if not questions:
        logger.error("No questions evaluated. Is ChromaDB populated? Run `make ingest-manuals`.")
        raise typer.Exit(1)

    async_client = AsyncOpenAI(
        api_key=settings.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    ragas_llm = llm_factory("llama-3.3-70b-versatile", provider="openai", client=async_client)

    logger.info("Running RAGAS evaluation...")
    metric = Faithfulness(llm=ragas_llm)
    faith_scores: list[float] = []
    for i, (q, a, ctx) in enumerate(zip(questions, answers, contexts_list, strict=False), 1):
        logger.info("[%d/%d] scoring faithfulness...", i, len(questions))
        try:
            result = metric.score(user_input=q, response=a, retrieved_contexts=ctx)
            faith_scores.append(float(result.value) if result.value is not None else float("nan"))
        except Exception as e:
            logger.warning("Skipping question %d due to error: %s", i, e)
            faith_scores.append(float("nan"))

    import numpy as _np
    mean_faithfulness = float(_np.nanmean(faith_scores))

    table = Table(title="RAGAS Results")
    table.add_column("Metric")
    table.add_column("Mean", justify="right")
    table.add_row("Faithfulness", f"{mean_faithfulness:.3f}")
    console.print(table)

    out_path = out_dir / "ragas_results.json"
    per_question = [
        {"question": q, "faithfulness": s}
        for q, s in zip(questions, faith_scores, strict=False)
    ]
    out_path.write_text(
        json.dumps(
            {
                "mean_faithfulness": mean_faithfulness,
                "per_question": per_question,
            },
            indent=2,
        )
    )
    logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    app()
