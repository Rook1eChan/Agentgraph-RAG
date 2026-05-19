#!/usr/bin/env python3
"""
Re-chunk corpus.json using sentence-level splitting + token merging.

Usage:
    python scripts/chunk_corpus.py
    python scripts/chunk_corpus.py --dataset hotpotqa --max-tokens 1000
"""

import sys
import json
import argparse
import logging
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import setting as settings
from src.chunking import SentenceChunker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Re-chunk corpus with sentence merging")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (overrides config)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=settings.CHUNK_MAX_TOKENS,
        help="Max tokens per chunk"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=settings.CHUNK_TOKENIZER_MODEL,
        help="Tokenizer model for tiktoken"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input corpus file (overrides config)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output corpus file (overrides config)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without writing"
    )
    args = parser.parse_args()

    if args.dataset:
        dataset = args.dataset
        corpus_file = f"{PROJECT_ROOT}/data/{dataset}/corpus.json"
        output_file = corpus_file
    else:
        dataset = settings.DATASET
        corpus_file = args.input or settings.DATA_CHUNKS_FILE
        output_file = args.output or settings.DATA_CHUNKS_FILE

    logger.info(f"Loading corpus from: {corpus_file}")
    with open(corpus_file, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    logger.info(f"Loaded {len(corpus)} documents")

    chunker = SentenceChunker(
        max_tokens=args.max_tokens,
        tokenizer_model=args.tokenizer
    )

    all_chunks = []
    doc_stats = []

    for doc in tqdm(corpus, desc="Chunking documents"):
        title = doc.get("title", doc.get("id", ""))
        text = doc.get("text", "")

        if not text.strip():
            doc_stats.append({"title": title, "chunks": 0, "skipped": True})
            continue

        chunks = chunker.chunk_text(text)

        for chunk in chunks:
            chunk["doc_title"] = title
            all_chunks.append(chunk)

        doc_stats.append({
            "title": title,
            "chunks": len(chunks),
            "num_tokens": sum(c["num_tokens"] for c in chunks),
        })

    total_tokens = sum(s["num_tokens"] for s in doc_stats)
    avg_tokens = total_tokens / len(doc_stats) if doc_stats else 0

    logger.info("=" * 60)
    logger.info(f"Re-chunking complete for dataset: {dataset}")
    logger.info(f"  Documents: {len(corpus)}")
    logger.info(f"  Total chunks: {len(all_chunks)}")
    logger.info(f"  Avg chunks per doc: {len(all_chunks) / len(corpus):.1f}")
    logger.info(f"  Avg tokens per chunk: {avg_tokens:.0f}")
    logger.info(f"  Total tokens: {total_tokens}")

    if args.dry_run:
        logger.info("Dry-run: no file written")
        for i, s in enumerate(doc_stats[:5]):
            logger.info(f"  [{i}] {s['title']}: {s['chunks']} chunks")
        if len(doc_stats) > 5:
            logger.info(f"  ... and {len(doc_stats) - 5} more documents")
        return

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False)

    logger.info(f"Written to: {output_file}")

    backup_path = output_path.with_suffix('.json.bak')
    if backup_path.exists():
        backup_path.unlink()
    with open(corpus_file, 'r', encoding='utf-8') as src:
        with open(backup_path, 'w', encoding='utf-8') as dst:
            dst.write(src.read())
    logger.info(f"Backup saved to: {backup_path}")


if __name__ == "__main__":
    main()