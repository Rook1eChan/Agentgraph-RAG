#!/usr/bin/env python3
"""
Build sentence-level embedding index for semantic search.

Usage:
    python scripts/build_index.py

Configuration is read from config/setting.py
"""

import os
import sys
import json
import re
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import setting as settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = re.split(r'[.!?\n]+', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]


def build_index(
        corpus_file: str,
        output_dir: str,
        embedding_provider: str = None,
        model_name: str = None,
        device: str = None,
        batch_size: int = 32,
):
    """Build sentence-level embedding index."""

    logger.info(f"Loading corpus from: {corpus_file}")
    with open(corpus_file, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    logger.info(f"Loaded {len(corpus)} corpus items")

    chunk_lookup = {c['title']: c for c in corpus}

    logger.info("Extracting sentences...")
    sentences = []
    sentence_to_chunk = []

    for chunk in tqdm(corpus, desc="Processing corpus"):
        chunk_sentences = split_sentences(chunk['text'])
        for sent in chunk_sentences:
            sentences.append(sent)
            sentence_to_chunk.append(chunk['title'])

    logger.info(f"Total sentences: {len(sentences)}")

    if embedding_provider == "api":
        logger.info(f"Using API embedding: {model_name}")
        import openai
        client = openai.OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        def get_embeddings(texts, model=model_name):
            resp = client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]
        
        embeddings = []
        for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding"):
            batch = sentences[i:i+batch_size]
            embeddings.extend(get_embeddings(batch))
        embeddings = np.array(embeddings)
        actual_model_name = model_name
    else:
        logger.info(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name, device=device)
        logger.info(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

        embeddings = model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        actual_model_name = model_name

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    index_file = output_path / "sentence_index.pkl"
    index_data = {
        'sentences': sentences,
        'embeddings': embeddings,
        'sentence_to_chunk': sentence_to_chunk,
        'chunks': chunk_lookup,
        'model_name': actual_model_name
    }

    logger.info(f"Saving index to: {index_file}")
    with open(index_file, 'wb') as f:
        pickle.dump(index_data, f)

    logger.info(f"Index built successfully!")
    logger.info(f"  Corpus items: {len(corpus)}")
    logger.info(f"  Sentences: {len(sentences)}")
    logger.info(f"  Embedding dim: {embeddings.shape[1]}")


def main():
    logger.info(f"\n===== AgentGraph-RAG: Build Index =====")
    build_index(
        corpus_file=settings.DATA_CHUNKS_FILE,
        output_dir=settings.DATA_INDEX_DIR,
        embedding_provider=settings.EMBEDDING_PROVIDER,
        model_name=settings.EMBEDDING_MODEL,
        device=settings.EMBEDDING_DEVICE,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
    )


if __name__ == "__main__":
    main()
