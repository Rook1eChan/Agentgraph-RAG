#!/usr/bin/env python3
"""
Build Dual-Layer Cognitive Navigation Graph from corpus.

Usage:
    python scripts/build_dual_graph.py
"""

import os
import sys
import logging
import importlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import setting as settings

dual_graph_module = importlib.import_module("src.graph.dual_graph")
build_dual_graph = dual_graph_module.build_dual_graph

logger = logging.getLogger(__name__)


def main():
    corpus_file = settings.DATA_CHUNKS_FILE
    output_dir = f"{settings.DATA_GRAPH_DIR}/{settings.DATASET}"

    logger.info(f"\n===== src: Build Dual-Layer Graph =====")
    logger.info(f"Building Dual-Layer Graph for dataset: {settings.DATASET}")
    logger.info(f"Corpus: {corpus_file}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)

    result = build_dual_graph(corpus_file, output_dir, spacy_model=settings.SPACY_MODEL)

    logger.info("=" * 60)
    logger.info("Dual-Layer Graph build complete!")
    logger.info(f"  Chunks: {result['chunks']}")
    logger.info(f"  Entities: {result['entities']}")
    logger.info(f"  Contains edges: {result['contains_edges']}")
    logger.info(f"  Co-occurrence edges: {result['cooccurrence_edges']}")


if __name__ == "__main__":
    main()
