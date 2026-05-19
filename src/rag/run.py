#!/usr/bin/env python3
"""Command-line interface for single query RAG."""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import setting as settings
from src.rag import RAGEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="RAG Query CLI")
    parser.add_argument("question", type=str, help="Question to ask")
    parser.add_argument(
        "--llm",
        type=str,
        default=None,
        help="LLM model name"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )
    args = parser.parse_args()

    logger.info(f"Initializing RAG engine (all tools)...")
    engine = RAGEngine(
        llm_model=args.llm,
        verbose=args.verbose
    )
    engine.initialize()

    logger.info(f"Question: {args.question}")
    result = engine.query(args.question)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("\n" + "=" * 60)
        print(f"Answer: {result['answer']}")
        print("=" * 60)
        if args.verbose or result.get('trajectory'):
            print(f"\nLoops: {result['loops']}")
            print(f"Trajectory:")
            for step in result.get('trajectory', []):
                print(f"  [{step['loop']}] {step['tool_name']}: {step.get('arguments', {})}")
                if args.verbose:
                    print(f"      Result: {step.get('tool_result', '')[:200]}...")

    engine.close()


if __name__ == "__main__":
    main()