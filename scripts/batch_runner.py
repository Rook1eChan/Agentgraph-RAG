#!/usr/bin/env python3
"""
Batch Runner for AgentGraph-RAG

Usage:
    python scripts/batch_runner.py
"""

import os
import sys
import json
import logging
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import setting as settings

from src import BaseAgent, ToolRegistry
from src.core.llm import LLMClient
from src.tools.semantic_search import SemanticSearchTool
from src.tools.read_chunk import ReadChunkTool
from src.tools.graph_hop import GraphHopTool
from src.tools.keyword_search import KeywordSearchTool
from src.agent.prompt import SYSTEM_PROMPT_ALL

logger = logging.getLogger(__name__)


class BatchRunner:
    """Batch runner with concurrent execution and checkpoint resume."""

    def __init__(
            self,
            verbose: bool = False,
            workers: int = None,
            limit: int = None,
    ):
        self.verbose = verbose

        self.questions_file = Path(settings.DATA_QUESTIONS_FILE)
        self.output_dir = Path(settings.get_output_dir())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.limit = limit or settings.BATCH_LIMIT
        self.num_workers = workers or settings.BATCH_WORKERS

        self.predictions_file = self.output_dir / "predictions.jsonl"
        self.write_lock = Lock()

        self.questions = self._load_questions()

        self._shared_tools, self._system_prompt = self._init_shared_tools()

    def _init_shared_tools(self) -> tuple:
        """Initialize shared tools (embedding model loaded only once)."""
        chunks_file = settings.DATA_CHUNKS_FILE
        index_dir = settings.DATA_INDEX_DIR
        graph_dir = f"{settings.DATA_GRAPH_DIR}/{settings.DATASET}"

        tools = ToolRegistry()
        tools.register(ReadChunkTool(chunks_file=chunks_file))

        # Always register keyword_search
        tools.register(KeywordSearchTool(
            chunks_file=chunks_file,
            graph_dir=graph_dir
        ))
        logger.info("KeywordSearch tool registered")

        # Register semantic_search if index exists
        index_file = Path(index_dir) / "sentence_index.pkl"
        if index_file.exists():
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            tools.register(SemanticSearchTool(
                chunks_file=chunks_file,
                index_dir=index_dir,
                model_name=settings.EMBEDDING_MODEL,
                device=settings.EMBEDDING_DEVICE
            ))
            logger.info("SemanticSearch tool registered")
        else:
            logger.info(f"Semantic index not found: {index_file}")

        # Register graph_hop if dual-graph exists
        dual_graph_file = Path(graph_dir) / "dual_graph_meta.json"
        if dual_graph_file.exists():
            logger.info(f"Loading Dual-Graph from: {graph_dir}")
            tools.register(GraphHopTool(
                graph_dir=graph_dir,
                embedding_model=settings.EMBEDDING_MODEL,
                device=settings.EMBEDDING_DEVICE
            ))
            logger.info("GraphHop tool registered")
        else:
            logger.info(f"Dual-graph not found: {dual_graph_file}")

        return tools, SYSTEM_PROMPT_ALL

    def _load_questions(self) -> List[Dict[str, Any]]:
        """Load questions from file."""
        with open(self.questions_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)

        if self.limit:
            questions = questions[:self.limit]

        return questions

    def _load_completed_qids(self) -> set:
        """Load completed question IDs for checkpoint resume."""
        completed_qids = set()

        if not self.predictions_file.exists():
            return completed_qids

        try:
            with open(self.predictions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if 'question' in data and 'pred_answer' in data:
                            qid = data.get('qid') or data.get('id')
                            if qid is not None and data.get('answered', True):
                                completed_qids.add(qid)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.info(f"Warning: Error loading completed data: {e}")

        return completed_qids

    def _append_prediction(self, prediction: Dict[str, Any]):
        """Append prediction to file (thread-safe), overwriting existing qid."""
        qid = prediction.get('qid')

        existing = []
        if self.predictions_file.exists():
            with open(self.predictions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if data.get('qid') != qid:
                            existing.append(data)

        existing.append(prediction)

        with self.write_lock:
            with open(self.predictions_file, 'w', encoding='utf-8') as f:
                for item in existing:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def _create_llm_client(self):
        """Create LLM client based on configuration."""
        api_key = settings.LLM_API_KEY
        if not api_key:
            raise ValueError("API key required. Set in /config/setting.py")

        return LLMClient(
            model=settings.LLM_MODEL,
            api_key=api_key,
            base_url=settings.LLM_BASE_URL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            reasoning_effort=settings.LLM_REASONING_EFFORT,
        )

    def _create_agent(self) -> BaseAgent:
        """Create agent instance with shared tools."""
        client = self._create_llm_client()

        return BaseAgent(
            llm_client=client,
            tools=self._shared_tools,
            system_prompt=self._system_prompt,
            max_loops=settings.AGENT_MAX_LOOPS,
            max_token_budget=settings.AGENT_MAX_TOKEN_BUDGET,
            verbose=self.verbose
        )

    def _process_one(self, item: Dict[str, Any], agent: BaseAgent) -> Dict[str, Any]:
        """Process one question."""
        qid = item.get('qid') or item.get('id')
        question = item.get('question', '')
        gold_answer = item.get('answer', item.get('gold_answer', ''))
        supporting_facts = item.get('supporting_facts', [])

        try:
            result = agent.run(question)

            return {
                'qid': qid,
                'question': question,
                'trajectory': result['trajectory'],
                'messages': result['messages'],
                'gold_answer': gold_answer,
                'pred_answer': result['answer'],
                'loops': result['loops'],
                'total_retrieved_tokens': result.get('total_retrieved_tokens', 0),
                'retrieval_logs': result.get('retrieval_logs', []),
                'chunks_read_count': result.get('chunks_read_count', 0),
                'chunks_read_ids': result.get('chunks_read_ids', []),
                'supporting_facts': supporting_facts,
                'answered': result.get('answered', True)
            }
        except Exception as e:
            return {
                'qid': qid,
                'question': question,
                'trajectory': [],
                'gold_answer': gold_answer,
                'pred_answer': f"Error: {str(e)}",
                'loops': 0,
                'total_retrieved_tokens': 0,
                'retrieval_logs': [],
                'chunks_read_count': 0,
                'chunks_read_ids': [],
                'supporting_facts': supporting_facts,
                'error': str(e),
                'answered': False
            }

    def run(self):
        """Run batch processing."""
        completed_qids = self._load_completed_qids()

        pending = [q for q in self.questions
                   if (q.get('qid') or q.get('id')) not in completed_qids]

        logger.info(f"Total questions: {len(self.questions)}")
        logger.info(f"Completed: {len(completed_qids)}")
        logger.info(f"Pending: {len(pending)}")

        if not pending:
            logger.info("All questions completed!")
            return

        logger.info(f"Starting with {self.num_workers} workers...")

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {}

            for item in pending:
                agent = self._create_agent()
                future = executor.submit(self._process_one, item, agent)
                futures[future] = item.get('qid') or item.get('id')

            with tqdm(total=len(pending), desc="Processing") as pbar:
                for future in as_completed(futures):
                    qid = futures[future]
                    try:
                        result = future.result()
                        self._append_prediction(result)
                    except Exception as e:
                        logger.info(f"Error processing {qid}: {e}")
                    pbar.update(1)

        logger.info(f"\nResults saved to: {self.predictions_file}")


def main():
    logger.info(f"\n===== AgentGraph-RAG: Batch Runner =====")
    runner = BatchRunner(
        verbose=settings.AGENT_VERBOSE,
        workers=settings.BATCH_WORKERS,
        limit=settings.BATCH_LIMIT,
    )

    runner.run()


if __name__ == "__main__":
    main()
