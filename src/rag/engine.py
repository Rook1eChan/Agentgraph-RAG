"""Simplified RAG Engine for experimental use."""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import setting as settings
from src import BaseAgent, ToolRegistry
from src.core.llm import LLMClient
from src.core.context import AgentContext
from src.tools.read_chunk import ReadChunkTool
from src.tools.keyword_search import KeywordSearchTool
from src.tools.semantic_search import SemanticSearchTool
from src.tools.graph_hop import GraphHopTool
from src.agent.prompt import SYSTEM_PROMPT_ALL

logger = logging.getLogger(__name__)


class RAGEngine:
    """Simplified RAG engine for dataset experiments."""

    def __init__(
        self,
        llm_model: str = None,
        llm_api_key: str = None,
        llm_base_url: str = None,
        verbose: bool = False,
    ):
        self.llm_model = llm_model or settings.LLM_MODEL
        self.llm_api_key = llm_api_key or settings.LLM_API_KEY
        self.llm_base_url = llm_base_url or settings.LLM_BASE_URL
        self.verbose = verbose

        self.tools = None
        self.agent = None
        self.system_prompt = None
        self.llm_client = None

    def initialize(self):
        """Initialize tools and agent."""
        if self.agent is not None:
            return

        chunks_file = settings.DATA_CHUNKS_FILE
        index_dir = settings.DATA_INDEX_DIR
        graph_dir = f"{settings.DATA_GRAPH_DIR}/{settings.DATASET}"

        self.tools = ToolRegistry()
        self.tools.register(ReadChunkTool(chunks_file=chunks_file))

        # Always register keyword_search
        self.tools.register(KeywordSearchTool(
            chunks_file=chunks_file,
            graph_dir=graph_dir
        ))
        logger.info("KeywordSearch tool registered")

        # Register semantic_search if index exists
        index_file = Path(index_dir) / "sentence_index.pkl"
        if index_file.exists():
            self.tools.register(SemanticSearchTool(
                chunks_file=chunks_file,
                index_dir=index_dir,
                model_name=settings.EMBEDDING_MODEL,
                device=settings.EMBEDDING_DEVICE
            ))
            logger.info("SemanticSearch tool registered")
        else:
            logger.warning(f"Semantic index not found: {index_file}")

        # Register graph_hop if dual-graph exists
        dual_graph_meta = Path(graph_dir) / "dual_graph_meta.json"
        if dual_graph_meta.exists():
            self.tools.register(GraphHopTool(
                graph_dir=graph_dir,
                embedding_model=settings.EMBEDDING_MODEL,
                device=settings.EMBEDDING_DEVICE
            ))
            logger.info("GraphHop tool registered")
        else:
            logger.warning(f"Dual-graph not found: {dual_graph_meta}")

        self.system_prompt = SYSTEM_PROMPT_ALL

        self.llm_client = LLMClient(
            model=self.llm_model,
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            reasoning_effort=settings.LLM_REASONING_EFFORT,
        )

        self.agent = BaseAgent(
            llm_client=self.llm_client,
            tools=self.tools,
            system_prompt=self.system_prompt,
            max_loops=settings.AGENT_MAX_LOOPS,
            max_token_budget=settings.AGENT_MAX_TOKEN_BUDGET,
            verbose=self.verbose
        )

        logger.info("RAG Engine initialized")

    def query(self, question: str) -> Dict[str, Any]:
        """Execute RAG query."""
        if self.agent is None:
            self.initialize()

        try:
            result = self.agent.run(question)
            return {
                "answer": result.get("answer", ""),
                "success": True,
                "loops": result.get("loops", 0),
                "trajectory": result.get("trajectory", []),
                "total_retrieved_tokens": result.get("total_retrieved_tokens", 0),
                "retrieval_logs": result.get("retrieval_logs", []),
                "chunks_read_count": result.get("chunks_read_count", 0),
            }
        except Exception as e:
            logger.error(f"RAG query error: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "success": False,
                "error": str(e),
                "loops": 0,
                "trajectory": [],
            }

    def close(self):
        """Cleanup resources."""
        self.agent = None
        self.tools = None
        self.llm_client = None