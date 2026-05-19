"""Graph Hop tool - Graph traversal for multi-hop reasoning."""

import os
import logging
import threading
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from .base import BaseTool

if TYPE_CHECKING:
    from ..core.context import AgentContext

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class GraphHopTool(BaseTool):
    """Graph traversal tool - find co-occurring entities."""
    
    _embedding_lock = threading.Lock()
    
    def __init__(
        self,
        graph_dir: str = None,
        embedding_model: str = None,
        device: str = None,
    ):
        self.graph_dir = graph_dir
        self._embedding_model_name = embedding_model
        self.device = device
        
        self._retriever = None
        self._embedding_model = None
        
        if not HAS_TIKTOKEN:
            raise ImportError("tiktoken required. Install: pip install tiktoken")
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
    
    @property
    def retriever(self):
        """Lazy load graph retriever."""
        if self._retriever is None:
            if self.graph_dir and os.path.exists(self.graph_dir):
                from ..graph.dual_graph import DualGraphRetriever
                self._retriever = DualGraphRetriever(self.graph_dir)
            else:
                return None
        return self._retriever
    
    @property
    def name(self) -> str:
        return "graph_hop"
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "graph_hop",
                "description": """Graph traversal - find co-occurring entities in the knowledge graph.

This tool finds entities that have co-occurred with the given entity in the same document chunks. It is the core tool for multi-hop reasoning.

WORKFLOW:
1. Input: a known entity name (must be exact)
2. Find all entities that co-occurred with it in chunks
3. Rank by co-occurrence frequency and semantic relevance
4. Return top related entities with their shared chunk IDs

WHEN TO USE:
- Multi-hop questions requiring entity chains (e.g., "X's father of Y")
- When you have found an entity and need to find related entities
- For traversing entity relationships in the graph

OUTPUT:
Returns list of related entities with:
- Entity name
- Co-occurrence weight (how many times they appeared together)
- Shared_Chunk_IDs (chunks containing both entities)""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_name": {
                            "type": "string",
                            "description": "The exact entity name to query (must match entities in the graph)"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of top related entities to return (default: 5, max: 10)",
                            "default": 5
                        }
                    },
                    "required": ["entity_name"]
                }
            }
        }
    
    def execute(self, context: "AgentContext", entity_name: str, top_k: int = 5) -> Tuple[str, Dict[str, Any]]:
        logger.info(f"GraphHop Input: entity_name={entity_name}, top_k={top_k}")
        
        top_k = min(top_k, 10)
        
        if self.retriever is None:
            logger.info("GraphHop Output: Graph not available")
            return "Graph not available. Please run build_dual_graph.py first.", {
                "retrieved_tokens": 0,
                "entities_found": 0
            }
        
        cooccurring = self.retriever.get_cooccurrence_entities(entity_name, top_k=top_k * 2)
        
        if not cooccurring:
            return f"No co-occurring entities found for: {entity_name}", {
                "retrieved_tokens": 0,
                "entities_found": 0
            }
        
        result_parts = []
        for entity, weight in cooccurring[:top_k]:
            shared_chunks = self.retriever.get_shared_chunks(entity)
            shared_chunk_ids = [str(cid) for cid in shared_chunks[:5]]
            
            result_parts.append(
                f"Entity: {entity} (Co-occurrence weight: {weight})\n"
                f"Shared_Chunk_IDs: {', '.join(shared_chunk_ids)}"
            )
        
        tool_result = "\n\n".join(result_parts)
        
        context.add_retrieval_log(
            tool_name="graph_hop",
            tokens=0,
            metadata={
                "query_entity": entity_name,
                "entities_found": len(cooccurring)
            }
        )
        
        tool_log = {
            "retrieved_tokens": 0,
            "entities_found": len(cooccurring)
        }
        
        logger.info(f"GraphHop Output: entities_found={len(cooccurring)}")
        
        return tool_result, tool_log
