"""Keyword search tool - entity-based matching via dual-graph."""

import json
import logging
import re
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

EXACT_WEIGHT = 0.7
INCLUDE_WEIGHT = 0.3


class KeywordSearchTool(BaseTool):
    """Keyword search using entity matching via dual-graph."""
    
    def __init__(self, chunks_file: str, graph_dir: str):
        self.chunks_file = chunks_file
        self.chunks = self._load_chunks()
        
        if not HAS_TIKTOKEN:
            raise ImportError("tiktoken required. Install: pip install tiktoken")
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        
        graph_file = f"{graph_dir}/dual_graph_meta.json"
        with open(graph_file, 'r', encoding='utf-8') as f:
            self.graph_data = json.load(f)
        
        self.chunk_to_entities = self.graph_data.get('chunk_to_entities', {})
        
        logger.info(f"KeywordSearchTool: loaded {len(self.chunks)} chunks, {len(self.chunk_to_entities)} chunk-entity mappings")
    
    def _load_chunks(self) -> List[Dict[str, Any]]:
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if data and isinstance(data[0], dict):
            return data
        
        chunks = []
        for item in data:
            if isinstance(item, str):
                parts = item.split(':', 1)
                if len(parts) == 2:
                    chunks.append({'id': parts[0], 'title': parts[0], 'text': parts[1]})
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'[.!?\n]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _match_entities(self, keywords: List[str]) -> Dict[int, float]:
        """Match keywords against entities in each chunk."""
        chunk_scores = {}
        
        for chunk_idx_str, entity_names in self.chunk_to_entities.items():
            chunk_idx = int(chunk_idx_str)
            score = 0
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                
                for ent in entity_names:
                    ent_lower = ent.lower()
                    
                    if keyword_lower == ent_lower:
                        score += EXACT_WEIGHT * len(keyword)
                    elif keyword_lower in ent_lower or ent_lower in keyword_lower:
                        score += INCLUDE_WEIGHT * len(keyword)
            
            if score > 0:
                chunk_scores[chunk_idx] = score
        
        return chunk_scores
    
    @property
    def name(self) -> str:
        return "keyword_search"
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "keyword_search",
                "description": """Search for document chunks using keyword-based entity matching (case-insensitive). 

The system first matches your keywords against entities extracted from documents, then finds chunks containing those entities.

IMPORTANT: Use SHORT, SPECIFIC terms (1-3 words maximum). Each keyword is matched independently.

Examples of GOOD keywords:
  - Entity names: "Albert Einstein", "Tesla", "Python", "Argentina"
  - Technical terms: "photosynthesis", "quantum mechanics"
  - Key concepts: "climate change", "GDP growth"

Examples of BAD keywords (DO NOT use):
  - Long phrases: "the person who invented the telephone" → use "Alexander Bell" instead
  - Questions: "when did World War 2 start" → use "World War 2", "1939" instead
  - Descriptions: "the country between France and Spain" → use "Andorra" instead
  - Full sentences: "how does the stock market work" → use "stock market", "trading" instead

RETURNS: Abbreviated snippets showing where matched entities appear in chunks. Use read_chunk to get full content.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of keywords to search. Each keyword should be 1-3 words maximum (e.g., ['Einstein', 'relativity theory', '1905'])."
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of top-ranked chunks to return (default: 5, max: 20)",
                            "default": 5
                        }
                    },
                    "required": ["keywords"]
                }
            }
        }
    
    def execute(self, context: 'AgentContext', keywords: List[str], top_k: int = 5) -> Tuple[str, Dict[str, Any]]:
        logger.debug(f"KeywordSearch Input: keywords={keywords}, top_k={top_k}")
        
        top_k = min(top_k, 20)
        
        chunk_scores = self._match_entities(keywords)
        
        if not chunk_scores:
            tool_result = f"No results found for keywords: {keywords}"
            tool_log = {"retrieved_tokens": 0, "chunks_found": 0}
            return tool_result, tool_log
        
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        top_chunk_idxs = [idx for idx, score in sorted_chunks[:top_k]]
        
        scored_chunks = []
        for idx in top_chunk_idxs:
            chunk = self.chunks[idx]
            scored_chunks.append({
                'chunk_id': chunk['title'],
                'score': chunk_scores[idx],
                'text': chunk['text']
            })
        
        result_parts = []
        for item in scored_chunks:
            sentences = self._split_sentences(item['text'])
            matched_sentences = []
            
            keywords_lower = [k.lower() for k in keywords]
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(kw in sentence_lower for kw in keywords_lower):
                    matched_sentences.append(sentence)
            
            if matched_sentences:
                matched_text = "... " + " ... ".join(matched_sentences[:5]) + " ..."
            else:
                matched_text = "(no exact sentence match)"
            
            result_parts.append(f"Chunk ID: {item['chunk_id']}, Score: {item['score']:.2f}, Matched: {matched_text}")
        
        tool_result = "\n\n".join(result_parts)
        
        all_matched_sentences = []
        for item in scored_chunks:
            sentences = self._split_sentences(item['text'])
            for sentence in sentences:
                if any(kw in sentence.lower() for kw in [k.lower() for k in keywords]):
                    all_matched_sentences.append(sentence)
        
        if all_matched_sentences:
            sentences_text = "\n".join(all_matched_sentences)
            retrieved_tokens = len(self.tokenizer.encode(sentences_text))
        else:
            retrieved_tokens = 0
        
        context.add_retrieval_log(
            tool_name="keyword_search",
            tokens=retrieved_tokens,
            metadata={
                "keywords": keywords,
                "chunks_found": len(scored_chunks),
                "chunk_ids": [c['chunk_id'] for c in scored_chunks]
            }
        )
        
        tool_log = {"retrieved_tokens": retrieved_tokens, "chunks_found": len(scored_chunks)}
        
        logger.info(f"KeywordSearch Output: chunks_found={len(scored_chunks)}, retrieved_tokens={retrieved_tokens}")
        
        return tool_result, tool_log