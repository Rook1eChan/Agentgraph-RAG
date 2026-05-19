"""Dual-Layer Cognitive Navigation Graph Builder.

Graph Structure:
- Chunk Nodes: Document chunks with text
- Entity Nodes: Extracted entities (using spaCy NER)

Edges:
- Contains Edges: Chunk <-> Entity (entity appears in chunk)
- Co-occurrence Edges: Entity <-> Entity (co-appear in same chunk, weighted by count)
"""

import json
import os
import logging
import numpy as np
from scipy import sparse
from collections import defaultdict
import spacy
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DualGraphBuilder:
    """Build Dual-Layer Cognitive Navigation Graph."""
    
    def __init__(self, corpus_file: str, output_dir: str, spacy_model: str = "en_core_web_sm"):
        self.corpus_file = corpus_file
        self.output_dir = output_dir
        
        if spacy_model not in ["en_core_web_sm", "en_core_web_trf"]:
            logger.warning(f"Unknown spacy model '{spacy_model}', defaulting to en_core_web_sm")
            spacy_model = "en_core_web_sm"
        
        # Configure GPU for spaCy (try, but fallback to CPU if fails)
        import torch
        if spacy_model == "en_core_web_trf" and torch.cuda.is_available():
            try:
                spacy.require_gpu(0)
                logger.info(f"Using GPU for spaCy model: {spacy_model}")
            except Exception as e:
                logger.warning(f"GPU not available for spaCy, using CPU: {e}")
        
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            logger.warning(f"spaCy model '{spacy_model}' not found. Downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", spacy_model], check=True)
            self.nlp = spacy.load(spacy_model)
        
        self.chunks = []
        self.entities = {}
        self.chunk_to_entities = defaultdict(list)
        self.entity_cooccurrence = defaultdict(int)
        self.entity_to_idx = {}
        self.chunk_to_idx = {}
    
    def build(self):
        """Build the Dual-Layer Graph."""
        logger.info(f"Loading corpus from: {self.corpus_file}")
        
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
        
        logger.info(f"Processing {len(corpus)} chunks...")
        
        self._load_chunks(corpus)
        self._extract_entities()
        self._build_contains_edges()
        self._build_cooccurrence_edges()
        self._save()
        
        return {
            "chunks": len(self.chunks),
            "entities": len(self.entities),
            "contains_edges": sum(len(v) for v in self.chunk_to_entities.values()),
            "cooccurrence_edges": len(self.entity_cooccurrence)
        }
    
    def _load_chunks(self, corpus):
        """Load chunks from corpus."""
        for i, chunk in enumerate(corpus):
            self.chunks.append({
                "id": i,
                "title": chunk.get("title", ""),
                "text": chunk.get("text", "")
            })
            self.chunk_to_idx[chunk.get("title", f"chunk_{i}")] = i
            self.chunk_to_idx[str(i)] = i
        
        logger.info(f"  Loaded {len(self.chunks)} chunks")
    
    def _extract_entities(self):
        """Extract entities from chunks using spaCy NER."""
        entity_mentions = defaultdict(list)

        logger.info("  Extracting entities...")

        texts = [(chunk["text"], idx) for idx, chunk in enumerate(self.chunks)]

        for doc, idx in tqdm(
            self.nlp.pipe(texts, as_tuples=True, batch_size=64),
            total=len(texts),
            desc="  NER"
        ):
            chunk_entities = set()
            for ent in doc.ents:
                if len(ent.text) > 2 and ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'EVENT', 'WORK_OF_ART', 'DATE']:
                    entity_name = ent.text.strip()
                    chunk_entities.add(entity_name)

                    if entity_name not in self.entity_to_idx:
                        self.entity_to_idx[entity_name] = len(self.entity_to_idx)

                    entity_mentions[entity_name].append({
                        "chunk_id": idx,
                        "type": ent.label_
                    })

            self.chunk_to_entities[idx] = list(chunk_entities)

        for entity_name, mentions in entity_mentions.items():
            entity_id = self.entity_to_idx[entity_name]
            chunk_ids = list(set([m["chunk_id"] for m in mentions]))

            self.entities[entity_name] = {
                "id": entity_id,
                "name": entity_name,
                "type": mentions[0]["type"] if mentions else "UNKNOWN",
                "chunk_ids": chunk_ids,
                "mention_count": len(mentions)
            }

        logger.info(f"  Extracted {len(self.entities)} unique entities")
    
    def _build_contains_edges(self):
        """Build Contains Edges: Chunk <-> Entity."""
        logger.info("  Building Contains Edges...")
        
        contains_edge_count = 0
        for chunk_id, entities in self.chunk_to_entities.items():
            contains_edge_count += len(entities)
        
        logger.info(f"    Contains edges: {contains_edge_count}")
    
    def _build_cooccurrence_edges(self):
        """Build Co-occurrence Edges: Entity <-> Entity."""
        logger.info("  Building Co-occurrence Edges...")
        
        for chunk_id, entities in self.chunk_to_entities.items():
            entity_list = list(entities)
            for i in range(len(entity_list)):
                for j in range(i + 1, len(entity_list)):
                    e1, e2 = entity_list[i], entity_list[j]
                    if e1 != e2:
                        pair = tuple(sorted([e1, e2]))
                        self.entity_cooccurrence[pair] += 1
        
        cooccurrence_edges = {k: v for k, v in self.entity_cooccurrence.items() if v >= 1}
        self.entity_cooccurrence = cooccurrence_edges
        
        logger.info(f"    Co-occurrence edges: {len(self.entity_cooccurrence)}")
    
    def _save(self):
        """Save graph to files."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        meta = {
            "n_chunks": len(self.chunks),
            "n_entities": len(self.entities),
            "n_contains_edges": sum(len(v) for v in self.chunk_to_entities.values()),
            "n_cooccurrence_edges": len(self.entity_cooccurrence),
            "chunks": self.chunks,
            "entities": self.entities,
            "chunk_to_entities": {str(k): v for k, v in self.chunk_to_entities.items()},
            "entity_cooccurrence": {f"{k[0]}|||{k[1]}": v for k, v in self.entity_cooccurrence.items()}
        }
        
        meta_file = os.path.join(self.output_dir, "dual_graph_meta.json")
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        logger.info(f"  Graph saved to: {self.output_dir}")


class DualGraphRetriever:
    """Retriever for Dual-Layer Graph."""
    
    def __init__(self, graph_dir: str):
        self.graph_dir = graph_dir
        
        meta_file = os.path.join(graph_dir, "dual_graph_meta.json")
        with open(meta_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        self.chunks = meta["chunks"]
        self.entities = meta["entities"]
        self.chunk_to_entities = {int(k): v for k, v in meta["chunk_to_entities"].items()}
        
        self.entity_cooccurrence = {}
        for k, v in meta["entity_cooccurrence"].items():
            parts = k.split("|||")
            self.entity_cooccurrence[(parts[0], parts[1])] = v
        
        self.entity_name_to_idx = {e["name"]: e["id"] for e in self.entities.values()}
        self.chunk_id_to_title = {c["id"]: c["title"] for c in self.chunks}
    
    def get_cooccurrence_entities(self, entity_name: str, top_k: int = 10):
        """Get co-occurring entities for a given entity."""
        if entity_name not in self.entity_name_to_idx:
            return []
        
        cooccurring = []
        for (e1, e2), weight in self.entity_cooccurrence.items():
            if e1 == entity_name:
                cooccurring.append((e2, weight))
            elif e2 == entity_name:
                cooccurring.append((e1, weight))
        
        cooccurring.sort(key=lambda x: -x[1])
        return cooccurring[:top_k]
    
    def get_shared_chunks(self, entity_name: str):
        """Get chunk IDs that contain the given entity."""
        if entity_name in self.entity_name_to_idx:
            entity_id = self.entity_name_to_idx[entity_name]
            return self.entities[entity_name].get("chunk_ids", [])
        return []
    
    def get_chunk_entities(self, chunk_id: int):
        """Get entities contained in a chunk."""
        return self.chunk_to_entities.get(chunk_id, [])


def build_dual_graph(corpus_file: str, output_dir: str, spacy_model: str = "en_core_web_sm"):
    """Build Dual-Layer Graph."""
    builder = DualGraphBuilder(corpus_file, output_dir, spacy_model=spacy_model)
    return builder.build()


def load_dual_graph(graph_dir: str):
    """Load Dual-Layer Graph."""
    return DualGraphRetriever(graph_dir)
