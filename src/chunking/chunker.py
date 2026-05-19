"""Sentence-level chunker with token-aware merging."""

import re
import logging
from typing import List, Dict, Any, Tuple

import tiktoken

logger = logging.getLogger(__name__)


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using Chinese and English punctuation."""
    sentences = re.split(r'[。！？.!?\n]+', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in text using tiktoken."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


class SentenceChunker:
    """Split text into chunks by sentence, merging until token threshold."""

    def __init__(self, max_tokens: int = 1000, tokenizer_model: str = "gpt-4o"):
        self.max_tokens = max_tokens
        self.tokenizer_model = tokenizer_model
        try:
            self.enc = tiktoken.encoding_for_model(tokenizer_model)
        except Exception:
            self.enc = tiktoken.get_encoding("cl100k_base")
        logger.info(f"SentenceChunker initialized: max_tokens={max_tokens}, model={tokenizer_model}")

    def _split_long_sentence(self, sent: str, max_sent_tokens: int = 800) -> List[str]:
        """Force-split a long sentence into smaller parts."""
        enc = self.enc
        words = sent.split()
        parts, current = [], []

        for word in words:
            test = " ".join(current + [word])
            if len(enc.encode(test)) <= max_sent_tokens:
                current.append(word)
            else:
                if current:
                    parts.append(" ".join(current))
                    current = [word]
                else:
                    current.append(word)

        if current:
            parts.append(" ".join(current))
        return parts

    def chunk(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """Merge sentences into chunks respecting token limit.

        Args:
            sentences: List of sentence strings.

        Returns:
            List of chunk dicts with keys: id, title, text
        """
        chunks = []
        current_chunk_sentences = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = len(self.enc.encode(sent))

            if sent_tokens > self.max_tokens * 0.9:
                if current_chunk_sentences:
                    chunks.append({
                        "id": len(chunks),
                        "title": current_chunk_sentences[0][:50],
                        "text": " ".join(current_chunk_sentences),
                        "num_sentences": len(current_chunk_sentences),
                        "num_tokens": current_tokens,
                    })
                    current_chunk_sentences = []
                    current_tokens = 0

                sub_parts = self._split_long_sentence(sent, int(self.max_tokens * 0.8))
                for part in sub_parts:
                    part_tokens = len(self.enc.encode(part))
                    if current_tokens + part_tokens + 1 <= self.max_tokens:
                        current_chunk_sentences.append(part)
                        current_tokens += part_tokens + 1
                    else:
                        chunks.append({
                            "id": len(chunks),
                            "title": current_chunk_sentences[0][:50] if current_chunk_sentences else part[:50],
                            "text": " ".join(current_chunk_sentences),
                            "num_sentences": len(current_chunk_sentences),
                            "num_tokens": current_tokens,
                        })
                        current_chunk_sentences = [part]
                        current_tokens = part_tokens

            elif not current_chunk_sentences:
                current_chunk_sentences.append(sent)
                current_tokens = sent_tokens
            elif current_tokens + sent_tokens + 1 <= self.max_tokens:
                current_chunk_sentences.append(sent)
                current_tokens += sent_tokens + 1
            else:
                chunks.append({
                    "id": len(chunks),
                    "title": current_chunk_sentences[0][:50],
                    "text": " ".join(current_chunk_sentences),
                    "num_sentences": len(current_chunk_sentences),
                    "num_tokens": current_tokens,
                })
                current_chunk_sentences = [sent]
                current_tokens = sent_tokens

        if current_chunk_sentences:
            chunks.append({
                "id": len(chunks),
                "title": current_chunk_sentences[0][:50],
                "text": " ".join(current_chunk_sentences),
                "num_sentences": len(current_chunk_sentences),
                "num_tokens": current_tokens,
            })

        return chunks

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Split and chunk a single text document.

        Args:
            text: Raw document text.

        Returns:
            List of chunk dicts.
        """
        sentences = split_sentences(text)
        return self.chunk(sentences)


def chunk_text(text: str, max_tokens: int = 1000, tokenizer_model: str = "gpt-4o") -> List[Dict[str, Any]]:
    """Convenience function for chunking a single text."""
    chunker = SentenceChunker(max_tokens=max_tokens, tokenizer_model=tokenizer_model)
    return chunker.chunk_text(text)