"""Solutions for hard_03_corpus_analyser.py"""
import math
from collections import Counter
from dataclasses import dataclass

@dataclass
class CorpusStats:
    document_count: int
    total_tokens: int
    vocabulary_size: int
    avg_document_length: float
    most_common: list[tuple[str, int]]

@dataclass
class TFIDFVector:
    document_id: int
    scores: dict[str, float]
    
    def top_terms(self, n: int = 10) -> list[tuple[str, float]]:
        sorted_terms = sorted(self.scores.items(), key=lambda x: -x[1])
        return sorted_terms[:n]

class CorpusAnalyser:
    def __init__(self):
        self._documents: list[list[str]] = []
        self._idf: dict[str, float] = {}
    
    def add_document(self, tokens: list[str]) -> None:
        self._documents.append(tokens)
    
    def compute_stats(self) -> CorpusStats:
        all_tokens = [t for doc in self._documents for t in doc]
        freq = Counter(all_tokens)
        return CorpusStats(
            document_count=len(self._documents),
            total_tokens=len(all_tokens),
            vocabulary_size=len(freq),
            avg_document_length=len(all_tokens) / max(len(self._documents), 1),
            most_common=freq.most_common(10)
        )
    
    def compute_idf(self) -> None:
        n_docs = len(self._documents)
        vocab = set(t for doc in self._documents for t in doc)
        for term in vocab:
            df = sum(1 for doc in self._documents if term in doc)
            self._idf[term] = math.log(n_docs / (1 + df)) + 1
    
    def compute_tfidf(self, doc_index: int) -> TFIDFVector:
        if not self._idf:
            self.compute_idf()
        doc = self._documents[doc_index]
        tf = Counter(doc)
        doc_len = len(doc)
        scores = {t: (c / doc_len) * self._idf.get(t, 0) for t, c in tf.items()}
        return TFIDFVector(doc_index, scores)
    
    def document_similarity(self, doc1: int, doc2: int) -> float:
        v1 = self.compute_tfidf(doc1).scores
        v2 = self.compute_tfidf(doc2).scores
        terms = set(v1) | set(v2)
        dot = sum(v1.get(t, 0) * v2.get(t, 0) for t in terms)
        norm1 = math.sqrt(sum(v ** 2 for v in v1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in v2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
    
    def distinctive_terms(self, doc_index: int, n: int = 10) -> list[tuple[str, float]]:
        return self.compute_tfidf(doc_index).top_terms(n)
