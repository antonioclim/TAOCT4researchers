"""Solutions for hard_02_nlp_pipeline.py"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

class PipelineStage(ABC):
    @abstractmethod
    def process(self, tokens: list[str]) -> list[str]:
        pass

class LowercaseStage(PipelineStage):
    def process(self, tokens: list[str]) -> list[str]:
        return [t.lower() for t in tokens]

class StopwordStage(PipelineStage):
    def __init__(self, stopwords: set[str]):
        self.stopwords = stopwords
    
    def process(self, tokens: list[str]) -> list[str]:
        return [t for t in tokens if t.lower() not in self.stopwords]

class MinLengthStage(PipelineStage):
    def __init__(self, min_length: int = 2):
        self.min_length = min_length
    
    def process(self, tokens: list[str]) -> list[str]:
        return [t for t in tokens if len(t) >= self.min_length]

@dataclass
class PipelineResult:
    original_tokens: list[str]
    processed_tokens: list[str]
    stages_applied: list[str]

class TextPipeline:
    def __init__(self):
        self._stages: list[tuple[str, PipelineStage]] = []
    
    def add_stage(self, name: str, stage: PipelineStage) -> "TextPipeline":
        self._stages.append((name, stage))
        return self
    
    def process(self, tokens: list[str]) -> PipelineResult:
        original = tokens.copy()
        processed = tokens
        applied = []
        for name, stage in self._stages:
            processed = stage.process(processed)
            applied.append(name)
        return PipelineResult(original, processed, applied)
