"""
Exercise: NLP Pipeline (Hard)

Build a complete NLP preprocessing pipeline.

Duration: 30-40 minutes
Difficulty: ★★★★☆

Author: Antonio Clim
Version: 1.0.0
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable


class PipelineStage(ABC):
    """Abstract base class for pipeline stages."""
    
    @abstractmethod
    def process(self, tokens: list[str]) -> list[str]:
        """Process tokens through this stage."""
        pass


class LowercaseStage(PipelineStage):
    """Convert tokens to lowercase."""
    
    def process(self, tokens: list[str]) -> list[str]:
        # TODO: Implement
        pass


class StopwordStage(PipelineStage):
    """Remove stopwords."""
    
    def __init__(self, stopwords: set[str]):
        self.stopwords = stopwords
    
    def process(self, tokens: list[str]) -> list[str]:
        # TODO: Implement
        pass


class MinLengthStage(PipelineStage):
    """Filter tokens by minimum length."""
    
    def __init__(self, min_length: int = 2):
        self.min_length = min_length
    
    def process(self, tokens: list[str]) -> list[str]:
        # TODO: Implement
        pass


@dataclass
class PipelineResult:
    original_tokens: list[str]
    processed_tokens: list[str]
    stages_applied: list[str]


class TextPipeline:
    """Configurable text processing pipeline."""
    
    def __init__(self):
        self._stages: list[tuple[str, PipelineStage]] = []
    
    def add_stage(self, name: str, stage: PipelineStage) -> "TextPipeline":
        """Add a processing stage (fluent interface)."""
        # TODO: Implement
        pass
    
    def process(self, tokens: list[str]) -> PipelineResult:
        """Run tokens through all stages."""
        # TODO: Implement
        pass


def run_tests() -> None:
    pipeline = TextPipeline()
    pipeline.add_stage("lowercase", LowercaseStage())
    pipeline.add_stage("minlength", MinLengthStage(3))
    
    result = pipeline.process(["The", "CAT", "is", "RUNNING"])
    print("Pipeline result:", result)
    print("All tests passed!")


if __name__ == "__main__":
    run_tests()
