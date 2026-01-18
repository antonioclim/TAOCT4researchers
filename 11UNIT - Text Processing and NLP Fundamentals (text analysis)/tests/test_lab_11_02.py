"""
Tests for Lab 11_02: NLP Fundamentals

This module provides comprehensive tests for the NLP fundamentals
laboratory components.

Run with: pytest tests/test_lab_11_02.py -v

Author: Antonio Clim
Version: 1.0.0
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    pass

# Import lab modules (path configured in conftest.py)
try:
    from lab_11_02_nlp_fundamentals import (
        WhitespaceTokeniser,
        RegexTokeniser,
        PorterStemmer,
        SnowballStemmer,
        Lemmatiser,
        StopwordFilter,
        FrequencyAnalyser,
        FrequencyAnalysis,
        NGramExtractor,
        TFIDFCalculator,
        TFIDFResult,
        SimpleSentimentAnalyser,
        TextAnalyser,
        TextAnalysisResult,
    )
    LAB_AVAILABLE = True
except ImportError:
    LAB_AVAILABLE = False


# =============================================================================
# TOKENISER TESTS
# =============================================================================

@pytest.mark.skipif(not LAB_AVAILABLE, reason="Lab module not available")
class TestWhitespaceTokeniser:
    """Tests for the WhitespaceTokeniser class."""
    
    def test_basic_tokenisation(self) -> None:
        """Test basic whitespace tokenisation."""
        tokeniser = WhitespaceTokeniser()
        tokens = tokeniser.tokenise("Hello World")
        assert tokens == ["Hello", "World"]
    
    def test_multiple_spaces(self) -> None:
        """Test handling of multiple spaces."""
        tokeniser = WhitespaceTokeniser()
        tokens = tokeniser.tokenise("Hello   World")
        assert tokens == ["Hello", "World"]
    
    def test_empty_string(self) -> None:
        """Test tokenising empty string."""
        tokeniser = WhitespaceTokeniser()
        tokens = tokeniser.tokenise("")
        assert tokens == []
    
    def test_preserves_punctuation(self) -> None:
        """Test that punctuation is preserved with words."""
        tokeniser = WhitespaceTokeniser()
        tokens = tokeniser.tokenise("Hello, World!")
        assert "Hello," in tokens


@pytest.mark.skipif(not LAB_AVAILABLE, reason="Lab module not available")
class TestRegexTokeniser:
    """Tests for the RegexTokeniser class."""
    
    def test_word_pattern(self) -> None:
        """Test default word pattern."""
        tokeniser = RegexTokeniser(r"\w+")
        tokens = tokeniser.tokenise("Hello, World!")
        assert tokens == ["Hello", "World"]
    
    def test_custom_pattern(self) -> None:
        """Test custom pattern."""
        tokeniser = RegexTokeniser(r"\d+")
        tokens = tokeniser.tokenise("Order 123 and 456")
        assert tokens == ["123", "456"]
    
    def test_no_matches(self) -> None:
        """Test when pattern doesn't match."""
        tokeniser = RegexTokeniser(r"\d+")
        tokens = tokeniser.tokenise("No numbers here")
        assert tokens == []


# =============================================================================
# STEMMER TESTS
# =============================================================================

@pytest.mark.skipif(not LAB_AVAILABLE, reason="Lab module not available")
class TestPorterStemmer:
    """Tests for the PorterStemmer class."""
    
    @pytest.fixture
    def stemmer(self) -> PorterStemmer:
        """Create a stemmer instance."""
        return PorterStemmer()
    
    @pytest.mark.nltk
    def test_stem_running(self, stemmer: PorterStemmer) -> None:
        """Test stemming 'running'."""
        try:
            result = stemmer.stem("running")
            assert result == "run"
        except ImportError:
            pytest.skip("NLTK not available")
    
    @pytest.mark.nltk
    def test_stem_tokens(self, stemmer: PorterStemmer) -> None:
        """Test stemming a list of tokens."""
        try:
            tokens = ["running", "runs", "ran"]
            result = stemmer.stem_tokens(tokens)
            assert len(result) == 3
        except ImportError:
            pytest.skip("NLTK not available")


# =============================================================================
# FREQUENCY ANALYSIS TESTS
# =============================================================================

@pytest.mark.skipif(not LAB_AVAILABLE, reason="Lab module not available")
class TestFrequencyAnalyser:
    """Tests for the FrequencyAnalyser class."""
    
    @pytest.fixture
    def analyser(self) -> FrequencyAnalyser:
        """Create an analyser instance."""
        return FrequencyAnalyser()
    
    def test_analyse_tokens(self, analyser: FrequencyAnalyser) -> None:
        """Test frequency analysis on token list."""
        tokens = ["the", "cat", "sat", "on", "the", "mat"]
        result = analyser.analyse(tokens)
        
        assert result.total_tokens == 6
        assert result.vocabulary_size == 5
        assert result.frequencies["the"] == 2
    
    def test_most_common(self, analyser: FrequencyAnalyser) -> None:
        """Test most common tokens."""
        tokens = ["a", "a", "a", "b", "b", "c"]
        result = analyser.analyse(tokens)
        
        most_common = result.most_common(2)
        assert most_common[0][0] == "a"
        assert most_common[0][1] == 3
    
    def test_relative_frequency(self, analyser: FrequencyAnalyser) -> None:
        """Test relative frequency calculation."""
        tokens = ["a", "b", "a", "a"]
        result = analyser.analyse(tokens)
        
        assert result.relative_frequency("a") == 0.75
    
    def test_empty_tokens(self, analyser: FrequencyAnalyser) -> None:
        """Test analysis of empty token list."""
        result = analyser.analyse([])
        
        assert result.total_tokens == 0
        assert result.vocabulary_size == 0


# =============================================================================
# N-GRAM TESTS
# =============================================================================

@pytest.mark.skipif(not LAB_AVAILABLE, reason="Lab module not available")
class TestNGramExtractor:
    """Tests for the NGramExtractor class."""
    
    def test_bigrams(self) -> None:
        """Test bigram extraction."""
        extractor = NGramExtractor(n=2)
        tokens = ["natural", "language", "processing"]
        bigrams = extractor.extract(tokens)
        
        assert ("natural", "language") in bigrams
        assert ("language", "processing") in bigrams
        assert len(bigrams) == 2
    
    def test_trigrams(self) -> None:
        """Test trigram extraction."""
        extractor = NGramExtractor(n=3)
        tokens = ["natural", "language", "processing", "is", "fun"]
        trigrams = extractor.extract(tokens)
        
        assert ("natural", "language", "processing") in trigrams
        assert len(trigrams) == 3
    
    def test_unigrams(self) -> None:
        """Test unigram extraction."""
        extractor = NGramExtractor(n=1)
        tokens = ["a", "b", "c"]
        unigrams = extractor.extract(tokens)
        
        assert len(unigrams) == 3
        assert all(len(ng) == 1 for ng in unigrams)
    
    def test_insufficient_tokens(self) -> None:
        """Test when tokens are fewer than n."""
        extractor = NGramExtractor(n=5)
        tokens = ["a", "b", "c"]
        result = extractor.extract(tokens)
        
        assert result == []
    
    def test_with_counts(self) -> None:
        """Test n-gram extraction with counts."""
        extractor = NGramExtractor(n=2)
        tokens = ["the", "cat", "the", "cat", "sat"]
        counts = extractor.extract_with_counts(tokens)
        
        assert counts[("the", "cat")] == 2
    
    def test_invalid_n(self) -> None:
        """Test that invalid n raises error."""
        with pytest.raises(ValueError):
            NGramExtractor(n=0)


# =============================================================================
# TF-IDF TESTS
# =============================================================================

@pytest.mark.skipif(not LAB_AVAILABLE, reason="Lab module not available")
class TestTFIDFCalculator:
    """Tests for the TFIDFCalculator class."""
    
    @pytest.fixture
    def calculator(self) -> TFIDFCalculator:
        """Create a calculator instance."""
        return TFIDFCalculator()
    
    @pytest.fixture
    def sample_corpus(self) -> list[list[str]]:
        """Provide sample corpus."""
        return [
            ["the", "cat", "sat", "on", "the", "mat"],
            ["the", "dog", "sat", "on", "the", "log"],
            ["the", "cat", "chased", "the", "dog"],
        ]
    
    def test_fit(self, calculator: TFIDFCalculator, sample_corpus: list[list[str]]) -> None:
        """Test fitting to corpus."""
        calculator.fit(sample_corpus)
        # No exception means success
        assert True
    
    def test_transform_without_fit(self, calculator: TFIDFCalculator) -> None:
        """Test transform without fit raises error."""
        with pytest.raises(ValueError):
            calculator.transform(0)
    
    def test_transform(self, calculator: TFIDFCalculator, sample_corpus: list[list[str]]) -> None:
        """Test TF-IDF transformation."""
        calculator.fit(sample_corpus)
        result = calculator.transform(0)
        
        assert isinstance(result, TFIDFResult)
        assert result.document_id == 0
        assert "cat" in result.scores
    
    def test_distinctive_terms(
        self,
        calculator: TFIDFCalculator,
        sample_corpus: list[list[str]]
    ) -> None:
        """Test that unique terms have higher TF-IDF."""
        calculator.fit(sample_corpus)
        result = calculator.transform(0)
        
        # "mat" only appears in doc 0, should have high score
        # "the" appears everywhere, should have lower score
        top_terms = result.top_terms(3)
        assert len(top_terms) <= 3
    
    def test_fit_transform(
        self,
        calculator: TFIDFCalculator,
        sample_corpus: list[list[str]]
    ) -> None:
        """Test fit_transform convenience method."""
        results = calculator.fit_transform(sample_corpus)
        
        assert len(results) == 3
        assert all(isinstance(r, TFIDFResult) for r in results)


# =============================================================================
# SENTIMENT ANALYSIS TESTS
# =============================================================================

@pytest.mark.skipif(not LAB_AVAILABLE, reason="Lab module not available")
class TestSimpleSentimentAnalyser:
    """Tests for the SimpleSentimentAnalyser class."""
    
    @pytest.fixture
    def analyser(self) -> SimpleSentimentAnalyser:
        """Create an analyser instance."""
        return SimpleSentimentAnalyser()
    
    def test_positive_sentiment(self, analyser: SimpleSentimentAnalyser) -> None:
        """Test positive sentiment detection."""
        tokens = ["wonderful", "amazing", "great"]
        score = analyser.analyse(tokens)
        assert score > 0
    
    def test_negative_sentiment(self, analyser: SimpleSentimentAnalyser) -> None:
        """Test negative sentiment detection."""
        tokens = ["terrible", "awful", "bad"]
        score = analyser.analyse(tokens)
        assert score < 0
    
    def test_neutral_sentiment(self, analyser: SimpleSentimentAnalyser) -> None:
        """Test neutral text."""
        tokens = ["the", "cat", "sat", "on", "mat"]
        score = analyser.analyse(tokens)
        assert score == 0
    
    def test_mixed_sentiment(self, analyser: SimpleSentimentAnalyser) -> None:
        """Test mixed positive and negative."""
        tokens = ["good", "bad"]
        score = analyser.analyse(tokens)
        assert score == 0  # Balanced
    
    def test_empty_tokens(self, analyser: SimpleSentimentAnalyser) -> None:
        """Test empty token list."""
        score = analyser.analyse([])
        assert score == 0.0


# =============================================================================
# TEXT ANALYSER TESTS
# =============================================================================

@pytest.mark.skipif(not LAB_AVAILABLE, reason="Lab module not available")
class TestTextAnalyser:
    """Tests for the TextAnalyser class."""
    
    def test_basic_analysis(self) -> None:
        """Test basic text analysis."""
        analyser = TextAnalyser(use_stemming=False, remove_stopwords=False)
        result = analyser.analyse("The quick brown fox")
        
        assert isinstance(result, TextAnalysisResult)
        assert len(result.tokens) > 0
    
    def test_with_stopword_removal(self) -> None:
        """Test analysis with stopword removal."""
        analyser = TextAnalyser(use_stemming=False, remove_stopwords=True)
        result = analyser.analyse("The quick brown fox")
        
        # Normalised tokens should be fewer due to stopword removal
        assert len(result.normalised_tokens) <= len(result.tokens)
    
    def test_result_contains_frequencies(self) -> None:
        """Test that result contains frequency analysis."""
        analyser = TextAnalyser()
        result = analyser.analyse("cat cat dog")
        
        assert isinstance(result.frequencies, FrequencyAnalysis)
    
    def test_result_contains_bigrams(self) -> None:
        """Test that result contains bigrams."""
        analyser = TextAnalyser()
        result = analyser.analyse("the quick brown fox")
        
        assert isinstance(result.bigrams, Counter)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

@pytest.mark.skipif(not LAB_AVAILABLE, reason="Lab module not available")
class TestNLPIntegration:
    """Integration tests for NLP pipeline."""
    
    def test_tokenise_and_count(self) -> None:
        """Test tokenisation followed by frequency counting."""
        tokeniser = RegexTokeniser(r"\w+")
        analyser = FrequencyAnalyser()
        
        text = "The cat sat on the mat"
        tokens = tokeniser.tokenise(text.lower())
        result = analyser.analyse(tokens)
        
        assert result.frequencies["the"] == 2
    
    def test_tokenise_and_ngrams(self) -> None:
        """Test tokenisation followed by n-gram extraction."""
        tokeniser = RegexTokeniser(r"\w+")
        extractor = NGramExtractor(n=2)
        
        text = "natural language processing"
        tokens = tokeniser.tokenise(text)
        bigrams = extractor.extract(tokens)
        
        assert ("natural", "language") in bigrams
    
    def test_full_pipeline(self) -> None:
        """Test complete NLP pipeline."""
        # This tests the full flow without NLTK dependency
        tokeniser = RegexTokeniser(r"\w+")
        analyser = FrequencyAnalyser()
        extractor = NGramExtractor(n=2)
        
        text = "Machine learning is great. Machine learning is powerful."
        tokens = tokeniser.tokenise(text.lower())
        
        freq = analyser.analyse(tokens)
        bigrams = extractor.extract_with_counts(tokens)
        
        assert freq.frequencies["machine"] == 2
        assert freq.frequencies["learning"] == 2
        assert bigrams[("machine", "learning")] == 2
