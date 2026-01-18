"""
Lab 11_02: NLP Fundamentals

This laboratory introduces core Natural Language Processing concepts
using NLTK and spaCy libraries.

Sections:
    §1. Tokenisation (~80 lines)
    §2. Normalisation (~80 lines)
    §3. Text Features (~100 lines)
    §4. Basic NLP Tasks (~90 lines)

Learning Objectives:
    LO3: Build text preprocessing pipelines: tokenisation, normalisation, stopwords
    LO4: Apply NLP techniques: stemming, lemmatisation, POS tagging, n-grams
    LO5: Analyse text corpora using frequency analysis and TF-IDF

Duration: 40 minutes
Difficulty: ★★★★☆

Dependencies:
    - nltk>=3.8
    - Run: python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); 
           nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"

Author: Antonio Clim
Version: 1.0.0
Date: January 2025
"""

from __future__ import annotations

import logging
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# §1. TOKENISATION
# =============================================================================

class Tokeniser(ABC):
    """
    Abstract base class for tokenisers.
    
    Defines the interface for all tokeniser implementations.
    """
    
    @abstractmethod
    def tokenise(self, text: str) -> list[str]:
        """
        Tokenise text into a list of tokens.
        
        Args:
            text: The input text to tokenise.
        
        Returns:
            List of tokens.
        """
        pass


class WhitespaceTokeniser(Tokeniser):
    """
    Simple whitespace-based tokeniser.
    
    Splits text on whitespace characters. Fast but naive;
    does not handle punctuation or contractions.
    
    Example:
        >>> tokeniser = WhitespaceTokeniser()
        >>> tokeniser.tokenise("Hello, world!")
        ['Hello,', 'world!']
    """
    
    def tokenise(self, text: str) -> list[str]:
        """Split on whitespace."""
        return text.split()


class RegexTokeniser(Tokeniser):
    """
    Regex-based tokeniser with configurable pattern.
    
    Uses regular expressions to extract tokens, providing
    more control than whitespace splitting.
    
    Attributes:
        pattern: Compiled regex pattern for token matching.
    
    Example:
        >>> tokeniser = RegexTokeniser(r'\\w+')
        >>> tokeniser.tokenise("Hello, world!")
        ['Hello', 'world']
    """
    
    def __init__(self, pattern: str = r"\w+") -> None:
        """
        Initialise with regex pattern.
        
        Args:
            pattern: Regex pattern to match tokens.
        """
        self.pattern = re.compile(pattern)
        logger.debug("RegexTokeniser initialised with pattern: %s", pattern)
    
    def tokenise(self, text: str) -> list[str]:
        """Extract tokens matching the pattern."""
        return self.pattern.findall(text)


class WordTokeniser(Tokeniser):
    """
    Word tokeniser using NLTK's word_tokenize.
    
    Handles contractions, punctuation and other edge cases
    better than simple approaches.
    
    Example:
        >>> tokeniser = WordTokeniser()
        >>> tokeniser.tokenise("Don't stop!")
        ['Do', "n't", 'stop', '!']
    """
    
    def __init__(self) -> None:
        """Initialise word tokeniser (lazy NLTK import)."""
        self._tokenize: Callable[[str], list[str]] | None = None
    
    def _ensure_nltk(self) -> None:
        """Lazy import of NLTK tokeniser."""
        if self._tokenize is None:
            try:
                from nltk.tokenize import word_tokenize
                self._tokenize = word_tokenize
            except ImportError as e:
                logger.error("NLTK not available: %s", e)
                raise
    
    def tokenise(self, text: str) -> list[str]:
        """Tokenise using NLTK word_tokenize."""
        self._ensure_nltk()
        assert self._tokenize is not None
        return self._tokenize(text)


class SentenceTokeniser(Tokeniser):
    """
    Sentence tokeniser using NLTK's sent_tokenize.
    
    Handles abbreviations, decimal points and other
    challenges in sentence boundary detection.
    
    Example:
        >>> tokeniser = SentenceTokeniser()
        >>> tokeniser.tokenise("Hello. How are you?")
        ['Hello.', 'How are you?']
    """
    
    def __init__(self) -> None:
        """Initialise sentence tokeniser."""
        self._tokenize: Callable[[str], list[str]] | None = None
    
    def _ensure_nltk(self) -> None:
        """Lazy import of NLTK tokeniser."""
        if self._tokenize is None:
            try:
                from nltk.tokenize import sent_tokenize
                self._tokenize = sent_tokenize
            except ImportError as e:
                logger.error("NLTK not available: %s", e)
                raise
    
    def tokenise(self, text: str) -> list[str]:
        """Tokenise into sentences."""
        self._ensure_nltk()
        assert self._tokenize is not None
        return self._tokenize(text)


# =============================================================================
# §2. NORMALISATION
# =============================================================================

class Stemmer(ABC):
    """Abstract base class for stemmers."""
    
    @abstractmethod
    def stem(self, word: str) -> str:
        """Reduce word to its stem."""
        pass
    
    def stem_tokens(self, tokens: list[str]) -> list[str]:
        """Stem a list of tokens."""
        return [self.stem(token) for token in tokens]


class PorterStemmer(Stemmer):
    """
    Porter stemmer wrapper for NLTK.
    
    Applies the Porter stemming algorithm to reduce words
    to their stems through suffix stripping rules.
    
    Example:
        >>> stemmer = PorterStemmer()
        >>> stemmer.stem("running")
        'run'
    """
    
    def __init__(self) -> None:
        """Initialise Porter stemmer."""
        self._stemmer = None
    
    def _ensure_nltk(self) -> None:
        """Lazy import of NLTK stemmer."""
        if self._stemmer is None:
            try:
                from nltk.stem import PorterStemmer as NLTKPorter
                self._stemmer = NLTKPorter()
            except ImportError as e:
                logger.error("NLTK not available: %s", e)
                raise
    
    def stem(self, word: str) -> str:
        """Apply Porter stemming."""
        self._ensure_nltk()
        return self._stemmer.stem(word)


class SnowballStemmer(Stemmer):
    """
    Snowball stemmer with multilingual support.
    
    Attributes:
        language: Target language for stemming.
    
    Example:
        >>> stemmer = SnowballStemmer("english")
        >>> stemmer.stem("universities")
        'univers'
    """
    
    def __init__(self, language: str = "english") -> None:
        """
        Initialise Snowball stemmer.
        
        Args:
            language: Language for stemming rules.
        """
        self.language = language
        self._stemmer = None
    
    def _ensure_nltk(self) -> None:
        """Lazy import of NLTK stemmer."""
        if self._stemmer is None:
            try:
                from nltk.stem import SnowballStemmer as NLTKSnowball
                self._stemmer = NLTKSnowball(self.language)
            except ImportError as e:
                logger.error("NLTK not available: %s", e)
                raise
    
    def stem(self, word: str) -> str:
        """Apply Snowball stemming."""
        self._ensure_nltk()
        return self._stemmer.stem(word)


class Lemmatiser:
    """
    WordNet-based lemmatiser.
    
    Reduces words to their lemmas (dictionary forms) using
    WordNet. More accurate than stemming but slower.
    
    Example:
        >>> lemmatiser = Lemmatiser()
        >>> lemmatiser.lemmatise("better", pos="a")
        'good'
    """
    
    POS_MAP = {
        "n": "n",  # noun
        "v": "v",  # verb
        "a": "a",  # adjective
        "r": "r",  # adverb
    }
    
    def __init__(self) -> None:
        """Initialise lemmatiser."""
        self._lemmatizer = None
    
    def _ensure_nltk(self) -> None:
        """Lazy import of NLTK lemmatiser."""
        if self._lemmatizer is None:
            try:
                from nltk.stem import WordNetLemmatizer
                self._lemmatizer = WordNetLemmatizer()
            except ImportError as e:
                logger.error("NLTK not available: %s", e)
                raise
    
    def lemmatise(self, word: str, pos: str = "n") -> str:
        """
        Lemmatise a word.
        
        Args:
            word: The word to lemmatise.
            pos: Part of speech (n=noun, v=verb, a=adj, r=adv).
        
        Returns:
            The lemma (base form).
        """
        self._ensure_nltk()
        return self._lemmatizer.lemmatize(word, pos=pos)
    
    def lemmatise_tokens(self, tokens: list[str], pos: str = "n") -> list[str]:
        """Lemmatise a list of tokens."""
        return [self.lemmatise(token, pos) for token in tokens]


class StopwordFilter:
    """
    Stopword filtering for text preprocessing.
    
    Removes common function words that carry little
    semantic content.
    
    Attributes:
        stopwords: Set of stopwords to filter.
        language: Language of stopword list.
    
    Example:
        >>> filt = StopwordFilter()
        >>> filt.filter_tokens(["the", "cat", "is", "running"])
        ['cat', 'running']
    """
    
    def __init__(self, language: str = "english", additional: set[str] | None = None) -> None:
        """
        Initialise stopword filter.
        
        Args:
            language: Language for stopword list.
            additional: Additional custom stopwords.
        """
        self.language = language
        self._stopwords: set[str] | None = None
        self._additional = additional or set()
    
    def _ensure_nltk(self) -> None:
        """Lazy import of NLTK stopwords."""
        if self._stopwords is None:
            try:
                from nltk.corpus import stopwords
                self._stopwords = set(stopwords.words(self.language))
                self._stopwords.update(self._additional)
            except ImportError as e:
                logger.error("NLTK not available: %s", e)
                raise
    
    @property
    def stopwords(self) -> set[str]:
        """Get the stopword set."""
        self._ensure_nltk()
        assert self._stopwords is not None
        return self._stopwords
    
    def is_stopword(self, word: str) -> bool:
        """Check if a word is a stopword."""
        return word.lower() in self.stopwords
    
    def filter_tokens(self, tokens: list[str]) -> list[str]:
        """Remove stopwords from token list."""
        return [t for t in tokens if t.lower() not in self.stopwords]
    
    def add_stopwords(self, words: Iterable[str]) -> None:
        """Add custom stopwords."""
        self._ensure_nltk()
        assert self._stopwords is not None
        self._stopwords.update(words)


# =============================================================================
# §3. TEXT FEATURES
# =============================================================================

@dataclass
class FrequencyAnalysis:
    """
    Results of frequency analysis.
    
    Attributes:
        frequencies: Counter of token frequencies.
        total_tokens: Total number of tokens.
        vocabulary_size: Number of unique tokens.
    """
    
    frequencies: Counter[str]
    total_tokens: int
    vocabulary_size: int
    
    def most_common(self, n: int = 10) -> list[tuple[str, int]]:
        """Get n most common tokens."""
        return self.frequencies.most_common(n)
    
    def relative_frequency(self, token: str) -> float:
        """Get relative frequency of a token."""
        if self.total_tokens == 0:
            return 0.0
        return self.frequencies[token] / self.total_tokens


class FrequencyAnalyser:
    """
    Analyses token frequencies in text.
    
    Computes frequency distributions, vocabulary statistics
    and common token rankings.
    
    Example:
        >>> analyser = FrequencyAnalyser()
        >>> result = analyser.analyse(["the", "cat", "sat", "on", "the", "mat"])
        >>> result.frequencies["the"]
        2
    """
    
    def analyse(self, tokens: list[str]) -> FrequencyAnalysis:
        """
        Perform frequency analysis on tokens.
        
        Args:
            tokens: List of tokens to analyse.
        
        Returns:
            FrequencyAnalysis with statistics.
        """
        freq = Counter(tokens)
        return FrequencyAnalysis(
            frequencies=freq,
            total_tokens=len(tokens),
            vocabulary_size=len(freq)
        )
    
    def analyse_text(self, text: str, tokeniser: Tokeniser | None = None) -> FrequencyAnalysis:
        """
        Analyse frequency in raw text.
        
        Args:
            text: Raw text to analyse.
            tokeniser: Tokeniser to use (defaults to RegexTokeniser).
        
        Returns:
            FrequencyAnalysis with statistics.
        """
        tok = tokeniser or RegexTokeniser(r"\w+")
        tokens = tok.tokenise(text.lower())
        return self.analyse(tokens)


class NGramExtractor:
    """
    Extracts n-grams from token sequences.
    
    N-grams are contiguous sequences of n items, useful
    for capturing local context and collocations.
    
    Attributes:
        n: Size of n-grams to extract.
    
    Example:
        >>> extractor = NGramExtractor(n=2)
        >>> extractor.extract(["natural", "language", "processing"])
        [('natural', 'language'), ('language', 'processing')]
    """
    
    def __init__(self, n: int = 2) -> None:
        """
        Initialise n-gram extractor.
        
        Args:
            n: Size of n-grams (2 for bigrams, 3 for trigrams, etc.).
        """
        if n < 1:
            raise ValueError("n must be at least 1")
        self.n = n
        logger.debug("NGramExtractor initialised with n=%d", n)
    
    def extract(self, tokens: list[str]) -> list[tuple[str, ...]]:
        """
        Extract n-grams from token list.
        
        Args:
            tokens: List of tokens.
        
        Returns:
            List of n-gram tuples.
        """
        if len(tokens) < self.n:
            return []
        return [tuple(tokens[i:i + self.n]) for i in range(len(tokens) - self.n + 1)]
    
    def extract_with_counts(self, tokens: list[str]) -> Counter[tuple[str, ...]]:
        """
        Extract n-grams with frequency counts.
        
        Args:
            tokens: List of tokens.
        
        Returns:
            Counter of n-gram frequencies.
        """
        ngrams = self.extract(tokens)
        return Counter(ngrams)


@dataclass
class TFIDFResult:
    """
    TF-IDF computation result for a document.
    
    Attributes:
        document_id: Identifier for the document.
        scores: Dictionary mapping terms to TF-IDF scores.
    """
    
    document_id: int
    scores: dict[str, float] = field(default_factory=dict)
    
    def top_terms(self, n: int = 10) -> list[tuple[str, float]]:
        """Get n terms with highest TF-IDF scores."""
        sorted_terms = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_terms[:n]


class TFIDFCalculator:
    """
    Calculates TF-IDF (Term Frequency-Inverse Document Frequency) scores.
    
    TF-IDF weights terms by their frequency in a document relative
    to their frequency across the corpus, highlighting distinctive terms.
    
    Example:
        >>> calc = TFIDFCalculator()
        >>> corpus = [["the", "cat"], ["the", "dog"], ["cat", "dog"]]
        >>> calc.fit(corpus)
        >>> result = calc.transform(0)
        >>> result.scores["cat"]  # Higher than "the"
    """
    
    def __init__(self) -> None:
        """Initialise TF-IDF calculator."""
        self._documents: list[list[str]] = []
        self._vocabulary: set[str] = set()
        self._idf: dict[str, float] = {}
        self._fitted = False
    
    def fit(self, documents: list[list[str]]) -> None:
        """
        Fit the calculator to a corpus.
        
        Computes IDF values for all terms in the corpus.
        
        Args:
            documents: List of tokenised documents.
        """
        self._documents = documents
        self._vocabulary = set()
        
        # Build vocabulary
        for doc in documents:
            self._vocabulary.update(doc)
        
        # Compute IDF for each term
        n_docs = len(documents)
        for term in self._vocabulary:
            doc_freq = sum(1 for doc in documents if term in doc)
            # Add 1 to avoid division by zero
            self._idf[term] = math.log(n_docs / (1 + doc_freq)) + 1
        
        self._fitted = True
        logger.debug("TFIDFCalculator fitted with %d documents, %d terms",
                     n_docs, len(self._vocabulary))
    
    def transform(self, doc_index: int) -> TFIDFResult:
        """
        Compute TF-IDF scores for a document.
        
        Args:
            doc_index: Index of document in the fitted corpus.
        
        Returns:
            TFIDFResult with term scores.
        
        Raises:
            ValueError: If calculator not fitted or index out of range.
        """
        if not self._fitted:
            raise ValueError("Calculator must be fitted before transform")
        if doc_index < 0 or doc_index >= len(self._documents):
            raise ValueError(f"Document index {doc_index} out of range")
        
        doc = self._documents[doc_index]
        tf = Counter(doc)
        doc_len = len(doc)
        
        scores = {}
        for term, count in tf.items():
            tf_score = count / doc_len if doc_len > 0 else 0
            idf_score = self._idf.get(term, 0)
            scores[term] = tf_score * idf_score
        
        return TFIDFResult(document_id=doc_index, scores=scores)
    
    def fit_transform(self, documents: list[list[str]]) -> list[TFIDFResult]:
        """
        Fit and transform all documents.
        
        Args:
            documents: List of tokenised documents.
        
        Returns:
            List of TFIDFResult for each document.
        """
        self.fit(documents)
        return [self.transform(i) for i in range(len(documents))]


# =============================================================================
# §4. BASIC NLP TASKS
# =============================================================================

@dataclass
class POSTaggedToken:
    """
    A token with its part-of-speech tag.
    
    Attributes:
        token: The original token text.
        pos: Part-of-speech tag.
        tag: Fine-grained tag (if available).
    """
    
    token: str
    pos: str
    tag: str = ""


class POSTagger:
    """
    Part-of-speech tagger using NLTK.
    
    Assigns grammatical categories to tokens.
    
    Example:
        >>> tagger = POSTagger()
        >>> tagged = tagger.tag(["The", "cat", "sat"])
        >>> tagged[1].pos
        'NN'
    """
    
    def __init__(self) -> None:
        """Initialise POS tagger."""
        self._pos_tag = None
    
    def _ensure_nltk(self) -> None:
        """Lazy import of NLTK tagger."""
        if self._pos_tag is None:
            try:
                from nltk import pos_tag
                self._pos_tag = pos_tag
            except ImportError as e:
                logger.error("NLTK not available: %s", e)
                raise
    
    def tag(self, tokens: list[str]) -> list[POSTaggedToken]:
        """
        Tag tokens with parts of speech.
        
        Args:
            tokens: List of tokens to tag.
        
        Returns:
            List of POSTaggedToken objects.
        """
        self._ensure_nltk()
        tagged = self._pos_tag(tokens)
        return [POSTaggedToken(token=t, pos=p, tag=p) for t, p in tagged]
    
    def tag_text(self, text: str) -> list[POSTaggedToken]:
        """
        Tokenise and tag text.
        
        Args:
            text: Raw text to process.
        
        Returns:
            List of POSTaggedToken objects.
        """
        tokeniser = WordTokeniser()
        tokens = tokeniser.tokenise(text)
        return self.tag(tokens)


class SimpleSentimentAnalyser:
    """
    Simple lexicon-based sentiment analyser.
    
    Uses positive and negative word lists to compute
    a basic sentiment score.
    
    Attributes:
        positive_words: Set of positive sentiment words.
        negative_words: Set of negative sentiment words.
    
    Example:
        >>> analyser = SimpleSentimentAnalyser()
        >>> analyser.analyse(["good", "great", "bad"])
        0.333...  # (2 positive - 1 negative) / 3
    """
    
    # Basic sentiment lexicons
    DEFAULT_POSITIVE = {
        "good", "great", "excellent", "wonderful", "fantastic", "amazing",
        "positive", "happy", "love", "best", "beautiful", "perfect", "nice"
    }
    
    DEFAULT_NEGATIVE = {
        "bad", "terrible", "awful", "horrible", "poor", "negative", "worst",
        "hate", "ugly", "sad", "wrong", "disappointing", "failure"
    }
    
    def __init__(
        self,
        positive_words: set[str] | None = None,
        negative_words: set[str] | None = None
    ) -> None:
        """
        Initialise sentiment analyser.
        
        Args:
            positive_words: Custom positive word set.
            negative_words: Custom negative word set.
        """
        self.positive_words = positive_words or self.DEFAULT_POSITIVE
        self.negative_words = negative_words or self.DEFAULT_NEGATIVE
    
    def analyse(self, tokens: list[str]) -> float:
        """
        Compute sentiment score for tokens.
        
        Args:
            tokens: List of tokens to analyse.
        
        Returns:
            Sentiment score in range [-1, 1].
        """
        if not tokens:
            return 0.0
        
        pos_count = sum(1 for t in tokens if t.lower() in self.positive_words)
        neg_count = sum(1 for t in tokens if t.lower() in self.negative_words)
        
        return (pos_count - neg_count) / len(tokens)
    
    def analyse_text(self, text: str) -> float:
        """
        Analyse sentiment of raw text.
        
        Args:
            text: Raw text to analyse.
        
        Returns:
            Sentiment score in range [-1, 1].
        """
        tokeniser = RegexTokeniser(r"\w+")
        tokens = tokeniser.tokenise(text)
        return self.analyse(tokens)


@dataclass
class TextAnalysisResult:
    """
    Comprehensive text analysis result.
    
    Attributes:
        original_text: The input text.
        tokens: Tokenised text.
        normalised_tokens: Tokens after normalisation.
        frequencies: Token frequency analysis.
        bigrams: Bigram frequency analysis.
        sentiment: Sentiment score.
    """
    
    original_text: str
    tokens: list[str]
    normalised_tokens: list[str]
    frequencies: FrequencyAnalysis
    bigrams: Counter[tuple[str, ...]]
    sentiment: float


class TextAnalyser:
    """
    Comprehensive text analyser combining multiple NLP techniques.
    
    Provides a unified interface for tokenisation, normalisation,
    frequency analysis, n-gram extraction and sentiment analysis.
    
    Example:
        >>> analyser = TextAnalyser()
        >>> result = analyser.analyse("The quick brown fox is good.")
        >>> result.sentiment
        0.166...
    """
    
    def __init__(
        self,
        use_stemming: bool = False,
        remove_stopwords: bool = True
    ) -> None:
        """
        Initialise text analyser.
        
        Args:
            use_stemming: Whether to apply stemming.
            remove_stopwords: Whether to remove stopwords.
        """
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        
        self._tokeniser = RegexTokeniser(r"\w+")
        self._stemmer = PorterStemmer() if use_stemming else None
        self._stopword_filter = StopwordFilter() if remove_stopwords else None
        self._freq_analyser = FrequencyAnalyser()
        self._bigram_extractor = NGramExtractor(n=2)
        self._sentiment_analyser = SimpleSentimentAnalyser()
    
    def analyse(self, text: str) -> TextAnalysisResult:
        """
        Perform comprehensive analysis on text.
        
        Args:
            text: Raw text to analyse.
        
        Returns:
            TextAnalysisResult with all analysis components.
        """
        # Tokenise
        tokens = self._tokeniser.tokenise(text.lower())
        
        # Normalise
        normalised = tokens.copy()
        
        if self._stopword_filter:
            normalised = self._stopword_filter.filter_tokens(normalised)
        
        if self._stemmer:
            normalised = self._stemmer.stem_tokens(normalised)
        
        # Analyse
        frequencies = self._freq_analyser.analyse(normalised)
        bigrams = self._bigram_extractor.extract_with_counts(normalised)
        sentiment = self._sentiment_analyser.analyse(tokens)
        
        return TextAnalysisResult(
            original_text=text,
            tokens=tokens,
            normalised_tokens=normalised,
            frequencies=frequencies,
            bigrams=bigrams,
            sentiment=sentiment
        )


# =============================================================================
# DEMONSTRATION FUNCTIONS
# =============================================================================

def demonstrate_tokenisation() -> None:
    """Demonstrate tokenisation approaches."""
    text = "Dr. Smith's analysis won't be ready until 3:00 p.m."
    
    logger.info("Text: %s", text)
    
    ws_tok = WhitespaceTokeniser()
    logger.info("Whitespace: %s", ws_tok.tokenise(text))
    
    regex_tok = RegexTokeniser(r"\w+")
    logger.info("Regex (\\w+): %s", regex_tok.tokenise(text))


def demonstrate_normalisation() -> None:
    """Demonstrate text normalisation."""
    tokens = ["Running", "runs", "ran", "universities", "better"]
    
    stemmer = PorterStemmer()
    logger.info("Stemmed: %s", stemmer.stem_tokens([t.lower() for t in tokens]))


def demonstrate_tfidf() -> None:
    """Demonstrate TF-IDF calculation."""
    corpus = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "sat", "on", "the", "log"],
        ["the", "cat", "chased", "the", "dog"]
    ]
    
    calc = TFIDFCalculator()
    results = calc.fit_transform(corpus)
    
    for result in results:
        logger.info("Document %d top terms: %s", 
                    result.document_id, result.top_terms(3))


if __name__ == "__main__":
    logger.info("=== Tokenisation ===")
    demonstrate_tokenisation()
    
    logger.info("\n=== Normalisation ===")
    demonstrate_normalisation()
    
    logger.info("\n=== TF-IDF ===")
    demonstrate_tfidf()
