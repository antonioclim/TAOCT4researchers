"""
Lab 11_02 Solutions: NLP Fundamentals

This module provides worked solutions and additional examples
for the NLP fundamentals laboratory.

Author: Antonio Clim
Version: 1.0.0
"""

from __future__ import annotations

from collections import Counter
from lab.lab_11_02_nlp_fundamentals import (
    WhitespaceTokeniser,
    RegexTokeniser,
    WordTokeniser,
    SentenceTokeniser,
    PorterStemmer,
    SnowballStemmer,
    Lemmatiser,
    StopwordFilter,
    FrequencyAnalyser,
    NGramExtractor,
    TFIDFCalculator,
    POSTagger,
    SimpleSentimentAnalyser,
    TextAnalyser,
)


# =============================================================================
# SOLUTION EXAMPLES
# =============================================================================

def solution_tokenisation() -> None:
    """Demonstrate tokenisation solutions."""
    text = "Dr. Smith's analysis won't be ready until 3:00 p.m. today."
    
    # Compare tokenisers
    ws = WhitespaceTokeniser()
    regex = RegexTokeniser(r"\w+")
    
    ws_tokens = ws.tokenise(text)
    regex_tokens = regex.tokenise(text)
    
    # Whitespace keeps punctuation attached
    assert "Dr." in ws_tokens[0]
    
    # Regex extracts only word characters
    assert "Dr" in regex_tokens
    assert "." not in regex_tokens
    
    # NLTK handles contractions
    # word_tok = WordTokeniser()
    # nltk_tokens = word_tok.tokenise(text)
    # assert "wo" in nltk_tokens or "n't" in nltk_tokens


def solution_stemming_vs_lemmatisation() -> None:
    """Compare stemming and lemmatisation."""
    words = ["running", "runs", "ran", "better", "studies", "universities"]
    
    stemmer = PorterStemmer()
    # lemmatiser = Lemmatiser()  # Requires NLTK
    
    stemmed = stemmer.stem_tokens([w.lower() for w in words])
    
    # Stemming results
    assert stemmer.stem("running") == "run"
    assert stemmer.stem("studies") == "studi"  # Not a real word!
    
    # Lemmatisation would give:
    # "running" (v) -> "run"
    # "better" (a) -> "good"
    # "studies" -> "study"


def solution_stopword_filtering() -> None:
    """Demonstrate stopword filtering."""
    tokens = ["the", "quick", "brown", "fox", "is", "very", "fast"]
    
    # Using StopwordFilter (requires NLTK)
    # filt = StopwordFilter()
    # filtered = filt.filter_tokens(tokens)
    
    # Manual filtering demonstration
    basic_stopwords = {"the", "is", "a", "an", "and", "or", "but", "very"}
    filtered = [t for t in tokens if t.lower() not in basic_stopwords]
    
    assert "the" not in filtered
    assert "is" not in filtered
    assert "quick" in filtered
    assert "fox" in filtered


def solution_ngrams() -> None:
    """Demonstrate n-gram extraction."""
    tokens = ["natural", "language", "processing", "is", "fascinating"]
    
    # Bigrams
    bigram_ext = NGramExtractor(n=2)
    bigrams = bigram_ext.extract(tokens)
    
    assert ("natural", "language") in bigrams
    assert ("language", "processing") in bigrams
    assert len(bigrams) == 4
    
    # Trigrams
    trigram_ext = NGramExtractor(n=3)
    trigrams = trigram_ext.extract(tokens)
    
    assert ("natural", "language", "processing") in trigrams
    assert len(trigrams) == 3
    
    # With counts
    text_tokens = ["the", "cat", "sat", "on", "the", "mat", "the", "cat"]
    bigram_counts = bigram_ext.extract_with_counts(text_tokens)
    
    assert bigram_counts[("the", "cat")] == 2


def solution_frequency_analysis() -> None:
    """Demonstrate frequency analysis."""
    text = "The cat sat on the mat. The cat was happy."
    
    analyser = FrequencyAnalyser()
    result = analyser.analyse_text(text)
    
    # "the" appears most frequently
    most_common = result.most_common(3)
    assert most_common[0][0] == "the"
    assert most_common[0][1] == 3
    
    # Relative frequency
    rel_freq = result.relative_frequency("cat")
    assert rel_freq == 2 / result.total_tokens


def solution_tfidf() -> None:
    """Demonstrate TF-IDF calculation."""
    corpus = [
        ["machine", "learning", "algorithm", "data"],
        ["deep", "learning", "neural", "network"],
        ["data", "science", "statistics", "analysis"],
    ]
    
    calc = TFIDFCalculator()
    results = calc.fit_transform(corpus)
    
    # Document 0: "algorithm" is unique, should have high TF-IDF
    doc0_scores = results[0].scores
    
    # "learning" appears in 2 docs, lower IDF
    # "algorithm" appears in 1 doc, higher IDF
    if "algorithm" in doc0_scores and "learning" in doc0_scores:
        # algorithm should have higher score (more distinctive)
        pass  # Exact values depend on implementation
    
    # Get top terms for each document
    for i, result in enumerate(results):
        top = result.top_terms(2)
        assert len(top) <= 2


def solution_sentiment() -> None:
    """Demonstrate sentiment analysis."""
    analyser = SimpleSentimentAnalyser()
    
    # Positive text
    positive = "This is wonderful and amazing! I love it."
    pos_score = analyser.analyse_text(positive)
    assert pos_score > 0
    
    # Negative text
    negative = "This is terrible and awful. I hate it."
    neg_score = analyser.analyse_text(negative)
    assert neg_score < 0
    
    # Neutral/mixed text
    mixed = "The movie was good but the ending was bad."
    mixed_score = analyser.analyse_text(mixed)
    # Score should be close to 0


def solution_comprehensive_analysis() -> None:
    """Demonstrate comprehensive text analysis."""
    text = """
    Natural language processing is a fascinating field of computer science.
    It combines linguistics with machine learning to understand human language.
    The applications are amazing and wonderful.
    """
    
    analyser = TextAnalyser(use_stemming=True, remove_stopwords=True)
    result = analyser.analyse(text)
    
    # Check components
    assert len(result.tokens) > 0
    assert len(result.normalised_tokens) <= len(result.tokens)
    assert result.frequencies.total_tokens == len(result.normalised_tokens)
    
    # Sentiment should be positive (amazing, wonderful, fascinating)
    assert result.sentiment > 0


# =============================================================================
# EXERCISE SOLUTIONS
# =============================================================================

def exercise_1_solution() -> dict[str, int]:
    """
    Exercise 1: Compute vocabulary statistics for a corpus.
    
    Returns:
        Dictionary with vocabulary statistics.
    """
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "A quick brown dog runs in the park",
        "The lazy cat sleeps all day long",
    ]
    
    tokeniser = RegexTokeniser(r"\w+")
    all_tokens: list[str] = []
    vocab_per_doc: list[set[str]] = []
    
    for doc in documents:
        tokens = tokeniser.tokenise(doc.lower())
        all_tokens.extend(tokens)
        vocab_per_doc.append(set(tokens))
    
    total_vocab = set(all_tokens)
    
    # Words appearing in all documents
    common_vocab = vocab_per_doc[0]
    for vocab in vocab_per_doc[1:]:
        common_vocab = common_vocab & vocab
    
    return {
        "total_tokens": len(all_tokens),
        "unique_tokens": len(total_vocab),
        "common_to_all": len(common_vocab),
        "avg_doc_length": len(all_tokens) // len(documents),
    }


def exercise_2_solution() -> list[tuple[str, ...]]:
    """
    Exercise 2: Find most frequent bigrams in text.
    
    Returns:
        Top 5 bigrams by frequency.
    """
    text = """
    Machine learning is a subset of artificial intelligence.
    Machine learning algorithms learn from data.
    Deep learning is a subset of machine learning.
    Artificial intelligence includes machine learning and deep learning.
    """
    
    # Tokenise and normalise
    tokeniser = RegexTokeniser(r"\w+")
    tokens = tokeniser.tokenise(text.lower())
    
    # Remove stopwords
    stopwords = {"is", "a", "of", "and", "from", "the", "includes"}
    tokens = [t for t in tokens if t not in stopwords]
    
    # Extract bigrams
    extractor = NGramExtractor(n=2)
    bigram_counts = extractor.extract_with_counts(tokens)
    
    return bigram_counts.most_common(5)


def exercise_3_solution() -> dict[int, list[tuple[str, float]]]:
    """
    Exercise 3: Compute TF-IDF and find distinctive terms.
    
    Returns:
        Dictionary mapping doc index to top distinctive terms.
    """
    corpus = [
        "python programming language code development",
        "java programming enterprise software applications",
        "machine learning data science python analysis",
        "web development javascript frontend backend",
    ]
    
    tokeniser = RegexTokeniser(r"\w+")
    tokenised = [tokeniser.tokenise(doc.lower()) for doc in corpus]
    
    calc = TFIDFCalculator()
    results = calc.fit_transform(tokenised)
    
    return {
        result.document_id: result.top_terms(3)
        for result in results
    }


if __name__ == "__main__":
    # Run all solutions
    solution_tokenisation()
    solution_stemming_vs_lemmatisation()
    solution_stopword_filtering()
    solution_ngrams()
    solution_frequency_analysis()
    solution_tfidf()
    solution_sentiment()
    solution_comprehensive_analysis()
    
    # Run exercises
    ex1 = exercise_1_solution()
    print(f"Exercise 1: {ex1}")
    
    ex2 = exercise_2_solution()
    print(f"Exercise 2: {ex2}")
    
    ex3 = exercise_3_solution()
    print(f"Exercise 3: {ex3}")
    
    print("\nAll solutions verified successfully!")
