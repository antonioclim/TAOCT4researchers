# Unit 11: Homework Assignment

## Text Processing and NLP Fundamentals

---

## Overview

This homework assignment assesses your understanding and application of text processing techniques, regular expressions and natural language processing fundamentals. You will complete a series of tasks that progress from basic pattern matching through to corpus analysis.

**Submission Requirements:**
- All Python files must include type hints
- Use Google-style docstrings
- Code must pass provided test cases
- Include a brief report (500 words) discussing your approach

**Due Date:** End of Week 11
**Total Points:** 100

---

## Part A: Regular Expression Mastery (30 points)

### Task A1: Pattern Development (10 points)

Develop regular expression patterns for the following extraction tasks. Each pattern must handle edge cases appropriately.

**A1.1 Academic Citation Pattern (3 points)**

Create a pattern to extract citations in the format: `(Author, Year)` or `(Author et al., Year)`.

Examples to match:
- `(Smith, 2020)`
- `(Jones et al., 2019)`
- `(García-López, 2021)`

Examples NOT to match:
- `(2020)` (missing author)
- `Smith, 2020` (missing parentheses)

```python
def extract_citations(text: str) -> list[tuple[str, str]]:
    """
    Extract academic citations from text.
    
    Args:
        text: Input text containing citations.
    
    Returns:
        List of (author, year) tuples.
    """
    # Your implementation here
    pass
```

**A1.2 UK Postcode Pattern (3 points)**

Create a pattern to validate UK postcodes in standard format.

Valid formats:
- `SW1A 1AA`
- `M1 1AA`
- `B33 8TH`
- `CR2 6XH`

```python
def validate_uk_postcode(postcode: str) -> bool:
    """
    Validate a UK postcode.
    
    Args:
        postcode: Postcode string to validate.
    
    Returns:
        True if valid, False otherwise.
    """
    # Your implementation here
    pass
```

**A1.3 Log Entry Parser (4 points)**

Create a pattern with named groups to parse log entries in the format:
`[TIMESTAMP] LEVEL: Component - Message`

```python
@dataclass
class LogEntry:
    timestamp: str
    level: str
    component: str
    message: str

def parse_log_entry(line: str) -> LogEntry | None:
    """
    Parse a structured log entry.
    
    Args:
        line: Log line to parse.
    
    Returns:
        LogEntry if pattern matches, None otherwise.
    """
    # Your implementation here
    pass
```

### Task A2: Text Extraction Pipeline (10 points)

Implement a class that extracts structured data from unstructured text documents.

```python
@dataclass
class ExtractedData:
    emails: list[str]
    urls: list[str]
    dates: list[str]
    phone_numbers: list[str]
    monetary_amounts: list[tuple[str, float]]  # (currency, amount)

class DocumentExtractor:
    """
    Extracts structured data from documents.
    
    Your implementation should:
    1. Extract all email addresses
    2. Extract all URLs (http and https)
    3. Extract dates in multiple formats (ISO, UK, US)
    4. Extract phone numbers (international and domestic)
    5. Extract monetary amounts with currency symbols
    """
    
    def extract(self, document: str) -> ExtractedData:
        """Extract all structured data from document."""
        # Your implementation here
        pass
```

### Task A3: Search and Replace Operations (10 points)

Implement a text sanitisation class that performs multiple search-and-replace operations.

```python
class TextSanitiser:
    """
    Sanitises text by applying configurable transformations.
    
    Requirements:
    1. Redact email addresses (replace with [EMAIL])
    2. Redact phone numbers (replace with [PHONE])
    3. Normalise whitespace
    4. Remove HTML tags whilst preserving content
    5. Fix common OCR errors (configurable mappings)
    """
    
    def __init__(self, ocr_corrections: dict[str, str] | None = None):
        """Initialise with optional OCR corrections."""
        pass
    
    def sanitise(self, text: str) -> str:
        """Apply all sanitisation transformations."""
        pass
```

---

## Part B: NLP Pipeline Implementation (40 points)

### Task B1: Custom Tokeniser (10 points)

Implement a configurable tokeniser that handles domain-specific requirements.

```python
@dataclass
class TokeniserConfig:
    preserve_case: bool = False
    handle_contractions: bool = True
    handle_hyphenation: bool = True
    handle_numbers: bool = True
    custom_patterns: list[str] = field(default_factory=list)

class ConfigurableTokeniser:
    """
    Tokeniser with configurable behaviour.
    
    Features:
    1. Optional case preservation
    2. Contraction handling (don't -> do, n't OR don't)
    3. Hyphenated word handling (state-of-the-art -> options)
    4. Number recognition (preserve decimals, dates)
    5. Custom pattern recognition (URLs, emails, etc.)
    """
    
    def __init__(self, config: TokeniserConfig):
        """Initialise with configuration."""
        pass
    
    def tokenise(self, text: str) -> list[str]:
        """Tokenise text according to configuration."""
        pass
```

### Task B2: Text Normalisation Pipeline (15 points)

Implement a complete text normalisation pipeline with configurable stages.

```python
@dataclass
class NormalisationConfig:
    lowercase: bool = True
    remove_stopwords: bool = True
    stemming: bool = False
    lemmatisation: bool = True
    min_token_length: int = 2
    custom_stopwords: set[str] = field(default_factory=set)

class NormalisationPipeline:
    """
    Text normalisation pipeline.
    
    Stages (configurable order):
    1. Tokenisation
    2. Case normalisation
    3. Stopword removal (with customisation)
    4. Stemming OR lemmatisation (not both)
    5. Minimum length filtering
    6. Optional custom transformations
    
    Requirements:
    - Pipeline must be immutable (create new tokens, don't modify)
    - Each stage must log its transformation
    - Support adding custom transformation functions
    """
    
    def __init__(self, config: NormalisationConfig):
        """Initialise pipeline with configuration."""
        pass
    
    def add_custom_stage(
        self, 
        name: str, 
        func: Callable[[list[str]], list[str]]
    ) -> None:
        """Add a custom processing stage."""
        pass
    
    def process(self, text: str) -> list[str]:
        """Process text through all pipeline stages."""
        pass
    
    def process_batch(self, texts: list[str]) -> list[list[str]]:
        """Process multiple texts efficiently."""
        pass
```

### Task B3: Feature Extraction Suite (15 points)

Implement a comprehensive feature extraction class for text analysis.

```python
@dataclass
class TextFeatures:
    token_count: int
    vocabulary_size: int
    avg_token_length: float
    frequency_distribution: Counter[str]
    bigram_distribution: Counter[tuple[str, str]]
    trigram_distribution: Counter[tuple[str, str, str]]
    tfidf_scores: dict[str, float] | None  # Only for corpus context

class FeatureExtractor:
    """
    Extracts numerical features from text.
    
    Features:
    1. Basic statistics (token count, vocabulary size)
    2. Frequency distributions (unigrams, bigrams, trigrams)
    3. TF-IDF scores (when fitted to corpus)
    4. Lexical diversity metrics
    5. N-gram collocation strength
    
    Requirements:
    - Support both single document and corpus modes
    - Implement fit/transform pattern for TF-IDF
    - Calculate type-token ratio
    """
    
    def __init__(self, ngram_range: tuple[int, int] = (1, 3)):
        """Initialise with n-gram range."""
        pass
    
    def fit(self, corpus: list[list[str]]) -> None:
        """Fit to corpus for TF-IDF calculation."""
        pass
    
    def extract(self, tokens: list[str]) -> TextFeatures:
        """Extract features from tokenised document."""
        pass
    
    def type_token_ratio(self, tokens: list[str]) -> float:
        """Calculate lexical diversity measure."""
        pass
```

---

## Part C: Corpus Analysis Project (30 points)

### Task C1: Comparative Corpus Analysis

For this task, you will analyse and compare two text corpora. You may use provided sample texts or source your own (minimum 5 documents per corpus, 500+ words each).

**Requirements:**

1. **Data Preparation (5 points)**
   - Load and preprocess both corpora
   - Apply consistent normalisation
   - Document any data cleaning decisions

2. **Frequency Analysis (10 points)**
   - Compute term frequencies for each corpus
   - Identify most common terms (excluding stopwords)
   - Calculate vocabulary overlap between corpora
   - Visualise frequency distributions (optional for bonus)

3. **TF-IDF Analysis (10 points)**
   - Compute TF-IDF scores across combined corpus
   - Identify distinctive terms for each corpus
   - Rank documents by similarity to each other

4. **Report (5 points)**
   - Interpret your findings (300-500 words)
   - Discuss what distinguishes the two corpora
   - Reflect on preprocessing decisions and their impact

```python
class CorpusAnalyser:
    """
    Comparative corpus analysis tool.
    
    Your implementation should support:
    1. Loading corpora from files or strings
    2. Preprocessing with configurable pipeline
    3. Frequency and TF-IDF analysis
    4. Comparison metrics between corpora
    """
    
    def __init__(self, pipeline: NormalisationPipeline):
        """Initialise with preprocessing pipeline."""
        pass
    
    def load_corpus(self, documents: list[str], name: str) -> None:
        """Load and preprocess a corpus."""
        pass
    
    def compare(self) -> dict[str, Any]:
        """Compare loaded corpora and return analysis results."""
        pass
    
    def distinctive_terms(self, corpus_name: str, n: int = 20) -> list[tuple[str, float]]:
        """Get terms most distinctive to a corpus."""
        pass
```

---

## Part D: Reflection and Analysis (Bonus - 10 points)

### Task D1: Method Comparison

Write a 300-word analysis comparing different approaches to text normalisation. Your analysis should address:

1. **Stemming vs Lemmatisation Trade-offs**
   - When is stemming preferable to lemmatisation?
   - What are the computational costs of each approach?
   - How do errors in each method affect downstream tasks?

2. **Tokenisation Strategies**
   - Compare whitespace, regex and NLTK tokenisation
   - Discuss domain-specific tokenisation challenges
   - When might subword tokenisation be appropriate?

3. **Practical Recommendations**
   - For sentiment analysis, which normalisation approach would you recommend?
   - For information retrieval, how would your choices differ?
   - How does corpus size affect these decisions?

### Task D2: Error Analysis

Identify and document at least five cases where your regex patterns or NLP pipeline produces incorrect or unexpected results. For each case:

1. Describe the input that causes the problem
2. Explain why the error occurs
3. Propose a solution or mitigation strategy

This exercise develops debugging skills essential for production text processing systems.

### Task D3: Performance Benchmarking

Benchmark your implementations on a corpus of at least 1,000 documents:

1. Measure tokenisation speed (tokens per second)
2. Compare stemmer performance (Porter vs Snowball)
3. Evaluate TF-IDF computation time scaling

Present your results in a table and discuss implications for large-scale text processing. Consider memory usage as well as execution time.

---

## Submission Checklist

Before submitting, verify:

- [ ] All code files include complete type hints
- [ ] All public functions have Google-style docstrings
- [ ] Code passes linting (ruff or flake8)
- [ ] All test cases pass
- [ ] Report is included (500 words minimum)
- [ ] Files are organised in the correct directory structure

**Directory Structure:**
```
homework/
├── part_a/
│   ├── task_a1_patterns.py
│   ├── task_a2_extractor.py
│   └── task_a3_sanitiser.py
├── part_b/
│   ├── task_b1_tokeniser.py
│   ├── task_b2_pipeline.py
│   └── task_b3_features.py
├── part_c/
│   ├── corpus_analyser.py
│   ├── data/
│   │   ├── corpus_a/
│   │   └── corpus_b/
│   └── report.md
└── tests/
    └── test_homework.py
```

---

## Grading Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| Part A: Patterns | 30 | Correctness, edge case handling, code quality |
| Part B: Pipeline | 40 | Functionality, design, documentation |
| Part C: Analysis | 30 | Methodology, interpretation, code quality |
| **Total** | **100** | |

**Bonus Points (up to 10):**
- Visualisations of analysis results
- Additional corpus comparison metrics
- Performance optimisation for large corpora

---

## Resources

- Lecture notes: `theory/lecture_notes.md`
- Lab solutions: `lab/solutions/`
- Cheatsheet: `resources/cheatsheet.md`
- Test framework: `tests/conftest.py`

---

© 2025 Antonio Clim. All rights reserved.
