# 11UNIT: Text Processing and NLP Fundamentals

## Lecture Notes

---

## 1. The Computational Treatment of Text

### 1.1 Text as Structured Data

Text represents one of the most abundant yet challenging forms of data confronting computational researchers. Unlike numerical datasets with inherent mathematical properties, textual data requires transformation before computational manipulation becomes tractable. This transformation—from raw character sequences to structured representations amenable to analysis—constitutes the central concern of text processing.

The fundamental insight underlying computational text analysis is that linguistic patterns, despite their apparent complexity, exhibit regularities susceptible to algorithmic detection. As observed in literary studies, computational text analysis has "uncovered authorship patterns and thematic developments across centuries of literature" (Jockers & Underwood, 2016, p. 292). Such discoveries emerge not from reading in the traditional sense but from systematic pattern extraction across corpora of unprecedented scale.

Text processing occupies a distinctive position in computational methodology: it bridges the qualitative richness of linguistic expression with the quantitative rigour of algorithmic analysis. The researcher who masters these techniques gains access to research questions previously beyond reach—questions involving hundreds of thousands of documents, millions of social media posts, or centuries of historical records.

### 1.2 Levels of Textual Analysis

Computational text processing operates across multiple levels of linguistic organisation:

**Character level**: The fundamental units of written language. Processing at this level involves encoding schemes (ASCII, UTF-8, UTF-16), character classification (alphabetic, numeric, punctuation) and character-by-character transformation (case normalisation, accent removal).

**Token level**: Meaningful units extracted through tokenisation. Tokens typically correspond to words but may include punctuation, numbers or domain-specific entities. Token-level processing includes stemming, lemmatisation and part-of-speech tagging.

**Sequence level**: Patterns spanning multiple tokens. N-grams capture local co-occurrence patterns; phrase detection identifies meaningful multi-word expressions; sentence segmentation establishes clause boundaries.

**Document level**: Aggregate properties of complete texts. Document-level features include length statistics, vocabulary richness, topic distributions and sentiment scores.

**Corpus level**: Relationships across document collections. Corpus-level analysis includes frequency distributions, document similarity, topic modelling and author attribution.

Effective text processing requires selecting the appropriate level for a given research question whilst recognising dependencies between levels—document-level features presuppose token-level processing, which in turn depends upon character-level encoding.

---

## 2. Regular Expressions: The Language of Patterns

### 2.1 Theoretical Foundations

Regular expressions constitute a formal language for specifying character patterns. Rooted in automata theory, regular expressions describe the class of regular languages—precisely those languages recognisable by finite automata. This theoretical grounding ensures that regex pattern matching operates in linear time with respect to input length, a property essential for processing large corpora.

The practical utility of regular expressions extends far beyond theoretical elegance. They provide a concise notation for expressing patterns that would require extensive procedural code. A pattern like `\b[A-Z][a-z]+\b` captures capitalised words in a single expression; implementing equivalent logic procedurally would require loops, conditionals and character classification.

### 2.2 Metacharacter Semantics

Regular expression metacharacters function as operators within the pattern language:

**Anchors** assert position without consuming characters:
- `^` matches the start of a line (or string in single-line mode)
- `$` matches the end of a line (or string before final newline)
- `\b` matches word boundaries (transitions between \w and \W)
- `\B` matches non-boundary positions

**Character classes** match sets of characters:
- `[abc]` matches any single character from the set {a, b, c}
- `[a-z]` matches any lowercase letter (range notation)
- `[^abc]` matches any character except a, b or c (negation)
- `\d`, `\w`, `\s` match digits, word characters, whitespace respectively
- `\D`, `\W`, `\S` match their complements

**Quantifiers** specify repetition:
- `*` matches zero or more occurrences (greedy)
- `+` matches one or more occurrences (greedy)
- `?` matches zero or one occurrence (greedy)
- `{n}` matches exactly n occurrences
- `{n,m}` matches between n and m occurrences (inclusive)
- `{n,}` matches n or more occurrences

By default, quantifiers are greedy—they match as many characters as possible whilst permitting the overall pattern to succeed. Appending `?` produces non-greedy (lazy) quantifiers that match minimally.

**Grouping and alternation**:
- `(...)` creates a capturing group, enabling extraction
- `(?:...)` creates a non-capturing group for structure without capture
- `(?P<name>...)` creates a named capturing group
- `|` denotes alternation (logical OR between alternatives)

### 2.3 Advanced Constructs

**Lookahead assertions** examine context without consuming:
- `(?=...)` positive lookahead: succeeds if pattern matches ahead
- `(?!...)` negative lookahead: succeeds if pattern does not match ahead

**Lookbehind assertions** examine preceding context:
- `(?<=...)` positive lookbehind: succeeds if pattern matches behind
- `(?<!...)` negative lookbehind: succeeds if pattern does not match behind

These zero-width assertions enable context-sensitive matching. For example, `(?<=\$)\d+` matches numbers preceded by dollar signs without including the dollar sign in the match.

**Flags** modify pattern interpretation:
- `re.IGNORECASE` (or `re.I`): case-insensitive matching
- `re.MULTILINE` (or `re.M`): `^` and `$` match line boundaries
- `re.DOTALL` (or `re.S`): `.` matches newlines
- `re.VERBOSE` (or `re.X`): permits whitespace and comments in patterns

### 2.4 The Python re Module

Python's `re` module provides the standard interface for regex operations:

```python
import re

# Compilation (optional but efficient for repeated use)
pattern = re.compile(r'\b\d{4}\b')

# Searching
match = pattern.search(text)      # First match or None
matches = pattern.findall(text)   # All non-overlapping matches
iterator = pattern.finditer(text) # Iterator of Match objects

# Substitution
result = pattern.sub(replacement, text)  # Replace all matches
result = pattern.subn(replacement, text) # Replace with count

# Splitting
parts = pattern.split(text)  # Split on pattern matches
```

The raw string prefix `r"..."` prevents Python from interpreting backslashes, ensuring they reach the regex engine intact.

---

## 3. Unicode and Encoding

### 3.1 Character Encoding Fundamentals

Character encoding maps between human-readable symbols and their digital representations. Understanding encoding is essential because text files are ultimately byte sequences; interpretation requires knowing which encoding was used.

**ASCII** (American Standard Code for Information Interchange) encodes 128 characters in 7 bits, covering English letters, digits and common punctuation. Its limitation to English-centric characters proved inadequate for international computing.

**Unicode** provides a universal character set assigning unique code points (integers) to over 140,000 characters across all writing systems. A code point is written as U+XXXX (e.g., U+00E9 for é). Unicode is an abstract character repertoire; it says nothing about byte representation.

**UTF-8** encodes Unicode code points using variable-length byte sequences (1–4 bytes). ASCII characters use single bytes identical to their ASCII values, ensuring backward compatibility. UTF-8 dominates web content and is the recommended encoding for text files.

**UTF-16** uses 2 or 4 bytes per character. It is common on Windows systems and in Java/JavaScript strings. Byte order marks (BOM) may indicate endianness.

### 3.2 Unicode Normalisation

Identical-appearing text may have distinct Unicode representations. The character "é" can be represented as:
- U+00E9 (precomposed: Latin Small Letter E with Acute)
- U+0065 U+0301 (decomposed: e + Combining Acute Accent)

These representations are canonically equivalent but have different byte sequences. Comparison operations may fail unexpectedly without normalisation.

The Unicode Consortium defines four normalisation forms:
- **NFC** (Canonical Decomposition, followed by Canonical Composition): prefers composed characters
- **NFD** (Canonical Decomposition): fully decomposed form
- **NFKC** (Compatibility Decomposition, followed by Canonical Composition): normalises compatibility characters
- **NFKD** (Compatibility Decomposition): fully decomposed with compatibility normalisation

For text processing, NFC is typically appropriate for storage and comparison; NFKD provides aggressive normalisation useful for searching.

```python
import unicodedata

text = "café"  # May contain composed or decomposed é
normalised = unicodedata.normalize('NFC', text)
```

---

## 4. Tokenisation: Segmenting Text

### 4.1 The Tokenisation Problem

Tokenisation—the segmentation of text into meaningful units—appears deceptively simple. Splitting on whitespace produces reasonable results for well-formatted English prose but fails for:

- **Contractions**: "don't" → ["don", "'t"] or ["do", "n't"] or ["don't"]?
- **Hyphenated compounds**: "state-of-the-art" → one token or four?
- **Abbreviations**: "Dr. Smith" → sentence boundary or title?
- **Numbers**: "3.14159" → one token (number) or three?
- **URLs and emails**: "user@example.com" → how many tokens?
- **Non-English text**: Chinese lacks whitespace entirely

Tokenisation decisions propagate through downstream processing. Different tokenisation schemes yield different vocabularies, frequency counts and analytical results.

### 4.2 Tokenisation Strategies

**Whitespace tokenisation** splits on space characters. Simple and fast but inadequate for most applications.

**Rule-based tokenisation** applies language-specific heuristics. NLTK's word tokenisers handle English contractions; spaCy's tokeniser uses extensive exception lists.

**Subword tokenisation** breaks words into smaller units. Byte Pair Encoding (BPE) and WordPiece learn vocabularies from corpora, handling unknown words gracefully. These methods underlie modern neural language models.

**Sentence tokenisation** identifies sentence boundaries. Abbreviations, decimal points and ellipses complicate period-based approaches. NLTK's Punkt tokeniser uses unsupervised learning to identify abbreviations.

```python
from nltk.tokenize import word_tokenize, sent_tokenize

text = "Dr. Smith's analysis won't be ready until 3:00 p.m."
words = word_tokenize(text)  # ['Dr.', 'Smith', "'s", 'analysis', ...]
sentences = sent_tokenize(text)  # [entire string - no sentence break]
```

---

## 5. Text Normalisation

### 5.1 The Normalisation Imperative

Raw text exhibits variation irrelevant to many analytical purposes. "Running", "RUNNING" and "running" likely convey equivalent meaning; treating them as distinct types inflates vocabulary size and fragments frequency counts. Normalisation reduces this variation, mapping surface forms to canonical representations.

The degree of normalisation involves trade-offs. Aggressive normalisation reduces vocabulary size and improves statistical power but may eliminate meaningful distinctions. "US" (country) and "us" (pronoun) differ significantly; case folding erases this difference.

### 5.2 Stemming

Stemming applies rule-based affix removal to reduce words to stems. The Porter stemmer, published in 1980, remains widely used. Its cascade of suffix-stripping rules transforms words like "computational" → "comput".

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
stemmer.stem("computational")  # 'comput'
stemmer.stem("computation")    # 'comput'
stemmer.stem("computing")      # 'comput'
```

Stemming's rule-based nature produces occasional anomalies: "university" and "universe" may stem identically despite semantic dissimilarity. The Snowball stemmer provides improved accuracy and multilingual support.

### 5.3 Lemmatisation

Lemmatisation employs dictionary lookup to find canonical forms (lemmas). Unlike stemming, it produces valid words: "better" → "good", "was" → "be". Accurate lemmatisation requires part-of-speech information—"meeting" as noun lemmatises to "meeting" whilst as verb it lemmatises to "meet".

```python
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatiser = WordNetLemmatizer()
lemmatiser.lemmatize("running", pos=wordnet.VERB)  # 'run'
lemmatiser.lemmatize("running", pos=wordnet.NOUN)  # 'running'
```

Lemmatisation produces more interpretable results than stemming but incurs computational overhead and requires linguistic resources.

### 5.4 Stopword Removal

Stopwords—high-frequency function words like "the", "is" and "of"—often contribute little to content analysis. Removing them reduces data dimensionality and highlights content words.

```python
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
tokens = [t for t in tokens if t.lower() not in stop_words]
```

Custom stopword lists may be necessary for domain-specific applications. Legal text analysis might retain "shall" and "hereby" despite their stopword-like frequency in that domain.

---

## 6. Text Feature Extraction

### 6.1 Bag-of-Words Representation

The bag-of-words model represents documents as vectors of word frequencies, discarding word order. A document becomes a sparse vector in a high-dimensional vocabulary space.

```python
from collections import Counter

def bag_of_words(tokens: list[str]) -> dict[str, int]:
    return Counter(tokens)
```

Despite its simplicity, bag-of-words supports effective text classification, clustering and information retrieval. Its limitations—ignoring syntax, semantics and word order—motivate more sophisticated representations.

### 6.2 N-gram Models

N-grams capture local word sequences. Bigrams (n=2) record word pairs; trigrams (n=3) record triples. N-grams partially preserve word order and capture collocations.

```python
from nltk import ngrams

text_tokens = ["natural", "language", "processing"]
bigrams = list(ngrams(text_tokens, 2))
# [('natural', 'language'), ('language', 'processing')]
```

Higher-order n-grams capture more context but suffer from sparsity—most possible n-grams never appear in any corpus.

### 6.3 TF-IDF Weighting

Term Frequency–Inverse Document Frequency (TF-IDF) weights terms by both local and global frequency. A term appearing frequently in one document but rarely across the corpus receives high weight; terms appearing everywhere receive low weight.

**Term Frequency (TF)**: Raw count or normalised frequency of term t in document d.

**Inverse Document Frequency (IDF)**: Logarithmic measure of term rarity:
$$\text{IDF}(t) = \log\frac{N}{|\{d : t \in d\}|}$$
where N is total documents and the denominator counts documents containing t.

**TF-IDF**: The product TF × IDF assigns high weights to distinctive terms.

Consider how decomposition transforms an approach to sentiment analysis—determining the emotional tone of a text. Rather than assessing sentiment as a unified whole, a computational approach decomposes the task into distinct steps: preprocessing the text to normalise format, splitting it into sentences, analysing sentiment at the word level, aggregating to determine sentence-level sentiment and finally computing overall document sentiment.

This decomposition principle applies equally to TF-IDF computation: tokenise, count, compute document frequencies, calculate IDF and finally weight terms.

---

## 7. Part-of-Speech Tagging

### 7.1 Linguistic Foundations

Part-of-speech (POS) tagging assigns grammatical categories to tokens: noun, verb, adjective, adverb, preposition and others. The Penn Treebank tagset defines 36 tags for English, distinguishing singular nouns (NN) from plural (NNS), base-form verbs (VB) from past tense (VBD).

POS information enables more accurate text processing:
- Lemmatisation requires POS to distinguish verb from noun forms
- Named entity recognition uses POS patterns (proper noun sequences)
- Phrase extraction identifies noun phrases and verb phrases
- Sentiment analysis weights adjectives and adverbs differently from nouns

### 7.2 Tagging Approaches

Modern POS taggers employ statistical or neural methods trained on annotated corpora. spaCy uses convolutional neural networks; NLTK provides both rule-based and statistical taggers.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The quick brown fox jumps over the lazy dog.")
for token in doc:
    print(f"{token.text}: {token.pos_}")
# The: DET, quick: ADJ, brown: ADJ, fox: NOUN, ...
```

State-of-the-art taggers achieve approximately 97% accuracy on standard benchmarks—sufficient for most research applications but imperfect on domain-specific terminology or informal text.

---

## 8. Building Text Processing Pipelines

### 8.1 Pipeline Architecture

Professional text processing organises transformations into modular pipelines. Each stage accepts input from its predecessor and produces output for its successor:

```
Raw Text → Encoding Normalisation → Tokenisation → Case Folding → 
Stopword Removal → Stemming/Lemmatisation → Feature Extraction → Output
```

Pipeline stages are independent units testable in isolation. This modularity facilitates debugging, experimentation and reuse.

### 8.2 Design Principles

**Immutability**: Each stage produces new data rather than modifying input. This prevents subtle bugs from shared state.

**Composability**: Stages conform to consistent interfaces (text in, text or tokens out), enabling flexible recombination.

**Configurability**: Pipeline parameters (stopword lists, stemmer choice, n-gram size) are external configuration rather than hardcoded values.

**Logging**: Each stage records its transformations, enabling reproducibility and debugging.

A list comprehension like `[x**2 for x in range(10) if x % 2 == 0]` generates a list containing the squares of even numbers from 0 to 9, combining iteration, filtering and transformation in a single expression (Hudak, 1989, p. 384). This functional style translates naturally to text processing pipelines where each stage transforms its input through filtering and mapping operations.

---

## 9. Research Applications

### 9.1 Domain-Specific Considerations

Different research domains require tailored text processing approaches:

**Historical documents**: OCR errors necessitate error-tolerant matching; archaic spelling requires historical normalisation; multilingual documents demand language detection.

**Social media**: Informal spelling, hashtags, mentions and emoji require specialised tokenisers; short text length limits statistical methods.

**Scientific literature**: Citation extraction, equation handling and technical terminology challenge standard tools; domain vocabularies require custom resources.

**Legal text**: Formal language, defined terms and structured references follow domain conventions; cross-reference resolution requires document-level processing.

### 9.2 Index Structures for Text

Efficient text search at scale requires specialised data structures. Domain-specific indices represent how research domains have developed specialised indexing structures optimised for particular data types. Examples include B-trees for database applications, suffix trees for text processing and R-trees for spatial data.

Suffix trees enable efficient substring search, pattern matching and repeat detection—operations fundamental to computational biology and plagiarism detection. Understanding when to employ such structures distinguishes scalable text processing from naive approaches.

---

## 10. Summary

Text processing transforms raw character sequences into structured representations suitable for computational analysis. This transformation proceeds through multiple stages: encoding normalisation ensures consistent byte interpretation; tokenisation segments continuous text into discrete units; normalisation reduces surface variation through stemming or lemmatisation; feature extraction produces numerical representations for machine learning.

Regular expressions provide powerful pattern-matching capabilities essential for text extraction and validation. Unicode awareness prevents encoding-related failures. NLP techniques—tokenisation, POS tagging, named entity recognition—extract linguistic structure from unstructured text.

These competencies enable research across disciplines: literary scholars analyse stylistic patterns across centuries of text; social scientists quantify sentiment in millions of social media posts; historians extract named entities from digitised archives. The techniques presented herein constitute foundational infrastructure for such computational inquiry.

---

## References

Hudak, P. (1989). Conception, evolution, and application of functional programming languages. *ACM Computing Surveys*, 21(3), 359–411.

Jockers, M. L., & Underwood, T. (2016). Text-mining the humanities. In S. Schreibman, R. Siemens, & J. Unsworth (Eds.), *A new companion to digital humanities* (pp. 291–306). Wiley-Blackwell.

---

© 2025 Antonio Clim. All rights reserved.
