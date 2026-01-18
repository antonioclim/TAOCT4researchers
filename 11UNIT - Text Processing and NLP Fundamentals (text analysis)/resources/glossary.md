# Unit 11: Glossary

## Text Processing and NLP Fundamentals

This glossary defines key terms and concepts covered in Unit 11.

---

## A

### Anchor
A regex construct that matches a position rather than a character. Common anchors include `^` (start of string), `$` (end of string), and `\b` (word boundary). Anchors are zero-width assertions—they do not consume characters from the input string.

### ASCII
American Standard Code for Information Interchange. A character encoding standard using 7 bits to represent 128 characters, including English letters, digits, punctuation and control characters. ASCII forms the basis for many modern encodings and is a subset of UTF-8.

---

## B

### Bag-of-Words (BoW)
A text representation model that treats documents as unordered collections of words, disregarding grammar and word order. Each document becomes a vector of word frequencies. Despite its simplicity, BoW serves as an effective baseline for text classification and information retrieval tasks.

### Bigram
A contiguous sequence of two items (typically words) extracted from a text. Bigrams capture local context and common collocations. For example, "natural language" is a bigram in NLP texts. Part of the broader family of n-grams.

---

## C

### Capturing Group
A regex construct using parentheses `()` that both groups pattern elements and captures the matched text for extraction. Captured groups can be referenced by number (`\1`, `\2`) or by name when using named groups `(?P<name>...)`.

### Character Class
A regex construct using square brackets `[]` to match any single character from a specified set. For example, `[aeiou]` matches any vowel, whilst `[0-9]` matches any digit. Negation with `[^...]` matches characters NOT in the set.

### Code Point
A numerical value in the Unicode standard that maps to a specific character. Code points are written as U+XXXX (hexadecimal). For example, U+0041 represents 'A' and U+00E9 represents 'é'.

### Collocation
A sequence of words that co-occur more frequently than expected by chance. Collocations include compound terms ("ice cream"), phrasal verbs ("give up"), and idioms ("kick the bucket"). N-gram analysis helps identify collocations in corpora.

### Corpus (pl. Corpora)
A large, structured collection of texts used for linguistic analysis, machine learning training, or statistical study. Corpora may be general (news articles) or domain-specific (medical literature). Proper corpus construction requires careful sampling and annotation.

---

## D

### Document Frequency (DF)
The number of documents in a corpus that contain a particular term. Used in TF-IDF calculation to weight terms by their rarity. Terms appearing in many documents have high DF and low IDF.

---

## E

### Encoding
The method by which characters are represented as bytes for storage or transmission. Common encodings include UTF-8 (variable-length, ASCII-compatible), UTF-16 (2 or 4 bytes), and Latin-1 (single-byte Western European). Encoding mismatches cause mojibake—garbled text.

---

## F

### Finite Automaton
A computational model consisting of states and transitions, used to recognise patterns in strings. Regular expressions describe languages that can be recognised by finite automata. This theoretical foundation ensures regex matching operates in linear time.

---

## G

### Greedy Matching
The default behaviour of regex quantifiers, which match as many characters as possible whilst still allowing the overall pattern to succeed. Non-greedy (lazy) matching, indicated by `?` after the quantifier, matches as few characters as possible.

---

## H

### Hapax Legomenon
A word that appears exactly once in a corpus. Hapax legomena are significant in authorship attribution and vocabulary analysis. The proportion of hapaxes indicates lexical diversity—creative or specialised texts often have higher hapax ratios.

---

## I

### Inverse Document Frequency (IDF)
A measure of term rarity across a corpus, calculated as the logarithm of total documents divided by document frequency. IDF weights terms by their distinctiveness—common terms like "the" have low IDF, whilst rare terms have high IDF.

---

## L

### Lemma
The canonical or dictionary form of a word. "run" is the lemma for "running", "runs", and "ran". Unlike stems, lemmas are valid words found in dictionaries.

### Lemmatisation
The process of reducing words to their lemmas using morphological analysis and dictionary lookup. More accurate than stemming but slower and requiring part-of-speech information for disambiguation. "better" lemmatises to "good" (as adjective).

### Lookahead
A regex assertion that checks whether a pattern exists ahead of the current position without consuming characters. Positive lookahead `(?=...)` succeeds if the pattern matches; negative lookahead `(?!...)` succeeds if it does not match.

### Lookbehind
A regex assertion that checks whether a pattern exists behind the current position. Positive lookbehind `(?<=...)` requires the pattern to precede; negative lookbehind `(?<!...)` requires it not to precede. Lookbehind patterns must have fixed length in Python.

---

## M

### Metacharacter
A character with special meaning in regular expressions, as opposed to a literal character. The regex metacharacters are: `. ^ $ * + ? { } [ ] \ | ( )`. To match a metacharacter literally, escape it with backslash.

---

## N

### Named Group
A regex capturing group with an assigned name, written as `(?P<name>...)` in Python. Named groups improve code readability and can be accessed by name rather than number: `match.group('name')`.

### N-gram
A contiguous sequence of n items from a text. Unigrams (n=1) are individual tokens; bigrams (n=2) are pairs; trigrams (n=3) are triples. N-grams capture local context and are used for language modelling, text classification and collocation detection.

### Normalisation (Text)
The process of transforming text into a consistent, canonical form. Text normalisation may include case folding, Unicode normalisation, accent removal, spelling correction and abbreviation expansion. Normalisation reduces vocabulary size and improves matching.

### Normalisation (Unicode)
The process of converting Unicode text to a standard form to ensure equivalent strings compare as equal. NFC (composed) combines characters where possible; NFD (decomposed) separates base characters from combining marks. NFKC and NFKD additionally normalise compatibility characters.

---

## P

### Part-of-Speech (POS) Tagging
The process of assigning grammatical categories (noun, verb, adjective, etc.) to tokens. POS tags enable syntax-aware processing such as lemmatisation, phrase extraction and named entity recognition. Common tagsets include the Penn Treebank tags.

### Pattern (Regex)
A string specifying a search pattern using regular expression syntax. Patterns may include literal characters, metacharacters, quantifiers, groups and assertions. In Python, patterns are typically written as raw strings (`r"..."`) to avoid backslash interpretation.

### Porter Stemmer
An influential stemming algorithm published by Martin Porter in 1980. It applies a cascade of suffix-stripping rules to reduce English words to stems. Though sometimes producing non-words ("studies" → "studi"), it remains widely used for its speed and simplicity.

---

## Q

### Quantifier
A regex construct specifying how many times a pattern element may repeat. Basic quantifiers include `*` (zero or more), `+` (one or more), `?` (zero or one). Bounded quantifiers `{n}`, `{n,}`, `{n,m}` specify exact counts or ranges.

---

## S

### Stemming
Rule-based reduction of words to their stems by removing affixes. Faster than lemmatisation but less accurate—may produce non-words and conflate unrelated terms. The Porter and Snowball stemmers are commonly used implementations.

### Stopword
A high-frequency word with low semantic content, typically filtered during preprocessing. Common English stopwords include "the", "is", "at", "which". Domain-specific corpora may require customised stopword lists.

---

## T

### Term Frequency (TF)
The count of a term's occurrences within a document, often normalised by document length. High TF indicates a term's importance within a specific document. Combined with IDF to produce TF-IDF weights.

### TF-IDF
Term Frequency–Inverse Document Frequency. A numerical statistic reflecting a term's importance within a document relative to a corpus. High TF-IDF indicates a term that is frequent locally but rare globally—a distinctive term for that document.

### Token
A unit of text produced by tokenisation, typically a word, punctuation mark or number. Tokens serve as the basic units for subsequent NLP processing. Token boundaries depend on the tokenisation strategy employed.

### Tokenisation
The segmentation of text into discrete units (tokens). Word tokenisation produces word-level tokens; sentence tokenisation identifies sentence boundaries. Challenges include handling contractions, abbreviations, hyphenation and non-whitespace-delimited languages.

### Type-Token Ratio (TTR)
A measure of lexical diversity calculated as vocabulary size divided by total tokens. Higher TTR indicates more varied vocabulary. TTR is sensitive to text length—longer texts tend toward lower TTR as common words accumulate.

---

## U

### Unicode
An international standard defining characters from virtually all writing systems, assigning each a unique code point. Unicode separates character identity (code points) from byte representation (encodings like UTF-8). Essential for processing multilingual text.

### UTF-8
A variable-length Unicode encoding using 1–4 bytes per character. ASCII characters use single bytes, maintaining backward compatibility. UTF-8 is the dominant encoding for web content and the recommended default for text files.

---

## W

### Word Boundary
A position in text between a word character (`\w`) and a non-word character (`\W`), or at the start/end of the string adjacent to a word character. Matched by the regex anchor `\b`. Useful for matching whole words: `\bcat\b` matches "cat" but not "category".

---

© 2025 Antonio Clim. All rights reserved.
