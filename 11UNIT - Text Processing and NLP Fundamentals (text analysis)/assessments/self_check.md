# Unit 11: Self-Assessment Checklist

## Text Processing and NLP Fundamentals

Use this checklist to assess your understanding before the quiz. Each section corresponds to key competencies required for successful completion of this unit.

---

## Regular Expressions

### Metacharacters
- [ ] I can explain what each metacharacter does: `. ^ $ * + ? { } [ ] \ | ( )`
- [ ] I understand the difference between `\d`, `\w`, and `\s` and their negated forms
- [ ] I know how to escape literal metacharacters using backslash
- [ ] I can distinguish between metacharacter behaviour inside and outside character classes

### Quantifiers
- [ ] I can use `*`, `+`, `?`, and `{n,m}` correctly in patterns
- [ ] I understand greedy vs non-greedy matching and when each applies
- [ ] I can make quantifiers non-greedy by appending `?`
- [ ] I can specify exact repetition counts using curly brace notation

### Groups and Assertions
- [ ] I can create capturing groups with `()` for extraction
- [ ] I can use named groups `(?P<name>...)` for improved readability
- [ ] I understand lookahead `(?=)` and lookbehind `(?<=)` assertions
- [ ] I can apply negative lookahead `(?!)` and lookbehind `(?<!)` assertions
- [ ] I know when to use non-capturing groups `(?:...)` for efficiency

### Python re Module
- [ ] I know when to use `search()` vs `match()` vs `findall()` vs `finditer()`
- [ ] I can use `re.sub()` for replacements with backreferences
- [ ] I understand regex flags: IGNORECASE, MULTILINE, VERBOSE, DOTALL
- [ ] I can compile patterns for improved performance in repeated use

---

## String Operations

- [ ] I can use `split()`, `join()`, and `strip()` effectively for text manipulation
- [ ] I understand string encoding and decoding between bytes and strings
- [ ] I can work with Unicode text and handle encoding errors appropriately
- [ ] I know the four Unicode normalisation forms (NFC, NFD, NFKC, NFKD) and their uses
- [ ] I can apply string translation tables for character mapping

---

## NLP Fundamentals

### Tokenisation
- [ ] I can tokenise text into words and sentences using NLTK
- [ ] I understand challenges: contractions, abbreviations, hyphenation, URLs
- [ ] I can implement custom tokenisation rules for domain-specific needs
- [ ] I recognise when subword tokenisation may be appropriate

### Normalisation
- [ ] I can apply stemming using Porter or Snowball stemmer
- [ ] I understand when to use lemmatisation instead of stemming
- [ ] I can remove stopwords appropriately for the analysis context
- [ ] I know the trade-offs between aggressive and conservative normalisation

### Text Features
- [ ] I can compute term frequencies from a document
- [ ] I understand and can calculate TF-IDF weights
- [ ] I can extract n-grams (bigrams, trigrams) from text
- [ ] I understand the difference between bag-of-words and sequence-based representations

---

## Pipeline Design

- [ ] I can design modular preprocessing pipelines with clear interfaces
- [ ] I understand that the order of operations affects results
- [ ] I can document design decisions and justify methodological choices
- [ ] I can adapt pipelines for different text domains (literary, scientific, social media)

---

## Quick Self-Test

Answer these questions to verify your readiness:

1. What pattern matches "colour" or "color"? → `colou?r`
2. What does `\b` match? → Word boundary (transition between word and non-word characters)
3. Difference between stemming and lemmatisation? → Rule-based suffix removal vs dictionary-based canonical form lookup
4. What does high IDF mean? → The term is rare across the corpus, appearing in few documents
5. Why use raw strings for regex patterns? → Prevents Python from interpreting backslashes as escape sequences

---

## Reflection Prompts

Consider these questions for deeper understanding:

- How would you modify a tokeniser for processing social media text with hashtags and mentions?
- What preprocessing steps would you change when analysing historical documents versus modern news articles?
- When might you choose not to remove stopwords from your analysis?

---

© 2025 Antonio Clim. All rights reserved.
