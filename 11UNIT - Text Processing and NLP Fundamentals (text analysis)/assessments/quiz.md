# Unit 11: Quiz

## Text Processing and NLP Fundamentals

**Instructions**: Answer all questions. For multiple-choice questions, select the best answer. For short-answer questions, provide concise responses.

**Time Limit**: 20 minutes
**Total Points**: 100

---

## Section A: Multiple Choice (60 points)

### Question 1 (10 points)

Which regex pattern correctly matches a word boundary?

A) `\s`
B) `\b`
C) `\w`
D) `^`

---

### Question 2 (10 points)

What is the difference between `*` and `+` quantifiers in regular expressions?

A) `*` matches one or more; `+` matches zero or more
B) `*` matches zero or more; `+` matches one or more
C) `*` is greedy; `+` is non-greedy
D) `*` matches digits; `+` matches letters

---

### Question 3 (10 points)

Which Python `re` module function returns ALL non-overlapping matches as a list?

A) `re.search()`
B) `re.match()`
C) `re.findall()`
D) `re.split()`

---

### Question 4 (10 points)

What is the primary difference between stemming and lemmatisation?

A) Stemming is faster; lemmatisation is slower
B) Stemming uses rules to remove affixes; lemmatisation uses dictionary lookup
C) Stemming produces valid words; lemmatisation may produce invalid words
D) Stemming requires POS tags; lemmatisation does not

---

### Question 5 (10 points)

In TF-IDF, a term with high IDF indicates that the term:

A) Appears frequently in many documents
B) Appears rarely across the corpus
C) Is a stopword
D) Has high term frequency

---

### Question 6 (10 points)

Which Unicode normalisation form produces precomposed characters (e.g., é as a single code point)?

A) NFD
B) NFC
C) NFKD
D) ASCII

---

## Section B: Short Answer (40 points)

### Question 7 (10 points)

Write a regex pattern that matches UK postcodes in the format "SW1A 1AA" (where the first part is 2-4 alphanumeric characters and the second part is a digit followed by two letters).

**Your answer:**

```
Pattern: _______________________
```

---

### Question 8 (10 points)

Explain why the following two strings might not compare as equal in Python, even though they appear identical:

```python
s1 = "café"      # 4 characters
s2 = "cafe\u0301" # 5 characters (e + combining acute accent)
```

How would you ensure they compare as equal?

**Your answer:**

---

### Question 9 (10 points)

Given the following corpus, calculate the IDF (Inverse Document Frequency) for the term "python":

```
Document 1: ["python", "programming", "code"]
Document 2: ["java", "programming", "enterprise"]
Document 3: ["python", "data", "science"]
```

Show your calculation. Use the formula: IDF(t) = log(N / df(t)) where N is total documents and df(t) is document frequency of term t.

**Your answer:**

---

### Question 10 (10 points)

Design a simple text preprocessing pipeline for sentiment analysis. List the stages in order and briefly explain why each is necessary.

**Your answer:**

---

---

## Section C: Practical Application (Bonus - 20 points)

### Question 11 (10 points)

Given the following text, identify and explain three potential challenges for an automated text processing system:

```
"Dr. Smith's analysis (cf. Fig. 3.2) shows that U.S. GDP grew by 2.5% in Q4 2024—
a significant improvement over last year's -0.3% decline!"
```

Consider tokenisation, named entity recognition and numeric extraction challenges.

**Your answer:**

---

### Question 12 (10 points)

Design a preprocessing pipeline for analysing customer reviews. Specify:
- The order of operations
- Which techniques you would apply
- Justification for each choice

Your pipeline should prepare text for sentiment classification.

**Your answer:**

---

## Study Notes

### Key Formulas

**Term Frequency (TF):**
```
TF(t, d) = count(t in d) / total_tokens(d)
```

**Inverse Document Frequency (IDF):**
```
IDF(t) = log(N / df(t))
```
Where N = total documents, df(t) = documents containing term t

**TF-IDF:**
```
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

### Common Pitfalls

1. **Regex greediness**: Remember that `.*` is greedy by default; use `.*?` for non-greedy matching
2. **Unicode normalisation**: Always normalise Unicode before comparison; composed and decomposed forms may appear identical but differ in byte representation
3. **Stopword removal timing**: Remove stopwords after tokenisation but before stemming or lemmatisation
4. **Case sensitivity**: Lowercase text early in the pipeline unless case carries semantic information

### Performance Tips

1. Compile regex patterns once, reuse many times
2. Use `re.finditer()` instead of `re.findall()` for memory efficiency on large texts
3. Consider lazy loading for NLTK resources
4. Batch process documents when computing TF-IDF

---

## Answer Key

*(For instructor use)*

**Section A:**
1. B - The `\b` metacharacter matches word boundaries
2. B - `*` matches zero or more occurrences; `+` requires at least one
3. C - `re.findall()` returns all non-overlapping matches as a list
4. B - Stemming uses algorithmic rules; lemmatisation uses dictionary lookup
5. B - High IDF indicates the term is rare across the corpus
6. B - NFC (Canonical Composition) produces precomposed characters

**Section B:**
7. `[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}` (or similar valid pattern)
   - Explanation: First part has 1-2 letters, then a digit, optionally another alphanumeric, then space(s), then digit followed by two letters

8. The strings use different Unicode representations (precomposed vs decomposed). The first string uses a precomposed é (U+00E9), whilst the second uses e (U+0065) followed by a combining acute accent (U+0301). Use `unicodedata.normalize('NFC', s1) == unicodedata.normalize('NFC', s2)` to ensure equality by converting both to the same canonical form.

9. IDF("python") = log(3/2) ≈ 0.405
   - N = 3 (total documents)
   - df("python") = 2 (appears in documents 1 and 3)
   - IDF = log(3/2) = log(1.5) ≈ 0.405

10. Sample preprocessing pipeline for sentiment analysis:
    1. **Tokenisation** - Split text into individual words to enable word-level analysis
    2. **Lowercasing** - Normalise case to reduce vocabulary size and ensure consistent matching
    3. **Stopword removal** - Remove common function words that add noise without sentiment value
    4. **Lemmatisation** - Reduce words to base forms to group related terms (preferred over stemming for accuracy)
    5. **Feature extraction** - Convert to numerical representation (bag-of-words or TF-IDF vectors)

**Section C (Bonus):**
11. Challenges in the given text:
    - Abbreviations: "Dr.", "U.S.", "cf.", "Fig." contain periods that may be confused with sentence boundaries
    - Numbers with context: "2.5%", "-0.3%", "Q4 2024", "3.2" require special handling to preserve meaning
    - Special punctuation: Em dash (—), parentheses and quotation marks need appropriate handling

12. Review preprocessing pipeline example:
    1. HTML/special character removal (clean web artifacts)
    2. Tokenisation (word-level, preserve contractions)
    3. Lowercasing (normalise case)
    4. Stopword removal (remove noise)
    5. Negation handling (preserve "not good" semantics)
    6. Lemmatisation (normalise word forms)
    7. Feature extraction (TF-IDF or embeddings)

---

© 2025 Antonio Clim. All rights reserved.
