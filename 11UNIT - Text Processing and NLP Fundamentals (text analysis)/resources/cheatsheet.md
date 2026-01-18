# Unit 11: Cheatsheet

## Text Processing and NLP Fundamentals

---

## Regular Expressions Quick Reference

### Metacharacters
| Char | Meaning | Example |
|------|---------|---------|
| `.` | Any character (except newline) | `a.c` → abc, aXc |
| `^` | Start of string/line | `^Hello` |
| `$` | End of string/line | `World$` |
| `*` | Zero or more | `ab*c` → ac, abc, abbc |
| `+` | One or more | `ab+c` → abc, abbc |
| `?` | Zero or one | `colou?r` → color, colour |
| `\|` | Alternation (OR) | `cat\|dog` |
| `()` | Grouping/capture | `(ab)+` → ab, abab |
| `[]` | Character class | `[aeiou]` → vowels |
| `\` | Escape | `\.` → literal dot |

### Character Classes
| Pattern | Matches |
|---------|---------|
| `\d` | Digit [0-9] |
| `\D` | Non-digit |
| `\w` | Word char [a-zA-Z0-9_] |
| `\W` | Non-word char |
| `\s` | Whitespace |
| `\S` | Non-whitespace |
| `\b` | Word boundary |
| `[a-z]` | Lowercase letters |
| `[^abc]` | NOT a, b, or c |

### Quantifiers
| Pattern | Meaning |
|---------|---------|
| `{n}` | Exactly n times |
| `{n,}` | n or more times |
| `{n,m}` | Between n and m |
| `*?` | Non-greedy zero+ |
| `+?` | Non-greedy one+ |

### Groups and Assertions
| Pattern | Meaning |
|---------|---------|
| `(...)` | Capturing group |
| `(?:...)` | Non-capturing group |
| `(?P<n>...)` | Named group |
| `(?=...)` | Positive lookahead |
| `(?!...)` | Negative lookahead |
| `(?<=...)` | Positive lookbehind |
| `(?<!...)` | Negative lookbehind |

---

## Python re Module

```python
import re

# Search (first match)
match = re.search(r'\d+', text)
if match:
    print(match.group())  # Matched string
    print(match.start())  # Start position

# Find all
matches = re.findall(r'\d+', text)  # List of strings

# Find with details
for m in re.finditer(r'\d+', text):
    print(m.group(), m.span())

# Replace
result = re.sub(r'\d+', 'NUM', text)

# Split
parts = re.split(r'[,;]', text)

# Compile (for reuse)
pattern = re.compile(r'\d+', re.IGNORECASE)
```

### Flags
| Flag | Short | Effect |
|------|-------|--------|
| `re.IGNORECASE` | `re.I` | Case-insensitive |
| `re.MULTILINE` | `re.M` | ^ $ match line boundaries |
| `re.DOTALL` | `re.S` | . matches newlines |
| `re.VERBOSE` | `re.X` | Allow whitespace/comments |

---

## String Methods

```python
# Case
s.lower(), s.upper(), s.title(), s.capitalize()

# Whitespace
s.strip(), s.lstrip(), s.rstrip()
s.split(), s.split(','), s.splitlines()

# Search
s.find('x'), s.rfind('x'), s.count('x')
s.startswith('x'), s.endswith('x')

# Modify
s.replace('old', 'new')
' '.join(['a', 'b', 'c'])  # 'a b c'

# Testing
s.isalpha(), s.isdigit(), s.isalnum()
s.isspace(), s.isupper(), s.islower()
```

---

## Unicode

```python
import unicodedata

# Normalisation
text = unicodedata.normalize('NFC', text)  # Composed
text = unicodedata.normalize('NFD', text)  # Decomposed

# Remove accents
def remove_accents(s):
    nfkd = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))

# Character info
unicodedata.name('é')  # 'LATIN SMALL LETTER E WITH ACUTE'
unicodedata.category('A')  # 'Lu' (Letter, uppercase)
```

---

## Tokenisation

```python
# Simple
tokens = text.split()  # Whitespace
tokens = re.findall(r'\w+', text)  # Words only

# NLTK
from nltk.tokenize import word_tokenize, sent_tokenize
words = word_tokenize(text)
sentences = sent_tokenize(text)

# spaCy
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
tokens = [token.text for token in doc]
```

---

## Normalisation

```python
# Stemming
from nltk.stem import PorterStemmer, SnowballStemmer
stemmer = PorterStemmer()
stem = stemmer.stem('running')  # 'run'

# Snowball (multi-language)
stemmer = SnowballStemmer('english')
stem = stemmer.stem('connection')  # 'connect'

# Lemmatisation
from nltk.stem import WordNetLemmatizer
lemmatiser = WordNetLemmatizer()
lemma = lemmatiser.lemmatize('better', pos='a')  # 'good'

# Stopwords
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
filtered = [w for w in tokens if w.lower() not in stop_words]
```

---

## spaCy Pipeline

```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup.")

# Access tokens
for token in doc:
    print(token.text, token.pos_, token.dep_, token.lemma_)

# Named entities
for ent in doc.ents:
    print(ent.text, ent.label_)  # Apple ORG, U.K. GPE

# Sentence segmentation
for sent in doc.sents:
    print(sent.text)
```

---

## Text Features

```python
from collections import Counter

# Frequency
freq = Counter(tokens)
top_10 = freq.most_common(10)

# N-grams
def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

bigrams = ngrams(tokens, 2)

# TF-IDF (sklearn)
from sklearn.feature_extraction.text import TfidfVectorizer
vectoriser = TfidfVectorizer()
tfidf = vectoriser.fit_transform(documents)

# Get feature names
feature_names = vectoriser.get_feature_names_out()
```

---

## Common Patterns

```python
# Email
r'\b[\w.+-]+@[\w.-]+\.\w{2,}\b'

# URL
r'https?://[\w.-]+(?:/[\w./?&=-]*)?'

# Date (ISO)
r'\d{4}-\d{2}-\d{2}'

# Phone
r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'

# UK Postcode
r'[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}'

# IP Address (IPv4)
r'\b(?:\d{1,3}\.){3}\d{1,3}\b'

# Hashtag
r'#\w+'

# Mention
r'@\w+'
```

---

## Pipeline Example

```python
def preprocess(text):
    # 1. Lowercase
    text = text.lower()
    # 2. Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # 3. Tokenise
    tokens = text.split()
    # 4. Remove stopwords
    tokens = [t for t in tokens if t not in stop_words]
    # 5. Lemmatise
    tokens = [lemmatiser.lemmatize(t) for t in tokens]
    return tokens
```

---

© 2025 Antonio Clim. All rights reserved.
