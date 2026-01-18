# Week 11: Learning Objectives

## Text Processing and NLP Fundamentals

---

## Overview

This document specifies the measurable learning objectives for Week 11, aligned with Bloom's Taxonomy. Each objective includes assessment criteria and example evidence of achievement.

---

## Primary Objectives

### Objective 1: Explain Regular Expression Concepts

**Bloom's Level**: Understand

**Statement**: Explain regular expression syntax, metacharacters and matching semantics with accurate technical vocabulary.

**Sub-objectives**:

1.1. Describe the function of each regex metacharacter (. ^ $ * + ? { } [ ] \ | ( ))
1.2. Explain the difference between greedy and non-greedy quantifiers
1.3. Distinguish capturing groups from non-capturing groups
1.4. Describe lookahead and lookbehind assertions
1.5. Explain how regex flags modify pattern behaviour

**Assessment Criteria**:

| Criterion | Unsatisfactory | Satisfactory | Excellent |
|-----------|---------------|--------------|-----------|
| Metacharacter knowledge | Fewer than half identified | Most identified correctly | All identified with examples |
| Quantifier understanding | Confuses greedy/non-greedy | Correct explanation | Demonstrates with edge cases |
| Group comprehension | Cannot distinguish types | Explains capturing groups | Explains named groups and backreferences |
| Assertion knowledge | Cannot explain | Describes basic lookahead | Explains all four assertion types |

**Evidence of Achievement**:
- Written explanation of regex components
- Correct interpretation of complex patterns
- Pattern design with appropriate construct selection

---

### Objective 2: Implement Text Extraction with Regex

**Bloom's Level**: Apply

**Statement**: Implement text extraction and validation using regular expression patterns in Python.

**Sub-objectives**:

2.1. Write patterns to extract structured data (emails, dates, URLs)
2.2. Apply appropriate re module functions (search, findall, finditer, sub)
2.3. Use groups to extract specific portions of matches
2.4. Implement validation patterns with anchors
2.5. Handle edge cases through pattern refinement

**Assessment Criteria**:

| Criterion | Unsatisfactory | Satisfactory | Excellent |
|-----------|---------------|--------------|-----------|
| Pattern correctness | Patterns fail on test cases | Patterns pass most cases | Patterns handle edge cases |
| Function selection | Incorrect function choice | Appropriate function use | Optimal function with justification |
| Group usage | No extraction capability | Basic group extraction | Named groups with clean code |
| Validation patterns | Incomplete validation | Correct anchoring | Comprehensive with lookahead |

**Evidence of Achievement**:
- Working extraction scripts for structured data
- Validation functions with comprehensive test coverage
- Refactored patterns demonstrating iterative improvement

---

### Objective 3: Build Text Preprocessing Pipelines

**Bloom's Level**: Apply

**Statement**: Build text preprocessing pipelines incorporating tokenisation, normalisation and stopword removal.

**Sub-objectives**:

3.1. Implement word and sentence tokenisation
3.2. Apply case normalisation appropriately
3.3. Integrate stopword removal with customisation
3.4. Handle Unicode normalisation for multilingual text
3.5. Compose pipeline stages into coherent workflows

**Assessment Criteria**:

| Criterion | Unsatisfactory | Satisfactory | Excellent |
|-----------|---------------|--------------|-----------|
| Tokenisation | Whitespace-only | Uses proper tokeniser | Custom tokeniser for domain |
| Normalisation | Inconsistent case | Proper case folding | Unicode-aware normalisation |
| Stopwords | No removal | Standard list applied | Custom domain-specific list |
| Pipeline design | Ad hoc processing | Sequential stages | Configurable, logged pipeline |

**Evidence of Achievement**:
- Preprocessing pipeline handling diverse text inputs
- Documentation of pipeline stages and rationale
- Test cases demonstrating pipeline correctness

---

### Objective 4: Apply NLP Techniques

**Bloom's Level**: Apply

**Statement**: Apply NLP techniques including stemming, lemmatisation, POS tagging and n-gram extraction.

**Sub-objectives**:

4.1. Apply stemming algorithms (Porter, Snowball)
4.2. Implement lemmatisation with POS awareness
4.3. Perform POS tagging using NLTK or spaCy
4.4. Extract n-grams of specified lengths
4.5. Select appropriate technique for given requirements

**Assessment Criteria**:

| Criterion | Unsatisfactory | Satisfactory | Excellent |
|-----------|---------------|--------------|-----------|
| Stemming | Cannot apply | Correct stemmer use | Compares stemmer outputs |
| Lemmatisation | Ignores POS | Basic lemmatisation | POS-informed lemmatisation |
| POS tagging | Cannot perform | Correct tag assignment | Tag interpretation and filtering |
| N-grams | Incorrect extraction | Correct n-gram lists | Efficient extraction with analysis |

**Evidence of Achievement**:
- Code demonstrating all four techniques
- Comparative analysis of stemming vs lemmatisation
- N-gram frequency analysis of sample corpus

---

### Objective 5: Analyse Text Corpora

**Bloom's Level**: Analyse

**Statement**: Analyse text corpora using frequency analysis and TF-IDF weighting to identify significant terms.

**Sub-objectives**:

5.1. Compute term frequency distributions
5.2. Calculate TF-IDF weights for document collections
5.3. Interpret TF-IDF weights to identify distinctive terms
5.4. Compare term distributions across subcorpora
5.5. Evaluate the impact of preprocessing on analysis results

**Assessment Criteria**:

| Criterion | Unsatisfactory | Satisfactory | Excellent |
|-----------|---------------|--------------|-----------|
| Frequency analysis | Incorrect counting | Correct frequencies | Visualised distributions |
| TF-IDF computation | Formula errors | Correct calculation | Variant formula comparison |
| Interpretation | No insight | Basic term identification | Nuanced corpus characterisation |
| Comparative analysis | Single corpus only | Two corpora compared | Multi-corpus with statistical tests |

**Evidence of Achievement**:
- Frequency analysis report for sample corpus
- TF-IDF ranking with interpretation
- Comparative analysis of two or more document sets

---

### Objective 6: Design Text Processing Pipelines

**Bloom's Level**: Create

**Statement**: Design end-to-end text processing pipelines tailored to specific research applications.

**Sub-objectives**:

6.1. Assess requirements for given research questions
6.2. Select appropriate preprocessing stages
6.3. Implement pipeline with proper architecture
6.4. Validate pipeline against diverse inputs
6.5. Document design decisions and limitations

**Assessment Criteria**:

| Criterion | Unsatisfactory | Satisfactory | Excellent |
|-----------|---------------|--------------|-----------|
| Requirements analysis | Missing key requirements | Basic requirements captured | Comprehensive specification |
| Stage selection | Inappropriate choices | Reasonable selection | Optimal choices with justification |
| Implementation | Non-functional | Working pipeline | Resilient with error handling |
| Documentation | None or minimal | Basic documentation | Professional with examples |

**Evidence of Achievement**:
- Pipeline designed for specific research scenario
- Documentation of design rationale
- Test suite validating pipeline behaviour

---

## Supporting Objectives

### S1: Recall Regex Metacharacters

**Bloom's Level**: Remember

**Statement**: Recall common regex metacharacters and their functions from memory.

**Evidence**: Correctly identify at least 10 metacharacters with their meanings without reference.

---

### S2: Distinguish Stemming and Lemmatisation

**Bloom's Level**: Understand

**Statement**: Distinguish between stemming and lemmatisation approaches, identifying appropriate use cases.

**Evidence**: Written comparison explaining algorithmic differences, output characteristics and selection criteria.

---

### S3: Handle Unicode Text

**Bloom's Level**: Apply

**Statement**: Handle Unicode text with appropriate encoding specification and normalisation.

**Evidence**: Code correctly processing multilingual text with explicit encoding and normalisation.

---

### S4: Compare TF-IDF Weights

**Bloom's Level**: Analyse

**Statement**: Compare TF-IDF weights across document collections to characterise content differences.

**Evidence**: Analysis report identifying distinctive terms in contrasted corpora with interpretation.

---

## Alignment with Course Objectives

| Week 11 Objective | Course Objective |
|-------------------|------------------|
| O1: Regex concepts | Understand computational primitives |
| O2: Regex implementation | Implement algorithms correctly |
| O3: Preprocessing pipelines | Build modular software components |
| O4: NLP techniques | Apply domain-specific methods |
| O5: Corpus analysis | Analyse data systematically |
| O6: Pipeline design | Create research software |
| S1-S4 | Supporting competencies |

---

## Self-Assessment Checklist

Use this checklist to assess your progress:

### Regular Expressions (O1, O2)
- [ ] I can explain the function of each metacharacter
- [ ] I understand greedy vs non-greedy matching
- [ ] I can write patterns with capturing groups
- [ ] I can use lookahead and lookbehind assertions
- [ ] I can select appropriate re module functions

### Preprocessing (O3)
- [ ] I can implement tokenisation correctly
- [ ] I understand Unicode normalisation forms
- [ ] I can customise stopword lists
- [ ] I can compose pipeline stages

### NLP Techniques (O4)
- [ ] I can apply stemming algorithms
- [ ] I can perform POS-aware lemmatisation
- [ ] I can extract n-grams
- [ ] I can select appropriate techniques for requirements

### Corpus Analysis (O5, O6)
- [ ] I can compute term frequencies
- [ ] I can calculate TF-IDF weights
- [ ] I can interpret analysis results
- [ ] I can design complete pipelines

---

## Preparation for 13UNIT

The competencies developed in this unit prepare directly for machine learning text applications:

- Preprocessing pipelines produce clean input for ML models
- TF-IDF vectors serve as features for text classification
- N-gram features capture local patterns for various ML tasks
- Understanding text representation enables informed model selection

---

© 2025 Antonio Clim. All rights reserved.
