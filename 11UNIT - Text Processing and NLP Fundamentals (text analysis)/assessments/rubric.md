# Unit 11: Assessment Rubric

## Text Processing and NLP Fundamentals

This rubric provides detailed criteria for evaluating student work across laboratory assignments, homework exercises and the knowledge assessment quiz.

---

## Grading Scale

| Grade | Percentage | Description |
|-------|------------|-------------|
| Excellent | 90-100% | Exceeds expectations; demonstrates mastery of concepts and implementation |
| Good | 75-89% | Meets expectations; demonstrates competence with minor gaps |
| Satisfactory | 60-74% | Partially meets expectations; foundational understanding present |
| Needs Improvement | Below 60% | Does not meet expectations; significant gaps in understanding |

---

## Laboratory Assessment (40%)

### Lab 11_01: Regex and String Operations (20%)

| Criterion | Excellent (4) | Good (3) | Satisfactory (2) | Needs Improvement (1) |
|-----------|---------------|----------|------------------|----------------------|
| Pattern correctness | All patterns work correctly including edge cases; handles malformed input gracefully | Most patterns work; minor edge case issues that do not affect primary functionality | Patterns work for basic cases; struggles with complex scenarios | Patterns have significant errors affecting core functionality |
| Code quality | Clean, well-documented, complete type hints; follows PEP 8 conventions throughout | Good structure, minor documentation gaps; type hints mostly present | Basic structure, inconsistent documentation; partial type hinting | Poor structure, no documentation; no type hints |
| Function selection | Optimal re module function choices with clear justification; appropriate use of compilation | Appropriate choices for most scenarios; minor inefficiencies | Some inappropriate choices; confusion between similar functions | Incorrect function usage; fundamental misunderstanding of API |
| Unicode handling | Correct encoding throughout; proper normalisation form selection | Handles common encodings; basic normalisation | UTF-8 only; no normalisation consideration | Encoding errors; no awareness of Unicode issues |

### Lab 11_02: NLP Fundamentals (20%)

| Criterion | Excellent (4) | Good (3) | Satisfactory (2) | Needs Improvement (1) |
|-----------|---------------|----------|------------------|----------------------|
| Tokenisation | Custom tokeniser handles all edge cases including contractions, URLs, and domain-specific tokens | Standard tokeniser used correctly; awareness of limitations | Basic tokenisation only; no consideration of edge cases | Incorrect tokenisation; splits on inappropriate boundaries |
| Normalisation | Complete pipeline with justified choices; appropriate stemmer/lemmatiser selection | Good pipeline, minor gaps in justification; correct tool selection | Basic normalisation only; no justification for choices | Missing key steps; inappropriate tool selection |
| Feature extraction | TF-IDF and n-grams implemented correctly with proper mathematical formulation | Basic features computed correctly; minor calculation issues | Partial implementation; conceptual gaps | Significant errors in implementation |
| Pipeline integration | Modular design with clear interfaces; components reusable across contexts | Functional pipeline; some coupling between stages | Linear pipeline only; no modularity | Monolithic code; no separation of concerns |

---

## Homework Assessment (40%)

### Part A: Regex Mastery (30 points)

| Criterion | Points | Description |
|-----------|--------|-------------|
| Pattern accuracy | 0-15 | Patterns correctly match required formats; no false positives or negatives |
| Edge case handling | 0-10 | Patterns handle boundary conditions, empty strings, and malformed input |
| Code documentation | 0-5 | Clear docstrings explaining pattern logic; test cases provided |

### Part B: NLP Pipeline (40 points)

| Criterion | Points | Description |
|-----------|--------|-------------|
| Architecture | 0-15 | Clean, modular design with separation of concerns; follows single-responsibility principle |
| Functionality | 0-15 | All required features implemented; correct output for test cases |
| Documentation | 0-10 | Complete docstrings with type hints; methodology documented |

### Part C: Corpus Analysis (30 points)

| Criterion | Points | Description |
|-----------|--------|-------------|
| Implementation | 0-15 | Correct analysis methods; appropriate statistical measures |
| Interpretation | 0-10 | Insightful conclusions drawn from data; limitations acknowledged |
| Report quality | 0-5 | Clear, well-structured writing; visualisations where appropriate |

---

## Quiz Assessment (20%)

The quiz comprises conceptual questions assessing understanding of core principles. See quiz.md for detailed marking scheme and question breakdown.

| Question Type | Count | Points Each | Focus Area |
|---------------|-------|-------------|------------|
| Multiple choice | 6 | 2 | Regex metacharacters, tokenisation concepts, NLP terminology |
| Short answer | 4 | 2 | Pattern writing, pipeline design, method comparison |

---

## Code Quality Standards

All submitted code must adhere to these standards:

- 100% type hint coverage on function signatures
- Google-style docstrings with Args, Returns, and Raises sections
- No linting errors from ruff or flake8
- Maximum 100 character line length
- British English in all documentation
- Meaningful variable and function names
- Appropriate use of constants for magic values

---

## Late Submission Policy

Submissions after the deadline incur a 10% penalty per day, up to a maximum of three days. Submissions more than three days late receive zero marks unless prior arrangements have been made.

---

## Academic Integrity

All work must be original. Use of AI assistants must be declared and the extent of assistance documented. Plagiarism from online sources or other students results in zero marks and potential disciplinary action.

---

© 2025 Antonio Clim. All rights reserved.
