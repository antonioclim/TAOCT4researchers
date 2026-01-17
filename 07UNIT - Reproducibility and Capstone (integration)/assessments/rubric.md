# Week 7 Grading Rubric

## 📋 Overview

This rubric provides detailed grading criteria for Week 7 assessments including laboratory exercises, homework assignments and the capstone project.

---

## Laboratory Exercises Rubric

### Lab 7.01: Reproducibility Toolkit (40 points)

| Criterion | Excellent (100%) | Good (80%) | Satisfactory (60%) | Needs Improvement (40%) | Incomplete (0%) |
|-----------|-----------------|------------|-------------------|------------------------|-----------------|
| **Seed Management** (10 pts) | Correctly implements seed management for all libraries with state capture/restore | Implements seed management with minor issues | Basic seed setting works but state management incomplete | Seeds set but not consistently applied | Not implemented |
| **Data Manifests** (10 pts) | Complete manifest with SHA-256, metadata and verification | Manifest implementation with minor omissions | Basic hashing implemented | Hash calculation present but incomplete | Not implemented |
| **Environment Capture** (10 pts) | Comprehensive environment capture including versions and hardware | Captures most environment details | Basic package list generated | Minimal environment information | Not implemented |
| **Code Quality** (10 pts) | Type hints, docstrings, logging throughout; passes ruff and mypy | Minor style issues; mostly type-hinted | Some type hints; basic documentation | Few type hints; minimal docs | No quality standards met |

### Lab 7.02: Testing and CI/CD (40 points)

| Criterion | Excellent (100%) | Good (80%) | Satisfactory (60%) | Needs Improvement (40%) | Incomplete (0%) |
|-----------|-----------------|------------|-------------------|------------------------|-----------------|
| **Unit Tests** (10 pts) | Comprehensive tests with edge cases; >90% coverage | Good test coverage; most edge cases | Basic tests pass; >70% coverage | Some tests present; <70% coverage | No tests or tests fail |
| **Fixtures & Mocking** (10 pts) | Effective use of fixtures at appropriate scopes; mocking used correctly | Good fixture design; minor mocking issues | Basic fixtures work; limited mocking | Fixtures present but poorly designed | No fixtures or mocking |
| **CI/CD Configuration** (10 pts) | Complete GitHub Actions workflow with matrix testing and quality checks | Workflow runs successfully with minor gaps | Basic workflow that runs tests | Workflow present but has errors | No CI/CD configuration |
| **Code Quality** (10 pts) | Type hints, docstrings, logging throughout; passes ruff and mypy | Minor style issues; mostly type-hinted | Some type hints; basic documentation | Few type hints; minimal docs | No quality standards met |

### Lab 7.03: Project Scaffolder (20 points)

| Criterion | Excellent (100%) | Good (80%) | Satisfactory (60%) | Needs Improvement (40%) | Incomplete (0%) |
|-----------|-----------------|------------|-------------------|------------------------|-----------------|
| **Template Engine** (5 pts) | Supports variables, conditionals, loops and filters | Variables and conditionals work | Basic variable substitution | Template loading only | Not implemented |
| **Project Generation** (10 pts) | Generates complete project with all standard files and directories | Most files generated correctly | Basic structure created | Partial generation | Not implemented |
| **Customisation** (5 pts) | Configurable options with sensible defaults | Some configuration options | Limited customisation | No configuration | Not implemented |

---

## Homework Rubric (100 points)

### Part 1: Project Structure (25 points)

| Criterion | Points | Requirements |
|-----------|--------|--------------|
| **Directory Layout** | 10 | Follows standard Python project structure (src/, tests/, docs/) |
| **Configuration Files** | 8 | pyproject.toml with all required fields; .gitignore appropriate |
| **Package Initialisation** | 7 | Correct __init__.py files with version and exports |

### Part 2: README Documentation (20 points)

| Criterion | Points | Requirements |
|-----------|--------|--------------|
| **Project Description** | 5 | Clear, concise description of purpose and features |
| **Installation Instructions** | 5 | Complete steps including dependencies |
| **Usage Examples** | 5 | Working code examples with expected output |
| **Contributing Guidelines** | 5 | Clear instructions for contributors |

### Part 3: Code Quality (25 points)

| Criterion | Points | Requirements |
|-----------|--------|--------------|
| **Linting** | 8 | Passes ruff check with no errors |
| **Type Checking** | 8 | Passes mypy --strict with no errors |
| **Docstrings** | 5 | Google-style docstrings on all public functions/classes |
| **Type Hints** | 4 | Complete type annotations on all functions |

### Part 4: Testing (15 points)

| Criterion | Points | Requirements |
|-----------|--------|--------------|
| **Test Coverage** | 8 | ≥70% code coverage |
| **Test Quality** | 4 | Tests are meaningful and test actual functionality |
| **Fixtures** | 3 | Appropriate use of fixtures for shared setup |

### Part 5: Presentation (15 points)

| Criterion | Points | Requirements |
|-----------|--------|--------------|
| **Structure** | 5 | Logical flow: intro, methods, results, conclusion |
| **Technical Depth** | 5 | Demonstrates understanding of concepts |
| **Live Demo** | 5 | Working demonstration of key features |

---

## Capstone Project Rubric

The capstone project is evaluated across four dimensions totalling 100 points.

### Technical Excellence (35 points)

| Score Range | Description |
|-------------|-------------|
| **31-35** | Exceptional implementation demonstrating mastery. Code is clean, well-documented, fully tested (>90% coverage) and uses advanced techniques appropriately. |
| **25-30** | Strong implementation with minor issues. Good documentation and testing (>80% coverage). Uses course concepts effectively. |
| **18-24** | Adequate implementation meeting basic requirements. Documentation present but incomplete. Testing coverage >70%. |
| **10-17** | Partial implementation with significant gaps. Minimal documentation or testing. |
| **0-9** | Incomplete or non-functional implementation. |

### Research Rigour (30 points)

| Score Range | Description |
|-------------|-------------|
| **27-30** | Fully reproducible with comprehensive documentation. Environment specifications complete. Data manifests verify all inputs. Results can be regenerated exactly. |
| **21-26** | Mostly reproducible with minor gaps. Clear instructions provided. Most dependencies documented. |
| **15-20** | Partially reproducible. Some documentation gaps or missing dependencies. |
| **8-14** | Reproducibility issues present. Significant undocumented elements. |
| **0-7** | Not reproducible. Missing critical documentation or dependencies. |

### Communication (25 points)

| Score Range | Description |
|-------------|-------------|
| **23-25** | Exceptional presentation and documentation. Clear narrative, professional slides, engaging delivery. README is comprehensive and well-structured. |
| **18-22** | Good communication with minor issues. Clear explanations, adequate slides. README covers essential information. |
| **13-17** | Adequate communication meeting minimum requirements. Basic slides and documentation. |
| **7-12** | Communication gaps present. Unclear explanations or missing documentation. |
| **0-6** | Poor communication. Incomplete presentation or documentation. |

### Originality (10 points)

| Score Range | Description |
|-------------|-------------|
| **9-10** | Novel application or significant extension of course concepts. Creative problem-solving evident. |
| **7-8** | Some original elements beyond basic requirements. Shows independent thinking. |
| **5-6** | Meets requirements but follows examples closely. Limited original contribution. |
| **3-4** | Minimal original work. Closely follows provided templates. |
| **0-2** | No original contribution. Direct copy of examples. |

---

## Grade Boundaries

| Grade | Percentage | Description |
|-------|------------|-------------|
| A+ | 95-100% | Outstanding achievement exceeding expectations |
| A | 90-94% | Excellent work demonstrating mastery |
| A- | 85-89% | Very good work with minor imperfections |
| B+ | 80-84% | Good work meeting all requirements |
| B | 75-79% | Solid work with some areas for improvement |
| B- | 70-74% | Satisfactory work meeting basic requirements |
| C+ | 65-69% | Adequate work with notable gaps |
| C | 60-64% | Passing work with significant room for improvement |
| C- | 55-59% | Minimally passing work |
| D | 50-54% | Below expectations but shows some effort |
| F | <50% | Failing; does not meet minimum requirements |

---

## Late Submission Policy

| Days Late | Penalty |
|-----------|---------|
| 1 day | -10% |
| 2 days | -20% |
| 3 days | -30% |
| 4+ days | Not accepted without prior arrangement |

Extensions must be requested at least 24 hours before the deadline with valid justification.

---

## Academic Integrity

All submitted work must be original. The following constitute academic misconduct:

- Copying code from other students or online sources without attribution
- Using AI tools to generate substantial portions of code without disclosure
- Submitting work completed by another person
- Sharing solutions with other students before the deadline

Suspected violations will be reported to the academic integrity committee.

---

© 2025 Antonio Clim. All rights reserved.
