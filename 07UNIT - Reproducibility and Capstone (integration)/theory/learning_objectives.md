# Week 7: Learning Objectives

## Reproducibility and Capstone

---

## Overview

This document specifies the measurable learning objectives for Week 7, aligned with Bloom's Taxonomy. Each objective includes assessment criteria and example evidence of achievement.

---

## Primary Objectives

### Objective 1: Implement Thorough Testing

**Bloom's Level**: Apply

**Statement**: Implement comprehensive testing with pytest and configure CI/CD pipelines using GitHub Actions.

**Sub-objectives**:

1.1. Write unit tests following the Arrange-Act-Assert pattern
1.2. Create test fixtures using pytest's fixture system
1.3. Apply mocking to isolate components under test
1.4. Configure pytest with appropriate options and plugins
1.5. Set up GitHub Actions workflow for automated testing

**Assessment Criteria**:

| Criterion | Unsatisfactory | Satisfactory | Excellent |
|-----------|---------------|--------------|-----------|
| Test coverage | <50% | 50-80% | >80% |
| Test organisation | Scattered, no fixtures | Uses fixtures | Comprehensive fixture hierarchy |
| CI configuration | Missing or broken | Basic workflow | Multi-version matrix with coverage |
| Mocking usage | None | Basic mocks | Sophisticated isolation patterns |

**Evidence of Achievement**:
- pytest test suite with ≥80% coverage
- Working GitHub Actions workflow
- Tests that demonstrate isolation through mocking

---

### Objective 2: Build Reproducible Project Structure

**Bloom's Level**: Create

**Statement**: Build reproducible project structures with proper documentation, dependency management and seed control.

**Sub-objectives**:

2.1. Create standardised project layout following established conventions
2.2. Configure pyproject.toml with pinned dependencies
2.3. Implement comprehensive seed management
2.4. Write documentation at code, module and project levels
2.5. Create data manifests for integrity verification

**Assessment Criteria**:

| Criterion | Unsatisfactory | Satisfactory | Excellent |
|-----------|---------------|--------------|-----------|
| Project structure | Ad hoc organisation | Standard layout | Professional with all conventions |
| Dependencies | Missing or unpinned | Basic pinning | Versioned with dev extras |
| Seed management | None | Single library | Comprehensive multi-library |
| Documentation | Minimal | README and docstrings | Full documentation site |
| Data integrity | None | Basic manifest | Automated verification |

**Evidence of Achievement**:
- Project following standard src layout
- Complete pyproject.toml
- Reproducible results across independent runs
- README with installation, usage and contribution guidelines

---

### Objective 3: Conduct Peer Review

**Bloom's Level**: Evaluate

**Statement**: Conduct peer review using established criteria and provide constructive feedback following established conventions.

**Sub-objectives**:

3.1. Apply code review criteria systematically
3.2. Evaluate reproducibility of submitted projects
3.3. Assess documentation quality and completeness
3.4. Provide constructive, actionable feedback
3.5. Identify areas for improvement while acknowledging strengths

**Assessment Criteria**:

| Criterion | Unsatisfactory | Satisfactory | Excellent |
|-----------|---------------|--------------|-----------|
| Review thoroughness | Superficial | Covers main points | Comprehensive checklist |
| Feedback quality | Vague or harsh | Clear suggestions | Specific, actionable, constructive |
| Reproducibility check | Not attempted | Partial verification | Complete reproduction |
| Technical accuracy | Contains errors | Generally accurate | Expert-level insights |

**Evidence of Achievement**:
- Completed peer review using provided checklist
- Written feedback demonstrating technical understanding
- Documentation of reproduction attempt

---

## Supporting Objectives

### S1: Understand the Reproducibility Crisis

**Bloom's Level**: Understand

**Statement**: Explain the causes and consequences of the reproducibility crisis in computational research.

**Evidence**: Written explanation identifying at least three causes and three consequences.

---

### S2: Compare Testing Approaches

**Bloom's Level**: Analyse

**Statement**: Compare and contrast unit testing, integration testing and property-based testing.

**Evidence**: Analysis table or essay comparing approaches across dimensions including scope, execution speed, coverage and maintenance burden.

---

### S3: Evaluate CI/CD Configurations

**Bloom's Level**: Evaluate

**Statement**: Evaluate CI/CD configurations for completeness and effectiveness.

**Evidence**: Critique of a provided CI configuration with specific improvement recommendations.

---

## Alignment with Course Objectives

| Week 7 Objective | Course Objective |
|------------------|------------------|
| O1: Testing | Implement algorithms with correctness guarantees |
| O2: Reproducibility | Produce research software meeting professional standards |
| O3: Peer Review | Evaluate computational approaches critically |
| S1: Crisis | Understand context of computational research |
| S2: Testing Approaches | Select appropriate tools for given problems |
| S3: CI/CD | Apply software engineering established conventions |

---

## Capstone Integration

The capstone project provides summative assessment of all Week 7 objectives. Projects will be evaluated on:

1. **Testing** (O1): Test suite coverage and quality
2. **Reproducibility** (O2): Ability to reproduce results from documentation alone
3. **Documentation** (O2): Clarity and completeness
4. **Peer Review** (O3): Quality of review provided to peers

---

## Self-Assessment Checklist

Use this checklist to assess your progress:

### Testing (O1)
- [ ] I can write unit tests using pytest
- [ ] I understand the Arrange-Act-Assert pattern
- [ ] I can create and use pytest fixtures
- [ ] I can mock external dependencies
- [ ] I can configure GitHub Actions for CI

### Reproducibility (O2)
- [ ] I can create standard project layouts
- [ ] I can configure pyproject.toml properly
- [ ] I can manage random seeds across libraries
- [ ] I can write effective documentation
- [ ] I can create data integrity manifests

### Peer Review (O3)
- [ ] I can apply review criteria systematically
- [ ] I can verify reproducibility of code
- [ ] I can provide constructive feedback
- [ ] I can identify both strengths and weaknesses

---

© 2025 Antonio Clim. All rights reserved.
