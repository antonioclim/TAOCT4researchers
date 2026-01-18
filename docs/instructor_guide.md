# Instructor Guide

## The Art of Computational Thinking for Researchers

**Extended Edition — Fourteen Instructional Units**

---

## Introduction

This guide provides pedagogical guidance for instructors delivering the TAOCT4researchers curriculum in formal educational settings. It addresses course planning, instructional strategies, assessment practices and common challenges.

**Note:** Use of these materials in formal teaching requires prior written authorisation from the author. Please contact via the repository issue tracker with the `[LICENCE]` tag.

---

## Course Planning

### Audience Analysis

Before delivery, assess your learners':

| Dimension | Questions to Consider |
|-----------|----------------------|
| **Prior Knowledge** | What programming experience do participants have? Have they completed introductory Python courses? |
| **Disciplinary Background** | What research domains are represented? What computational methods are common in their fields? |
| **Learning Goals** | Are participants seeking breadth or depth? Are there specific units of particular relevance? |
| **Time Constraints** | What is the available contact time? Can participants engage with materials outside sessions? |
| **Technical Resources** | Do participants have access to appropriate computing environments? |

### Pathway Selection

Based on audience analysis, select an appropriate pathway:

**Full Curriculum (143 hours)**
- Suitable for: Dedicated computational methods courses
- Format: Full semester or year-long programme
- Prerequisites: Basic Python proficiency

**Data Science Focus (96 hours)**
- Units: 01 → 02 → 03 → 04 → 05 → 06 → 10 → 11 → 13
- Suitable for: Research data analysis courses
- Emphasis: Numerical methods, visualisation, ML

**Software Engineering Focus (74 hours)**
- Units: 01 → 02 → 03 → 04 → 07 → 08 → 09 → 12
- Suitable for: Research software development
- Emphasis: Design patterns, testing, robustness

**Foundations Intensive (55 hours)**
- Units: 01 → 02 → 03 → 04 → 05 → 06 → 07
- Suitable for: Introduction to computational research
- Emphasis: Core skills and integration

---

## Instructional Strategies

### Session Structure

Recommended structure for a typical 2-hour session:

| Segment | Duration | Activity |
|---------|----------|----------|
| Opening | 5 min | Recap previous session, preview objectives |
| Theory | 20 min | Lecture delivery with interactive elements |
| Demonstration | 15 min | Live coding or worked example |
| Practice | 30 min | Guided laboratory work |
| Independent Work | 30 min | Exercise completion |
| Synthesis | 15 min | Discussion, Q&A, preview next session |
| Buffer | 5 min | Administrative matters |

### Active Learning Techniques

The curriculum supports various active learning approaches:

**Think-Pair-Share**
- Pose conceptual question from quiz materials
- Individual reflection (1 minute)
- Pair discussion (2 minutes)
- Class sharing (3 minutes)

**Live Coding**
- Project your screen and code alongside participants
- Make deliberate errors to demonstrate debugging
- Verbalise your thought process
- Pause frequently for questions

**Peer Instruction**
- Present multiple-choice concept question
- Individual response (clickers or hands)
- If <70% correct, pair discussion then re-vote
- If ≥70% correct, brief explanation and proceed

**Collaborative Problem Solving**
- Assign medium-difficulty exercises to pairs
- Rotate pairs across sessions
- Designate one "driver" (typing) and one "navigator" (directing)
- Switch roles midway through exercise

### Scaffolding Strategies

For units with high cognitive load:

1. **Worked Examples First**
   - Complete a similar problem before assigning practice
   - Annotate key decision points
   - Highlight common pitfalls

2. **Fading Scaffolds**
   - Begin with partial solutions requiring completion
   - Gradually reduce provided code
   - Final exercises require full implementation

3. **Chunking Content**
   - Break lengthy laboratories into sub-tasks
   - Provide checkpoints with test verification
   - Allow completion across multiple sessions

---

## Assessment Guidance

### Formative Assessment

Use formative assessment to guide instruction:

| Tool | Purpose | Frequency |
|------|---------|-----------|
| Quiz questions | Check conceptual understanding | Each session |
| Laboratory checkpoints | Verify procedural skills | During labs |
| Self-assessment | Encourage metacognition | End of each unit |
| Minute papers | Gather real-time feedback | As needed |

### Summative Assessment

For graded courses, suggested weighting:

| Component | Weight | Notes |
|-----------|--------|-------|
| Laboratory completion | 40% | Verified via test suites |
| Quizzes | 20% | Can be administered online |
| Exercises | 20% | Subset of medium/hard exercises |
| Capstone project | 20% | Unit 07 integration project |

### Rubric Usage

Each unit includes assessment rubrics. When grading:

1. Review rubric criteria before evaluating
2. Apply criteria consistently across submissions
3. Provide specific feedback referencing rubric dimensions
4. Calibrate with co-instructors if team-teaching

---

## Common Challenges

### Technical Issues

| Challenge | Mitigation |
|-----------|-----------|
| Environment setup failures | Provide pre-configured virtual environment; use cloud-based alternatives (Google Colab, GitHub Codespaces) |
| Dependency conflicts | Pin versions in requirements.txt; provide troubleshooting guide |
| Performance limitations | Reduce dataset sizes for classroom use; pre-compute expensive results |
| Network restrictions | Prepare offline alternatives; cache external resources |

### Pedagogical Challenges

| Challenge | Strategy |
|-----------|----------|
| **Wide skill variance** | Use tiered exercises (easy/medium/hard); pair stronger with developing learners |
| **Conceptual difficulties** | Multiple representations (visual, verbal, code); additional worked examples |
| **Disengagement** | Connect to participants' research domains; invite guest researchers |
| **Time pressure** | Identify core vs. optional content; provide take-home materials |

### Unit-Specific Notes

**Unit 01 (Epistemology)**
- Abstract concepts may challenge participants
- Emphasise practical relevance through AST lab
- Use visualisations extensively

**Unit 03 (Complexity)**
- Mathematical notation may intimidate some learners
- Focus on intuition before formal definitions
- Empirical benchmarking provides concrete grounding

**Unit 08 (Recursion/DP)**
- Recursion is a common stumbling point
- Use extensive tracing and visualisation
- Start with familiar examples (factorial, Fibonacci)

**Unit 13 (Machine Learning)**
- Participants may have inflated expectations from media
- Emphasise rigorous evaluation over flashy results
- Connect to reproducibility concerns from Unit 07

**Unit 14 (Parallel Computing)**
- Debugging parallel code is difficult
- Start with embarrassingly parallel problems
- Emphasise profiling before parallelising

---

## Classroom Setup

### Physical Space

Optimal classroom configuration:

- Projector/screen visible from all positions
- Individual workstations (one per participant)
- Instructor workstation with screen sharing capability
- Whiteboard for diagrams and explanations
- Power outlets accessible to all positions

### Virtual Delivery

For online instruction:

- Use screen sharing with annotation capability
- Provide breakout rooms for pair work
- Share code via collaborative editors (VS Code Live Share)
- Record sessions for asynchronous review
- Maintain chat channel for questions

### Computing Environment

Recommended setup:

```bash
# Create standardised environment
python -m venv taoct_env
source taoct_env/bin/activate
pip install -r requirements.txt

# Verify installation
make check
```

Consider providing:
- Pre-configured virtual machine image
- Docker container with all dependencies
- Cloud-based notebook environment

---

## Materials Preparation

### Before the Course

1. **Review all materials** — Ensure familiarity with content
2. **Test all code** — Verify examples run in target environment
3. **Prepare supplementary examples** — Domain-specific illustrations
4. **Configure assessment tools** — Quiz platforms, submission systems
5. **Establish communication channels** — Discussion forums, office hours

### Before Each Session

1. **Review unit objectives** — What should participants achieve?
2. **Select exercises** — Which difficulty levels are appropriate?
3. **Prepare demonstrations** — Test live coding examples
4. **Anticipate questions** — Review FAQ and troubleshooting guides
5. **Check technology** — Projector, environment, network

### After Each Session

1. **Collect feedback** — Minute papers, informal discussions
2. **Review submissions** — Identify common difficulties
3. **Adjust pacing** — Modify subsequent sessions as needed
4. **Update materials** — Note improvements for future iterations

---

## Inclusive Teaching

### Accessibility Considerations

- Ensure visualisations have text alternatives
- Provide transcripts for any audio/video content
- Allow flexible deadlines when appropriate
- Offer multiple submission formats

### Supporting Diverse Backgrounds

- Avoid jargon without explanation
- Acknowledge multiple valid approaches
- Encourage questions at all levels
- Create psychologically safe environment for errors

### International Learners

- Speak clearly and at moderate pace
- Provide written materials alongside verbal instruction
- Allow additional time for non-native English speakers
- Be aware of cultural differences in participation norms

---

## Continuous Improvement

### Gathering Feedback

| Method | Timing | Purpose |
|--------|--------|---------|
| Pre-course survey | Before start | Assess prior knowledge, expectations |
| Mid-course check | Week 7 | Identify issues early |
| End-of-unit reflection | Each unit | Unit-specific feedback |
| Final evaluation | Course end | Overall assessment |

### Iterating on Materials

If you identify improvements:

1. Document the issue or suggestion
2. Test proposed changes in isolation
3. Submit via repository contribution process
4. Share insights with author for curriculum updates

---

## Resources

### Supplementary Materials

- [Curriculum Overview](curriculum.md)
- [Pacing Guide](pacing.md)
- Unit-specific cheatsheets in `resources/` directories
- Interactive visualisations in `assets/animations/` directories

### Professional Development

Recommended reading for computational education:

- Guzdial, M. (2015). *Learner-Centered Design of Computing Education*
- Wilson, G. (2019). *Teaching Tech Together*
- Brown, N.C.C., & Wilson, G. (2018). Ten quick tips for teaching programming

### Support

For instructor support:

- Repository Issues: https://github.com/antonioclim/TAOCT4researchers/issues
- Tag instructor queries with `[INSTRUCTOR]`

---

## Licence Reminder

Teaching or presenting these materials to third parties requires prior written consent from the author. Please review [LICENCE.md](../LICENCE.md) and contact via official channels for educational use permissions.

---

*Last updated: January 2026*
