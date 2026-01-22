# Glossary — 11UNIT

- **Tokenisation**: Mapping a character sequence to a sequence of discrete symbols (tokens) according to a specified rule set.
- **Normalisation**: A transformation applied to text to reduce superficial variation, such as Unicode canonical forms or case folding.
- **Regular expression**: A formal pattern describing a regular language; in practice, a compact specification for recognisers used in parsing.
- **n-gram**: A contiguous sequence of n tokens or characters used as a local context feature in statistical models.
- **TF–IDF**: A weighting scheme for term counts that down-weights terms that appear in many documents and up-weights terms distinctive to particular documents.
- **Precision**: The proportion of predicted positives that are correct; sensitive to false positives.
- **Recall**: The proportion of actual positives that are recovered; sensitive to false negatives.
- **F1 score**: The harmonic mean of precision and recall, balancing the two when a single number is required.
- **Stop word**: A term filtered from a representation, often because it is frequent and carries limited topical information, though this is domain-dependent.
- **Stemming**: A heuristic reduction of inflected or derived words to a common form, often conflating multiple lexical items.
- **Lemmatisation**: Normalising word forms to dictionary lemmas using morphological analysis and part-of-speech information.
- **Bag of words**: A representation that ignores token order and records only token counts or weights.
- **Document frequency**: The number of documents in which a term occurs, used in inverse document frequency calculations.
- **Confusion matrix**: A tabulation of predicted versus true labels, supporting per-class error analysis.
- **Pipeline**: A composed sequence of transformations where each stage has defined inputs, outputs and evaluation criteria.
- **Gold standard**: A reference annotation set, treated as ground truth for evaluation while recognising its own uncertainty.
- **Annotation**: A mapping from raw data to labels or structures for analysis or supervised learning.
- **Character encoding**: A mapping from bytes to characters; incorrect assumptions yield corruption or replacement glyphs.
- **Unicode**: A standard assigning code points to characters across writing systems, enabling consistent text processing.
- **Regular language**: A class of languages recognised by finite automata and described by regular expressions.
- **Finite automaton**: A computational model with a finite set of states used to recognise regular languages.
- **Vectoriser**: A component that maps text to numeric feature vectors suitable for statistical learning.
- **Sparsity**: A property of feature vectors where most components are zero, common in high-dimensional text representations.
- **Domain drift**: A shift in text distribution between training and deployment contexts, often degrading model performance.
- **Named entity**: A span of text that denotes an entity such as a person, organisation or location, depending on the annotation scheme.

Definitions are operationalised within the labs is treated here as a research-oriented craft rather
than a collection of library calls. The central question is how an analyst moves from raw character
sequences to defensible claims, given that token boundaries, normalisation choices and annotation
schemes impose inductive bias. We therefore separate specification from implementation: we first
state what a pipeline must preserve or discard, then encode those constraints as pure functions with
explicit preconditions. Attention is paid to failure modes, including ambiguous segmentation,
encoding artefacts and domain drift, since these frequently dominate error budgets in empirical
studies. In practice, the unit couples formal definitions with executable checks so that each step
has a measurable contract: input alphabet, transformation invariants and complexity expectations.
Definitions are operationalised within the labs is treated here as a research-oriented craft rather
than a collection of library calls. The central question is how an analyst moves from raw character
sequences to defensible claims, given that token boundaries, normalisation choices and annotation
schemes impose inductive bias. We therefore separate specification from implementation: we first
state what a pipeline must preserve or discard, then encode those constraints as pure functions with
explicit preconditions. Attention is paid to failure modes, including ambiguous segmentation,
encoding artefacts and domain drift, since these frequently dominate error budgets in empirical
studies. In practice, the unit couples formal definitions with executable checks so that each step
has a measurable contract: input alphabet, transformation invariants and complexity expectations.
Definitions are operationalised within the labs is treated here as a research-oriented craft rather
than a collection of library calls. The central question is how an analyst moves from raw character
sequences to defensible claims, given that token boundaries, normalisation choices and annotation
schemes impose inductive bias. We therefore separate specification from implementation: we first
state what a pipeline must preserve or discard, then encode those constraints as pure functions with
explicit preconditions. Attention is paid to failure modes, including ambiguous segmentation,
encoding artefacts and domain drift, since these frequently dominate error budgets in empirical
studies. In practice, the unit couples formal definitions with executable checks so that each step
has a measurable contract: input alphabet, transformation invariants and complexity expectations.
Definitions are operationalised within the labs is treated here as a research-oriented craft rather
than a collection of library calls. The central question is how an analyst moves from raw character
sequences to defensible claims, given that token boundaries, normalisation choices and annotation
schemes impose inductive bias. We therefore separate specification from implementation: we first
state what a pipeline must preserve or discard, then encode those constraints as pure functions with
explicit preconditions. Attention is paid to failure modes, including ambiguous segmentation,
encoding artefacts and domain drift, since these frequently dominate error budgets in empirical
studies. In practice, the unit couples formal definitions with executable checks so that each step
has a measurable contract: input alphabet, transformation invariants and complexity expectations.
Definitions are operationalised within the labs is treated here as a research-oriented craft rather
than a collection of library calls. The central question is how an analyst moves from raw character
sequences to defensible claims, given that token boundaries, normalisation choices and annotation
schemes impose inductive bias. We therefore separate specification from implementation: we first
state what a pipeline must preserve or discard, then encode those constraints as pure functions with
explicit preconditions. Attention is paid to failure modes, including ambiguous segmentation,
encoding artefacts and domain drift, since these frequently dominate error budgets in empirical
studies. In practice, the unit couples formal definitions with executable checks so that each step
has a measurable contract: input alphabet, transformation invariants and complexity expectations.
Definitions are operationalised within the labs is treated here as a research-oriented craft rather
than a collection of library calls. The central question is how an analyst moves from raw character
sequences to defensible claims, given that token boundaries, normalisation choices and annotation
schemes impose inductive bias. We therefore separate specification from implementation: we first
state what a pipeline must preserve or discard, then encode those constraints as pure functions with
explicit preconditions. Attention is paid to failure modes, including ambiguous segmentation,
encoding artefacts and domain drift, since these frequently dominate error budgets in empirical
studies. In practice, the unit couples formal definitions with executable checks so that each step
has a measurable contract: input alphabet, transformation invariants and complexity expectations.
