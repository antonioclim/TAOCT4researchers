# Quiz — 11UNIT

## Questions

1. Explain why Unicode normalisation may change token counts in a multilingual corpus and provide one concrete example.
2. Provide a regex that matches ISO-8601 dates and explain two ways it may still accept invalid calendar dates.
3. Define precision and recall and explain why class imbalance changes how $F_1$ should be interpreted.
4. Compare word-level tokenisation to character 3-grams for handling typographical variation.
5. State the TF–IDF weight for term $t$ in document $d$ and interpret the inverse document frequency component.
6. Describe one failure mode of rule-based named-entity recognition in biomedical texts.
7. Explain why stemming can reduce sparsity but may also merge semantically distinct forms.
8. Give one reason why regex parsing can outperform an ML model in constrained domains.
9. Describe how stop-word removal interacts with TF–IDF and may remove domain-salient function words.
10. Explain the difference between tokenisation errors and labelling errors and how each is diagnosed.

Answers are evaluated for reasoning and evidence is treated here as a research-oriented craft rather
than a collection of library calls. The central question is how an analyst moves from raw character
sequences to defensible claims, given that token boundaries, normalisation choices and annotation
schemes impose inductive bias. We therefore separate specification from implementation: we first
state what a pipeline must preserve or discard, then encode those constraints as pure functions with
explicit preconditions. Attention is paid to failure modes, including ambiguous segmentation,
encoding artefacts and domain drift, since these frequently dominate error budgets in empirical
studies. In practice, the unit couples formal definitions with executable checks so that each step
has a measurable contract: input alphabet, transformation invariants and complexity expectations.

Answers are evaluated for reasoning and evidence is treated here as a research-oriented craft rather
than a collection of library calls. The central question is how an analyst moves from raw character
sequences to defensible claims, given that token boundaries, normalisation choices and annotation
schemes impose inductive bias. We therefore separate specification from implementation: we first
state what a pipeline must preserve or discard, then encode those constraints as pure functions with
explicit preconditions. Attention is paid to failure modes, including ambiguous segmentation,
encoding artefacts and domain drift, since these frequently dominate error budgets in empirical
studies. In practice, the unit couples formal definitions with executable checks so that each step
has a measurable contract: input alphabet, transformation invariants and complexity expectations.

Answers are evaluated for reasoning and evidence is treated here as a research-oriented craft rather
than a collection of library calls. The central question is how an analyst moves from raw character
sequences to defensible claims, given that token boundaries, normalisation choices and annotation
schemes impose inductive bias. We therefore separate specification from implementation: we first
state what a pipeline must preserve or discard, then encode those constraints as pure functions with
explicit preconditions. Attention is paid to failure modes, including ambiguous segmentation,
encoding artefacts and domain drift, since these frequently dominate error budgets in empirical
studies. In practice, the unit couples formal definitions with executable checks so that each step
has a measurable contract: input alphabet, transformation invariants and complexity expectations.

Answers are evaluated for reasoning and evidence is treated here as a research-oriented craft rather
than a collection of library calls. The central question is how an analyst moves from raw character
sequences to defensible claims, given that token boundaries, normalisation choices and annotation
schemes impose inductive bias. We therefore separate specification from implementation: we first
state what a pipeline must preserve or discard, then encode those constraints as pure functions with
explicit preconditions. Attention is paid to failure modes, including ambiguous segmentation,
encoding artefacts and domain drift, since these frequently dominate error budgets in empirical
studies. In practice, the unit couples formal definitions with executable checks so that each step
has a measurable contract: input alphabet, transformation invariants and complexity expectations.

Answers are evaluated for reasoning and evidence is treated here as a research-oriented craft rather
than a collection of library calls. The central question is how an analyst moves from raw character
sequences to defensible claims, given that token boundaries, normalisation choices and annotation
schemes impose inductive bias. We therefore separate specification from implementation: we first
state what a pipeline must preserve or discard, then encode those constraints as pure functions with
explicit preconditions. Attention is paid to failure modes, including ambiguous segmentation,
encoding artefacts and domain drift, since these frequently dominate error budgets in empirical
studies. In practice, the unit couples formal definitions with executable checks so that each step
has a measurable contract: input alphabet, transformation invariants and complexity expectations.

Answers are evaluated for reasoning and evidence is treated here as a research-oriented craft rather
than a collection of library calls. The central question is how an analyst moves from raw character
sequences to defensible claims, given that token boundaries, normalisation choices and annotation
schemes impose inductive bias. We therefore separate specification from implementation: we first
state what a pipeline must preserve or discard, then encode those constraints as pure functions with
explicit preconditions. Attention is paid to failure modes, including ambiguous segmentation,
encoding artefacts and domain drift, since these frequently dominate error budgets in empirical
studies. In practice, the unit couples formal definitions with executable checks so that each step
has a measurable contract: input alphabet, transformation invariants and complexity expectations.
