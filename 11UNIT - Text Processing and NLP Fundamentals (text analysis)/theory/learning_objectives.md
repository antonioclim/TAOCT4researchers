# Learning objectives — 11UNIT

## Measurable outcomes

Learning outcomes are framed as observable competences is treated here as a research-oriented craft
rather than a collection of library calls. The central question is how an analyst moves from raw
character sequences to defensible claims, given that token boundaries, normalisation choices and
annotation schemes impose inductive bias. We therefore separate specification from implementation:
we first state what a pipeline must preserve or discard, then encode those constraints as pure
functions with explicit preconditions. Attention is paid to failure modes, including ambiguous
segmentation, encoding artefacts and domain drift, since these frequently dominate error budgets in
empirical studies. In practice, the unit couples formal definitions with executable checks so that
each step has a measurable contract: input alphabet, transformation invariants and complexity
expectations.

Learning outcomes are framed as observable competences is treated here as a research-oriented craft
rather than a collection of library calls. The central question is how an analyst moves from raw
character sequences to defensible claims, given that token boundaries, normalisation choices and
annotation schemes impose inductive bias. We therefore separate specification from implementation:
we first state what a pipeline must preserve or discard, then encode those constraints as pure
functions with explicit preconditions. Attention is paid to failure modes, including ambiguous
segmentation, encoding artefacts and domain drift, since these frequently dominate error budgets in
empirical studies. In practice, the unit couples formal definitions with executable checks so that
each step has a measurable contract: input alphabet, transformation invariants and complexity
expectations.

Learning outcomes are framed as observable competences is treated here as a research-oriented craft
rather than a collection of library calls. The central question is how an analyst moves from raw
character sequences to defensible claims, given that token boundaries, normalisation choices and
annotation schemes impose inductive bias. We therefore separate specification from implementation:
we first state what a pipeline must preserve or discard, then encode those constraints as pure
functions with explicit preconditions. Attention is paid to failure modes, including ambiguous
segmentation, encoding artefacts and domain drift, since these frequently dominate error budgets in
empirical studies. In practice, the unit couples formal definitions with executable checks so that
each step has a measurable contract: input alphabet, transformation invariants and complexity
expectations.

- Derive and implement tokenisation rules for at least two domains and document differences in vocabulary size and n-gram distributions.
- Write, test and benchmark a set of regular expressions that extract structured records from semi-structured text, reporting false positives and false negatives.
- Compute unigram, bigram and trigram statistics and interpret them using appropriate normalisation (relative frequencies, log-odds or TF–IDF).
- Construct a TF–IDF representation for a document collection and fit a linear classifier, reporting $F_1$ and a confusion matrix.
- Implement a minimal named-entity recogniser using deterministic rules and evaluate it against a provided gold standard.
