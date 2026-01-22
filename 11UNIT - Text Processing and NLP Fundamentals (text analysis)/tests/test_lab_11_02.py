from pathlib import Path
from lab.lab_11_02_nlp_fundamentals import load_corpus_csv, analyse_corpus

def test_analyse_corpus_vocab_positive():
    df = load_corpus_csv(Path("resources/datasets/sample_corpus.csv"))
    res = analyse_corpus(df)
    assert res.n_docs == len(df)
    assert res.vocab_size > 0
