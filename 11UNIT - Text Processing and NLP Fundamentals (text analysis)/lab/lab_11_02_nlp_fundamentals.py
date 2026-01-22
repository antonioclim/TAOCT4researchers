"""
Tokenisation, n-gram statistics and TF–IDF baselines.

The emphasis is on transparent baselines: sparse representations and linear models that
support error analysis. This is a deliberate pedagogic choice for research contexts,
where understanding failure often matters more than marginal gains.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


@dataclass(frozen=True)
class CorpusResult:
    vocab_size: int
    n_docs: int


def load_corpus_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"doc_id", "text", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return df


def analyse_corpus(df: pd.DataFrame) -> CorpusResult:
    texts = df["text"].astype(str).tolist()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True)
    X = vec.fit_transform(texts)
    return CorpusResult(vocab_size=len(vec.vocabulary_), n_docs=X.shape[0])


def fit_baseline(df: pd.DataFrame) -> None:
    texts = df["text"].astype(str).tolist()
    y = df["label"].astype(str).tolist()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    y_hat = clf.predict(X)
    print(classification_report(y, y_hat))
    print("Confusion matrix:")
    print(confusion_matrix(y, y_hat, labels=sorted(set(y))))


def demo() -> None:
    df = load_corpus_csv(Path("resources/datasets/sample_corpus.csv"))
    res = analyse_corpus(df)
    print(f"Documents: {res.n_docs}")
    print(f"Vocabulary size (1–2 grams): {res.vocab_size}")
    fit_baseline(df)


def run_pipeline(in_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_corpus_csv(in_path)
    res = analyse_corpus(df)
    (out_dir / "corpus_summary.txt").write_text(
        f"n_docs={res.n_docs}\nvocab_size={res.vocab_size}\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="11UNIT corpus analysis pipeline")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("--analyse", type=Path, help="Input CSV with doc_id,text,label")
    parser.add_argument("--out", type=Path, default=Path("output"), help="Output directory")
    args = parser.parse_args(argv)

    if args.demo:
        demo()
        return 0
    if args.analyse:
        run_pipeline(args.analyse, args.out)
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
