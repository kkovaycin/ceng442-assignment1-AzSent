# -*- coding: utf-8 -*-
from gensim.models import Word2Vec, FastText
import pandas as pd
from pathlib import Path

FILES = [
    "labeled-sentiment_2col.xlsx",
    "test__1__2col.xlsx",
    "train__3__2col.xlsx",
    "train-00000-of-00001_2col.xlsx",
    "merged_dataset_CSV__1__2col.xlsx",
]

def load_sentences(files):
    sents = []
    for f in files:
        df = pd.read_excel(f, usecols=["cleaned_text"])
        sents.extend(df["cleaned_text"].astype(str).str.split().tolist())
    return sents

if __name__ == "__main__":
    sentences = load_sentences(FILES)
    Path("embeddings").mkdir(exist_ok=True)

    # Word2Vec
    w2v = Word2Vec(
        sentences=sentences,
        vector_size=300,
        window=5,
        min_count=3,
        sg=1,            # skip-gram
        negative=10,
        epochs=10
    )
    w2v.save("embeddings/word2vec.model")

    # FastText
    ft = FastText(
        sentences=sentences,
        vector_size=300,
        window=5,
        min_count=3,
        sg=1,
        min_n=3,
        max_n=6,
        epochs=10
    )
    ft.save("embeddings/fasttext.model")

    print("Saved embeddings to embeddings/word2vec.model and embeddings/fasttext.model")
