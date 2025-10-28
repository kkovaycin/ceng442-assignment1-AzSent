# -*- coding: utf-8 -*-
import pandas as pd
from gensim.models import Word2Vec, FastText
from numpy import dot
from numpy.linalg import norm
from pathlib import Path

# load models
w2v = Word2Vec.load("embeddings/word2vec.model")
ft  = FastText.load("embeddings/fasttext.model")

seed_words = ["yaxşı","pis","çox","bahalı","ucuz","mükəmməl","dəhşət","<PRICE>","<RATING_POS>"]
syn_pairs  = [("yaxşı","əla"), ("bahalı","qiymətli"), ("ucuz","sərfəli")]
ant_pairs  = [("yaxşı","pis"), ("bahalı","ucuz")]

def lexical_coverage(model, tokens):
    vocab = model.wv.key_to_index
    return sum(1 for t in tokens if t in vocab) / max(1, len(tokens))

FILES = [
    "labeled-sentiment_2col.xlsx",
    "test__1__2col.xlsx",
    "train__3__2col.xlsx",
    "train-00000-of-00001_2col.xlsx",
    "merged_dataset_CSV__1__2col.xlsx",
]

def read_tokens(f):
    df = pd.read_excel(f, usecols=["cleaned_text"])
    return [t for row in df["cleaned_text"].astype(str) for t in row.split()]

def pair_sim(model, pairs):
    vals=[]
    for a,b in pairs:
        try: vals.append(model.wv.similarity(a,b))
        except KeyError: pass
    return sum(vals)/len(vals) if vals else float('nan')

def neighbors(model, word, k=5):
    try: return [w for w,_ in model.wv.most_similar(word, topn=k)]
    except KeyError: return []

if __name__ == "__main__":
    lines = []
    lines.append("== Lexical coverage (per dataset) ==")
    for f in FILES:
        toks = read_tokens(f)
        cov_w2v = lexical_coverage(w2v, toks)
        cov_ftv = lexical_coverage(ft, toks)  # FT vocab coverage (not subword gen)
        lines.append(f"{f}: W2V={cov_w2v:.3f}, FT={cov_ftv:.3f}")

    syn_w2v = pair_sim(w2v, syn_pairs)
    syn_ft  = pair_sim(ft,  syn_pairs)
    ant_w2v = pair_sim(w2v, ant_pairs)
    ant_ft  = pair_sim(ft,  ant_pairs)

    lines.append("\n== Similarity (↑ good for synonyms; ↓ good for antonyms) ==")
    lines.append(f"Synonyms: W2V={syn_w2v:.3f}, FT={syn_ft:.3f}")
    lines.append(f"Antonyms: W2V={ant_w2v:.3f}, FT={ant_ft:.3f}")
    lines.append(f"Separation (Syn - Ant): W2V={(syn_w2v - ant_w2v):.3f}, FT={(syn_ft - ant_ft):.3f}")

    lines.append("\n== Nearest neighbors (qualitative) ==")
    for w in seed_words:
        lines.append(f"  {w}\n    W2V: {neighbors(w2v, w)}\n    FT : {neighbors(ft,  w)}")

    Path("results").mkdir(exist_ok=True)
    with open("results/compare.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("\n[OK] Wrote results/compare.txt")
