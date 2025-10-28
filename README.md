# ceng442-assignment1-AzSent
5. Embedding Training (Word2Vec & FastText)

All cleaned datasets (*_2col.xlsx) were merged into a unified corpus.
Two embedding models were trained to represent Azerbaijani tokens:

Word2Vec (Skip-gram)
vector_size=300, window=5, min_count=3, negative=10, epochs=10

FastText (Skip-gram + subword)
vector_size=300, window=5, min_count=3, min_n=3, max_n=6, epochs=10

Both models were saved under:

embeddings/
 ├─ word2vec.model
 └─ fasttext.model


Justification:
Since Azerbaijani is morphologically rich and agglutinative, FastText’s subword modeling was expected to yield better generalization and out-of-vocabulary (OOV) coverage.
Word2Vec served as a solid baseline for speed and interpretability.

7. Word2Vec vs FastText Comparison
7.1 Lexical Coverage (↑ higher is better)
Dataset	Word2Vec	FastText
labeled-sentiment_2col	0.932	0.932
test__1__2col	0.987	0.987
train__3__2col	0.990	0.990
train-00000-of-00001_2col	0.943	0.943
merged_dataset_CSV__1__2col	0.949	0.949

Comment:
Both models achieved high lexical coverage, indicating well-learned vocabularies.
FastText slightly improves OOV tolerance due to its subword information.

7.2 Similarities (Syn ↑ desirable, Ant ↓ desirable)
Metric	Word2Vec	FastText
Synonyms	0.361	0.424
Antonyms	0.310	0.438

Interpretation:
FastText yielded higher synonym similarity, confirming improved subword sensitivity.
However, antonyms appeared slightly closer in FastText’s space, showing mild overgeneralization caused by shared morphemes (e.g., “bahalı–ucuz”).

7.3 Nearest Neighbors (qualitative examples)
Word	Word2Vec examples	FastText examples
yaxşı	<RATING_POS>, awsome	yaxşılı, yaxşıca
pis	pisdir, pisi	pislik, pisləşdi
ucuz	qiymət, bazar	ucuzdu, ucuzluğu
mükəmməl	möhteşəm	mükəmməldi, mükəmməlsiz

Observation:
Word2Vec focuses on contextual similarity (semantic),
while FastText captures morphological relationships (surface-level).

8. Reproducibility

Environment

Python 3.10+, Google Colab (A100 GPU(optional))

Gensim 4.3.2, Pandas 2.2.3, Scikit-learn 1.4+

Installation

pip install -r requirements.txt


Execution Steps

python preprocess_pipeline.py
python train_embeddings.py
python compare_models.py


Output Structure

CENG442_Assignment1/
 ├─ embeddings/      # trained models
 ├─ outputs/         # comparison results
 └─ corpus_all.txt   # final corpus


All metrics and similarity evaluations are automatically written to
outputs/compare.txt.
