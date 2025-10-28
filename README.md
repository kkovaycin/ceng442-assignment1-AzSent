# ceng442-assignment1-AzSent
🧠 Embedding Training (Word2Vec & FastText)

The preprocessed Azerbaijani text data from all cleaned datasets were merged into a single corpus. Two models were trained to generate meaningful embeddings:

Word2Vec (Skip-gram)
vector_size=300, window=5, min_count=3, negative=10, epochs=10

FastText (Skip-gram + subword)
vector_size=300, window=5, min_count=3, min_n=3, max_n=6, epochs=10

Both models were saved under:

embeddings/
 ├─ word2vec.model
 └─ fasttext.model


🧩 Justification:
Azerbaijani is an agglutinative and morphologically rich language.
FastText’s subword mechanism was expected to improve coverage and handle unseen or rare word forms, while Word2Vec served as a strong, interpretable baseline.

⚖️ Comparison of Word2Vec and FastText
📊 Lexical Coverage
Dataset	Word2Vec	FastText
labeled-sentiment_2col	0.932	0.932
test__1__2col	0.987	0.987
train__3__2col	0.990	0.990
train-00000-of-00001_2col	0.943	0.943
merged_dataset_CSV__1__2col	0.949	0.949

✅ Both models achieved high lexical coverage, but FastText provided slightly better out-of-vocabulary (OOV) tolerance thanks to its subword structure.

🧩 Similarity Results
Metric	Word2Vec	FastText
Synonyms	0.361	0.424
Antonyms	0.310	0.438

💬 Interpretation:
FastText reached higher synonym similarity, proving its morphological sensitivity.
However, antonyms appeared slightly closer in the vector space due to shared subwords (e.g., bahalı–ucuz).
Word2Vec, on the other hand, preserved clearer semantic separation.

🔍 Nearest Neighbor Examples
Word	Word2Vec Neighbors	FastText Neighbors
yaxşı	<RATING_POS>, awsome	yaxşılı, yaxşıca
pis	pisdir, pisi	pislik, pisləşdi
ucuz	qiymət, bazar	ucuzdu, ucuzluğu
mükəmməl	möhteşəm	mükəmməldi, mükəmməlsiz

🧠 Observation:
Word2Vec captures contextual meaning, while FastText groups morphological variants together.
Thus, Word2Vec emphasizes semantic closeness, whereas FastText highlights word-form relationships.

🔁 Reproducibility

Environment:
Python 3.10+, Google Colab (A100 GPU)
Libraries: Gensim 4.3.2, Pandas 2.2.3, Regex, FTFY, Scikit-learn 1.4+

⚙️ Installation
pip install -r requirements.txt

▶️ Execution Steps
python preprocess_pipeline.py
python train_embeddings.py
python compare_models.py


Project Outputs:

CENG442_Assignment1/
 ├─ embeddings/     # Trained models
 ├─ outputs/        # Comparison results
 └─ corpus_all.txt  # Final domain-tagged corpus


🧾 All similarity, coverage, and neighbor analysis results are automatically saved in
outputs/compare.txt.
