# ceng442-assignment1-AzSent
🧠 Embedding Training (Word2Vec & FastText)

The preprocessed Azerbaijani text data from all cleaned datasets were merged into a single corpus.
Two models were trained to generate distributed representations of words:

Word2Vec (Skip-gram)

vector_size=300

window=5

min_count=3

negative=10

epochs=10

FastText (Skip-gram + subword)

vector_size=300

window=5

min_count=3

min_n=3, max_n=6

epochs=10

📁 Model Files:

embeddings/
 ├─ word2vec.model
 └─ fasttext.model


Azerbaijani is an agglutinative and morphologically rich language.
Therefore, FastText’s subword-based representation was expected to better handle rare and unseen word forms, while Word2Vec served as a strong contextual baseline.

⚖️ Comparison of Word2Vec and FastText
📊 Lexical Coverage
Dataset	Word2Vec	FastText
labeled-sentiment_2col	0.932	0.932
test__1__2col	0.987	0.987
train__3__2col	0.990	0.990
train-00000-of-00001_2col	0.943	0.943
merged_dataset_CSV__1__2col	0.949	0.949

✅ Both models achieved high lexical coverage, but FastText offers slightly higher tolerance for out-of-vocabulary (OOV) words.

🧩 Similarity Evaluation
Metric	Word2Vec	FastText
Synonyms	0.361	0.424
Antonyms	0.310	0.438

💬 Interpretation:
FastText reached higher synonym similarity due to its subword understanding,
while Word2Vec maintained better antonym separation — which is beneficial for sentiment-based tasks.

🔍 Nearest Neighbor Examples
Word	Word2Vec Neighbors	FastText Neighbors
yaxşı	<RATING_POS>, awsome	yaxşılı, yaxşıca
pis	pisdir, pisi	pislik, pisləşdi
ucuz	qiymət, bazar	ucuzdu, ucuzluğu
mükəmməl	möhteşəm	mükəmməldi, mükəmməlsiz

🧠 Observation:
Word2Vec focuses on contextual semantics,
whereas FastText better captures morphological variations within the same root form.

🔁 Reproducibility
⚙️ Installation

To recreate the environment, install all required dependencies:

pip install -r requirements.txt

▶️ Execution Steps

Run the project step-by-step from the main directory:

1️⃣ Preprocess the datasets

python preprocess_pipeline.py


2️⃣ Train the Word2Vec & FastText models

python train_embeddings.py


3️⃣ Compare the trained models

python compare_models.py


📂 Project Output Structure

CENG442_Assignment1/
 ├─ embeddings/     # Trained Word2Vec & FastText models
 ├─ outputs/        # Comparison results (coverage, similarity)
 ├─ raw_data/       # Original unprocessed datasets
 ├─ corpus_all.txt  # Final merged corpus
 ├─ requirements.txt
 └─ README.md


🧾 All metrics and qualitative results (coverage, synonym/antonym scores, nearest neighbors)
are automatically saved inside:

outputs/compare.txt

