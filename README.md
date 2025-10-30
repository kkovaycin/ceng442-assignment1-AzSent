# ceng442-assignment1-AzSent

### 🎯 Data & Goal

In this project, the primary objective is to clean Azerbaijani sentiment datasets given, to map labels to sentiment values in range from 0 to 1, more specifically {0.0, 0.5, 1.0},and save to 2-column excel files, to generate a cleaned and combined corpus, and train Word2Vec and FastText with the cleaned data. 
It is chosen to assign 0.5 for neutral sentiment due to its continuous position between negative=0.0 and positive=1.0 as midpoint, besides it makes available for regressions for the future.

### 🧹 Preprocessing

Several rules are applied in preprocessing to clean the data. Emojis are mapped as `EMO_POS` or `EMO_NEG`. Broken encodings are fixed with the help of functions `ftfy.fix_text()` and `html.unescape()`, HTML tags also removed.
Replacements below are done:
  - HTML tags → space  
  - URLs → `URL`  
  - Emails → `EMAIL`  
  - Phone Numbers → `PHONE`

Hashtags are removed and camelCases are split. User mentions are removed.
Lowercasing is applied Azerbaijani-aware:
- `I` -> `ı`
- `İ` -> `i`
- `i ̇"` -> `i`

 Multiple punctuations are reduced to single punctuation. Numbers are replaced with `<NUM>` token. Unnecessary characters are replaced with spaces and multiple spaces are collapsed. After these processes, the split text is tokenized to Azerbaijani letters and other allowed tokens. Character repetitions are reduced to at most 2 characters. Slangs are normalized with the help of `SLANG_MAP`. Negators are detected and marked as `_NEG` including the following 3 tokens. One-letter tokens are removed except `o` and `e`. Lastly, tokens are joined to create the cleaned sentence.

Here there are some examples from input and output data to compare preprocessing results:
- An example for hashtag removing:
   - Before preprocessing: `#pulsuzchorak gəlin dəstək olaq həyata keçirək`
   - After preprocessing: `pulsuzchorak gəlin dəstək olaq həyata keçirək`
 
- An example for user and numeric value labeling:
   - Before preprocessing: `@Fariz qardaş 2-ci pleyeri qoş işləyəcək, yəqin 1-ci pleyerlə baxırsan.`
   - After preprocessing: `user qardaş <NUM> ci pleyeri qoş işləyəcək yəqin <NUM> ci pleyerlə baxırsan`

- An example for repetition reduction:
    - Before preprocessing: `aaaa bu film dublyajjjj gozleyirdim sagolunnnnn`
    - After preprocessing: `aa bu film dublyajj gozleyirdim sagolunn`

- Some other examples:
   - Before preprocessing: `Özünüzdən #amerika kəşf etməyin...))`
   - After preprocessing: `özünüzdən amerika kəşf etməyin`
 
   - Before preprocessing: `Leo Messi....8 Ballandorlu Dünya Çempionu...gedin yatin və həzz alin ama sakitcə Leooooo Messi`
   - After preprocessing: `leo messi <NUM> ballandorlu dünya çempionu gedin yatin və həzz alin ama sakitcə leoo messi`

   - Before preprocessing: `Çox pis.! 3 ədəd telefonum var. İos və androidlə yoxladım. Hamısında eyniləşdirmə uğursuz olur. Gedin düz əməlli proqram təminatçısı tapın işə götürün.!`
   - After preprocessing: `<RATING_NEG> <NUM> ədəd telefonum var ios və androidlə yoxladım hamısında eyniləşdirmə uğursuz olur gedin düz əməlli proqram təminatçısı tapın işə götürün`

### 🛠️ Mini Challenges

There were some mini challenges that were implemented to enhance the quality of preprocessing and to observe linguistic effects. They are listed below:
- With calling `re.sub('([a-z])([A-Z])', r'\1 \2', m.group(1))` when normalizing hashtags, camelCases are split and it helped us to improve token coverage:
   - Before: `Messi'nin qazandığı kuboklar və müsbət statistikaları həqiqətən dəhşətdir! O, dünyanın ən yaxşı futbolçularından biridir və əfsanəvi bir karyerası var. Adətən futbol oynamağı xoşlayan birisi olmasam da, Messi kimi əməkdaşların üçün hər zaman böyük bir hörmət hiss edirəm. Fußbol dünyası üçün bir möcüzədir! #Messi #LeoMessi #LionelMessi #futbol   #futbolçu #tarix`
   - After: `messi'nin qazandığı kuboklar və müsbət statistikaları həqiqətən dəhşətdir o dünyanın ən yaxşı futbolçularından biridir və əfsanəvi bir karyerası var adətən futbol oynamağı xoşlayan birisi olmasam da messi kimi əməkdaşların üçün hər zaman böyük bir hörmət hiss edirəm fu bol dünyası üçün bir möcüzədir messi leo messi lionel messi futbol futbol çu tarix`

- Tagging the next three tokens with tag `_NEG` after the negator altered nearest neighbors significantly:
   - Before: `Çox gözəl vaxtlar idi Xoşqədəm yoxdu Zaur yox idi Bayramlar Baboslar hər şey təbiidir`
   - After: `çox gözəl vaxtlar idi xoşqədəm yoxdu zaur yox idi_NEG bayramlar_NEG baboslar_NEG hər şey təbiidir`

 - Slang mapping is used to improve the recognition of common Azerbaijani tokens to increase word consistency
   
    Slang map: `{"slm":"salam","tmm":"tamam","sagol":"sağol","cox":"çox","yaxsi":"yaxşı"}`
   
   - Before: `Halal olsun qardas sene, cox sagol.`
   - After: `halal olsun qardas sene çox sağol`

     
   - Before: `Dovlet cox yaxsi idare eledi veziyyeti.`
   - After: `dovlet çox yaxşı idare eledi veziyyeti`

### 🏷️ Domain-Aware

In the normalization process, domains are detected to make the models aware of differences between texts.
First, domain hints are compiled via regex:
```
# 3 instance of hint detection regexes
NEWS_HINTS   = re.compile(r"\b(apa|trend|azertac|reuters|bloomberg|dha|aa)\b", re.I)
SOCIAL_HINTS = re.compile(r"\b(rt)\b|@|#|(?:😂|😍|😊|👍|👎|😡|🙂)")
REV_HINTS    = re.compile(r"\b(azn|manat|qiymət|aldım|ulduz|çox yaxşı|çox pis)\b", re.I)
#...
```
Then, according to the hints gathered, domain tags are detected:
```
#...
if NEWS_HINTS.search(s): return "news"
if SOCIAL_HINTS.search(s): return "social"
if REV_HINTS.search(s):   return "reviews"
return "general"
#...
```
For review tag, extra normalization is applied:
```
#...
if domain == "reviews":
  s = PRICE_RE.sub(" <PRICE> ", cleaned)
  s = STARS_RE.sub(lambda m: f" <STARS_{m.group(1)}> ", cleaned)
  s = POS_RATE.sub(" <RATING_POS> ", s)
  s = NEG_RATE.sub(" <RATING_NEG> ", s)
  return " ".join(s.split())
#...
```
Ultimately, each line in the corpus is prefixed with concatenation of `"dom"` and its tag:
```
f"dom{domain} " + line
```
Here are some examples from corpus for different domains:
- `domgeneral prezident xankəndidə qəbul keçirdi`
- `domgeneral bu dəyərli məlumata görə çox sağ olun`
- `domgeneral helal olsun azerbaycan turkuysen kendi dilinde konus`
- `domreviews çox gözəl proqramdır num ulduz verirəm`
- `domreviews həmişə sifariş verdikdə istifadə etdiyim çatdırılma proqramıdır minimum sifariş məbləği num manatdır ilk sifarişdə isə num manat endirim verir çox razıyam`
- `domreviews salam internetimiz çox pis işləyir günlərlə internet olmur daima kəsilir num gün tamam olmamış bitir`

'''
'''
### 🧠 Embedding Training (Word2Vec & FastText)

### 📘 Overview
The preprocessed Azerbaijani text data from all cleaned datasets were merged into a single corpus.  
Two models were trained to generate **distributed representations of words**:

---
#### ⚙️ Embedding Configuration
Both **Word2Vec** and **FastText** were trained using the Skip-gram architecture  
with shared core parameters: `vector_size=300`, `window=5`, `min_count=3`, `epochs=10`.  
Word2Vec employed **negative sampling (negative=10)**,  
while FastText additionally modeled **subword information (min_n=3, max_n=6)**.

---


#### 📁 Model Files
embeddings/
 ├─ embeddings.txt

---

#### 🌍 Language Context
Azerbaijani is an **agglutinative and morphologically rich language**.  
Therefore, **FastText’s subword-based representation** was expected to better handle **rare and unseen word forms**,  
while **Word2Vec** served as a **strong contextual baseline**.

---

### ⚖️ Comparison of Word2Vec and FastText

#### 📊 Lexical Coverage
| Dataset | Word2Vec | FastText |
|:---------|:----------:|:----------:|
| labeled-sentiment_2col | 0.932 | 0.932 |
| test__1__2col | 0.987 | 0.987 |
| train__3__2col | 0.990 | 0.990 |
| train-00000-of-00001_2col | 0.943 | 0.943 |
| merged_dataset_CSV__1__2col | 0.949 | 0.949 |

✅ **Both models achieved high lexical coverage**,  
but **FastText** offers slightly higher tolerance for **out-of-vocabulary (OOV)** words.

---

#### 🧩 Similarity Evaluation
| Metric | Word2Vec | FastText |
|:--------|:----------:|:----------:|
| Synonyms | 0.361 | **0.424** |
| Antonyms | **0.310** | 0.438 |

---

#### 💬 Interpretation
- **FastText** reached higher **synonym similarity** due to its **subword understanding**.  
- **Word2Vec** maintained better **antonym separation**, which is beneficial for **sentiment-based tasks**.

---

#### 🔍 Nearest Neighbor Examples
| Word | Word2Vec Neighbors | FastText Neighbors |
|:------|:-------------------|:-------------------|
| **yaxşı** | `<RATING_POS>`, awesome | yaxşılı, yaxşıca |
| **pis** | pisdir, pisi | pislik, pisləşdi |
| **ucuz** | qiymət, bazar | ucuzdu, ucuzluğu |
| **mükəmməl** | möhteşəm | mükəmməldi, mükəmməlsiz |

---

### 🧠 Observation
- **Word2Vec** focuses on **contextual semantics**.  
- **FastText** captures **morphological variations** within the same root form, improving linguistic robustness.

---

### 🔁 Reproducibility

#### ⚙️ Installation
To recreate the environment, install all required dependencies:
pip install -r requirements.txt

---

#### ▶️ Execution Steps
Run the project step-by-step from the main directory:

1️⃣ **Preprocess the datasets**
python preprocess_pipeline.py

2️⃣ **Train the Word2Vec & FastText models**
python train_embeddings.py

3️⃣ **Compare the trained models**
python compare_models.py

---

### 📂 Project Output Structure
CENG442_Assignment1/
 ├─ embeddings/     # Trained Word2Vec & FastText models  
 ├─ outputs/        # Comparison results (coverage, similarity)  
 ├─ raw_data/       # Original unprocessed datasets  
 ├─ corpus_all.txt  # Final merged corpus  
 ├─ requirements.txt  
 └─ README.md  

---

### 🧾 Results
All metrics and qualitative results (coverage, synonym/antonym scores, nearest neighbors)  
are automatically saved inside:
outputs/
'''

