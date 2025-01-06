# %%
import nltk
import time
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasketch import MinHash, MinHashLSH
from typing import List
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import spacy
from rapidfuzz import fuzz, process
from nltk.corpus import brown
import random

nlp = spacy.load('en_core_web_sm')
nlp.max_length = 100000000

# nltk.download('brown')
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('wordnet')

stop_words = set(stopwords.words("english"))

mult = 1000

def generate_random_texts(n: int, avg_length: int) -> List[str]:
    """
    Generate n random texts using coherent phrases from Brown corpus

    Args:
        n: Number of texts to generate
        avg_length: Approximate number of words per text
    """
    # Get sentences from Brown corpus
    sentences = brown.sents(categories=['news', 'editorial', 'reviews'])

    texts = []
    for _ in range(n):
        # Generate new random length for each text
        target_length = avg_length * random.randint(1, mult)

        text = []
        word_count = 0
        while word_count < target_length:
            sent = random.choice(sentences)
            text.extend(sent)
            word_count += len(sent)

        text = ' '.join(text[:target_length]).lower()
        texts.append(text)

    return texts

repeated_text_0_1 = "This is the first text about natural language processing. "
semantic_text_2 = "This processing text is the very first course item about doing language. "
similar__text_3 = "This text is the very first one about natural language processing. "

random_texts = generate_random_texts(2, 100)

demo_texts = [
    # (repeated_text_0_1 * mult) + generate_random_texts(1, 50),
    (repeated_text_0_1) + random_texts[0],
    (repeated_text_0_1) + random_texts[1],
    # repeated_text_0_1 * mult,
    # semantic_text_2 * mult,
    # similar__text_3 * mult,
    # "This is the second text. It also discusses natural language processing." * mult,
    # "Natural language processing is a field of computer science." * mult,
    # *generate_random_texts(300, 50),
]

print(demo_texts[0])
print("-" * 25)
print(demo_texts[1])


#%%

####### Tokenization ############

def get_tokens(text: str) -> List[str]:
    """Tokenize using spaCy's pipeline, more strict with stop words"""
    docs = nlp.pipe([text.lower()], disable=['ner'])  # Only disable NER since we don't use named entities
    res = [
        token.lemma_ for doc in docs
        for token in doc
        if (not token.is_stop or token.pos_ in {'VERB', 'NUM', 'ADJ'}) and token.is_alpha
    ]
    return res

def nltk_get_pos_tag(tag: str) -> str:
    """Convert NLTK part of speech tag to WordNet tag"""
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag[0], wordnet.NOUN)

def nltk_get_tokens(text: str, encode: bool = False) -> List[str]:
    """
    Tokenize using NLTK, less strict with stop words
    Significantly faster and less memory intensive than spaCy
    """
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.wordpunct_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    pos_tagged_tokens = nltk.pos_tag(tokens)

    result = []
    for token, pos in pos_tagged_tokens:
        lemmatized = lemmatizer.lemmatize(token, nltk_get_pos_tag(pos))
        if encode:
            result.append(lemmatized.encode("utf8"))
        else:
            result.append(lemmatized)
    return result

####### Similarity ############

def compute_precision_from_jaccard(
    jaccard_similarity: float, len_a: int, len_b: int
) -> float:
    shorter_string_len = min(len_a, len_b)
    intersection = jaccard_similarity * (len_a + len_b) / (1 + jaccard_similarity)
    return intersection / shorter_string_len

def compute_recall_from_jaccard(
    jaccard_similarity: float, len_a: int, len_b: int
) -> float:
    longer_string_len = max(len_a, len_b)
    intersection = jaccard_similarity * (len_a + len_b) / (1 + jaccard_similarity)
    return intersection / longer_string_len

def compute_cosine_similarity(a: str, b: str) -> float:
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([a, b])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])
    return float(similarity[0, 0])

def get_classic_jaccard(tokens_a: List[str], tokens_b: List[str]) -> float:
    intersection = set(tokens_a) & set(tokens_b)
    union = set(tokens_a) | set(tokens_b)
    return len(intersection) / len(union)

###### Dedupe ############

def get_duplicate_candidates_cosine(tokenized_corpus: List[List[str]]) -> set[int]:
    candidates = set()
    # Join tokens back into strings for TfidfVectorizer
    texts = [' '.join(tokens) for tokens in tokenized_corpus]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(texts)

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if cosine_similarity(tfidf[i:i+1], tfidf[j:j+1])[0, 0] > 0.8:
                candidates.add(i)
                candidates.add(j)
    return candidates

def get_duplicate_candidates_classic_jaccard(tokenized_corpus: List[List[str]]) -> set[int]:
    candidates = set()
    for i in range(len(tokenized_corpus)):
        for j in range(i + 1, len(tokenized_corpus)):
            item_i = tokenized_corpus[i]
            item_j = tokenized_corpus[j]

            jaccard = get_classic_jaccard(item_i, item_j)
            precision = compute_precision_from_jaccard(jaccard, len(item_i), len(item_j))

            print(f"Items {i}/{j} ({len(item_i)}/{len(item_j)}): Jaccard: {jaccard:.2f}, Precision: {precision:.2f}")

            if precision > 0.8:
                candidates.add(i)
                candidates.add(j)
    return candidates

def get_duplicate_candidates_minhash(tokenized_encoded_corpus: List[List[str]]) -> set[int]:
    """
    Use MinHashLSH to find duplicate candidates
    This is the fastest method when there are many documents
    """
    minhashes = MinHash.bulk(tokenized_encoded_corpus, num_perm=128)
    lsh = MinHashLSH(threshold=0.8, num_perm=128)

    candidates = set()
    for i, m in enumerate(minhashes):
        lsh.insert(i, m)
        result = lsh.query(m)
        if len(result) > 1:
            candidates.update(result)

    return candidates

def get_duplicates(tokenized_corpus: List[List[str]]) -> set[int]:
    """
    Use rapidfuzz.process.cdist to efficiently find duplicate candidates
    """
    # Calculate all pairwise similarities at once
    distances = process.cdist(tokenized_corpus, tokenized_corpus, scorer=fuzz.partial_ratio)
    duplicates = set()
    for i in range(len(tokenized_corpus)):
        for j in range(i + 1, len(tokenized_corpus)):
            if distances[i][j] > 90:
                duplicates.add(i)
                duplicates.add(j)

    return duplicates

####### Test ############

start = time.time()
nltk_tokenized_corpus = [nltk_get_tokens(text) for text in demo_texts]
tokenization_time_1 = time.time() - start
nltk_tokenized_corpus_encoded = [[token.encode("utf8") for token in text] for text in nltk_tokenized_corpus]
print(f"NLTK: {tokenization_time_1:.2f}s")

# print(nltk_tokenized_corpus[0])
# print(nltk_tokenized_corpus[1])
# print(nltk_tokenized_corpus[2])
# print(nltk_tokenized_corpus[3])

# start = time.time()
# spacy_tokenized_corpus = [get_tokens(text) for text in test_corpus]
# corpus_time_2 = time.time() - start
# print(f"Spacy: {corpus_time_2:.2f}s")

start = time.time()
result2 = get_duplicate_candidates_minhash(nltk_tokenized_corpus_encoded)
time2 = time.time() - start
print(f"Bulk: {time2:.2f}s")
print(result2)

start = time.time()
result1 = get_duplicate_candidates_classic_jaccard(nltk_tokenized_corpus)
time1 = time.time() - start
print(f"Classic: {time1:.2f}s")
print(result1)

start = time.time()
result4 = get_duplicates(nltk_tokenized_corpus)
time4 = time.time() - start
print(f"Rapidfuzz: {time4:.2f}s")
print(result4)

# %%
