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

nlp = spacy.load('en_core_web_sm')
nlp.max_length = 100000000

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('wordnet')

mult = 1000
demo_texts = [
    "This is the first text about natural language processing but this one runs on and has a lot of additional words." * mult,
    "This is the first text about natural language ||| processing.<><>><><<><><" * mult,
    "This is one of the very first texts ever about natural language processing." * mult,
    "Blah blah blah" * mult,
    "This is the second text. It also discusses natural language processing." * mult,
    "Natural language processing is a field of computer science." * mult,
]

####### Tokenization ############

def lemmatize_tokens(text: str) -> List[str]:
    """Lemmatize tokens using spaCy's pipeline with streaming"""
    docs = nlp.pipe([text], disable=['ner'])  # Only disable NER since we don't use named entities
    return [
        token.lemma_ for doc in docs
        for token in doc
        if (not token.is_stop or token.pos_ in {'VERB', 'NUM', 'ADJ'}) and token.is_alpha
    ]

def get_tokens(text: str) -> List[str]:
    """Tokenize using spaCy's pipeline, more strict with stop words"""
    res = lemmatize_tokens(text.lower())
    return res

def nltk_get_pos_tag(tag: str) -> str:
    """Convert NLTK part of speech tag to WordNet tag"""
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag[0], wordnet.NOUN)

def nltk_lemmatize_tokens(tokens: List[str]) -> List[str]:
    """Lemmatize tokens using POS tagging for accuracy, less strict with stop words"""
    lemmatizer = WordNetLemmatizer()
    pos_tags = nltk.pos_tag(tokens)
    return [lemmatizer.lemmatize(token, nltk_get_pos_tag(pos))
            for token, pos in pos_tags]

def nltk_get_tokens(text: str) -> List[str]:
    """Tokenize using NLTK, less strict with stop words"""
    tokens = nltk.wordpunct_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    res = nltk_lemmatize_tokens(tokens)
    return res

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
            if get_classic_jaccard(tokenized_corpus[i], tokenized_corpus[j]) > 0.8:
                candidates.add(i)
                candidates.add(j)
    return candidates

def get_duplicate_candidates_bulk(tokenized_corpus: List[List[str]]) -> set[int]:
    token_arrays = [[token.encode("utf8") for token in tokens]
                   for tokens in tokenized_corpus]
    minhashes = list(MinHash.bulk(token_arrays, num_perm=128))
    lsh = MinHashLSH(threshold=0.8, num_perm=128)

    candidates = set()
    for i, m in enumerate(minhashes):
        lsh.insert(i, m)
        result = lsh.query(m)
        if len(result) > 1:
            candidates.update(result)

    return candidates

####### Test ############

test_corpus = demo_texts
tokenized_corpus = [get_tokens(text) for text in demo_texts]

start = time.time()
result2 = get_duplicate_candidates_bulk(tokenized_corpus)
time2 = time.time() - start
print(f"Bulk: {time2:.2f}s")
print(result2)

start = time.time()
result3 = get_duplicate_candidates_cosine(tokenized_corpus)
time3 = time.time() - start
print(f"Cosine: {time3:.2f}s")
print(result3)

start = time.time()
result1 = get_duplicate_candidates_classic_jaccard(tokenized_corpus)
time1 = time.time() - start
print(f"Classic: {time1:.2f}s")
print(result1)

print(result1 == result2 == result3)
