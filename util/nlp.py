# %%
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasketch import MinHash, MinHashLSH
from typing import List
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import spacy
from rapidfuzz import fuzz, process

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 100000000

# nltk.download('brown')
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('wordnet')

stop_words = set(stopwords.words("english"))

mult = 100

####### Tokenization ############


def get_tokens(text: str) -> List[str]:
    """Tokenize using spaCy's pipeline, more strict with stop words"""
    docs = nlp.pipe([text.lower()], disable=["ner"])  # Only disable NER since we don't use named entities
    res = [
        token.lemma_
        for doc in docs
        for token in doc
        if (not token.is_stop or token.pos_ in {"VERB", "NUM", "ADJ"}) and token.is_alpha
    ]
    return res


def nltk_get_pos_tag(tag: str) -> str:
    """Convert NLTK part of speech tag to WordNet tag"""
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }
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


####### Similarity ############################################################


def compute_precision_from_jaccard(jaccard_similarity: float, len_a: int, len_b: int) -> float:
    shorter_string_len = min(len_a, len_b)
    intersection = jaccard_similarity * (len_a + len_b) / (1 + jaccard_similarity)
    return intersection / shorter_string_len


def compute_cosine_similarity(a: str, b: str) -> float:
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([a, b])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])
    return float(similarity[0, 0])


def compute_simple_jaccard_precision(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Compute Jaccard similarity between two token lists."""
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    intersection = set_a & set_b
    precision_basis = min(len(set_a), len(set_b))
    if not precision_basis:
        return 0.0
    return len(intersection) / precision_basis


###### Dedupe ################################################################


def get_duplicate_candidates_cosine(tokenized_corpus: List[List[str]]) -> set[int]:
    """
    Use cosine similarity to find duplicate candidates
    """
    print("\nGetting duplicate candidates with cosine similarity")
    candidates = set()
    # Join tokens back into strings for TfidfVectorizer
    texts = [" ".join(tokens) for tokens in tokenized_corpus]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(texts)

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if cosine_similarity(tfidf[i : i + 1], tfidf[j : j + 1])[0, 0] > 0.8:
                candidates.add(i)
                candidates.add(j)
    return candidates


def get_duplicate_candidates_simple_jaccard_precision(
    tokenized_corpus: List[List[str]],
) -> set[int]:
    """
    Use simple Jaccard similarity to find duplicate candidates by precision
    """
    print("\nGetting duplicate candidates with simple jaccard precision")
    candidates = set()
    for i in range(len(tokenized_corpus)):
        for j in range(i + 1, len(tokenized_corpus)):
            item_i = tokenized_corpus[i]
            item_j = tokenized_corpus[j]

            precision = compute_simple_jaccard_precision(item_i, item_j)

            if precision > 0.8:
                print(f"Items {i}/{j} ({len(item_i)}/{len(item_j)}): Precision: {precision:.2f}")
                candidates.add(i)
                candidates.add(j)
    return candidates


def get_duplicate_candidates_minhash_precision(
    tokenized_encoded_corpus: List[List[str]],
) -> set[int]:
    """
    Use MinHash to compute Jaccard similarities between all pairs
    and filter based on precision threshold
    """
    print("\nGetting duplicate candidates with minhash precision")
    minhashes = MinHash.bulk(tokenized_encoded_corpus, num_perm=128)

    candidates = set()
    for i, m1 in enumerate(minhashes):
        for j in range(i + 1, len(minhashes)):
            m2 = minhashes[j]
            jaccard = m1.jaccard(m2)
            item_i = tokenized_encoded_corpus[i]
            item_j = tokenized_encoded_corpus[j]
            precision = compute_precision_from_jaccard(jaccard, len(set(item_i)), len(set(item_j)))

            if precision > 0.8:
                print(f"Intersection: {len(set(item_i) & set(item_j))}")
                print(
                    f"Items {i}/{j}  ({len(set(item_i))}/{len(set(item_j))}) of ({len(item_i)}/{len(item_j)}): Jaccard: {jaccard:.2f}, Precision: {precision:.2f}"
                )
                candidates.add(i)
                candidates.add(j)

    return candidates


def get_duplicate_candidates_minhash(
    tokenized_encoded_corpus: List[List[str]],
) -> set[int]:
    """
    Use MinHashLSH to find duplicate candidates
    This is the fastest method when there are many documents
    """
    print("\nGetting duplicate candidates with minhash jaccard")
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
    print("Deduping with rapidfuzz")

    # noinspection PyTypeChecker
    # Calculate all pairwise similarities at once
    distances = process.cdist(tokenized_corpus, tokenized_corpus, scorer=fuzz.partial_ratio)
    duplicates = set()
    for i in range(len(tokenized_corpus)):
        for j in range(i + 1, len(tokenized_corpus)):
            if distances[i][j] > 90:
                duplicates.add(i)
                duplicates.add(j)

    return duplicates
