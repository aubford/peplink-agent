# %%
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasketch import MinHash, MinHashLSH
from typing import List, Literal
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.corpus import brown
from rapidfuzz import fuzz, process
import spacy
import random
from util.viz import plot_item_frequency, plot_number_dist

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 100000000

# nltk.download('brown')
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('wordnet')


def generate_random_texts(n: int, avg_length: int, mult: int = 100) -> List[str]:
    """
    Generate n random texts using coherent phrases from Brown corpus

    Args:
        n: Number of texts to generate
        avg_length: Approximate number of words per text
    """
    # Get sentences from Brown corpus
    sentences = brown.sents(categories=["news", "editorial", "reviews"])

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

        text = " ".join(text[:target_length]).lower()
        texts.append(text)

    return texts


####### Tokenization ############


def spacy_get_tokens(text: str) -> List[str]:
    """Tokenize using spaCy's pipeline, more strict with stop words"""
    docs = nlp.pipe(
        [text.lower()], disable=["ner"]
    )  # Only disable NER since we don't use named entities
    res = [
        token.lemma_
        for doc in docs
        for token in doc
        if (not token.is_stop or token.pos_ in {"VERB", "NUM", "ADJ"})
        and token.is_alpha
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


def nltk_get_tokens(text: str, chunk_size: int = 1) -> List[str]:
    """
    Tokenize using NLTK, less strict with stop words
    Significantly faster and less memory intensive than spaCy

    Args:
        chunk_size: Number of words to chunk together as one token
    """
    stop_words = set(stopwords.words("english"))
    stop_words.update(
        [
            "um",
            "uh",
            "uhm",
            "let",
            "go",
            "yeah",
            "ok",
            "okay",
            "okay",
            "stuff",
            "really",
            "alot",
            "lot",
            "thing",
            "well",
        ]
    )

    tokens = nltk.wordpunct_tokenize(text.lower())
    tokens = [
        token
        for token in tokens
        if token.isalnum() and len(token) > 1 and token not in stop_words
    ]

    pos_tagged_tokens = nltk.pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    wordset = [
        lemmatizer.lemmatize(token, nltk_get_pos_tag(pos))
        for token, pos in pos_tagged_tokens
    ]

    return [
        " ".join(wordset[i : i + chunk_size])
        for i in range(0, len(wordset), chunk_size)
    ]


####### Similarity ############################################################


def compute_precision_from_jaccard(
    jaccard_similarity: float, len_a: int, len_b: int
) -> float:
    shorter_string_len = min(len_a, len_b)
    intersection = jaccard_similarity * (len_a + len_b) / (1 + jaccard_similarity)
    return intersection / shorter_string_len


def compute_cosine_similarity(a: str, b: str) -> float:
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([a, b])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])
    return float(similarity[0, 0])


def compute_simple_precision(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Compute Jaccard similarity between two token lists."""
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    intersection = set_a & set_b
    precision_basis = min(len(set_a), len(set_b))
    if not precision_basis:
        return 0.0
    return len(intersection) / precision_basis


def compute_simple_jaccard(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Compute Jaccard similarity between two token lists."""
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    intersection = set_a & set_b
    union = set_a | set_b
    if not union:
        return 0.0
    return round(len(intersection) / len(union), 2)


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


def get_duplicate_candidates_simple_precision(
    tokenized_corpus: List[List[str]],
    report: Literal["plot", "print", None] = None,
) -> set[int]:
    """
    Use simple precision to find duplicate candidates
    """
    print("\nGetting duplicate candidates with simple precision")
    jaccards = []
    similarities = []
    candidates = set()
    for i in range(len(tokenized_corpus)):
        for j in range(i + 1, len(tokenized_corpus)):
            item_i = tokenized_corpus[i]
            item_j = tokenized_corpus[j]

            if report == "print":
                jaccard = compute_simple_jaccard(item_i, item_j)
                jaccards.append(jaccard)

            if report:
                precision = compute_simple_precision(item_i, item_j)
                similarities.append(round(precision, 2))

            if precision > 0.7:
                candidates.add(i)
                candidates.add(j)

    if report == "plot":
        plot_number_dist(similarities)
    elif report == "print":
        print(f"Precisions: {similarities}")
        print(f"Jaccards: {jaccards}")
    return candidates


def get_duplicate_candidates_minhash_precision(
    tokenized_encoded_corpus: List[List[str]],
    report: Literal["plot", "print", None] = None,
) -> set[int]:
    """
    Use MinHash to compute Jaccard similarities between all pairs
    and filter based on precision threshold
    """
    print("\nGetting duplicate candidates with minhash precision")
    minhashes = MinHash.bulk(tokenized_encoded_corpus, num_perm=1024)
    similarities = []
    candidates = set()
    for i, m1 in enumerate(minhashes):
        for j in range(i + 1, len(minhashes)):
            m2 = minhashes[j]
            jaccard = m1.jaccard(m2)
            item_i = tokenized_encoded_corpus[i]
            item_j = tokenized_encoded_corpus[j]
            precision = compute_precision_from_jaccard(
                jaccard, len(set(item_i)), len(set(item_j))
            )

            if report:
                similarities.append(round(precision, 2))

            if precision > 0.7:
                candidates.add(i)
                candidates.add(j)

    if report == "plot":
        plot_number_dist(similarities)
    elif report == "print":
        print(f"Precisions: {similarities}")
    return candidates


def get_duplicate_candidates_minhash(
    tokenized_encoded_corpus: List[List[str]],
) -> set[int]:
    """
    Use MinHashLSH to find duplicate candidates
    This is the fastest method when there are many documents
    """
    print("\nGetting duplicate candidates with minhash jaccard")
    minhashes = MinHash.bulk(tokenized_encoded_corpus, num_perm=1024)
    lsh = MinHashLSH(threshold=0.7, num_perm=1024)

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
    distances = process.cdist(
        tokenized_corpus, tokenized_corpus, scorer=fuzz.partial_ratio
    )
    duplicates = set()
    for i in range(len(tokenized_corpus)):
        for j in range(i + 1, len(tokenized_corpus)):
            if distances[i][j] > 60:
                duplicates.add(i)
                duplicates.add(j)

    return duplicates


# %%
