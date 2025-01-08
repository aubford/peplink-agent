# %%
import nltk
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasketch import MinHash, MinHashLSH
from typing import List, Literal, Tuple, TypeVar, NamedTuple, Set
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.corpus import brown
from rapidfuzz import fuzz, process
import itertools
import spacy
import random

from util.util_main import print_replace
from util.viz import plot_item_frequency, plot_number_dist
from dataclasses import dataclass
from functools import cached_property

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
    *,
    threshold: float = 0.7,
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

            precision = compute_simple_precision(item_i, item_j)
            if report:
                similarities.append(round(precision, 2))

            if precision > threshold:
                candidates.add(i)
                candidates.add(j)

    print(f"Simple Precision Comparisons: {len(similarities)}")
    print(f"Simple Precision Candidates: {len(candidates)}")
    if report == "plot":
        plot_number_dist(similarities)
    elif report == "print":
        print(f"Simple Precisions: {similarities}")
        print(f"Simple Jaccards: {jaccards}")
    return candidates


@dataclass
class TokenizedDoc:
    """Associates tokenized text with original document ID"""

    doc_id: str
    tokens: List[str]

    @cached_property
    def encoded_tokens(self) -> List[bytes]:
        return [token.encode("utf8") for token in self.tokens]

    def __init__(self, doc_id: str, text: str):
        self.doc_id = doc_id
        self.tokens = nltk_get_tokens(text)


def filter_exact_duplicates_minhash(
    docs: List[TokenizedDoc],
    *,
    threshold: float = 0.98,
) -> List[TokenizedDoc]:
    """Returns filtered corpus with exact duplicates removed"""
    minhashes = MinHash.bulk([doc.encoded_tokens for doc in docs], num_perm=1024)
    lsh = MinHashLSH(threshold=threshold, num_perm=1024)

    to_remove = []
    for i, m in enumerate(minhashes):
        # insert lazily for speed
        lsh.insert(i, m)
        result = set(lsh.query(m))
        if len(result) > 1:
            # Check if we can merge with an existing set
            for existing_set in to_remove:
                if result - {i} == existing_set:
                    existing_set.add(i)
                    break
            else:
                # No matching set found, append new set
                to_remove.append(result)

    # Keep one representative from each group
    indices_to_remove = set()
    for group in to_remove:
        group.pop()  # Keep one representative
        indices_to_remove.update(group)
    # Return filtered corpus
    result = [doc for i, doc in enumerate(docs) if i not in indices_to_remove]
    print(f"Filtered corpus length from {len(docs)} to {len(result)}")
    return result


def get_duplicate_candidates_simple_precision(
    docs: List[TokenizedDoc],
    *,
    threshold: float = 0.8,
    report: Literal["plot", "print", None] = None,
) -> List[Tuple[TokenizedDoc, TokenizedDoc]]:
    """Returns pairs of documents that are potential duplicates using simple precision."""
    print(f"\nGetting duplicate pairs with simple precision from docs: {len(docs)}")

    candidates = []
    similarities = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            precision = compute_simple_precision(docs[i].tokens, docs[j].tokens)

            if report:
                similarities.append(round(precision, 2))

            if precision > threshold:
                candidates.append((docs[i], docs[j]))

    if report:
        print(f"Simple Precision Comparisons: {len(similarities)}")
        if report == "plot":
            plot_number_dist(similarities)
        elif report == "print":
            print(f"Simple Precisions: {similarities}")

    print(f"Simple Precision Candidates: {len(candidates)}")
    return candidates


def get_duplicate_candidates_minhash_precision(
    docs: List[TokenizedDoc],
    *,
    threshold: float = 0.7,
    report: Literal["plot", "print", None] = None,
) -> List[Tuple[TokenizedDoc, TokenizedDoc]]:
    """Returns pairs of documents that are potential duplicates"""
    minhashes = MinHash.bulk([doc.encoded_tokens for doc in docs], num_perm=1024)

    candidates = []
    similarities = []
    for i, m1 in enumerate(minhashes):
        for j in range(i + 1, len(minhashes)):
            m2 = minhashes[j]
            jaccard = m1.jaccard(m2)
            item_i = docs[i].tokens
            item_j = docs[j].tokens
            precision = compute_precision_from_jaccard(
                jaccard, len(set(item_i)), len(set(item_j))
            )

            if report:
                similarities.append(round(precision, 2))

            if precision > threshold:
                candidates.append((docs[i], docs[j]))

    if report:
        print(f"Minhash Comparisons: {len(similarities)}")
        if report == "plot":
            plot_number_dist(similarities)
        elif report == "print":
            print(f"Minhash Precisions: {similarities}")
    print(f"Minhash Candidates: {len(candidates)}")
    return candidates


call_counter = itertools.count(1)


def counting_scorer(s1: str, s2: str, **kwargs) -> float:
    current_call = next(call_counter)
    print_replace(
        f"Processed {current_call} comparisons. Next item lengths: {len(s1)}, {len(s2)}"
    )
    print(f"\nSTR_1: {s1[:200]}")
    print(f"\nSTR_2: {s2[:200]}")
    score = fuzz.partial_ratio(s1, s2)
    print(f"\nScore: {score}\n")
    return score


def get_duplicates(
    candidate_pairs: List[Tuple[TokenizedDoc, TokenizedDoc]]
) -> Set[str]:  # Returns doc_ids to remove
    """Returns set of document IDs that are duplicates"""
    print(f"\nGetting duplicates. Pairs to compare: {len(candidate_pairs)}")
    tokenized_docs_a, tokenized_docs_b = zip(*candidate_pairs)
    strings_a = [" ".join(doc.tokens) for doc in tokenized_docs_a]
    strings_b = [" ".join(doc.tokens) for doc in tokenized_docs_b]

    distances = process.cpdist(strings_a, strings_b, scorer=counting_scorer, workers=9)
    duplicates = set()
    for idx, (doc_a, doc_b) in enumerate(candidate_pairs):
        if distances[idx] > 90:
            if len(doc_a.tokens) < len(doc_b.tokens):
                duplicates.add(doc_a.doc_id)
            else:
                duplicates.add(doc_b.doc_id)

    print(f"Found ({len(duplicates)}) duplicates")
    return duplicates
