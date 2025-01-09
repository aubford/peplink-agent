# %%
import nltk
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasketch import MinHash, MinHashLSH
from typing import List, Literal, Tuple, Set
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.corpus import brown
from rapidfuzz import fuzz, process
import itertools
import spacy
import random
from textwrap import dedent
from config import RotatingFileLogWriter
import pandas as pd
from functools import wraps
from util.util_main import print_replace
from util.viz import  plot_number_dist
from dataclasses import dataclass

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 100000000

# nltk.download('brown')
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('wordnet')

logger = RotatingFileLogWriter("nlp")


####### ANALYSIS TOOLS #########################################################


def timer(func_name: str = None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            name = func_name or func.__name__
            print(f"{name} time: {elapsed:.2f}s")
            return result

        return wrapper

    return decorator


def log_write_pair(doc_a: str, doc_b: str, msg: str = "Duplicate found"):
    return dedent(
        f"""
        ::::{msg}::::
        {doc_a.replace('\n', ' ').replace('\r', '')}
        {'-' * 100}
        {doc_b.replace('\n', ' ').replace('\r', '')}
    """
    )


def generate_random_texts(n: int, avg_length: int, mult: int = 100) -> List[str]:
    sentences = brown.sents(categories=["news", "editorial", "reviews"])
    texts = []
    for _ in range(n):
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


####### Tokenization ###########################################################


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


def nltk_get_tokens(text: str) -> List[str]:
    """
    Tokenize using NLTK, less strict with stop words
    Significantly faster and less memory intensive than spaCy
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

    return [
        lemmatizer.lemmatize(token, nltk_get_pos_tag(pos))
        for token, pos in pos_tagged_tokens
    ]


def chunk_wordset(wordset: List[str], ngram: int) -> List[str]:
    return [
        " ".join(wordset[i : i + ngram])
        for i in range(0, len(wordset), ngram)
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
    logger.print_header("Getting duplicate candidates with cosine similarity")
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


@dataclass
class TokenizedDoc:
    """Associates tokenized text with original document ID"""

    doc_id: str
    tokens: List[str]

    def __init__(self, df_row: pd.Series):
        self.doc_id = df_row["id"]
        self.original_text = df_row["page_content"]
        self.df_row = df_row
        self.tokens = nltk_get_tokens(self.original_text)

    def get_chunked_tokens(self, ngram: int = 1) -> List[str]:
        if ngram == 1:
            return self.tokens
        return chunk_wordset(self.tokens, ngram)

    def get_encoded_tokens(self, ngram: int = 1) -> List[bytes]:
        if ngram == 1:
            return [token.encode("utf8") for token in self.tokens]
        return [token.encode("utf8") for token in self.get_chunked_tokens(ngram)]


@timer("Tokenization")
def tokenize_documents(df: pd.DataFrame) -> List[TokenizedDoc]:
    """Tokenize all documents in the dataframe."""
    logger.print_header(f"Tokenize {len(df)} docs")
    tokenized_docs = []
    for _, row in df.iterrows():
        doc = TokenizedDoc(row)
        tokenized_docs.append(doc)
    logger.print(f"*Tokenization Complete: {len(tokenized_docs)}")
    return tokenized_docs


filter_logger = RotatingFileLogWriter("nlp-filter")
@timer("Filter")
def filter_exact_duplicates_minhash(
    docs: List[TokenizedDoc],
    *,
    threshold: float = 0.98,
    ngram: int = 2,
) -> List[TokenizedDoc]:
    """Returns filtered corpus with exact duplicates removed"""
    logger.print_header(f"Filter exact duplicates for: {len(docs)} docs")
    logger.print(f"N-Gram: {ngram}")
    minhashes = MinHash.bulk(
        [doc.get_encoded_tokens(ngram) for doc in docs], num_perm=1024
    )
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

    removed_docs = [docs[i] for i in indices_to_remove]
    filter_logger.info("=" * 100)
    filter_logger.log_header(
        f"Filtered corpus length from {len(docs)} to {len(result)}. Removed Items:\n{"\n".join([doc.original_text.strip() for doc in removed_docs])}"
    )
    logger.print(f"*Filtered corpus from {len(docs)} to {len(result)}")
    return result



candidate_logger = RotatingFileLogWriter("nlp-candidate")
@timer("Candidates")
def get_duplicate_candidates_simple_precision(
    docs: List[TokenizedDoc],
    *,
    ngram: int = 1,
    threshold: float = 0.8,
    report: Literal["plot", "print", None] = None,
) -> List[Tuple[TokenizedDoc, TokenizedDoc]]:
    """Returns pairs of documents that are potential duplicates using simple precision."""
    logger.print_header(f"Get Candidates (precision) for: {len(docs)} docs")
    logger.print(f"N-Gram: {ngram}")

    candidates = []
    similarities = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            precision = compute_simple_precision(
                docs[i].get_chunked_tokens(ngram),
                docs[j].get_chunked_tokens(ngram),
            )

            if report:
                similarities.append(round(precision, 2))

            if precision > threshold:
                candidate_logger.info(
                    log_write_pair(
                        docs[i].original_text, docs[j].original_text, "Candidate Found"
                    )
                )
                candidates.append((docs[i], docs[j]))

    if report:
        logger.print(f"Simple Precision Comparisons: {len(similarities)}")
        if report == "plot":
            plot_number_dist(similarities)
        elif report == "print":
            logger.print(f"Simple Precisions: {similarities}")

    logger.print(f"*Simple Precision Complete. Num candidates: {len(candidates)}")
    return candidates


@timer("Candidates Minhash")
def get_duplicate_candidates_minhash_precision(
    docs: List[TokenizedDoc],
    *,
    threshold: float = 0.7,
    report: Literal["plot", "print", None] = None,
) -> List[Tuple[TokenizedDoc, TokenizedDoc]]:
    """Returns pairs of documents that are potential duplicates"""
    logger.print_header(f"Get Candidates (minhash) for: {len(docs)} docs")
    minhashes = MinHash.bulk([doc.get_encoded_tokens(2) for doc in docs], num_perm=1024)

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
        logger.print(f"Minhash Comparisons: {len(similarities)}")
        if report == "plot":
            plot_number_dist(similarities)
        elif report == "print":
            logger.print(f"Minhash Precisions: {similarities}")
    logger.print(f"Minhash Complete. Num candidates: {len(candidates)}")
    return candidates


duplicate_logger = RotatingFileLogWriter("nlp-duplicates")
@timer("Confirm Duplicates")
def confirm_duplicates(
    candidate_pairs: List[Tuple[TokenizedDoc, TokenizedDoc]],
    *,
    threshold: int = 90,
) -> Set[str]:
    """Returns set of document IDs that are duplicates"""
    call_counter = itertools.count(1)
    high_scores = 0

    def _progress_scorer(s1: str, s2: str, **kwargs) -> float:
        # high scores will always be much higher than duplicates found since these will flag the same item several times
        nonlocal high_scores
        print_replace(
            f"Processed {next(call_counter)} comparisons. Flagged comparisons: {high_scores}. Next item lengths: {len(s1)}, {len(s2)}"
        )

        start_time = time.time()
        score = fuzz.partial_ratio(s1, s2)
        duration = time.time() - start_time

        if score > 90:
            high_scores += 1

        if duration > 3:
            duplicate_logger.info(
                log_write_pair(
                    s1,
                    s2,
                    msg=f"Slow comparison ({duration:.2f}s).  Lengths: {len(s1)}, {len(s2)}",
                )
            )

        return score

    logger.print_header(f"Getting duplicates for: {len(candidate_pairs)} pairs")
    tokenized_docs_a, tokenized_docs_b = zip(*candidate_pairs)
    strings_a = [" ".join(doc.tokens) for doc in tokenized_docs_a]
    strings_b = [" ".join(doc.tokens) for doc in tokenized_docs_b]

    distances = process.cpdist(strings_a, strings_b, scorer=_progress_scorer, workers=6)
    duplicates = set()
    for idx, (doc_a, doc_b) in enumerate(candidate_pairs):
        if distances[idx] > threshold:
            duplicate_logger.info(
                log_write_pair(doc_a.original_text, doc_b.original_text)
            )
            if len(doc_a.tokens) < len(doc_b.tokens):
                duplicates.add(doc_a.doc_id)
            else:
                duplicates.add(doc_b.doc_id)

    logger.print(f"\n*Confirm Dupes Complete: Found ({len(duplicates)}) duplicates")
    return duplicates

# %%
