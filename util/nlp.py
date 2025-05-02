import nltk
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasketch import MinHash
from typing import List, Literal, Tuple
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.corpus import brown
from rapidfuzz import distance
import spacy
import random
import json
from textwrap import dedent
import pandas as pd
from functools import wraps
from dataclasses import dataclass

####### ANALYSIS TOOLS #########################################################


def timer(func_name: str | None = None):
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


def get_write_pair_log_text(doc_a: str, doc_b: str, msg: str = "Duplicate found"):
    return dedent(
        f"""
        ::::{msg}::::
        {doc_a.replace('\n', ' ').replace('\r', '')}
        {'-' * 100}
        {doc_b.replace('\n', ' ').replace('\r', '')}"""
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
DEFAULT_DISFLUENCIES = {
    "um",
    "uh",
    "uhm",
    "hmm",
    "hm",
    "bye",
}
DEFAULT_STOPWORDS = set(stopwords.words("english"))
DEFAULT_STOPWORDS.update(DEFAULT_DISFLUENCIES)


def spacy_get_tokens(text: str) -> List[str]:
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 100000000
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


def nltk_tokenize(text: str, stop_words: set[str] = DEFAULT_STOPWORDS) -> list[str]:
    tokens = nltk.wordpunct_tokenize(text.lower())
    return [
        token
        for token in tokens
        if token.isalnum() and len(token) > 1 and token not in stop_words
    ]


def nltk_get_lemmatized_tokens(text: str) -> List[str]:
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
    return [" ".join(wordset[i : i + ngram]) for i in range(0, len(wordset), ngram)]


####### Transform #############################################################

from flashtext import KeywordProcessor


def remove_keywords(text: str, keywords: set[str] = DEFAULT_DISFLUENCIES) -> str:
    """
    Remove keywords from text by replacing them with underscores and then removing the underscores and companion spaces or commas.
    """
    keyword_processor = KeywordProcessor()
    for word in keywords:
        keyword_processor.add_keyword(word, "__")
    replaced_with_underscores = keyword_processor.replace_keywords(text)
    return (
        replaced_with_underscores.replace("__, ", "")
        .replace("__,", "")
        .replace(",__", "")
        .replace(" __", "")
        .replace("__ ", "")
        .replace("__", "")
    )


def get_keywords(text: str, keywords: set[str] = DEFAULT_STOPWORDS) -> list[str]:
    keyword_processor = KeywordProcessor()
    keyword_processor.add_keywords_from_list(keywords)
    return keyword_processor.extract_keywords(text)


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
    candidates = set()
    # Join tokens back into strings for TfidfVectorizer
    texts = [" ".join(tokens) for tokens in tokenized_corpus]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(texts)

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if cosine_similarity(tfidf[i : i + 1], tfidf[j : j + 1])[0, 0] > 0.8:  # type: ignore
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
        self.tokens = nltk_get_lemmatized_tokens(self.original_text)

    def get_chunked_tokens(self, ngram: int = 1, shift: int | None = None) -> list[str]:
        if ngram == 1:
            return self.tokens
        tokens = self.tokens
        if shift:
            tokens = tokens[shift:]
        return chunk_wordset(tokens, ngram)

    def get_encoded_tokens(
        self, ngram: int = 1, shift: int | None = None
    ) -> list[bytes]:
        if ngram == 1:
            return [token.encode("utf8") for token in self.tokens]
        return [token.encode("utf8") for token in self.get_chunked_tokens(ngram, shift)]


# currently unused
@timer("Candidates Minhash")
def get_duplicate_candidates_minhash_precision(
    docs: List[TokenizedDoc],
    *,
    threshold: float = 0.7,
    report: Literal["plot", "print", None] = None,
) -> List[Tuple[TokenizedDoc, TokenizedDoc]]:
    """Returns pairs of documents that are potential duplicates"""
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
    return candidates


STOP_ENTITIES = ["pepwave", "peplink"]


def normalize_entities_and_themes(
    df: pd.DataFrame, similarity_threshold: float = 0.95
) -> pd.DataFrame:
    """
    Normalize the entities column by merging nearly identical elements
    using JARO_WINKLER distance comparison.

    Args:
        df: DataFrame with entities column
        similarity_threshold: Threshold for considering two strings similar (default: 0.95)

    Returns:
        DataFrame with normalized entities
    """

    def skip_stop_entity(entity: str, threshold: float = 0.9) -> bool:
        should_skip = any(
            1 - distance.JaroWinkler.distance(entity, stop_entity) > threshold
            for stop_entity in STOP_ENTITIES
        )
        if should_skip:
            print(f"Skipping stop entity: {entity}")
        return should_skip

    df["entities_pre_normalization"] = df["entities"]
    df = df.copy()
    df["entities"] = df["entities"].apply(
        lambda x: json.loads(x) if pd.notna(x) else []
    )
    df["entities"] = df["entities"].apply(lambda x: [str(item).lower() for item in x])
    df["entities"] = df["entities"].apply(
        lambda x: [item for item in x if not skip_stop_entity(item, 0.9)]
    )

    def normalize_list(elements) -> tuple[list, list]:
        # Track all merges to report later
        merges = []

        # Step 1: Group similar items
        similarity_groups = []
        processed = set()

        # Create groups of similar items
        for i, item in enumerate(elements):
            if i in processed or not item:
                continue

            # Start a new group with this item
            group = [item]
            processed.add(i)

            # Find similar items
            for j, other_item in enumerate(elements[i + 1 :], i + 1):
                if j in processed or not other_item:
                    continue

                # Calculate similarity using JARO_WINKLER
                try:
                    similarity = 1 - distance.JaroWinkler.distance(
                        str(item).lower(), str(other_item).lower()
                    )

                    if similarity >= similarity_threshold:
                        group.append(other_item)
                        processed.add(j)
                except Exception as e:
                    print(f"Error comparing {item} and {other_item}: {str(e)}")
                    continue

            similarity_groups.append(group)

        # Step 2: Merge each group, keeping the shortest lowercase version
        normalized = []
        for group in similarity_groups:
            if len(group) == 1:
                normalized.append(group[0])
            else:
                # Find item with shortest lowercase form
                try:
                    shortest = min(group, key=lambda x: len(str(x).lower()))

                    # Record the merge
                    if len(group) > 1:
                        merges.append(
                            {
                                "merged_items": sorted(group, key=lambda x: str(x)),
                                "into": shortest,
                                "similarity_threshold": similarity_threshold,
                            }
                        )
                except Exception as e:
                    print(f"Error finding shortest item in {group}: {str(e)}")
                    shortest = group[0]  # Fall back to first item

                normalized.append(shortest)

        return normalized, merges

    all_merges = []
    normalized_entities = []

    # Apply normalization to entities column
    for idx, row_entities in enumerate(df["entities"]):
        try:
            norm_entities, merges = normalize_list(row_entities)
            normalized_entities.append(norm_entities)
            all_merges.extend(merges)
        except Exception as e:
            print(f"Error normalizing entities in row {idx}: {str(e)}")
            print(f"Value: {row_entities}")
            normalized_entities.append(row_entities)  # Keep original on error

    df["entities"] = normalized_entities

    print(f"\n===== Entity Normalization Report =====")
    print(f"Using similarity threshold: {similarity_threshold}")
    print(f"Total merges performed: {len(all_merges)}")

    for i, merge in enumerate(all_merges, 1):
        items_str = ", ".join(f'"{item}"' for item in merge["merged_items"])
        print(f"  {i}. Merged [{items_str}] → \"{merge['into']}\"")

    print("===============================================\n")

    return df
