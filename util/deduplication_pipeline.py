import itertools
from typing import List, Literal, Set, Tuple
import pandas as pd
from datasketch import MinHash, MinHashLSH
from rapidfuzz import fuzz, process
from config import RotatingFileLogWriter
from util.util_main import dedupe_df_ids, print_replace
from util.viz import plot_number_dist
from util.nlp import (
    compute_simple_precision,
    get_write_pair_log_text,
    timer,
    TokenizedDoc,
)
from difflib import SequenceMatcher
import nltk


def prep_str_for_matching(s: str) -> str:
    tokens = nltk.wordpunct_tokenize(s.lower())
    tokens = [token for token in tokens if token.isalnum() and len(token) > 1]
    return " ".join(tokens)


class DeduplicationPipeline:
    def __init__(self, name: str):
        self.logger = RotatingFileLogWriter(f"deduplication_pipeline-{name}")
        self.filter_logger = RotatingFileLogWriter(f"dedupe-filter-{name}")
        self.candidate_logger = RotatingFileLogWriter(f"dedupe-candidates-{name}")
        self.duplicate_logger = RotatingFileLogWriter(f"dedupe-duplicates-{name}")

    def _tokenize_documents(self, df: pd.DataFrame) -> List[TokenizedDoc]:
        """Tokenize all documents in the dataframe."""
        self.logger.log_and_print_header(f"Tokenize {len(df)} docs")
        tokenized_docs = []
        for _, row in df.iterrows():
            doc = TokenizedDoc(row)
            tokenized_docs.append(doc)
        self.logger.log_and_print(f"*Tokenization Complete: {len(tokenized_docs)}")
        return tokenized_docs

    def _log_duplicate_group(self, group: List[int], docs: List[TokenizedDoc]) -> None:
        group_docs = [docs[i] for i in group]
        self.filter_logger.info(
            f"Group:\n{'\n'.join([
                doc.original_text.replace('\n', ' ').strip() for doc in group_docs
            ])}"
        )

    @timer("Filter exact duplicates")
    def filter_exact_duplicates_minhash(
        self,
        docs: List[TokenizedDoc],
        *,
        threshold: float = 0.98,
        min_unique_tokens: int = 30,
    ) -> List[TokenizedDoc]:
        """
        Returns filtered corpus with exact duplicates removed.
        This is a set comparison but uses bigrams to account for some word order.
        """
        self.logger.log_and_print_header(f"Filter exact duplicates for: {len(docs)} docs")
        self.logger.log_and_print(f"N-Gram: 2, Threshold: {threshold}")

        # Ignore docs with too few unique tokens
        valid_docs: List[TokenizedDoc] = []
        valid_indices: List[int] = []
        for i, doc in enumerate(docs):
            # Convert to set to count unique tokens
            if len(set(doc.tokens)) >= min_unique_tokens:
                valid_docs.append(doc)
                valid_indices.append(i)

        # Process only valid docs with MinHash
        minhashes = MinHash.bulk([doc.get_encoded_tokens(ngram=2) for doc in valid_docs], num_perm=1024)
        # Shift the minhashes by 1 word to account for bigram shifted overlap issue
        shifted_minhashes = MinHash.bulk(
            [doc.get_encoded_tokens(ngram=2, shift=1) for doc in valid_docs], num_perm=1024
        )
        # initialize the lsh
        lsh = MinHashLSH(threshold=threshold, num_perm=1024)

        to_remove = []
        for i, m in enumerate(minhashes):
            result = lsh.query(m)
            # if no result check if there is a match when shifted by 1 word to compensate for bigram shift issue
            if not result:
                shifted_m = shifted_minhashes[i]
                result = lsh.query(shifted_m)
                if result:
                    lsh.insert(i, shifted_m)
                else:
                    lsh.insert(i, m)
            else:
                # insert lazily for efficiency
                lsh.insert(i, m)

            # a result of length >0 means there are duplicates
            if result:
                result = set(result)
                # check if any of the existing sets in to_remove are a subset of the result
                # this allows us to insert lazily above
                for existing_set in to_remove:
                    if result == existing_set:
                        existing_set.add(i)
                        break
                else:
                    to_remove.append(result | {i})

        # Map back to original indices for removal
        indices_to_remove = set()
        for group in to_remove:
            # Convert group indices back to original doc indices
            original_group = {valid_indices[i] for i in group}
            self._log_duplicate_group(original_group, docs)
            original_group.pop()  # Keep one representative
            indices_to_remove.update(original_group)

        # Return filtered corpus
        filtered_corpus = [doc for i, doc in enumerate(docs) if i not in indices_to_remove]

        self.logger.log_and_print(f"*Filtered corpus from {len(docs)} to {len(filtered_corpus)}")
        return filtered_corpus

    @timer("Get candidates")
    def get_duplicate_candidates_simple_precision(
        self,
        docs: List[TokenizedDoc],
        *,
        ngram: int = 1,
        threshold: float = 0.8,
        report: Literal["plot", "print", None] = None,
    ) -> List[Tuple[TokenizedDoc, TokenizedDoc]]:
        """
        Returns pairs of documents that are potential duplicates using simple precision.
        This is a set comparison, so word order/count is irrelevant except for the ngram factor.
        """
        self.logger.log_and_print_header(f"Get Candidates (precision) for: {len(docs)} docs")
        self.logger.log_and_print(f"N-Gram: {ngram}, Threshold: {threshold}")

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
                    self.candidate_logger.info(
                        get_write_pair_log_text(
                            docs[i].original_text,
                            docs[j].original_text,
                            "Candidate Found",
                        )
                    )
                    candidates.append((docs[i], docs[j]))

        if report:
            self.logger.log_and_print(f"Simple Precision Comparisons: {len(similarities)}")
            if report == "plot":
                plot_number_dist(similarities)
            elif report == "print":
                self.logger.log_and_print(f"Simple Precisions: {similarities}")

        self.logger.log_and_print(f"*Simple Precision Complete. Num candidates: {len(candidates)}")
        return candidates

    def _log_sequence_matches(self, str1: str, str2: str) -> None:
        # str1 = prep_str_for_matching(str1)
        # str2 = prep_str_for_matching(str2)
        matcher = SequenceMatcher(None, str1, str2)
        matches = []
        for block in matcher.get_matching_blocks():
            if block.size > 0:  # Only include non-empty matches
                match_text = str1[block.a : block.a + block.size]
                matches.append(match_text)

        joined_matches = "], [".join(matches) if matches else "No matches found"
        self.duplicate_logger.info(f"Matches: [ {joined_matches} ]")

    @timer("Confirm duplicates")
    def _confirm_duplicates(
        self,
        candidate_pairs: List[Tuple[TokenizedDoc, TokenizedDoc]],
        *,
        threshold: int = 90,
    ) -> Set[str]:
        """Returns set of document IDs that are duplicates"""

        if not candidate_pairs:
            return set()
        call_counter = itertools.count(1)
        high_scores = 0

        def _progress_scorer(s1: str, s2: str, **kwargs) -> float:
            # high scores will always be much higher than duplicates found since these will flag the same item several times
            nonlocal high_scores
            print_replace(
                f"Processed {next(call_counter)} comparisons. Flagged comparisons: {high_scores}. Next item lengths: {len(s1)}, {len(s2)}"
            )

            # start_time = time.time()
            score = fuzz.partial_ratio(s1, s2)
            # duration = time.time() - start_time

            if score > threshold:
                high_scores += 1

            # if duration > 25:
            #     duplicate_logger.info(
            #         log_write_pair(
            #             s1,
            #             s2,
            #             msg=f"Slow comparison ({duration:.2f}s).  Lengths: {len(s1)}, {len(s2)}",
            #         )
            #     )

            return score

        self.logger.log_and_print_header(f"Getting duplicates for: {len(candidate_pairs)} pairs")
        self.logger.log_and_print(f"Threshold: {threshold}")
        tokenized_docs_a, tokenized_docs_b = zip(*candidate_pairs)
        strings_a = [" ".join(doc.tokens) for doc in tokenized_docs_a]
        strings_b = [" ".join(doc.tokens) for doc in tokenized_docs_b]

        distances = process.cpdist(strings_a, strings_b, scorer=_progress_scorer, workers=8)
        duplicates = set()
        for idx, (doc_a, doc_b) in enumerate(candidate_pairs):
            if distances[idx] > threshold:
                self.duplicate_logger.info(get_write_pair_log_text(doc_a.original_text, doc_b.original_text))
                self._log_sequence_matches(doc_a.original_text, doc_b.original_text)
                self.duplicate_logger.info(
                    get_write_pair_log_text(" ".join(doc_a.tokens), " ".join(doc_b.tokens), "Tokenized:")
                )
                self._log_sequence_matches(" ".join(doc_a.tokens), " ".join(doc_b.tokens))

                if len(doc_a.tokens) < len(doc_b.tokens):
                    duplicates.add(doc_a.doc_id)
                else:
                    duplicates.add(doc_b.doc_id)

        self.logger.log_and_print(f"\n*Confirm Duplicates Complete: Found ({len(duplicates)}) duplicates")
        return duplicates

    def _dedupe_df_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.log_and_print_header("Removing duplicate IDs")
        self.logger.log_and_print(f"\nStarting docs: {len(df)}")
        deduped_df = dedupe_df_ids(df)
        self.logger.log_and_print(f"Final docs: {len(deduped_df)}")
        return deduped_df

    def _apply_deduplication(
        self,
        df: pd.DataFrame,
        filtered_docs: List[TokenizedDoc],
        *,
        precision_threshold: float = 0.7,
        precision_ngram: int = 2,
        duplicate_threshold: int = 95,
        report_candidates: Literal["plot", "print", None] = None,
    ) -> pd.DataFrame:
        duplicate_candidates = self.get_duplicate_candidates_simple_precision(
            filtered_docs,
            threshold=precision_threshold,
            ngram=precision_ngram,
            report=report_candidates,
        )
        duplicate_doc_ids = self._confirm_duplicates(duplicate_candidates, threshold=duplicate_threshold)

        filtered_docs_ids_deduped = [doc.doc_id for doc in filtered_docs if doc.doc_id not in duplicate_doc_ids]
        self.logger.log_and_print(f"Filtered doc ids after deduplication: {len(filtered_docs_ids_deduped)}")

        df_deduped = df[df["id"].isin(filtered_docs_ids_deduped)]
        self.logger.log_and_print(f"Rows after deduplication: {len(df_deduped)}")
        return df_deduped

    def run(
        self,
        df: pd.DataFrame,
        *,
        precision_threshold: float = 0.7,
        precision_ngram: int = 2,
        duplicate_threshold: int = 95,
        report_candidates: Literal["plot", "print", None] = None,
    ) -> pd.DataFrame:
        df = df.drop_duplicates(subset=["page_content"], keep="first")
        tokenized_docs = self._tokenize_documents(df)
        filtered_docs = self.filter_exact_duplicates_minhash(tokenized_docs, threshold=0.98)
        return self._apply_deduplication(
            df,
            filtered_docs,
            precision_threshold=precision_threshold,
            precision_ngram=precision_ngram,
            duplicate_threshold=duplicate_threshold,
            report_candidates=report_candidates,
        )

    def run_dedupe_chunks(
        self,
        df: pd.DataFrame,
        *,
        precision_threshold: float = 0.95,
        precision_ngram: int = 1,
        duplicate_threshold: int = 95,
        report_candidates: Literal["plot", "print", None] = None,
    ) -> pd.DataFrame:
        df = df.drop_duplicates(subset=["page_content"], keep="first")
        tokenized_docs = self._tokenize_documents(df)
        return self._apply_deduplication(
            df,
            tokenized_docs,
            precision_threshold=precision_threshold,
            precision_ngram=precision_ngram,
            duplicate_threshold=duplicate_threshold,
            report_candidates=report_candidates,
        )
