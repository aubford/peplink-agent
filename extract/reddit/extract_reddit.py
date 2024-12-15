from langchain_community.document_loaders import RedditPostsLoader
from langchain_core.documents import Document
from typing import List
from toolz import keyfilter
from extract.base_extractor import BaseExtractor
from util.util import serialize_document, deduplicate_page_content


class RedditPostExtractor(BaseExtractor):
    def __init__(self, search_queries: List[str]):
        super().__init__("reddit")
        self.loader = RedditPostsLoader(
            client_id=self.config.get("REDDIT_CLIENT_ID"),
            client_secret=self.config.get("REDDIT_CLIENT_SECRET"),
            user_agent="Mozilla/5.0 (compatible; MyBot/1.0; +https://www.example.com)",
            categories=["hot", "new", "top", "rising"],
            mode="subreddit",
            search_queries=search_queries,
            number_posts=5000
        )

    @staticmethod
    def serialize_doc(document: Document) -> dict:
        serialized_document = serialize_document(document)
        metadata = serialized_document["metadata"]
        post_author = metadata['post_author']
        if post_author is not None:
            pick_keys = {'id',
                         'total_karma',
                         'verified',
                         'fullname',
                         'has_subscribed',
                         'has_verified_email',
                         'hide_from_robots',
                         'accept_followers',
                         'awardee_karma',
                         'awarder_karma',
                         'comment_karma',
                         'is_blocked',
                         'is_employee',
                         'is_gold',
                         'is_mod',
                         'link_karma',
                         'name'}
            filtered = keyfilter(
                pick_keys.__contains__,
                post_author.__dict__
            )
            metadata['post_author'] = filtered
        return serialized_document

    def extract(self) -> List[dict]:
        documents = self.loader.load()
        print(f"Fetched {len(documents)} documents")
        deduplicated = deduplicate_page_content(documents)
        print(f"Deduplicated to {len(deduplicated)} documents")
        serialized = [self.serialize_doc(document) for document in deduplicated]
        return serialized


extractor = RedditPostExtractor(['Peplink'])
data = extractor.extract()
extractor.write_json(data, "peplink")
