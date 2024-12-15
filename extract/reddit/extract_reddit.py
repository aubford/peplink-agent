from langchain_community.document_loaders import RedditPostsLoader
from langchain_core.documents import Document
from typing import List
from toolz import keyfilter
from config import config, ConfigType
from extract.base_extractor import BaseExtractor
from util.util import serialize_document


class RedditPostExtractor(BaseExtractor):
    def __init__(self, search_queries: List[str], _config: ConfigType):
        super().__init__("reddit")
        self.loader = RedditPostsLoader(
            client_id=_config.get("REDDIT_CLIENT_ID"),
            client_secret=_config.get("REDDIT_CLIENT_SECRET"),
            user_agent="Mozilla/5.0 (compatible; MyBot/1.0; +https://www.example.com)",
            categories=["hot", "new", "top", "rising"],
            mode="subreddit",
            search_queries=search_queries,
            number_posts=10000
        )

    @staticmethod
    def serialize_doc(document: Document) -> dict:
        serialized_document = serialize_document(document)
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
        metadata = serialized_document["metadata"]
        filtered = keyfilter(
            pick_keys.__contains__,
            metadata['post_author'].__dict__
        )
        metadata['post_author'] = filtered
        return serialized_document

    def extract(self) -> List[dict]:
        documents = self.loader.load()
        serialized = [self.serialize_doc(document) for document in documents]
        return serialized


extractor = RedditPostExtractor(['Peplink'], config)
data = extractor.extract()
extractor.write_json(data, "peplink")
