from extract.reddit.ForkedRedditPostsLoader import ForkedRedditPostsLoader
from extract.base_extractor import BaseExtractor, Ldoc
from util.util import serialize_document
import time

class RedditPostExtractor(BaseExtractor):
    def __init__(self, subreddit: str):
        super().__init__("reddit")
        self.loader = ForkedRedditPostsLoader(
            client_id=self.config.get("REDDIT_CLIENT_ID"),
            client_secret=self.config.get("REDDIT_CLIENT_SECRET"),
            user_agent="Mozilla/5.0 (compatible; MyBot/1.0; +https://www.example.com)",
            categories=["hot", "new", "top", "rising"],
            search_queries=[subreddit]
        )
        self.subreddit = subreddit

    def extract(self) -> None:
        stream_key = self.start_stream(Ldoc, identifier=self.subreddit)
        for doc in self.loader.lazy_load():
            self.stream_item(serialize_document(doc), stream_key)
            time.sleep(1)
        self.end_stream(stream_key)
