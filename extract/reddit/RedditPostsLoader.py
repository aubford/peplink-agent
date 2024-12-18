from __future__ import annotations
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader

if TYPE_CHECKING:
    import praw


def _dependable_praw_import() -> praw:
    try:
        import praw
    except ImportError:
        raise ImportError(
            "praw package not found, please install it with `pip install praw`"
        )
    return praw


class RedditPostsLoader(BaseLoader):
    """Load `Reddit` posts.

    Read posts on a subreddit.
    First, you need to go to
    https://www.reddit.com/prefs/apps/
    and create your application
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        user_agent: str,
        search_queries: Sequence[str],
        mode: str,
        categories: Sequence[str] = ["new"],
    ):
        """
        Initialize with client_id, client_secret, user_agent, search_queries, mode,
            categories.
        Example: https://www.reddit.com/r/learnpython/

        Args:
            client_id: Reddit client id.
            client_secret: Reddit client secret.
            user_agent: Reddit user agent.
            search_queries: The search queries.
            mode: The mode.
            categories: The categories. Default: ["new"]
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.search_queries = search_queries
        self.mode = mode
        self.categories = categories

    def load(self) -> List[Document]:
        """Load reddits."""
        praw = _dependable_praw_import()

        reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent,
        )

        results: List[Document] = []

        if self.mode == "subreddit":
            for search_query in self.search_queries:
                for category in self.categories:
                    docs = self._subreddit_posts_loader(
                        search_query=search_query, category=category, reddit=reddit
                    )
                    results.extend(docs)

        elif self.mode == "username":
            for search_query in self.search_queries:
                for category in self.categories:
                    docs = self._user_posts_loader(
                        search_query=search_query, category=category, reddit=reddit
                    )
                    results.extend(docs)

        else:
            raise ValueError(
                "mode not correct, please enter 'username' or 'subreddit' as mode"
            )

        return results

    def _subreddit_posts_loader(
        self, search_query: str, category: str, reddit: praw.reddit.Reddit
    ) -> Iterable[Document]:
        subreddit = reddit.subreddit(search_query)
        method = getattr(subreddit, category)
        cat_posts = method()

        """Format reddit posts into a string."""
        for post in cat_posts:
            # Get comments
            post.comments.replace_more(limit=None)  # Replace all MoreComments objects
            comments = []
            for comment in post.comments.list():
                try:
                    comments.append({
                        'body': comment.body,
                        'score': comment.score,
                        'id': comment.id,
                        'author': comment.author.name if comment.author else '[deleted]',
                        'created_utc': comment.created_utc,
                        'is_submitter': comment.is_submitter,
                        'parent_id': comment.parent_id,
                        'permalink': comment.permalink
                    })
                except Exception as e:
                    print(f"Error processing comment {comment.id}: {e}")
                    continue

            metadata = {
                "post_subreddit": post.subreddit_name_prefixed,
                "post_category": category,
                "post_title": post.title,
                "post_score": post.score,
                "post_id": post.id,
                "post_url": post.url,
                "post_author": post.author,
                "post_comments": comments
            }
            yield Document(
                page_content=post.selftext,
                metadata=metadata,
            )

    def _user_posts_loader(
        self, search_query: str, category: str, reddit: praw.reddit.Reddit
    ) -> Iterable[Document]:
        user = reddit.redditor(search_query)
        method = getattr(user.submissions, category)
        cat_posts = method()

        """Format reddit posts into a string."""
        for post in cat_posts:
            metadata = {
                "post_subreddit": post.subreddit_name_prefixed,
                "post_category": category,
                "post_title": post.title,
                "post_score": post.score,
                "post_id": post.id,
                "post_url": post.url,
                "post_author": post.author,
            }
            yield Document(
                page_content=post.selftext,
                metadata=metadata,
            )
