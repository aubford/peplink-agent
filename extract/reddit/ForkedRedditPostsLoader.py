from __future__ import annotations
from collections.abc import Iterator
from typing import Any, Iterable, List, Sequence
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from toolz import keyfilter
import praw


class ForkedRedditPostsLoader(BaseLoader):
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
            categories: The categories. Default: ["new"]
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.search_queries = search_queries
        self.categories = categories
        self.seen = set()

    def load(self) -> List[Document]:
        """Load reddits."""
        reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent,
        )

        results: List[Document] = []
        for search_query in self.search_queries:
            for category in self.categories:
                docs = self._subreddit_posts_loader(
                    search_query=search_query, category=category, reddit=reddit
                )
                results.extend(docs)

        return results

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load reddit posts."""
        reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent,
        )

        for search_query in self.search_queries:
            for category in self.categories:
                yield from self._subreddit_posts_loader(
                    search_query=search_query, category=category, reddit=reddit
                )

    def _subreddit_posts_loader(
        self, search_query: str, category: str, reddit: praw.reddit.Reddit
    ) -> Iterable[Document]:
        subreddit = reddit.subreddit(search_query)
        method = getattr(subreddit, category)
        cat_posts = method()

        """Format reddit posts into a string."""
        for post in cat_posts:
            if post.id in self.seen:
                continue
            metadata = {
                "subreddit": post.subreddit_name_prefixed,
                "subreddit_id": post.subreddit_id,
                "category": category,
                "title": post.title,
                "score": post.score,
                "id": post.id,
                "url": post.url,
                "post_author": self._format_author(post.author),
                "comments": self._get_comments(post.comments, roots_only=True),
            }
            self.seen.add(post.id)
            yield Document(
                page_content=post.selftext,
                metadata=metadata,
            )

    def _get_comments(self, post_comments: Any, roots_only: bool = False) -> List[dict]:
        post_comments.replace_more(limit=None)  # Replace all MoreComments objects
        comments = []
        for comment in post_comments.list():
            if comment.author is None or (roots_only and not comment.is_root):
                continue
            try:
                comments.append(self._format_comment(comment))
            except Exception as e:
                print(f"Error processing comment {comment.id}: {e}")
                continue
        return comments

    def _format_comment(self, comment: praw.reddit.Comment) -> dict:
        return {
            "id": comment.id,
            "body": comment.body,
            "score": comment.score,
            "comment_author": self._format_author(comment.author),
            "created_utc": comment.created_utc,
            "edited": comment.edited,
            "distinguished": comment.distinguished,
            "stickied": comment.stickied,
            "saved": comment.saved,
            "is_submitter": comment.is_submitter,
            "link_id": comment.link_id,
            "parent_id": comment.parent_id,
            "permalink": comment.permalink,
            "replies": self._get_comments(comment.replies),
        }

    def _format_author(self, author: praw.reddit.Submission) -> dict:
        if author is not None:
            result = {}
            pick_keys = {
                "id",
                "verified",
                "fullname",
                "has_subscribed",
                "has_verified_email",
                "accept_followers",
                "awardee_karma",
                "awarder_karma",
                "comment_karma",
                "total_karma",
                "link_karma",
                "is_suspended",
                "is_blocked",
                "is_employee",
                "is_gold",
                "is_mod",
                "name",
            }

            # Explicitly access each attribute to force loading
            for key in pick_keys:
                try:
                    result[key] = getattr(author, key)
                except (AttributeError, Exception):
                    continue

            return result
