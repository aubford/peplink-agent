import pytest
from transform.reddit.reddit_transform import RedditTransform, RedditComment
from pathlib import Path


# noinspection PyTypeChecker
class TestRedditTransform:

    @pytest.fixture
    def document_partial(self) -> dict:
        return {
            "page_content": "This is the main post content.",
            "metadata": {
                "id": "main_post",
                "title": "Test Post Title",
                "subreddit": "test_subreddit",
                "category": "test_category",
                "score": 100,
                "url": "https://reddit.com/r/test_subreddit/test_post",
                "post_author": {
                    "id": "test_author",
                    "name": "Test Author",
                    "verified": True,
                    "fullname": "t2_test_author",
                    "has_subscribed": True,
                    "has_verified_email": True,
                    "accept_followers": True,
                    "awardee_karma": 100,
                    "awarder_karma": 50,
                    "comment_karma": 500,
                    "total_karma": 1000,
                    "link_karma": 400,
                    "is_suspended": False,
                    "is_blocked": False,
                    "is_employee": False,
                    "is_gold": False,
                    "is_mod": False,
                },
            },
        }

    @staticmethod
    def comment_author_partial() -> dict:
        return {
            "id": "test_comment_author",
            "name": "Test Comment Author",
            "verified": True,
            "fullname": "t2_test_comment_author",
            "has_subscribed": True,
            "has_verified_email": True,
            "accept_followers": True,
            "awardee_karma": 200,
            "awarder_karma": 100,
            "comment_karma": 1000,
            "total_karma": 2000,
            "link_karma": 800,
            "is_suspended": False,
            "is_blocked": False,
            "is_employee": False,
            "is_gold": True,
            "is_mod": True,
        }

    @staticmethod
    def _id(id: str = "42", is_parent: bool = False) -> str:
        return f"t1_{id}" if is_parent else id

    def comment_or_reply(
        self, *, id: str = "42", score: int = 2, parent_id: str = "42", karma: int = 50
    ) -> RedditComment:
        return {
            "id": self._id(id),
            "parent_id": self._id(parent_id, True),
            "body": self.get_body(),
            "score": score,
            "comment_author": {
                **self.comment_author_partial(),
                "is_blocked": False,
                "comment_karma": karma,
                "is_gold": False,
            },
            "replies": [],
        }

    def high_quality_blocked_reply(self, *, id: str = "42", parent_id: str = "42") -> RedditComment:
        return {
            "id": self._id(id),
            "parent_id": self._id(parent_id, True),
            "body": self.get_body(),
            "score": 10000,
            "comment_author": {
                "is_blocked": True,
                "comment_karma": 100000,
                "is_gold": True,
            },
            "replies": [],
        }

    def low_quality_reply(self, *, id: str = "42", parent_id: str = "42") -> RedditComment:
        return {
            "id": self._id(id),
            "parent_id": self._id(parent_id, True),
            "body": self.get_body(False),
            "score": 2,
            "comment_author": {
                "is_blocked": False,
                "comment_karma": 50,
                "is_gold": True,
            },
            "replies": [],
        }

    @staticmethod
    def get_body(length_ok: bool = True) -> str:
        return (
            "This is a high quality reply with more than 40 words to ensure that it meets the length requirement for quality comment."
            * 2
            if length_ok
            else "Short, Low quality reply"
        )

    def test_transform_comment_nesting(self):
        transformer = RedditTransform()
        comment: RedditComment = {
            "id": self._id(),
            "parent_id": "t1_mef345",
            "body": self.get_body(),
            "score": 3,
            "comment_author": {
                **self.comment_author_partial(),
                "is_blocked": False,
                "comment_karma": 500,
                "is_gold": False,
            },
            "replies": [
                {
                    "id": self._id("1"),
                    "parent_id": self._id(is_parent=True),
                    "body": self.get_body(),
                    "score": 1,
                    "comment_author": {
                        "is_blocked": False,
                        "comment_karma": 1000,
                        "is_gold": True,
                    },
                    "replies": [
                        self.comment_or_reply(parent_id="1"),
                        self.high_quality_blocked_reply(parent_id="1"),
                        self.low_quality_reply(parent_id="1"),
                    ],
                },
                self.comment_or_reply(parent_id="1"),
                self.high_quality_blocked_reply(parent_id="1"),
                self.low_quality_reply(parent_id="1"),
            ],
        }

        expected_xml = (
            f"<comment> {self.get_body()}\n"
            f"  <reply> {self.get_body()}\n"
            f"    <reply> {self.get_body()} </reply>\n"
            "  </reply>\n"
            "</comment>"
        )

        actual_xml = transformer.transform_comment(comment)
        assert actual_xml == expected_xml

    def test_transform_comment_returns_none_for_bad_comments(self):
        transformer = RedditTransform()

        bad_comment: RedditComment = {
            "id": self._id(True),
            "parent_id": "t1_vwx234",
            "body": "Bad comment",
            "score": 1,
            "comment_author": {
                **self.comment_author_partial(),
                "is_blocked": True,
                "comment_karma": 5,
                "is_gold": False,
            },
            "replies": [
                self.low_quality_reply(),
            ],
        }

        actual_xml = transformer.transform_comment(bad_comment)
        assert actual_xml is None

    def test_transform_comment_keeps_comments_with_good_descendants(self):
        transformer = RedditTransform()

        comment_with_good_descendants: RedditComment = {
            "id": self._id(),
            "parent_id": "t1_efg456",
            "body": self.get_body(False),
            "score": 1,
            "comment_author": {
                **self.comment_author_partial(),
                "is_blocked": False,
                "comment_karma": 5,
                "is_gold": False,
            },
            "replies": [
                {
                    "id": "hij789",
                    "parent_id": self._id(is_parent=True),
                    "body": self.get_body(),
                    "score": 3,
                    "comment_author": {
                        "is_blocked": False,
                        "comment_karma": 200,
                        "is_gold": False,
                    },
                    "replies": [],
                }
            ],
        }

        expected_xml = f"<comment> {self.get_body(False)}\n" f"  <reply> {self.get_body()} </reply>\n" "</comment>"

        actual_xml = transformer.transform_comment(comment_with_good_descendants)
        assert actual_xml == expected_xml

    def test_transform_post_into_post_comments(self, document_partial: dict):
        transformer = RedditTransform()

        document = document_partial
        document["metadata"]["comments"] = [
            {
                "id": self._id(),
                "parent_id": "main_post",
                "body": self.get_body(),
                "score": 2,
                "comment_author": {
                    **self.comment_author_partial(),
                    "is_blocked": False,
                    "comment_karma": 500,
                    "is_gold": False,
                },
                "replies": [
                    self.comment_or_reply(),
                ],
            },
            self.comment_or_reply(score=2),
            self.comment_or_reply(score=2),
            self.comment_or_reply(score=-1),
            self.comment_or_reply(score=0),
            {
                "id": "short_comment_for_no_replies",
                "parent_id": "main_post",
                "body": "This is a reply with more than 20 words but less than 40 that doesn't meet the length requirement for comments with no quality replies.",
                "score": 2,
                "comment_author": {
                    **self.comment_author_partial(),
                    "is_blocked": False,
                    "comment_karma": 500,
                    "is_gold": True,
                },
                "replies": [
                    self.low_quality_reply(parent_id="short_comment_for_no_replies"),
                    self.high_quality_blocked_reply(parent_id="short_comment_for_no_replies"),
                ],
            },
        ]

        expected_content = (
            "## Reddit Post: Test Post Title\n"
            "\n"
            "This is the main post content.\n"
            "\n"
            "## Comments:\n"
            "\n"
            f"<comment> {self.get_body()}\n"
            f"  <reply> {self.get_body()} </reply>\n"
            "</comment>"
        )

        expected_content_no_replies = (
            "## Reddit Post: Test Post Title\n"
            "\n"
            "This is the main post content.\n"
            "\n"
            "## Comments:\n"
            "\n"
            f"<comment> {self.get_body()} </comment>"
        )

        actual_content = transformer.transform_post_into_post_comments(document, Path("file_path"))
        assert len(actual_content) == 3
        assert actual_content[0]["page_content"].strip() == expected_content
        assert actual_content[1]["page_content"].strip() == expected_content_no_replies
        assert actual_content[2]["page_content"].strip() == expected_content_no_replies
