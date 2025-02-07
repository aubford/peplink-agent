import pytest
from transform.reddit.reddit_transform import RedditTransform, RedditComment


# noinspection PyTypeChecker
class TestRedditTransform:

    def test_transform_comment(self):
        transformer = RedditTransform()

        high_quality_reply = {
            "id": "abc123",
            "parent_id": "t1_def456",
            "body": "This is a high quality reply to the reply with more than 100 characters to ensure it meets the length requirement.",
            "score": 3,
            "comment_author": {
                "is_blocked": False,
                "total_karma": 200,
                "is_gold": False,
            },
            "replies": [],
        }

        low_quality_blocked_reply = {
            "id": "ghi789",
            "parent_id": "t1_def456",
            "body": "Low quality reply",
            "score": 1,
            "comment_author": {
                "is_blocked": True,
                "total_karma": 100000,
                "is_gold": False,
            },
            "replies": [],
        }

        low_quality_reply = {
            "id": "jkl012",
            "parent_id": "t1_def456",
            "body": "Another low quality reply",
            "score": 0,
            "comment_author": {
                "is_blocked": False,
                "total_karma": 5,
                "is_gold": False,
            },
            "replies": [],
        }

        comment: RedditComment = {
            "id": "mno345",
            "parent_id": "t1_pqr678",  # parent is a submission
            "body": "This is a high quality comment with more than 100 characters to ensure it meets the length requirement for a quality comment.",
            "score": 10,
            "comment_author": {
                "is_blocked": False,
                "total_karma": 500,
                "is_gold": False,
            },
            "replies": [
                {
                    "id": "def456",
                    "parent_id": "t1_mno345",
                    "body": "This is a high quality reply with more than 100 characters to ensure it meets the length requirement for a quality reply.",
                    "score": 1,
                    "comment_author": {
                        "is_blocked": False,
                        "total_karma": 1000,
                        "is_gold": True,
                    },
                    "replies": [
                        high_quality_reply,
                        low_quality_blocked_reply,
                        low_quality_reply,
                    ],
                },
                high_quality_reply,
                low_quality_blocked_reply,
                low_quality_reply,
            ],
        }

        expected_xml = (
            "<comment> This is a high quality comment with more than 100 characters to ensure it meets the length requirement for a quality comment.\n"
            "  <reply> This is a high quality reply with more than 100 characters to ensure it meets the length requirement for a quality reply.\n"
            "    <reply> This is a high quality reply to the reply with more than 100 characters to ensure it meets the length requirement. </reply>\n"
            "  </reply>\n"
            "</comment>"
        )

        actual_xml = transformer.transform_comment(comment)
        assert actual_xml == expected_xml

    def test_transform_comment_returns_none_for_bad_comments(self):
        transformer = RedditTransform()

        bad_comment: RedditComment = {
            "id": "stu901",
            "parent_id": "t1_vwx234",
            "body": "Bad comment",
            "score": 0,
            "comment_author": {
                "is_blocked": True,
                "total_karma": 5,
                "is_gold": False,
            },
            "replies": [
                {
                    "id": "yz789a",
                    "parent_id": "t1_stu901",
                    "body": "Another bad reply",
                    "score": 0,
                    "comment_author": {
                        "is_blocked": False,
                        "total_karma": 5,
                        "is_gold": False,
                    },
                    "replies": [],
                }
            ],
        }

        actual_xml = transformer.transform_comment(bad_comment)
        assert actual_xml is None

    def test_transform_comment_keeps_good_descendants(self):
        transformer = RedditTransform()

        comment_with_good_descendants: RedditComment = {
            "id": "bcd123",
            "parent_id": "t1_efg456",
            "body": "Bad quality comment with more than 100 characters to ensure it meets the length requirement for a quality comment.",
            "score": 0,
            "comment_author": {
                "is_blocked": False,
                "total_karma": 5,
                "is_gold": False,
            },
            "replies": [
                {
                    "id": "hij789",
                    "parent_id": "t1_bcd123",
                    "body": "Good quality reply with more than 100 characters to ensure it meets the length requirement for a quality reply.",
                    "score": 3,
                    "comment_author": {
                        "is_blocked": False,
                        "total_karma": 200,
                        "is_gold": False,
                    },
                    "replies": [],
                }
            ],
        }

        expected_xml = (
            "<comment> Bad quality comment with more than 100 characters to ensure it meets the length requirement for a quality comment.\n"
            "  <reply> Good quality reply with more than 100 characters to ensure it meets the length requirement for a quality reply. </reply>\n"
            "</comment>"
        )

        actual_xml = transformer.transform_comment(comment_with_good_descendants)
        assert actual_xml == expected_xml

    def test_create_page_content(self):
        transformer = RedditTransform()

        document = {
            "page_content": "This is the main post content.",
            "metadata": {
                "title": "Test Post Title",
                "comments": [
                    {
                        "id": "comment1",
                        "parent_id": "t1_post1",
                        "body": "This is a high quality comment with more than 100 characters to ensure it meets the length requirement for a quality comment.",
                        "score": 10,
                        "comment_author": {
                            "is_blocked": False,
                            "total_karma": 500,
                            "is_gold": False,
                        },
                        "replies": [
                            {
                                "id": "reply1",
                                "parent_id": "t1_comment1",
                                "body": "This is a high quality reply with more than 100 characters to ensure it meets the length requirement for a quality reply.",
                                "score": 5,
                                "comment_author": {
                                    "is_blocked": False,
                                    "total_karma": 200,
                                    "is_gold": False,
                                },
                                "replies": [],
                            }
                        ],
                    }
                ],
            },
        }

        expected_content = (
            "## Reddit Post: Test Post Title\n"
            "\n"
            "This is the main post content.\n"
            "\n"
            "## Comments:\n"
            "\n"
            "<comment> This is a high quality comment with more than 100 characters to ensure it meets the length requirement for a quality comment.\n"
            "  <reply> This is a high quality reply with more than 100 characters to ensure it meets the length requirement for a quality reply. </reply>\n"
            "</comment>"
        )

        actual_content = transformer.create_page_content(document)
        assert actual_content.strip() == expected_content
