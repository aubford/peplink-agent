from __future__ import annotations
import json
import pandas as pd
from pathlib import Path
from transform.base_transform import BaseTransform, SubjectMatter
from util.util_main import set_string_columns
from typing import List, Optional, Union, TypedDict
from transform.reddit.get_score_cutoff import get_score_cutoff_percentile


class RedditAuthor(TypedDict):
    id: str
    verified: bool
    fullname: str
    has_subscribed: bool
    has_verified_email: bool
    accept_followers: bool
    awardee_karma: int
    awarder_karma: int
    comment_karma: int
    total_karma: int
    link_karma: int
    is_suspended: Optional[bool]
    is_blocked: bool
    is_employee: bool
    is_gold: bool
    is_mod: bool
    name: str


class RedditComment(TypedDict):
    id: str
    body: str
    score: int
    comment_author: RedditAuthor
    created_utc: float
    edited: Union[bool, float]
    distinguished: Optional[str]
    stickied: bool
    saved: bool
    is_submitter: bool
    link_id: str
    parent_id: str
    permalink: str
    replies: List[RedditComment]


class RedditTransform(BaseTransform):
    def __init__(self, folder_name: str, subject_matter: SubjectMatter):
        self.subject_matter = subject_matter
        self.folder_name = folder_name
        super().__init__()

    def transform_file(self, file_path: Path) -> pd.DataFrame:
        post_and_comments = []
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                author = data["metadata"]["post_author"]
                if not author:
                    print("No Author: Skipping")
                    continue

                if not data["page_content"]:
                    print("No Post Content: Skipping")
                    continue

                post_and_comments.extend(
                    self.transform_post_into_post_comments(data, file_path)
                )
        df = self.make_df(post_and_comments)

        # validate score is integer and not NaN or None
        df["score"] = pd.to_numeric(df["score"], errors="raise").astype("int64")

        set_string_columns(
            df,
            [
                "subreddit",
                "category",
                "post_title",
                "url",
                "post_id",
                "post_author_name",
                "comment_author_name",
                "post_author_id",
                "comment_author_id",
                "post_content",
                "comment_content",
            ],
            False,
        )

        # Count rows with None in any author_ column
        author_columns = [col for col in df.columns if col.startswith("author_")]
        rows_without_author_count = df[author_columns].isna().any(axis=1).sum()
        print(
            f"{file_path}: Found {rows_without_author_count} rows with None values in author columns"
        )

        # Reorder columns to put specified columns first
        first_cols = ["page_content", "post_title", "post_content", "comment_content"]
        df = self.reorder_columns(df, first_cols)

        return df

    @staticmethod
    def author_invalid(author):
        invalid = (
            not author or author.get("is_suspended", False) or author["is_blocked"]
        )
        if invalid:
            print(f"Author invalid, skipping\n{author}")
        return invalid

    def is_quality_comment_or_reply(
        self,
        comment: RedditComment,
        min_karma: int = 50,
        min_score: int = 2,
        min_length: int = 20,
    ) -> bool:
        """
        Check if a comment or reply is of high quality. High quality comments and replies are either: (defaults)
        - Post length meets min length (20 words)
            AND
        - User is not blocked
            AND
        - Meets min karma (50) and either meets min score (2) or post length is long (190 words)
            OR
        - Is gold, has min score + 1 (3) or very high karma (>1000)

        Note: Studies have shown that 25% of commenters have 2 or fewer karma points, 50% have 8 or fewer,
        and 75% have 86 or fewer. The mean karma was found to be 633, suggesting that the top 25% of users
        average well above 3,000 karma points.
        """
        author = comment["comment_author"]
        if self.author_invalid(author):
            return False
        score = comment["score"]
        word_count = len(comment["body"].split())
        return word_count >= min_length and (
            (
                author["comment_karma"] >= min_karma
                and (score >= min_score or word_count >= (min_length * 3) + 100)
            )
            or author.get("is_gold", False)
            or score >= min_score + 1
            or author["comment_karma"] > 1000
        )

    def transform_comment(self, comment: RedditComment) -> str | None:
        """
        Turn the comment's reply hierarchy into a string xml representation of a conversation that an LLM can understand.
        Start with the comment body and then follow the "replies" field recursively to build the conversation, filtering
        out low quality content along the way.

        Each comment is represented as a string with the following format:

        <comment>
        {comment_body}

          <reply>
          {body}

            <reply>
            {body}
            </reply>
          </reply>

          <reply>
          {body}
          </reply>
        </comment>

        Returns:
            str: A string in XML format representing the conversation in the comment section of the post.
        """

        high_quality_replies = [
            comment
            for comment in comment["replies"]
            if self.is_quality_comment_or_reply(comment)
        ]

        # If there are no high quality replies and the comment is not itself high quality, return None to be filtered out.
        # If a comment doesn't have quality replies, it needs to have enough content itself to potentially
        # have any information value, so we up the min length. We set the other params to initial since that filtering
        # has already been handled by filter_comments.
        if not high_quality_replies:
            if self.is_quality_comment_or_reply(
                comment, min_karma=0, min_score=1, min_length=40
            ):
                return self.format_comment_xml(comment["body"])
            else:
                return None

        # recursively prune the reply tree to only include high quality replies and their ancestors
        def is_quality_node(node: RedditComment) -> bool:
            return node in high_quality_replies or any(
                is_quality_node(r) for r in node["replies"]
            )

        # check if the reply is a direct child of the comment to build a reply tree instead of forest
        def is_child_node(parent: RedditComment, descendant: RedditComment) -> bool:
            return parent["id"] in descendant["parent_id"]

        # turn reply forest into a tree of high quality replies
        def prune_reply_tree(comment_or_reply: RedditComment) -> dict:
            return {
                "body": comment_or_reply["body"],
                "replies": [
                    prune_reply_tree(r)
                    for r in comment_or_reply["replies"]
                    if is_quality_node(r) and is_child_node(comment_or_reply, r)
                ],
            }

        pruned_comment_tree = prune_reply_tree(comment)

        return self.format_comment_xml(
            pruned_comment_tree["body"], pruned_comment_tree["replies"]
        )

    @staticmethod
    def filter_comments(comments: list[RedditComment]) -> list[RedditComment]:
        """Select the most upvoted comments with selectivity scaled by post engagement and score distribution."""

        if not comments:
            return []

        # first filter out any that have been downvoted
        comments = [c for c in comments if c["score"] > 0]
        if not comments:
            return []
        cutoff_percent = get_score_cutoff_percentile([c["score"] for c in comments])

        # Sort comments by score with fallbacks for ties:
        # 1. score (highest first)
        # 2. author's comment karma (highest first)
        # 3. author's gold status (True first)
        # 4. comment length (longest first)
        sorted_comments = sorted(
            comments,
            key=lambda x: (
                x["score"],
                x["comment_author"].get("comment_karma", 0),
                x["comment_author"].get("is_gold", False),
                len(x["body"]),
            ),
            reverse=True,
        )

        cutoff_index = max(1, round(len(sorted_comments) * cutoff_percent))
        return sorted_comments[:cutoff_index]

    def transform_post_into_post_comments(
        self, post: dict, file_path: Path
    ) -> list[dict]:
        """
        For each comment in the post, create a document with the page content string in this format:

        {post["title"]}

        {post["page_content"]}

        {Single comment and its replies from output of self.transform_comment}

        We use the comment as the first class citizen instead of the post since we are looking for answers to
        questions as opposed to questions themselves. In a representative sample, the post is typically a question and
        the comments are answers with each comment along with its post and replies documenting a single conversation
        about a given "answer" to the "question". We vet the reliability of comments and replies to comments by ensuring
        that they are from a reputable user or have been upvoted. If none of the comments have been upvoted, this is
        likely due to the post having little engagement, which doesn't necessarily mean we should skip it.
        This is where filter_comments comes in.
        """

        meta = post["metadata"]
        author = meta["post_author"]
        comments = meta["comments"]
        comments = self.filter_comments(comments)

        post_comments = []
        for comment in comments:
            transformed_comment_str = self.transform_comment(comment)
            if not transformed_comment_str:
                continue

            comment_author = comment["comment_author"]
            if self.author_invalid(comment_author):
                continue

            transformed_post = self.add_required_columns(
                columns={
                    # Post metadata
                    "subreddit": meta["subreddit"],
                    "category": meta["category"],
                    "url": meta["url"],
                    "post_id": meta["id"],
                    "post_title": meta["title"],
                    "post_content": post["page_content"],
                    "post_score": meta["score"],
                    "score": comment["score"],
                    "comment_date": comment["created_utc"],
                    # Author metadata. Allow deleted post authors but not comment authors.
                    "post_author_name": author.get("name", ""),
                    "post_author_id": author.get("id", ""),
                    "post_author_is_mod": author.get("is_mod", None),
                    "post_author_is_gold": author.get("is_gold", None),
                    "post_author_is_blocked": author.get("is_blocked", None),
                    "post_author_total_karma": author.get("total_karma", 0),
                    "post_author_comment_karma": author.get("comment_karma", 0),
                    "post_author_link_karma": author.get("link_karma", 0),
                    "post_author_verified": author.get("verified", None),
                    "post_author_has_verified_email": author.get(
                        "has_verified_email", None
                    ),
                    "post_author_has_subscribed": author.get("has_subscribed", None),
                    "post_author_is_employee": author.get("is_employee", None),
                    "comment_content": comment["body"],
                    "comment_author_name": comment_author["name"],
                    "comment_author_id": comment_author["id"],
                    "comment_author_is_mod": comment_author["is_mod"],
                    "comment_author_is_gold": comment_author["is_gold"],
                    "comment_author_is_blocked": comment_author["is_blocked"],
                    "comment_author_total_karma": comment_author["total_karma"],
                    "comment_author_comment_karma": comment_author["comment_karma"],
                    "comment_author_link_karma": comment_author["link_karma"],
                    "comment_author_verified": comment_author["verified"],
                    "comment_author_has_verified_email": comment_author[
                        "has_verified_email"
                    ],
                    "comment_author_has_subscribed": comment_author["has_subscribed"],
                    "comment_author_is_employee": comment_author["is_employee"],
                },
                page_content=self.create_page_content(
                    meta["title"], post["page_content"], transformed_comment_str
                ),
                file_path=file_path,
                doc_id=f"p_{meta['id']}_com_{comment['id']}",
            )
            post_comments.append(transformed_post)
        return post_comments


if __name__ == "__main__":
    transformer = RedditTransform(
        folder_name="reddit", subject_matter=SubjectMatter.PEPWAVE
    )
    transformer.transform()
    general_transformer = RedditTransform(
        folder_name="reddit_general", subject_matter=SubjectMatter.IT_NETWORKING
    )
    general_transformer.transform()
