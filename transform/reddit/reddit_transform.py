from __future__ import annotations
import json
import pandas as pd
from pathlib import Path
from transform.base_transform import BaseTransform
from util.util_main import set_string_columns
from typing import List, Optional, Union, TypedDict

# skipping fields:
# author_hide_from_robots
# author_accept_followers
# author_awarder_karma:  ignoring the karma constituents in favor of total karma
# author_awardee_karma
# author_link_karma
# author_comment_karma


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
    is_suspended: Optional[bool] = None
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
    folder_name = "reddit"

    def __init__(self):
        super().__init__()

    def transform_file(self, file_path: Path) -> pd.DataFrame:
        posts = []
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                meta = data["metadata"]
                author = meta["post_author"]
                if not author:
                    print("No Author: Skipping")
                    continue

                post = self.add_required_columns(
                    columns={
                        # Post metadata
                        "subreddit": meta["subreddit"],
                        "category": meta["category"],
                        "title": meta["title"],
                        "score": meta["score"],
                        "url": meta["url"],
                        # Author metadata
                        "author_name": author["name"],
                        "author_id": author["id"],
                        "author_is_mod": author["is_mod"],
                        "author_is_gold": author["is_gold"],
                        "author_is_blocked": author["is_blocked"],
                        "author_total_karma": author["total_karma"],
                        "author_verified": author["verified"],
                        "author_has_verified_email": author["has_verified_email"],
                        "author_has_subscribed": author["has_subscribed"],
                        "author_is_employee": author["is_employee"],
                    },
                    page_content=self.create_page_content(data),
                    file_path=file_path,
                    doc_id=meta["id"],
                )

                # filter out if no comments survive transform_comments

                posts.append(post)
        df = self.make_df(posts)

        # validate score is integer and not NaN or None
        df["score"] = pd.to_numeric(df["score"], errors="raise").astype("int64")

        set_string_columns(df, ["subreddit", "category", "title", "url", "author_name"], False)

        # Count rows with None in any author_ column
        author_columns = [col for col in df.columns if col.startswith("author_")]
        rows_without_author_count = df[author_columns].isna().any(axis=1).sum()
        print(f"{file_path}: Found {rows_without_author_count} rows with None values in author columns")

        return df

    @staticmethod
    def is_quality_comment(comment: RedditComment, min_karma: int = 40, min_score: int = 2) -> bool:
        author = comment["comment_author"]
        score = comment["score"]
        return (
            len(comment["body"]) > 100
            # if there is no is_blocked key that means author was deleted so we can consider them blocked
            and not author.get("is_blocked", True)
            and (
                (author["total_karma"] > min_karma and score >= min_score)
                or (score >= min_score + 1 or author.get("is_gold", False) or author["total_karma"] > 500)
            )
        )

    def select_high_quality_replies(
        self, replies: list[RedditComment], min_karma: int = 40, min_score: int = 2
    ) -> list[RedditComment]:
        """
        Given a list of comments/replies, return comments that meet the following criteria.

        - Comment.body is longer than 100 characters.
        - comment.author.is_blocked = False
        - comment.author.total_karma > min_karma
        - Comment.score greater than min_score.

        and one of the following are true:
        - Comment.score greater than min_score + 1.
        - comment.author.is_gold = True
        - comment.author.total_karma > 500
        """

        return [comment for comment in replies if self.is_quality_comment(comment, min_karma, min_score)]

    def transform_comment(self, comment: RedditComment) -> str | None:
        """
        Turn the comment's replies hierarchy into a string xml representation of a conversation that an LLM can understand.
        Start with the comment body and then follow the "replies" field recursively to build the conversation.

        Each comment is represented as a string with the following format:

        <comment> {comment_body}
            <reply> {body}
                <reply> {body} </reply>
            </reply>
            <reply> {body} </reply>
        </comment>

        Returns:
            str: A string in XML format representing the conversation in the comment section of the post.
        """

        high_quality_replies = self.select_high_quality_replies(comment["replies"])

        # if there are no high quality replies and the comment is not itself high quality, return None
        if not high_quality_replies:
            if self.is_quality_comment(comment, min_karma=10, min_score=1):
                return f"<comment> {comment['body']} </comment>"
            else:
                return None

        # recursively prune the reply tree to only include high quality replies and their ancestors
        def is_quality_node(node: RedditComment) -> bool:
            return node in high_quality_replies or any(is_quality_node(r) for r in node["replies"])

        # check if the reply is a direct child of the comment to build a reply tree instead of forest
        def is_child_node(comment_or_reply: RedditComment, reply: RedditComment) -> bool:
            return comment_or_reply["id"] in reply["parent_id"]

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

        pruned_comment = prune_reply_tree(comment)

        def build_xml(reply_comment: RedditComment, depth: int = 0) -> str:
            xml = f"{'  ' * depth}<reply> {reply_comment['body']}"
            if reply_comment["replies"]:
                xml += "\n"
                for r in reply_comment["replies"]:
                    xml += build_xml(r, depth + 1)
                xml += f"{'  ' * depth}</reply>\n"
            else:
                xml += " </reply>\n"
            return xml

        xml_str = f"<comment> {pruned_comment['body']}\n"
        for reply in pruned_comment["replies"]:
            xml_str += build_xml(reply, depth=1)
        xml_str += "</comment>"
        return xml_str

    def create_page_content(self, post: dict) -> str:
        """
        Create a page content string from a reddit post dict with this format:

        {post["title"]}

        {post["page_content"]}

        {
            List of comments using self.transform_comment in post["comments"]
        }

        """

        meta = post["metadata"]
        # Filter out None values from comments
        comments = list(filter(None, map(self.transform_comment, meta["comments"])))
        return f"## Reddit Post: {meta['title']}\n\n{post['page_content']}\n\n## Comments:\n\n{'\n\n'.join(comments)}"


if __name__ == "__main__":
    transformer = RedditTransform()
    transformer.transform()
