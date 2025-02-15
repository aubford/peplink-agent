from __future__ import annotations
from typing import Dict, List
from transform.base_transform import BaseMongoTransform, SubjectMatter
import pandas as pd
from util.util_main import set_string_columns


class MongoPepwaveTransform(BaseMongoTransform):
    """Transform for processing MongoDB Pepwave forum data. We don't need to filter this content as strictly as Reddit due to the quality of the community."""

    folder_name: str = "mongo"
    subject_matter: SubjectMatter = SubjectMatter.PEPWAVE

    def __init__(self):
        super().__init__("mongodb://localhost:27017", "pepwave")
        self.posts_collection = self.db["posts"]
        self.topics_collection = self.db["topics"]

    def get_comments(self) -> List[Dict]:
        """
        Retrieve posts where replyTo == topicId, making them first-class citizen posts.
        """
        return list(self.posts_collection.find({"$expr": {"$eq": ["$replyTo", "$topicId"]}}))

    def get_topic(self, topic_id: str) -> Dict[str, str] | None:
        """
        Retrieve topic information for a given topic ID.

        Args:
            topic_id: The ID of the topic to retrieve

        Returns:
            Dictionary containing topic title and content
        """
        topic = self.topics_collection.find_one({"_id": topic_id})
        if not topic:
            return None
        return {
            "topic_id": str(topic["_id"]),
            "topic_title": topic["title"],
            "topic_content": topic["topicTextContent"],
            "topic_category_id": topic["categoryId"],
            "topic_category_name": topic["categoryName"],
            "topic_created_at": topic["createdAt"],
            "topic_last_modified": topic["lastModified"],
            "topic_num_bookmarks": topic["numberOfBookmarks"],
            "topic_num_flags": topic["numberOfFlags"],
            "topic_num_likes": topic["numberOfLikes"],
            "topic_num_views": topic["numberOfViews"],
            "topic_summary": topic["summary"],
            "topic_tags": topic["tags"],
        }

    def build_reply_tree(self, parent_id: str) -> List[Dict]:
        """
        Recursively build a reply tree for a given post ID.
        """
        replies = list(self.posts_collection.find({"replyTo": parent_id}))
        for reply in replies:
            reply["replies"] = self.build_reply_tree(reply["_id"])
        return replies

    @staticmethod
    def generate_page_content(comment: dict, topic: dict) -> str:
        """
        Generate the page_content string by prepending the topic and appending replies in hierarchical order.
        """
        topic_title = topic["topic_title"]
        topic_content = topic["topic_content"]

        def format_replies(replies: List[Dict], depth: int = 0) -> str:
            formatted = ""
            for reply in replies:
                indent = "  " * depth
                formatted += f"{indent}<reply> {reply['postContent']}\n"
                formatted += format_replies(reply.get("replies", []), depth + 1)
                formatted += f"{indent}</reply>\n"
            return formatted

        comment_replies = comment.get("replies", [])
        if comment_replies:
            replies_formatted = format_replies(comment_replies)
            return f"## Topic: {topic_title}\n\n{topic_content}\n\n## Post:\n\n{comment['postContent']}\n\n## Replies:\n\n{replies_formatted}"
        else:
            return f"## Topic: {topic_title}\n\n{topic_content}\n\n## Post:\n\n{comment['postContent']}"

    def transform_db(self) -> pd.DataFrame:
        """
        Process all comments, build reply trees, and generate structured content.

        Returns:
            pd.DataFrame: DataFrame containing the transformed data with required columns
        """
        transformed_documents = []
        for comment in self.get_comments():
            # Skip comments that don't have a creator or topic
            creator = comment.get("creator")
            comment_id = comment["_id"]
            if not creator or not creator.get("name"):
                print(f"Comment {comment_id} has no creator, skipping...")
                continue

            topic = self.get_topic(comment["topicId"])
            if not topic:
                print(f"Topic for comment {comment_id} not found, skipping...")
                continue

            comment["replies"] = self.build_reply_tree(comment_id)
            transformed_doc = self.add_required_columns(
                columns={
                    "created_at": comment["createdAt"],
                    "last_modified": comment["lastModified"],
                    "number_of_likes": comment["numberOfLikes"],
                    "number_of_views": comment["numberOfViews"],
                    "creator_id": creator["id"],
                    "creator_name": creator["name"],
                    "creator_is_admin": creator["admin"],
                    "creator_is_moderator": creator["moderator"],
                    "creator_is_star": creator["star"],
                    "creator_is_leader": creator["leader"],
                    "creator_about": str(creator["about"]),
                    **topic,
                },
                page_content=self.generate_page_content(comment, topic),
                file_path="db_mongodb_pepwave",
                doc_id=comment_id,
            )
            transformed_documents.append(transformed_doc)

        df = self.make_df(transformed_documents)
        set_string_columns(
            df,
            ["creator_about", "creator_name", "topic_content", "topic_summary", "topic_title", "topic_category_name"],
        )
        return df


if __name__ == "__main__":
    transform = MongoPepwaveTransform()
    transform.transform()
