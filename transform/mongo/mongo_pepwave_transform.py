from __future__ import annotations
from transform.base_transform import BaseMongoTransform, SubjectMatter
import pandas as pd
from util.util_main import set_string_columns
from util.nlp import nltk_tokenize
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import nltk
from pathlib import Path
from matplotlib.ticker import MultipleLocator


tokenizer = AutoTokenizer.from_pretrained(
    "shahrukhx01/question-vs-statement-classifier"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "shahrukhx01/question-vs-statement-classifier"
)
question_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)


class MongoPepwaveTransform(BaseMongoTransform):
    """Transform for processing MongoDB Pepwave forum data. We don"t need to filter this content as strictly as Reddit due to the quality of the community."""

    folder_name: str = "mongo"
    subject_matter: SubjectMatter = SubjectMatter.PEPWAVE

    def __init__(self):
        super().__init__("mongodb://localhost:27017", "pepwave")
        self.posts_collection = self.db["posts"]
        self.topics_collection = self.db["topics"]

    def _get_comments(self) -> list[dict]:
        """
        Retrieve posts that are direct replies to topics (same as comments in Reddit)
        """
        return list(
            self.posts_collection.find(
                {
                    "$or": [
                        {"$expr": {"$eq": ["$replyTo", "$topicId"]}},
                        {"replyTo": None},
                    ]
                }
            )
        )

    def _get_topic(self, topic_id: str) -> dict | None:
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
            "post_id": str(topic["_id"]),
            "post_title": topic["title"],
            "post_content": topic["topicTextContent"],
            "post_category_id": topic["categoryId"],
            "post_category_name": topic["categoryName"],
            "post_created_at": topic["createdAt"],
            "post_last_modified": topic["lastModified"],
            "post_num_bookmarks": topic.get("numberOfBookmarks", 0),
            "post_num_flags": topic.get("numberOfFlags", 0),
            "post_num_likes": topic.get("numberOfLikes", 0),
            "post_num_views": topic.get("numberOfViews", 0),
            "post_tags": [tag["name"] for tag in topic.get("tags", [])],
        }

    def build_reply_tree(self, parent_id: str) -> list[dict]:
        """
        Recursively build a reply tree for a given post ID.
        """
        replies = list(self.posts_collection.find({"replyTo": parent_id}))
        for reply in replies:
            reply["replies"] = self.build_reply_tree(reply["_id"])
        return replies

    @staticmethod
    def _generate_page_content(comment: dict, topic: dict) -> str:
        """
        Generate the page_content string by prepending the topic and appending replies in hierarchical order.
        """
        post_title = topic["post_title"]
        post_content = topic["post_content"]
        post_tags = topic["post_tags"]

        def format_replies(replies: list[dict], depth: int = 0) -> str:
            formatted = ""
            for reply in replies:
                indent = "  " * depth
                formatted += f"{indent}<reply>\n{reply['postContent']}\n"
                formatted += format_replies(reply.get("replies", []), depth + 1)
                formatted += f"{indent}</reply>\n"
            return formatted

        comment_replies = comment.get("replies", [])
        tags = f"\n\n## Tags: {", ".join(post_tags)}" if post_tags else ""
        if comment_replies:
            replies_formatted = format_replies(comment_replies)
            return f"## Post\n\n ### Title: {post_title}\n\n ### Content:\n\n{post_content}\n\n ## Comments:\n\n<comment> {comment['postContent']} {replies_formatted}</comment>\n\n{tags}"
        else:
            return f"## Post\n\n ### Title: {post_title}\n\n ### Content:\n\n{post_content}\n\n ## Comments:\n\n<comment> {comment['postContent']} </comment>{tags}"

    def _total_replies_and_likes(
        self, replies_tree: list[dict]
    ) -> tuple[int, int, int]:
        """
        Recursively count total number of replies and likes in a reply tree.

        Returns:
            tuple[int, int]: (total_replies, total_likes)
        """
        total_replies_token_count = 0
        total_replies = len(replies_tree)
        total_likes = 0

        for reply in replies_tree:
            total_replies_token_count += len(nltk_tokenize(reply["postContent"]))
            total_likes += reply.get("numberOfLikes", 0)
            sub_replies, sub_likes, sub_replies_token_count = (
                self._total_replies_and_likes(reply.get("replies", []))
            )
            total_replies += sub_replies
            total_likes += sub_likes
            total_replies_token_count += sub_replies_token_count

        return total_replies, total_likes, total_replies_token_count

    @staticmethod
    def _plot_score_distribution(scores: np.ndarray, threshold: float) -> None:
        """
        Args:
            scores: Array of scores for all comments
            threshold: The cutoff threshold below which comments are filtered out
        """
        plt.figure(figsize=(10, 6))

        # Create histogram
        plt.hist(scores, bins=50, edgecolor="black", alpha=0.7)

        # Add threshold line
        plt.axvline(
            x=threshold, color="r", linestyle="--", label=f"Threshold ({threshold:.2f})"
        )

        # Add mean line
        mean_score = float(np.mean(scores))
        plt.axvline(
            x=mean_score, color="g", linestyle="--", label=f"Mean ({mean_score:.2f})"
        )
        # Calculate percentage of comments that will be filtered
        percent_filtered = (scores < threshold).mean() * 100

        plt.title(
            f"Comment Score Distribution\n{percent_filtered:.1f}% of comments below threshold"
        )
        plt.xlabel("Score")
        plt.ylabel("Number of Comments")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.show()

    @staticmethod
    def _print_features(features: np.ndarray, name: str):
        # Debug: Show first 30 rows
        print(
            f"------------------------ {name} FEATURES -------------------------------------"
        )

        # After scaling, verify the transformation
        print("\nFeature statistics:")
        for i in range(features.shape[1]):
            scaled_vals = features[:, i]
            print(f"Feature {i}:")
            print(f"  Mean: {np.mean(scaled_vals):.2f}")
            print(f"  Median: {np.median(scaled_vals):.2f}")
            print(f"  Range: {np.min(scaled_vals):.2f} to {np.max(scaled_vals):.2f}")
        print("\n")

    @staticmethod
    def _plot_distribution(
        values: np.ndarray, title: str, xlabel: str, upper_limit=None
    ) -> None:
        """
        Args:
            values: Array of values to plot
            title: Plot title
            xlabel: X-axis label
        """
        bins = 100
        max_value = np.max(values)
        mean = np.mean(values)
        std = np.std(values)

        # Create a single plot
        plt.figure(figsize=(15, 8))

        # Set range from 0 to 3 standard deviations (with minimum of 50)
        zoom_upper_limit = upper_limit if upper_limit else max(mean + 3 * std, 50)

        plt.hist(
            values,
            bins=bins,
            edgecolor="black",
            alpha=0.7,
            range=(-20.0, float(zoom_upper_limit)),
        )
        plt.axvline(
            x=float(mean), color="g", linestyle="--", label=f"Mean ({mean:.2f})"
        )
        plt.axvline(x=float(mean + std), color="r", linestyle="--", label=f"Mean ± Std")
        plt.axvline(x=float(mean - std), color="r", linestyle="--")

        plt.title(f"{title} (Max value: {max_value:.2f})")
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.gca().xaxis.set_major_locator(MultipleLocator(10))
        plt.gca().xaxis.set_minor_locator(MultipleLocator(2))

        plt.tight_layout()
        plt.show()

    def filter_transformed_docs(self, comments: list[dict]) -> list[dict]:
        """Filter outliers in a way that considers the following features collectively rather than
        applying independent thresholds. The objective is to remove comments that fall in
        the lower tail of the distribution based on a weighted scoring approach, ensuring
        a continuous measure of importance rather than a strict cutoff:

        - number_of_likes: Filter out low, importance weight: 10
        - total_reply_likes: Filter out low, importance weight: 6
        - post_num_likes: Filter out low, importance weight: 4
        - len(page_content): Filter out low, importance weight: 4
        - total_replies: Filter out low, importance weight: 4
        - post_num_views: Filter out low, importance weight: 2
        - creator_is_star: Keep if true, ignore this feature if false, importance weight: 1
        - creator_is_leader: Keep if true, ignore this feature if false, importance weight: 1

        Side note: We can't filter by grouping comments by topic like we can with Reddit because of the
        way that people use replies as comments in this forum. It's ambiguous whether a reply is an actual
        reply to a comment or to the topic because user typically use the reply button for the latest comment
        instead of replying to the original comment like they are supposed to, so we have to consider replies
        to comments to potentially be equally as valuable as the comment itself.
        """
        if not comments:
            return []

        # The topic is usually a question, so we care more about the comment + replies length.
        def pc_length(c: dict) -> int:
            post_content_length = len(nltk_tokenize(c["post_content"]))
            return len(nltk_tokenize(c["page_content"])) - post_content_length

        page_content_lengths = np.array([pc_length(c) for c in comments])
        # self._plot_distribution(
        #     page_content_lengths, "Page Content Length Distribution", "Length", 400
        # )

        # At a certain point, content getting longer provides diminishing returns.
        # This applies compression to the longer end of the distribution that asymptotically approaches the upper_limit.
        # The most important thing is that we have enough information and longer isn't always better.
        # Using mean as the threshold preserves more modal behavior for a balanced approach but maybe
        # it should just be fixed to e.g. 90 based on a guess of what "enough information" means?
        threshold = np.mean(page_content_lengths)
        upper_limit = np.std(page_content_lengths) * 1.2

        def compress_long_lengths(length_arr: np.ndarray) -> np.ndarray:
            lengths_to_compress = length_arr - threshold
            return threshold + (
                lengths_to_compress
                - (lengths_to_compress**2 / (upper_limit + lengths_to_compress))
            )

        page_content_lengths_transformed = np.where(
            page_content_lengths > threshold,
            compress_long_lengths(page_content_lengths),
            page_content_lengths,
        )

        # self._plot_distribution(
        #     page_content_lengths_transformed,
        #     "Transformed Page Content Length Distribution",
        #     "Length",
        #     400,
        # )

        features = {
            "number_of_likes": {
                "weight": 20,
                "values": np.array([c["number_of_likes"] for c in comments]),
            },
            "total_reply_likes": {
                "weight": 16,
                "values": np.array([c["total_reply_likes"] for c in comments]),
            },
            "page_content_length": {
                "weight": 22,
                "values": page_content_lengths_transformed,
            },
            "post_num_likes": {
                "weight": 5,
                "values": np.array([c["post_num_likes"] for c in comments]),
            },
            "post_num_views": {
                "weight": 3,
                "values": np.array([c["post_num_views"] for c in comments]),
            },
        }

        feature_matrix = np.column_stack(
            [feature["values"] for feature in features.values()]
        )
        # self._print_features(feature_matrix, "Original")

        # Use QuantileTransformer to transform to uniform distribution (0 to 1)
        scaler = QuantileTransformer(
            random_state=42,
            n_quantiles=min(len(comments), 1000),  # Adjust based on dataset size
            output_distribution="uniform",
        )
        scaled_features = scaler.fit_transform(feature_matrix)
        # self._print_features(scaled_features, "Scaled")

        # Calculate weighted scores
        weights = np.array([feature["weight"] for feature in features.values()])
        scores = np.zeros(len(comments))

        # Combine scaled features with weights
        for i in range(scaled_features.shape[1]):
            scores += scaled_features[:, i] * weights[i]

        # Add credential-based scores
        # Number of leaders who are not stars: 0
        # Number of stars who are not leaders: 7207
        # Total number of stars: 11638
        # Total number of moderators: 3949
        # Total number of leaders: 4431
        for i, comment in enumerate(comments):
            if comment["creator_is_star"] or comment["creator_is_moderator"]:
                scores[i] += 12
            if comment["creator_is_leader"] or comment["creator_is_moderator"]:
                # treat this as a bonus instead of an independent feature since all leaders a stars
                scores[i] += 6

        mean_score = np.mean(scores)
        std_score = np.std(scores)
        threshold = float(mean_score - std_score)

        self._plot_score_distribution(scores, threshold)

        keep_indices = np.where(scores >= threshold)[0]
        removed_indices = np.where(scores < threshold)[0]

        # Save removed comments as parquet file
        removed_comments = [comments[i] for i in removed_indices]
        print(f"Removed {len(removed_comments)} comments")
        removed_df = pd.DataFrame(removed_comments)
        removed_df.to_parquet(
            Path(__file__).parent / "removed_comments.parquet", index=False
        )

        return [comments[i] for i in keep_indices]

    @staticmethod
    def _contains_question(text: str) -> bool:
        """
        Check if a text contains a question using Huggingface model.
        """
        sentences = nltk.sent_tokenize(text)
        for sentence in sentences:
            # classifier seems to get confused by urls
            if "http" in sentence:
                continue
            result = question_classifier(sentence)
            if isinstance(result, list) and result[0].get("label") == "LABEL_1":
                return True
        return False

    def _skip_comment(
        self,
        comment: dict,
        topic: dict,
        total_replies_token_count: int,
    ) -> bool:
        """Rough filter on the frontend to remove low information data.
        Users typically use the reply button for the latest comment instead of replying to the original comment like
        they are supposed to, so we have to consider replies to comments to potentially be equally as valuable as the comment itself.
        """
        if topic["post_category_name"] in [
            "Feature Requests",
            "Beta Releases",
            "Announcements",
        ]:
            return True

        comment_token_count = len(nltk_tokenize(comment["postContent"]))
        total_token_count = comment_token_count + total_replies_token_count

        if total_token_count > 50:
            return False

        token_threshold = 17

        # If comment is a question, we will need another back and forth to get actual info usually.
        # Negate the question comment and scale the expected OP response to the question comment length,
        # then we still need another reply which is the original threshold.
        if self._contains_question(comment["postContent"]):
            token_threshold += comment_token_count + min(10, comment_token_count)

        creator = comment["creator"]
        has_cred = creator["star"] or creator["leader"]
        if has_cred or comment["numberOfLikes"] > 0:
            token_threshold = max(1, token_threshold / (comment["numberOfLikes"] + 2))
        return total_token_count < token_threshold

    def transform_db(self) -> pd.DataFrame:
        """
        Process all comments, build reply trees, and generate structured content.

        Returns:
            pd.DataFrame: DataFrame containing the transformed data with required columns
        """
        transformed_documents: list[dict] = []
        for comment in self._get_comments():
            # Skip comments that don't have a creator or topic
            creator = comment.get("creator")
            comment_id = comment["_id"]

            if not creator or not creator.get("name"):
                print(f"Comment {comment_id} has no creator, skipping...")
                continue

            topic = self._get_topic(comment["topicId"])
            if not topic:
                print(f"Topic for comment {comment_id} not found, skipping...")
                continue

            comment["replies"] = self.build_reply_tree(comment_id)
            total_replies, total_reply_likes, total_replies_token_count = (
                self._total_replies_and_likes(comment["replies"])
            )

            if self._skip_comment(comment, topic, total_replies_token_count):
                continue

            transformed_doc = self.add_required_columns(
                columns={
                    "created_at": comment["createdAt"],
                    "last_modified": comment["lastModified"],
                    "number_of_likes": comment.get("numberOfLikes", 0),
                    "number_of_views": comment.get("numberOfViews", 0),
                    "creator_id": creator["id"],
                    "creator_name": creator["name"],
                    "creator_is_admin": creator["admin"],
                    "creator_is_moderator": creator["moderator"],
                    "creator_is_star": creator["star"],
                    "creator_is_leader": creator["leader"],
                    "creator_about": str(creator["about"]),
                    "comment_content": comment["postContent"],
                    "total_replies": total_replies,
                    "total_reply_likes": total_reply_likes,
                    **topic,
                },
                page_content=self._generate_page_content(comment, topic),
                file_path="db_mongodb_pepwave",
                doc_id=comment_id,
            )
            transformed_documents.append(transformed_doc)

        transformed_documents = self.filter_transformed_docs(transformed_documents)
        df = self.make_df(transformed_documents)
        set_string_columns(
            df,
            [
                "creator_about",
                "creator_name",
                "post_content",
                "post_title",
                "post_category_name",
                "comment_content",
            ],
        )

        # Reorder columns to put specified columns first
        first_cols = ["page_content", "post_title", "post_content", "comment_content"]
        df = self.reorder_columns(df, first_cols)

        return df


if __name__ == "__main__":
    transform = MongoPepwaveTransform()
    transform.transform()
