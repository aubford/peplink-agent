import json
import pandas as pd
from pathlib import Path
from transform.base_transform import BaseTransform
from util.util_main import set_string_columns


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
                post = self.add_required_columns(
                    columns={
                        # Post metadata
                        "subreddit": meta["post_subreddit"],
                        "category": meta["post_category"],
                        "title": meta["post_title"],
                        "score": meta["post_score"],
                        "url": meta["post_url"],
                        # Author metadata
                        "author_name": author["name"],
                        "author_id": author.get("id", None),
                        "author_is_employee": author.get("is_employee", None),
                        "author_is_mod": author.get("is_mod", None),
                        "author_is_gold": author.get("is_gold", None),
                        "author_verified": author.get("verified", None),
                        "author_has_verified_email": author.get("has_verified_email", None),
                        "author_hide_from_robots": author.get("hide_from_robots", None),
                        "author_is_blocked": author.get("is_blocked", None),
                        "author_accept_followers": author.get("accept_followers", None),
                        "author_has_subscribed": author.get("has_subscribed", None),
                        # Author karma
                        "author_total_karma": author.get("total_karma", None),
                        "author_awardee_karma": author.get("awardee_karma", None),
                        "author_awarder_karma": author.get("awarder_karma", None),
                        "author_link_karma": author.get("link_karma", None),
                        "author_comment_karma": author.get("comment_karma", None),
                    },
                    page_content=data["page_content"],
                    file_path=file_path,
                    doc_id=data["metadata"]["post_id"],
                )

                posts.append(post)
        df = self.make_df(posts)

        # validate score is integer and not NaN or None
        df["score"] = pd.to_numeric(df["score"], errors="raise").astype("int64")

        set_string_columns(
            df, ["subreddit", "category", "title", "url", "author_name"], False
        )

        # Count rows with None in any author_ column
        author_columns = [col for col in df.columns if col.startswith("author_")]
        rows_without_author_count = df[author_columns].isna().any(axis=1).sum()
        print(
            f"{file_path}: Found {rows_without_author_count} rows with None values in author columns"
        )

        return df


if __name__ == "__main__":
    transformer = RedditTransform()
    transformer.transform()
