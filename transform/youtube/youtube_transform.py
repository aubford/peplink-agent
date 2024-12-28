import json
import pandas as pd
from pathlib import Path
from transform.base_transform import BaseTransform
from datetime import datetime


class YouTubeTransform(BaseTransform):
    """Transform YouTube video data from JSONL files into a structured DataFrame."""

    def __init__(self):
        super().__init__("youtube")

    def transform_file(self, file_path: Path) -> pd.DataFrame:
        """Transform YouTube video data from a JSONL file.

        Args:
            file_path: Path to the JSONL file containing YouTube video data

        Returns:
            DataFrame containing transformed YouTube video data
        """
        videos = []

        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                metadata = data["metadata"]
                snippet = metadata["snippet"]
                content_details = metadata["contentDetails"]
                statistics = metadata["statistics"]

                video = self.add_required_columns(
                    columns={
                        # Snippet information
                        "date": datetime.strptime(
                            snippet["publishedAt"], "%Y-%m-%dT%H:%M:%SZ"
                        ).date(),
                        "channel_id": snippet["channelId"],
                        "title": snippet["title"],
                        "description": snippet["description"],
                        "channel_title": snippet["channelTitle"],
                        # Content details
                        "duration": content_details["duration"],
                        # Statistics (favorite_count not included because it only has zeroes)
                        "view_count": statistics["viewCount"],
                        "like_count": statistics["likeCount"],
                        "comment_count": statistics["commentCount"],
                        "word_count": len(data["page_content"].split()),
                    },
                    page_content=data["page_content"],
                    file_path=file_path,
                    doc_id=metadata["id"],
                )
                videos.append(video)

        df = pd.DataFrame(videos)
        df["duration"] = pd.to_timedelta(df["duration"])
        df = df[df["duration"] >= pd.Timedelta(minutes=5)].reset_index(drop=True)
        df["duration"] = df["duration"].dt.total_seconds()
        df["view_count"] = (
            pd.to_numeric(df["view_count"], errors="coerce").fillna(0).astype("Int64")
        )
        df["like_count"] = (
            pd.to_numeric(df["like_count"], errors="coerce").fillna(0).astype("Int64")
        )
        df["comment_count"] = (
            pd.to_numeric(df["comment_count"], errors="coerce")
            .fillna(0)
            .astype("Int64")
        )

        return df

if __name__ == "__main__":
    transformer = YouTubeTransform()
    transformer.transform()
