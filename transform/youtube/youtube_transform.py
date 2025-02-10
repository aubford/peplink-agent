import json
import pandas as pd
from pathlib import Path
from transform.base_transform import BaseTransform, SubjectMatter
from datetime import datetime
from util.util_main import get_column_word_count, set_string_columns


class YouTubeTransform(BaseTransform):
    """Transform YouTube video data from JSONL files into a structured DataFrame."""

    folder_name = "youtube"
    # All videos should most-likely be related to Pepwave since we filter for the word "pep" in the page content (obviously could be improved)
    subject_matter = SubjectMatter.PEPWAVE

    def __init__(self):
        super().__init__()

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
                # Skip videos with empty page content
                if not data.get("page_content"):
                    continue

                metadata = data["metadata"]
                snippet = metadata["snippet"]
                content_details = metadata["contentDetails"]
                statistics = metadata["statistics"]

                video = self.add_required_columns(
                    columns={
                        # Snippet information
                        "date": datetime.strptime(snippet["publishedAt"], "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d"),
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
                    },
                    page_content=data["page_content"],
                    file_path=file_path,
                    doc_id=metadata["id"],
                )
                # Skip if video with same ID already exists
                if any(v["id"] == video["id"] for v in videos):
                    continue
                videos.append(video)
        df = self.make_df(videos)

        set_string_columns(df, ["description"])
        set_string_columns(df, ["title", "date", "channel_title", "duration", "channel_id"], False)

        df["word_count"] = get_column_word_count(df, "page_content")

        # filter out videos less than 3 minutes and word count less than 300
        df["duration"] = pd.to_timedelta(df["duration"])
        df = df[(df["duration"] >= pd.Timedelta(minutes=3)) & (df["word_count"] >= 300)]
        self.notify_dropped_rows(df, ">3 min and >300 words")

        # Filter for "pep" content unless from allowed sources
        df = self._filter_for_pep(df, file_path)

        # set duration to final state for persistence
        df["duration"] = df["duration"].dt.total_seconds().astype("int64")

        # clean count data
        df["view_count"] = pd.to_numeric(df["view_count"], errors="coerce").fillna(0).astype("int64")
        df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0).astype("int64")
        df["comment_count"] = pd.to_numeric(df["comment_count"], errors="coerce").fillna(0).astype("int64")

        return df

    def _filter_for_pep(self, df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
        sources_to_filter = [
            "Frontierus",
            "MobileInternetResourceCenter",
            "Technorv",
            "MobileMustHave",
            "5Gstore",
        ]
        file_name = str(file_path).lower()
        if any(source.lower() in file_name for source in sources_to_filter):
            df = df[df["page_content"].str.lower().str.contains("pep")]
            self.notify_dropped_rows(df, "contains 'pep'")

        return df


if __name__ == "__main__":
    transformer = YouTubeTransform()
    transformer.transform()
