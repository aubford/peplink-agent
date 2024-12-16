from typing import List
from extract.youtube.youtube_base_extractor import YouTubeBaseExtractor
from extract.youtube.VideoItem import VideoItem


class YouTubeVideosExtractor(YouTubeBaseExtractor):
    def __init__(self, file_id: str):
        """
        Initialize YouTubeExtractor for a specific channel.

        Args:
            username: YouTube channel username (e.g. "@channelname")
        """
        super().__init__(file_id)

    def extract(self, video_ids: List[str]) -> None:
        video_item_stream = self.start_stream(VideoItem, identifier=self.file_id)

        for video_id in video_ids:
            video = self.fetch_video(video_id)
            self.stream_item(video, video_item_stream)

        self.end_stream(video_item_stream)
