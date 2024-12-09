from typing import List, Optional
from googleapiclient.discovery import build
from langchain_community.document_loaders.youtube import YoutubeLoader
from extract.youtube.VideoItem import VideoItem
from config import config
from extract.base_extractor import BaseExtractor
import logging
from logging.handlers import RotatingFileHandler
import json
from pathlib import Path

# Configure rotating file handler
handler = RotatingFileHandler(
    'youtube_extractor.log',  # Log file name
    maxBytes=10*1024*1024,    # Maximum file size in bytes (10 MB)
    backupCount=0             # No backup files
)

# Set up logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Get the logger and set its level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

class YouTubeExtractor(BaseExtractor):
    def __init__(self, username: str):
        """
        Initialize YouTubeExtractor for a specific channel.

        Args:
            username: YouTube channel username (e.g. "@channelname")
        """
        super().__init__(source_name="youtube")
        self.username = username
        self.youtube_client = build("youtube", "v3", developerKey=config.get("YOUTUBE_API_KEY"))
        self.channel_id = self._get_channel_id()
        self.uploads_playlist_id = self._get_uploads_playlist_id()
        self.logger = logger

    def _get_channel_id(self) -> Optional[str]:
        """Get channel ID from username."""
        request = self.youtube_client.search().list(
            part='snippet',
            q=self.username,
            type='channel',
            maxResults=1
        )
        response = request.execute()

        if response['items']:
            return response['items'][0]['snippet']['channelId']
        return None

    def _get_uploads_playlist_id(self) -> Optional[str]:
        """Get uploads playlist ID for the channel."""
        if not self.channel_id:
            return None

        request = self.youtube_client.channels().list(
            part="contentDetails",
            id=self.channel_id
        )
        response = request.execute()
        return response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

    def fetch_videos_with_transcripts(self) -> None:
        """
        Fetch all public videos with their transcripts from the channel and stream them to files.
        Each video is processed and written immediately.
        """
        if not self.uploads_playlist_id:
            return

        video_item_stream = self.start_stream(VideoItem)

        next_page_token = None
        while True:
            playlist_request = self.youtube_client.playlistItems().list(
                part="contentDetails",
                playlistId=self.uploads_playlist_id,
                maxResults=50,
                pageToken=next_page_token,
            )
            playlist_response = playlist_request.execute()
            self.logger.debug(f"Playlist response: {playlist_response}")

            for item in playlist_response["items"]:
                video_id = item["contentDetails"]["videoId"]
                video_request = self.youtube_client.videos().list(
                    part="status,contentDetails,snippet,statistics,topicDetails,localizations,player,recordingDetails",
                    id=video_id
                )
                video_response = video_request.execute()
                self.logger.debug(f"Video response for {video_id}: {video_response}")

                if video_response["items"]:
                    video_item = video_response["items"][0]
                    is_public = video_item["status"]["privacyStatus"] == "public"

                    if is_public:
                        try:
                            loader = YoutubeLoader(video_id=video_id)
                            docs = loader.load()
                        except Exception as e:
                            print(f"Could not load transcript for video {video_id}")
                            self.logger.error(
                                f"Could not load transcript for video {video_id}: {e}")
                            continue

                        video_item["transcript"] = docs[0].page_content
                        self.stream_item(video_item, video_item_stream)

            next_page_token = playlist_response.get("nextPageToken")
            if not next_page_token:
                break

        self.end_all_streams()


def main():
    """Example usage of YouTubeExtractor."""
    extractor = YouTubeExtractor("@peplink")
    extractor.fetch_videos_with_transcripts()

if __name__ == "__main__":
    main()
