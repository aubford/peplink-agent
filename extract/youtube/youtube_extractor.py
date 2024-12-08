from typing import List, Optional
from googleapiclient.discovery import build
from langchain_community.document_loaders.youtube import YoutubeLoader
from extract.youtube.VideoItem import VideoItem
from config.config import config
from extract.base_extractor import BaseExtractor

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

    def fetch_videos_with_transcripts(self) -> List[dict]:
        """
        Fetch all public videos with their transcripts from the channel.

        Returns:
            List of video items with transcripts
        """
        if not self.uploads_playlist_id:
            return []

        videos_with_transcripts = []
        next_page_token = None

        while True:
            playlist_request = self.youtube_client.playlistItems().list(
                part="contentDetails",
                playlistId=self.uploads_playlist_id,
                maxResults=50,
                pageToken=next_page_token,
            )
            playlist_response = playlist_request.execute()

            for item in playlist_response["items"]:
                video_id = item["contentDetails"]["videoId"]
                video_request = self.youtube_client.videos().list(
                    part="status,contentDetails,snippet,statistics,topicDetails,localizations,player,recordingDetails",
                    id=video_id
                )
                video_response = video_request.execute()

                if video_response["items"]:
                    video_item = video_response["items"][0]
                    is_public = video_item["status"]["privacyStatus"] == "public"

                    if is_public:
                        try:
                            loader = YoutubeLoader(video_id=video_id)
                            docs = loader.load()
                            video_item["transcript"] = docs[0].page_content
                            video = VideoItem.model_validate(video_item)
                            videos_with_transcripts.append(video.model_dump())
                        except Exception as e:
                            print(f"Could not load transcript for video {video_id}: {str(e)}")
                            continue

            next_page_token = playlist_response.get("nextPageToken")
            if not next_page_token:
                break

        return videos_with_transcripts

    def save_videos(self, videos: List[dict]) -> None:
        """
        Save videos to JSON and Parquet files in the data directories.

        Args:
            videos: List of video items
        """
        identifier = self.username.replace('@', '')
        self.save_data(videos,identifier)

def main():
    """Example usage of YouTubeExtractor."""
    extractor = YouTubeExtractor("@peplink")
    videos = extractor.fetch_videos_with_transcripts()
    extractor.save_videos(videos)  # Will save to data/youtube by default

if __name__ == "__main__":
    main()
