from googleapiclient.discovery import build
from langchain_community.document_loaders.youtube import YoutubeLoader
from extract.base_extractor import BaseExtractor
from extract.youtube.VideoItem import VideoItem
from util.util import serialize_document, empty_document_dict
from abc import ABC


class YouTubeBaseExtractor(BaseExtractor, ABC):
    source_name = "youtube"

    def __init__(self, file_id: str):
        """
        Initialize base YouTube extractor with common functionality.
        Args:
            file_id: id for end of the filename
        """
        super().__init__()
        self.youtube_client = build(
            "youtube", "v3", developerKey=self.config.get("YOUTUBE_API_KEY")
        )
        self.file_id = file_id
        self.set_logger(f"{self.source_name}_{file_id}")

    def fetch_video(self, video_id: str) -> dict:
        """
        Fetch video details and transcript for a given video ID.

        Args:
            video_id: YouTube video ID

        Returns:
            Dictionary containing video metadata and transcript
        """
        video_request = self.youtube_client.videos().list(
            part="status,contentDetails,snippet,statistics,topicDetails,localizations,player,recordingDetails",
            id=video_id,
        )
        video_response = video_request.execute()
        self.logger.debug(f"Video response for {video_id}: {video_response}")

        if video_response["items"]:
            video_item = video_response["items"][0]
            # is_public = video_item["status"]["privacyStatus"] == "public"

            # if is_public:
            try:
                loader = YoutubeLoader(video_id=video_id)
                docs = loader.load()
            except Exception as e:
                self.logger.error(
                    f"Could not load transcript for video {video_id}: {str(e)[:300]}"
                )
                return empty_document_dict(video_item)

            serialized = (
                serialize_document(docs[0]) if docs else empty_document_dict(video_item)
            )
            serialized["metadata"] = video_item
            return serialized

    def fetch_videos_for_playlist(self, playlist_id: str) -> None:
        video_item_stream = self.start_stream(VideoItem, identifier=self.file_id)
        next_page_token = None
        while True:
            playlist_request = self.youtube_client.playlistItems().list(
                part="contentDetails",
                playlistId=playlist_id,
                maxResults=50,
                pageToken=next_page_token,
            )
            playlist_response = playlist_request.execute()
            self.logger.debug(f"Playlist response: {playlist_response}")

            for item in playlist_response["items"]:
                video_id = item["contentDetails"]["videoId"]
                video = self.fetch_video(video_id)
                self.stream_item(video, video_item_stream)

            next_page_token = playlist_response.get("nextPageToken")
            if not next_page_token:
                break

        self.end_stream(video_item_stream)
