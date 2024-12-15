from googleapiclient.discovery import build
from langchain_community.document_loaders.youtube import YoutubeLoader
from extract.youtube.VideoItem import VideoItem
from extract.base_extractor import BaseExtractor
from config import ConfigType
from util.util import serialize_document

class YouTubeExtractor(BaseExtractor):
    def __init__(self, username: str, config: ConfigType):
        """
        Initialize YouTubeExtractor for a specific channel.

        Args:
            username: YouTube channel username (e.g. "@channelname")
        """
        source_name = "youtube"
        super().__init__(source_name)
        self.username = username
        self.set_logger(f"{source_name}_{username}")
        self.youtube_client = build(
            "youtube", "v3", developerKey=config.get("YOUTUBE_API_KEY"))
        self.channel_id = None
        self.uploads_playlist_id = None

    def _get_channel_id(self) -> None:
        """Get channel ID from username."""
        request = self.youtube_client.search().list(
            part='snippet',
            q=self.username,
            type='channel',
            maxResults=10
        )
        response = request.execute()

        if response['items']:
            self.channel_id = response['items'][0]['snippet']['channelId']
        else:
            raise FileNotFoundError(f"Channel {self.username} not found.  Response: {response}")

    def get_uploads_playlist_id(self) -> str:
        """Get uploads playlist ID for the channel."""
        self._get_channel_id()

        request = self.youtube_client.channels().list(
            part="contentDetails",
            id=self.channel_id
        )
        response = request.execute()
        uploads_playlist_id = response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
        if not uploads_playlist_id:
            raise FileNotFoundError(
                f"Could not find uploads playlist for {self.username} with channel ID {self.channel_id}")
        self.uploads_playlist_id = uploads_playlist_id
        return self.uploads_playlist_id

    def fetch_videos_with_transcripts(self) -> None:
        """
        Fetch all public videos with their transcripts from the channel and stream them to files.
        Each video is processed and written immediately.
        """
        if not self.uploads_playlist_id:
            raise SystemError(
                f"Could not find uploads playlist for {self.username}")

        video_item_stream = self.start_stream(
            VideoItem, identifier=self.username)

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
                            self.logger.error(
                                f"Could not load transcript for video {video_id}: {str(e)[:300]}")
                            continue

                        video_item["transcript"] = serialize_document(docs[0]) if docs else None
                        self.stream_item(video_item, video_item_stream)

            next_page_token = playlist_response.get("nextPageToken")
            if not next_page_token:
                break

        self.end_all_streams()
