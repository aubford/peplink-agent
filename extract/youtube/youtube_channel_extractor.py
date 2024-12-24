from extract.youtube.youtube_base_extractor import YouTubeBaseExtractor


class YouTubeChannelExtractor(YouTubeBaseExtractor):
    def __init__(self, username: str):
        """
        Initialize YouTubeExtractor for a specific channel.

        Args:
            username: YouTube channel username (e.g. "@channelname")
        """
        super().__init__(username)
        self.username = username

    def _get_channel_id(self) -> str:
        """Get channel ID from username."""
        request = self.youtube_client.search().list(part="snippet", q=self.username, type="channel", maxResults=10)
        response = request.execute()

        if response["items"]:
            return response["items"][0]["snippet"]["channelId"]
        else:
            raise FileNotFoundError(f"Channel {self.username} not found.  Response: {response}")

    def get_uploads_playlist_id(self) -> str:
        """Get uploads playlist ID for the channel."""
        channel_id = self._get_channel_id()

        request = self.youtube_client.channels().list(part="contentDetails", id=channel_id)
        response = request.execute()
        uploads_playlist_id = response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
        if not uploads_playlist_id:
            raise FileNotFoundError(f"Could not find uploads playlist for {self.username} with channel ID {channel_id}")
        return uploads_playlist_id

    def extract(self) -> None:
        playlist_id = self.get_uploads_playlist_id()
        self.fetch_videos_for_playlist(playlist_id)
