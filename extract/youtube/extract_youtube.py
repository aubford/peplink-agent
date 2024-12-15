# %%
from extract.extractor_manager import ExtractorManager
from extract.youtube.youtube_extractor import YouTubeExtractor


class YouTubeExtractorManager(ExtractorManager):
    def __init__(self, channels):
        super().__init__([
            YouTubeExtractor(channel)
            for channel in channels
        ])


# %%

fg_store = YouTubeExtractor('@5Gstore')
fg_store.get_uploads_playlist_id()
# fg_store.fetch_videos_with_transcripts()

# %%

manager = YouTubeExtractorManager(['@MobileInternetResourceCenter', '@Frontierus', '@MobileMustHave', '@Technorv'])
manager.fetch_all()

# %%
print(manager.extractors)
manager.fetch_all()
