# %%
from extract.extractor_manager import ExtractorManager
from extract.youtube.youtube_channel_extractor import YouTubeChannelExtractor


class YouTubeExtractorManager(ExtractorManager):
    def __init__(self, channels):
        super().__init__([
            YouTubeChannelExtractor(channel)
            for channel in channels
        ])


# %%

# ['@MobileInternetResourceCenter', '@Frontierus', '@MobileMustHave', '@Technorv', '@Peplink', '@5Gstore']
manager = YouTubeExtractorManager(['@5Gstore'])
manager.fetch_all()

# %%
print(manager.extractors)
manager.fetch_all()
