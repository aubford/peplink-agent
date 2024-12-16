# %%
from extract.extractor_manager import ExtractorManager
from extract.youtube.youtube_channel_extractor import YouTubeChannelExtractor
from extract.youtube.youtube_videos_extractor import YouTubeVideosExtractor

class YouTubeExtractorManager(ExtractorManager):
    def __init__(self, channels):
        super().__init__([
            YouTubeChannelExtractor(channel)
            for channel in channels
        ])


# %%

# ['@NetworkDirection', '@MobileInternetResourceCenter', '@Frontierus', '@MobileMustHave', '@Technorv', '@Peplink', '@5Gstore']
manager = YouTubeExtractorManager(['@5Gstore', '@NetworkDirection'])
manager.fetch_all()

#%%


extractors = YouTubeVideosExtractor("extra_videos", video_ids=['0PbTi_Prpgs', '_IOZ8_cPgu8', 'oHQvWa6J8dU',])
extractors.extract()