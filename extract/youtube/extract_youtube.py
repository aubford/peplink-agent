# %%
from extract.extractor_manager import ExtractorManager
from extract.youtube.youtube_channel_extractor import YouTubeChannelExtractor
from extract.youtube.youtube_videos_extractor import YouTubePlaylistExtractor, YouTubeVideosExtractor


class YouTubeExtractorManager(ExtractorManager):
    def __init__(self, channels):
        super().__init__([YouTubeChannelExtractor(channel) for channel in channels])


# %%

# ['@NetworkDirection', '@MobileInternetResourceCenter', '@Frontierus', '@MobileMustHave', '@Technorv', '@Peplink', '@5Gstore']
manager = YouTubeExtractorManager(['@NetworkDirection', '@WestNetworksLLC'])
manager.fetch_all()

# %%

extractors = YouTubeVideosExtractor("extra_videos",
                                    video_ids=['0PbTi_Prpgs', '_IOZ8_cPgu8', 'oHQvWa6J8dU', 'k9ZigsW9il0',
                                               '0j6-QFnnwQk'])
extractors.extract()

# %%

extractors = YouTubeVideosExtractor("peplink_training_videos",
                                    video_ids=['mZAr7Z7eL48', 'GLtjyS4ELAA', 'Ny5z_4Pjz6c', 'Ow8sdUEb_eg',
                                               'KiFrxH46qM0', 'J1Jcgce7zrQ', 'iNwVqhp2QtY', 'GLtjyS4ELAA',
                                               'fsB5MqE7uOU', '1vvm0JiEwww', '-ILspN9YRsY'])
extractors.extract()

# %%

extractors = YouTubePlaylistExtractor("west_peplink_university_playlist",
                                      playlist_id='PLT8XvvJf-9vgah5id_2tW6GvSCOmV62h6')
extractors.extract()

# %%
