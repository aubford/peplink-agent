# %%
from load.base_load import BaseLoad


class RedditGeneralLoad(BaseLoad):
    folder_name = "reddit_general"

    def __init__(self):
        super().__init__()
