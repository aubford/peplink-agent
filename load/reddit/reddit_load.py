# %%
from load.base_load import BaseLoad


class RedditLoad(BaseLoad):
    folder_name = "reddit"

    def __init__(self):
        super().__init__()
