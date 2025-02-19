from typing import List
from load.base_load import BaseLoad
import pandas as pd


class HtmlLoad(BaseLoad):
    folder_name = "html"

    def __init__(self):
        super().__init__()

    def create_merged_df(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        df = pd.concat(dfs)
        # Add section as header in page_content.  Do this here instead of transform so we can experiment.
        mask = df["section"].str.strip().astype(bool)
        df.loc[mask, "page_content"] = (
            df.loc[mask, "section"] + " \n " + df.loc[mask, "page_content"]
        )
        df["images"] = df["images"].apply(list)
        return df
