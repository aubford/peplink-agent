#!/usr/bin/env python
# coding: utf-8

# # BiliBili
#
# >[Bilibili](https://www.bilibili.com/) is one of the most beloved long-form video sites in China.
#
#
# This loader leverages the [bilibili-api](https://github.com/Nemo2011/bilibili-api) to retrieve text transcripts from `Bilibili` videos. To effectively use this loader, it's essential to have the `sessdata`, `bili_jct`, and `buvid3` cookie parameters. These can be obtained by logging into [Bilibili](https://www.bilibili.com/), then extracting the values of `sessdata`, `bili_jct`, and `buvid3` from the browser's developer tools.
#
# If you choose to leave the cookie parameters blank, the Loader will still function, but it will only retrieve video information for the metadata and will not be able to fetch transcripts.
#
# For detailed instructions on obtaining these credentials, refer to the guide [here](https://nemo2011.github.io/bilibili-api/#/get-credential).
#
# The BiliBiliLoader provides a user-friendly interface for easily accessing transcripts of desired video content on Bilibili, making it an invaluable tool for those looking to analyze or utilize this media data.

# In[24]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  bilibili-api-python")


# In[ ]:


from langchain_community.document_loaders import BiliBiliLoader


# In[ ]:


SESSDATA = "<your sessdata>"
BUVID3 = "<your buvids>"
BILI_JCT = "<your bili_jct>"


# In[18]:


loader = BiliBiliLoader(
    [
        "https://www.bilibili.com/video/BV1g84y1R7oE/",
    ],
    sessdata=SESSDATA,
    bili_jct=BILI_JCT,
    buvid3=BUVID3,
)


# In[22]:


docs = loader.load()


# In[23]:


docs
