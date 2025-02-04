#!/usr/bin/env python
# coding: utf-8

# # Reddit
# 
# >[Reddit](https://www.reddit.com) is an American social news aggregation, content rating, and discussion website.
# 
# 
# This loader fetches the text from the Posts of Subreddits or Reddit users, using the `praw` Python package.
# 
# Make a [Reddit Application](https://www.reddit.com/prefs/apps/) and initialize the loader with with your Reddit API credentials.

# In[1]:


from langchain_community.document_loaders import RedditPostsLoader


# In[2]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  praw')


# In[5]:


# load using 'subreddit' mode
loader = RedditPostsLoader(
    client_id="YOUR CLIENT ID",
    client_secret="YOUR CLIENT SECRET",
    user_agent="extractor by u/Master_Ocelot8179",
    categories=["new", "hot"],  # List of categories to load posts from
    mode="subreddit",
    search_queries=[
        "investing",
        "wallstreetbets",
    ],  # List of subreddits to load posts from
    number_posts=20,  # Default value is 10
)

# # or load using 'username' mode
# loader = RedditPostsLoader(
#     client_id="YOUR CLIENT ID",
#     client_secret="YOUR CLIENT SECRET",
#     user_agent="extractor by u/Master_Ocelot8179",
#     categories=['new', 'hot'],
#     mode = 'username',
#     search_queries=['ga3far', 'Master_Ocelot8179'],         # List of usernames to load posts from
#     number_posts=20
#     )

# Note: Categories can be only of following value - "controversial" "hot" "new" "rising" "top"


# In[6]:


documents = loader.load()
documents[:5]

