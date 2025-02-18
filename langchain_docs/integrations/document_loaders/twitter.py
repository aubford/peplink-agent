#!/usr/bin/env python
# coding: utf-8

# # Twitter
#
# >[Twitter](https://twitter.com/) is an online social media and social networking service.
#
# This loader fetches the text from the Tweets of a list of `Twitter` users, using the `tweepy` Python package.
# You must initialize the loader with your `Twitter API` token, and you need to pass in the Twitter username you want to extract.

# In[1]:


from langchain_community.document_loaders import TwitterTweetLoader


# In[2]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  tweepy")


# In[3]:


loader = TwitterTweetLoader.from_bearer_token(
    oauth2_bearer_token="YOUR BEARER TOKEN",
    twitter_users=["elonmusk"],
    number_tweets=50,  # Default value is 100
)

# Or load from access token and consumer keys
# loader = TwitterTweetLoader.from_secrets(
#     access_token='YOUR ACCESS TOKEN',
#     access_token_secret='YOUR ACCESS TOKEN SECRET',
#     consumer_key='YOUR CONSUMER KEY',
#     consumer_secret='YOUR CONSUMER SECRET',
#     twitter_users=['elonmusk'],
#     number_tweets=50,
# )


# In[4]:


documents = loader.load()
documents[:5]
