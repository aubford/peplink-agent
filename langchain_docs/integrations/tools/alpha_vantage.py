#!/usr/bin/env python
# coding: utf-8

# # Alpha Vantage
# 
# >[Alpha Vantage](https://www.alphavantage.co) Alpha Vantage provides realtime and historical financial market data through a set of powerful and developer-friendly data APIs and spreadsheets. 
# 
# Use the ``AlphaVantageAPIWrapper`` to get currency exchange rates.

# In[1]:


import getpass
import os

os.environ["ALPHAVANTAGE_API_KEY"] = getpass.getpass()


# In[2]:


from langchain_community.utilities.alpha_vantage import AlphaVantageAPIWrapper


# In[3]:


alpha_vantage = AlphaVantageAPIWrapper()
alpha_vantage._get_exchange_rate("USD", "JPY")


# The `_get_time_series_daily` method returns the date, daily open, daily high, daily low, daily close, and daily volume of the global equity specified, covering the 100 latest data points.

# In[ ]:


alpha_vantage._get_time_series_daily("IBM")


# The `_get_time_series_weekly` method returns the last trading day of the week, weekly open, weekly high, weekly low, weekly close, and weekly volume of the global equity specified, covering 20+ years of historical data.

# In[ ]:


alpha_vantage._get_time_series_weekly("IBM")


# The `_get_quote_endpoint` method is a lightweight alternative to the time series APIs and returns the latest price and volume info for the specified symbol.

# In[6]:


alpha_vantage._get_quote_endpoint("IBM")


# The `search_symbol` method returns a list of symbols and the matching company information based on the text entered.

# In[ ]:


alpha_vantage.search_symbols("IB")


# The `_get_market_news_sentiment` method returns live and historical market news sentiment for a given asset.

# In[ ]:


alpha_vantage._get_market_news_sentiment("IBM")


# The `_get_top_gainers_losers` method returns the top 20 gainers, losers and most active stocks in the US market.

# In[ ]:


alpha_vantage._get_top_gainers_losers()


# The `run` method of the wrapper takes the following parameters: from_currency, to_currency. 
# 
# It Gets the currency exchange rates for the given currency pair.

# In[9]:


alpha_vantage.run("USD", "JPY")

