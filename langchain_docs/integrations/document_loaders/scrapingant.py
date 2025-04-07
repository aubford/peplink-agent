#!/usr/bin/env python
# coding: utf-8

# ---
# sidebar_label: ScrapingAnt
# ---
# 
# # ScrapingAnt
# ## Overview
# [ScrapingAnt](https://scrapingant.com/) is a web scraping API with headless browser capabilities, proxies, and anti-bot bypass. It allows for extracting web page data into accessible LLM markdown.
# 
# This particular integration uses only Markdown extraction feature, but don't hesitate to [reach out to us](mailto:support@scrapingant.com) if you need more features provided by ScrapingAnt, but not yet implemented in this integration.
# 
# ### Integration details
# 
# | Class                                                                                                                                                    | Package                                                                                        | Local | Serializable | JS support |
# |:---------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|:-----:|:------------:|:----------:|
# | [ScrapingAntLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.scrapingant.ScrapingAntLoader.html) | [langchain_community](https://python.langchain.com/api_reference/community/index.html) |   ❌   |      ❌       |     ❌      | 
# 
# ### Loader features
# |      Source       | Document Lazy Loading | Async Support |
# |:-----------------:|:---------------------:|:-------------:| 
# | ScrapingAntLoader |           ✅           |       ❌       | 
# 

# ## Setup
# 
# Install ScrapingAnt Python SDK and he required Langchain packages using pip:
# ```shell
# pip install scrapingant-client langchain langchain-community
# ```

# ## Instantiation

# In[6]:


from langchain_community.document_loaders import ScrapingAntLoader

scrapingant_loader = ScrapingAntLoader(
    ["https://scrapingant.com/", "https://example.com/"],  # List of URLs to scrape
    api_key="<YOUR_SCRAPINGANT_TOKEN>",  # Get your API key from https://scrapingant.com/
    continue_on_failure=True,  # Ignore unprocessable web pages and log their exceptions
)


# The ScrapingAntLoader also allows providing a dict - scraping config for customizing the scrape request. As it is based on the [ScrapingAnt Python SDK](https://github.com/ScrapingAnt/scrapingant-client-python) you can pass any of the [common arguments](https://github.com/ScrapingAnt/scrapingant-client-python) to the `scrape_config` parameter.

# In[5]:


from langchain_community.document_loaders import ScrapingAntLoader

scrapingant_config = {
    "browser": True,  # Enable browser rendering with a cloud browser
    "proxy_type": "datacenter",  # Select a proxy type (datacenter or residential)
    "proxy_country": "us",  # Select a proxy location
}

scrapingant_additional_config_loader = ScrapingAntLoader(
    ["https://scrapingant.com/"],
    api_key="<YOUR_SCRAPINGANT_TOKEN>",  # Get your API key from https://scrapingant.com/
    continue_on_failure=True,  # Ignore unprocessable web pages and log their exceptions
    scrape_config=scrapingant_config,  # Pass the scrape_config object
)


# ## Load
# 
# Use the `load` method to scrape the web pages and get the extracted markdown content.
# 

# In[ ]:


# Load documents from URLs as markdown
documents = scrapingant_loader.load()

print(documents)


# ## Lazy Load
# 
# Use the 'lazy_load' method to scrape the web pages and get the extracted markdown content lazily.

# In[ ]:


# Lazy load documents from URLs as markdown
lazy_documents = scrapingant_loader.lazy_load()

for document in lazy_documents:
    print(document)


# ## API reference
# 
# This loader is based on the [ScrapingAnt Python SDK](https://docs.scrapingant.com/python-client). For more configuration options, see the [common arguments](https://github.com/ScrapingAnt/scrapingant-client-python/tree/master?tab=readme-ov-file#common-arguments)
