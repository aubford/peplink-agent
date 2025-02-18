#!/usr/bin/env python
# coding: utf-8

# # Datadog Logs
#
# >[Datadog](https://www.datadoghq.com/) is a monitoring and analytics platform for cloud-scale applications.
#
# This loader fetches the logs from your applications in Datadog using the `datadog_api_client` Python package. You must initialize the loader with your `Datadog API key` and `APP key`, and you need to pass in the query to extract the desired logs.

# In[ ]:


from langchain_community.document_loaders import DatadogLogsLoader


# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  datadog-api-client")


# In[ ]:


DD_API_KEY = "..."
DD_APP_KEY = "..."


# In[ ]:


query = "service:agent status:error"

loader = DatadogLogsLoader(
    query=query,
    api_key=DD_API_KEY,
    app_key=DD_APP_KEY,
    from_time=1688732708951,  # Optional, timestamp in milliseconds
    to_time=1688736308951,  # Optional, timestamp in milliseconds
    limit=100,  # Optional, default is 100
)


# In[11]:


documents = loader.load()
documents
