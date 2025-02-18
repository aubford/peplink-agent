#!/usr/bin/env python
# coding: utf-8

# # Azure AI Data
#
# >[Azure AI Studio](https://ai.azure.com/) provides the capability to upload data assets to cloud storage and register existing data assets from the following sources:
# >
# >- `Microsoft OneLake`
# >- `Azure Blob Storage`
# >- `Azure Data Lake gen 2`
#
# The benefit of this approach over `AzureBlobStorageContainerLoader` and `AzureBlobStorageFileLoader` is that authentication is handled seamlessly to cloud storage. You can use either *identity-based* data access control to the data or *credential-based* (e.g. SAS token, account key). In the case of credential-based data access you do not need to specify secrets in your code or set up key vaults - the system handles that for you.
#
# This notebook covers how to load document objects from a data asset in AI Studio.

# In[ ]:


get_ipython().run_line_magic(
    "pip", "install --upgrade --quiet  azureml-fsspec, azure-ai-generative"
)


# In[1]:


from azure.ai.resources.client import AIClient
from azure.identity import DefaultAzureCredential
from langchain_community.document_loaders import AzureAIDataLoader


# In[ ]:


# Create a connection to your project
client = AIClient(
    credential=DefaultAzureCredential(),
    subscription_id="<subscription_id>",
    resource_group_name="<resource_group_name>",
    project_name="<project_name>",
)


# In[3]:


# get the latest version of your data asset
data_asset = client.data.get(name="<data_asset_name>", label="latest")


# In[ ]:


# load the data asset
loader = AzureAIDataLoader(url=data_asset.path)


# In[4]:


loader.load()


# ## Specifying a glob pattern
# You can also specify a glob pattern for more finegrained control over what files to load. In the example below, only files with a `pdf` extension will be loaded.

# In[5]:


loader = AzureAIDataLoader(url=data_asset.path, glob="*.pdf")


# In[6]:


loader.load()


# In[ ]:
