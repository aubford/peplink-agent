#!/usr/bin/env python
# coding: utf-8

# # Huawei OBS File
# The following code demonstrates how to load an object from the Huawei OBS (Object Storage Service) as document.

# In[ ]:


# Install the required package
# pip install esdk-obs-python


# In[1]:


from langchain_community.document_loaders.obs_file import OBSFileLoader


# In[2]:


endpoint = "your-endpoint"


# In[3]:


from obs import ObsClient

obs_client = ObsClient(
    access_key_id="your-access-key",
    secret_access_key="your-secret-key",
    server=endpoint,
)
loader = OBSFileLoader("your-bucket-name", "your-object-key", client=obs_client)


# In[ ]:


loader.load()


# ## Each Loader with Separate Authentication Information
# If you don't need to reuse OBS connections between different loaders, you can directly configure the `config`. The loader will use the config information to initialize its own OBS client.

# In[4]:


# Configure your access credentials\n
config = {"ak": "your-access-key", "sk": "your-secret-key"}
loader = OBSFileLoader(
    "your-bucket-name", "your-object-key", endpoint=endpoint, config=config
)


# In[ ]:


loader.load()


# ## Get Authentication Information from ECS
# If your langchain is deployed on Huawei Cloud ECS and [Agency is set up](https://support.huaweicloud.com/intl/en-us/usermanual-ecs/ecs_03_0166.html#section7), the loader can directly get the security token from ECS without needing access key and secret key. 

# In[5]:


config = {"get_token_from_ecs": True}
loader = OBSFileLoader(
    "your-bucket-name", "your-object-key", endpoint=endpoint, config=config
)


# In[ ]:


loader.load()


# ## Access a Publicly Accessible Object
# If the object you want to access allows anonymous user access (anonymous users have `GetObject` permission), you can directly load the object without configuring the `config` parameter.

# In[6]:


loader = OBSFileLoader("your-bucket-name", "your-object-key", endpoint=endpoint)


# In[ ]:


loader.load()

