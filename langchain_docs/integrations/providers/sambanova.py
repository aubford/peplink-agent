#!/usr/bin/env python
# coding: utf-8

# # SambaNova
#
# Customers are turning to [SambaNova](https://sambanova.ai/) to quickly deploy state-of-the-art AI capabilities to gain competitive advantage. Our purpose-built enterprise-scale AI platform is the technology backbone for the next generation of AI computing. We power the foundation models that unlock the valuable business insights trapped in data.
#
# Designed for AI, the SambaNova RDU was built with a revolutionary dataflow architecture. This design makes the RDU significantly more efficient for these workloads than GPUs as it eliminates redundant calls to memory, which are an inherent limitation of how GPUs function. This built-in efficiency is one of the features that makes the RDU capable of much higher performance than GPUs in a fraction of the footprint.
#
# On top of our architecture We have developed some platforms that allow companies and developers to get full advantage of the RDU processors and open source models.

# ### SambaNovaCloud
#
# SambaNova's [SambaNova Cloud](https://cloud.sambanova.ai/) is a platform for performing inference with open-source models
#
# You can obtain a free SambaNovaCloud API key [here](https://cloud.sambanova.ai/)

# ### SambaStudio
#
# SambaNova's [SambaStudio](https://docs.sambanova.ai/sambastudio/latest/sambastudio-intro.html) is a rich, GUI-based platform that provides the functionality to train, deploy, and manage models in SambaNova DataScale systems.

# ## Installation and Setup
#
# Install the integration package:
#
# ```bash
# pip install langchain-sambanova
# ```
#
# set your API key it as an environment variable:
#
# If you are a SambaNovaCloud user:
#
# ```bash
# export SAMBANOVA_API_KEY="your-sambanova-cloud-api-key-here"
# ```
#
# or if you are SambaStudio User
#
# ```bash
# export SAMBASTUDIO_API_KEY="your-sambastudio-api-key-here"
# ```

# ## Chat models

# In[2]:


from langchain_sambanova import ChatSambaNovaCloud

llm = ChatSambaNovaCloud(model="Meta-Llama-3.3-70B-Instruct", temperature=0.7)
llm.invoke("Tell me a joke about artificial intelligence.")


# In[ ]:


from langchain_sambanova import ChatSambaStudio

llm = ChatSambaStudio(model="Meta-Llama-3.3-70B-Instruct", temperature=0.7)
llm.invoke("Tell me a joke about artificial intelligence.")


# ## Embedding Models

# In[ ]:


from langchain_sambanova import SambaStudioEmbeddings

embeddings = SambaStudioEmbeddings(model="e5-mistral-7b-instruct")
embeddings.embed_query("What is the meaning of life?")


# API Reference [langchain-sambanova](https://python.langchain.com/api_reference/sambanova/index.html)
