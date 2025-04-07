#!/usr/bin/env python
# coding: utf-8

# # UnstructuredMarkdownLoader
# 
# This notebook provides a quick overview for getting started with UnstructuredMarkdown [document loader](https://python.langchain.com/docs/concepts/document_loaders). For detailed documentation of all __ModuleName__Loader features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.markdown.UnstructuredMarkdownLoader.html).
# 
# ## Overview
# ### Integration details
# 
# 
# | Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/document_loaders/file_loaders/unstructured/)|
# | :--- | :--- | :---: | :---: |  :---: |
# | [UnstructuredMarkdownLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.markdown.UnstructuredMarkdownLoader.html) | [langchain_community](https://python.langchain.com/api_reference/community/index.html) | ❌ | ❌ | ✅ | 
# ### Loader features
# | Source | Document Lazy Loading | Native Async Support
# | :---: | :---: | :---: | 
# | UnstructuredMarkdownLoader | ✅ | ❌ | 
# 
# ## Setup
# 
# To access UnstructuredMarkdownLoader document loader you'll need to install the `langchain-community` integration package and the `unstructured` python package.
# 
# ### Credentials
# 
# No credentials are needed to use this loader.

# To enable automated tracing of your model calls, set your [LangSmith](https://docs.smith.langchain.com/) API key:

# In[ ]:


# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"


# ### Installation
# 
# Install **langchain_community** and **unstructured**

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain_community unstructured')


# ## Initialization
# 
# Now we can instantiate our model object and load documents. 
# 
# You can run the loader in one of two modes: "single" and "elements". If you use "single" mode, the document will be returned as a single `Document` object. If you use "elements" mode, the unstructured library will split the document into elements such as `Title` and `NarrativeText`. You can pass in additional `unstructured` kwargs after mode to apply different `unstructured` settings.

# In[10]:


from langchain_community.document_loaders import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader(
    "./example_data/example.md",
    mode="single",
    strategy="fast",
)


# ## Load

# In[11]:


docs = loader.load()
docs[0]


# In[12]:


print(docs[0].metadata)


# ## Lazy Load

# In[7]:


page = []
for doc in loader.lazy_load():
    page.append(doc)
    if len(page) >= 10:
        # do some paged operation, e.g.
        # index.upsert(page)

        page = []
page[0]


# ## Load Elements
# 
# In this example we will load in the `elements` mode, which will return a list of the different elements in the markdown document:

# In[14]:


from langchain_community.document_loaders import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader(
    "./example_data/example.md",
    mode="elements",
    strategy="fast",
)

docs = loader.load()
len(docs)


# As you see there are 29 elements that were pulled from the `example.md` file. The first element is the title of the document as expected:

# In[16]:


docs[0].page_content


# ## API reference
# 
# For detailed documentation of all UnstructuredMarkdownLoader features and configurations head to the API reference: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.markdown.UnstructuredMarkdownLoader.html
