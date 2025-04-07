#!/usr/bin/env python
# coding: utf-8

# # Azure AI Search
# 
# [Azure AI Search](https://learn.microsoft.com/azure/search/search-what-is-azure-search) (formerly known as `Azure Search` and `Azure Cognitive Search`) is a cloud search service that gives developers infrastructure, APIs, and tools for information retrieval of vector, keyword, and hybrid queries at scale.
# 
# You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration

# ## Install Azure AI Search SDK
# 
# Use azure-search-documents package version 11.4.0 or later.

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  azure-search-documents')
get_ipython().run_line_magic('pip', 'install --upgrade --quiet  azure-identity')


# ## Import required libraries
# 
# `OpenAIEmbeddings` is assumed, but if you're using Azure OpenAI, import `AzureOpenAIEmbeddings` instead.

# In[2]:


import os

from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings


# ## Configure OpenAI settings
# Set variables for your OpenAI provider. You need either an [OpenAI account](https://platform.openai.com/docs/quickstart?context=python) or an [Azure OpenAI account](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource) to generate the embeddings. 

# In[3]:


# Option 1: use an OpenAI account
openai_api_key: str = "PLACEHOLDER FOR YOUR API KEY"
openai_api_version: str = "2023-05-15"
model: str = "text-embedding-ada-002"


# In[27]:


# Option 2: use an Azure OpenAI account with a deployment of an embedding model
azure_endpoint: str = "PLACEHOLDER FOR YOUR AZURE OPENAI ENDPOINT"
azure_openai_api_key: str = "PLACEHOLDER FOR YOUR AZURE OPENAI KEY"
azure_openai_api_version: str = "2023-05-15"
azure_deployment: str = "text-embedding-ada-002"


# ## Configure vector store settings
# 
# You need an [Azure subscription](https://azure.microsoft.com/en-us/free/search) and [Azure AI Search service](https://learn.microsoft.com/azure/search/search-create-service-portal) to use this vector store integration. No-cost versions are available for small and limited workloads.
#  
# Set variables for your Azure AI Search URL and admin API key. You can get these variables from the [Azure portal](https://portal.azure.com/#blade/HubsExtension/BrowseResourceBlade/resourceType/Microsoft.Search%2FsearchServices).

# In[24]:


vector_store_address: str = "YOUR_AZURE_SEARCH_ENDPOINT"
vector_store_password: str = "YOUR_AZURE_SEARCH_ADMIN_KEY"


# ## Create embeddings and vector store instances
#  
# Create instances of the OpenAIEmbeddings and AzureSearch classes. When you complete this step, you should have an empty search index on your Azure AI Search resource. The integration module provides a default schema.

# In[6]:


# Option 1: Use OpenAIEmbeddings with OpenAI account
embeddings: OpenAIEmbeddings = OpenAIEmbeddings(
    openai_api_key=openai_api_key, openai_api_version=openai_api_version, model=model
)


# In[29]:


# Option 2: Use AzureOpenAIEmbeddings with an Azure account
embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    azure_deployment=azure_deployment,
    openai_api_version=azure_openai_api_version,
    azure_endpoint=azure_endpoint,
    api_key=azure_openai_api_key,
)


# ## Create vector store instance
#  
# Create instance of the AzureSearch class using the embeddings from above

# In[30]:


index_name: str = "langchain-vector-demo"
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
)


# In[ ]:


# Specify additional properties for the Azure client such as the following https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core/README.md#configurations
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
    # Configure max retries for the Azure client
    additional_search_client_options={"retry_total": 4},
)


# ## Insert text and embeddings into vector store
#  
# This step loads, chunks, and vectorizes the sample document, and then indexes the content into a search index on Azure AI Search.

# In[31]:


from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("../../how_to/state_of_the_union.txt", encoding="utf-8")

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

vector_store.add_documents(documents=docs)


# ## Perform a vector similarity search
#  
# Execute a pure vector similarity search using the similarity_search() method:

# In[11]:


# Perform a similarity search
docs = vector_store.similarity_search(
    query="What did the president say about Ketanji Brown Jackson",
    k=3,
    search_type="similarity",
)
print(docs[0].page_content)


# ## Perform a vector similarity search with relevance scores
#  
# Execute a pure vector similarity search using the similarity_search_with_relevance_scores() method. Queries that don't meet the threshold requirements are exluded.

# In[12]:


docs_and_scores = vector_store.similarity_search_with_relevance_scores(
    query="What did the president say about Ketanji Brown Jackson",
    k=4,
    score_threshold=0.80,
)
from pprint import pprint

pprint(docs_and_scores)


# ## Perform a hybrid search
# 
# Execute hybrid search using the search_type or hybrid_search() method. Vector and nonvector text fields are queried in parallel, results are merged, and top matches of the unified result set are returned.

# In[13]:


# Perform a hybrid search using the search_type parameter
docs = vector_store.similarity_search(
    query="What did the president say about Ketanji Brown Jackson",
    k=3,
    search_type="hybrid",
)
print(docs[0].page_content)


# In[14]:


# Perform a hybrid search using the hybrid_search method
docs = vector_store.hybrid_search(
    query="What did the president say about Ketanji Brown Jackson", k=3
)
print(docs[0].page_content)


# ## Custom schemas and queries
# 
# This section shows you how to replace the default schema with a custom schema.
# 

# ### Create a new index with custom filterable fields 
# 
# This schema shows field definitions. It's the default schema, plus several new fields attributed as filterable. Because it's using the default vector configuration, you won't see vector configuration or vector profile overrides here. The name of the default vector profile is "myHnswProfile" and it's using a vector configuration of Hierarchical Navigable Small World (HNSW) for indexing and queries against the content_vector field.
# 
# There's no data for this schema in this step. When you execute the cell, you should get an empty index on Azure AI Search.

# In[15]:


from azure.search.documents.indexes.models import (
    ScoringProfile,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    TextWeights,
)

#  Replace OpenAIEmbeddings with AzureOpenAIEmbeddings if Azure OpenAI is your provider.
embeddings: OpenAIEmbeddings = OpenAIEmbeddings(
    openai_api_key=openai_api_key, openai_api_version=openai_api_version, model=model
)
embedding_function = embeddings.embed_query

fields = [
    SimpleField(
        name="id",
        type=SearchFieldDataType.String,
        key=True,
        filterable=True,
    ),
    SearchableField(
        name="content",
        type=SearchFieldDataType.String,
        searchable=True,
    ),
    SearchField(
        name="content_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=len(embedding_function("Text")),
        vector_search_profile_name="myHnswProfile",
    ),
    SearchableField(
        name="metadata",
        type=SearchFieldDataType.String,
        searchable=True,
    ),
    # Additional field to store the title
    SearchableField(
        name="title",
        type=SearchFieldDataType.String,
        searchable=True,
    ),
    # Additional field for filtering on document source
    SimpleField(
        name="source",
        type=SearchFieldDataType.String,
        filterable=True,
    ),
]

index_name: str = "langchain-vector-demo-custom"

vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=embedding_function,
    fields=fields,
)


# ### Add data and perform a query that includes a filter
# 
# This example adds data to the vector store based on the custom schema. It loads text into the title and source fields. The source field is filterable. The sample query in this section filters the results based on content in the source field.

# In[16]:


# Data in the metadata dictionary with a corresponding field in the index will be added to the index.
# In this example, the metadata dictionary contains a title, a source, and a random field.
# The title and the source are added to the index as separate fields, but the random value is ignored because it's not defined in the schema.
# The random field is only stored in the metadata field.
vector_store.add_texts(
    ["Test 1", "Test 2", "Test 3"],
    [
        {"title": "Title 1", "source": "A", "random": "10290"},
        {"title": "Title 2", "source": "A", "random": "48392"},
        {"title": "Title 3", "source": "B", "random": "32893"},
    ],
)


# In[17]:


res = vector_store.similarity_search(query="Test 3 source1", k=3, search_type="hybrid")
res


# In[18]:


res = vector_store.similarity_search(
    query="Test 3 source1", k=3, search_type="hybrid", filters="source eq 'A'"
)
res


# ### Create a new index with a scoring profile
# 
# Here's another custom schema that includes a scoring profile definition. A scoring profile is used for relevance tuning of nonvector content, which is helpful in hybrid search scenarios.

# In[19]:


from azure.search.documents.indexes.models import (
    FreshnessScoringFunction,
    FreshnessScoringParameters,
    ScoringProfile,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    TextWeights,
)

#  Replace OpenAIEmbeddings with AzureOpenAIEmbeddings if Azure OpenAI is your provider.
embeddings: OpenAIEmbeddings = OpenAIEmbeddings(
    openai_api_key=openai_api_key, openai_api_version=openai_api_version, model=model
)
embedding_function = embeddings.embed_query

fields = [
    SimpleField(
        name="id",
        type=SearchFieldDataType.String,
        key=True,
        filterable=True,
    ),
    SearchableField(
        name="content",
        type=SearchFieldDataType.String,
        searchable=True,
    ),
    SearchField(
        name="content_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=len(embedding_function("Text")),
        vector_search_profile_name="myHnswProfile",
    ),
    SearchableField(
        name="metadata",
        type=SearchFieldDataType.String,
        searchable=True,
    ),
    # Additional field to store the title
    SearchableField(
        name="title",
        type=SearchFieldDataType.String,
        searchable=True,
    ),
    # Additional field for filtering on document source
    SimpleField(
        name="source",
        type=SearchFieldDataType.String,
        filterable=True,
    ),
    # Additional data field for last doc update
    SimpleField(
        name="last_update",
        type=SearchFieldDataType.DateTimeOffset,
        searchable=True,
        filterable=True,
    ),
]
# Adding a custom scoring profile with a freshness function
sc_name = "scoring_profile"
sc = ScoringProfile(
    name=sc_name,
    text_weights=TextWeights(weights={"title": 5}),
    function_aggregation="sum",
    functions=[
        FreshnessScoringFunction(
            field_name="last_update",
            boost=100,
            parameters=FreshnessScoringParameters(boosting_duration="P2D"),
            interpolation="linear",
        )
    ],
)

index_name = "langchain-vector-demo-custom-scoring-profile"

vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
    fields=fields,
    scoring_profiles=[sc],
    default_scoring_profile=sc_name,
)


# In[20]:


# Adding same data with different last_update to show Scoring Profile effect
from datetime import datetime, timedelta

today = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S-00:00")
yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S-00:00")
one_month_ago = (datetime.utcnow() - timedelta(days=30)).strftime(
    "%Y-%m-%dT%H:%M:%S-00:00"
)

vector_store.add_texts(
    ["Test 1", "Test 1", "Test 1"],
    [
        {
            "title": "Title 1",
            "source": "source1",
            "random": "10290",
            "last_update": today,
        },
        {
            "title": "Title 1",
            "source": "source1",
            "random": "48392",
            "last_update": yesterday,
        },
        {
            "title": "Title 1",
            "source": "source1",
            "random": "32893",
            "last_update": one_month_ago,
        },
    ],
)


# In[21]:


res = vector_store.similarity_search(query="Test 1", k=3, search_type="similarity")
res

