#!/usr/bin/env python
# coding: utf-8

# # How to load web pages
#
# This guide covers how to [load](/docs/concepts/document_loaders/) web pages into the LangChain [Document](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html) format that we use downstream. Web pages contain text, images, and other multimedia elements, and are typically represented with HTML. They may include links to other pages or resources.
#
# LangChain integrates with a host of parsers that are appropriate for web pages. The right parser will depend on your needs. Below we demonstrate two possibilities:
#
# - [Simple and fast](/docs/how_to/document_loader_web#simple-and-fast-text-extraction) parsing, in which we recover one `Document` per web page with its content represented as a "flattened" string;
# - [Advanced](/docs/how_to/document_loader_web#advanced-parsing) parsing, in which we recover multiple `Document` objects per page, allowing one to identify and traverse sections, links, tables, and other structures.
#
# ## Setup
#
# For the "simple and fast" parsing, we will need `langchain-community` and the `beautifulsoup4` library:

# In[ ]:


get_ipython().run_line_magic("pip", "install -qU langchain-community beautifulsoup4")


# For advanced parsing, we will use `langchain-unstructured`:

# In[ ]:


get_ipython().run_line_magic("pip", "install -qU langchain-unstructured")


# ## Simple and fast text extraction
#
# If you are looking for a simple string representation of text that is embedded in a web page, the method below is appropriate. It will return a list of `Document` objects -- one per page -- containing a single string of the page's text. Under the hood it uses the `beautifulsoup4` Python library.
#
# LangChain document loaders implement `lazy_load` and its async variant, `alazy_load`, which return iterators of `Document objects`. We will use these below.

# In[1]:


import bs4
from langchain_community.document_loaders import WebBaseLoader

page_url = "https://python.langchain.com/docs/how_to/chatbots_memory/"

loader = WebBaseLoader(web_paths=[page_url])
docs = []
async for doc in loader.alazy_load():
    docs.append(doc)

assert len(docs) == 1
doc = docs[0]


# In[2]:


print(f"{doc.metadata}\n")
print(doc.page_content[:500].strip())


# This is essentially a dump of the text from the page's HTML. It may contain extraneous information like headings and navigation bars. If you are familiar with the expected HTML, you can specify desired `<div>` classes and other parameters via BeautifulSoup. Below we parse only the body text of the article:

# In[3]:


loader = WebBaseLoader(
    web_paths=[page_url],
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(class_="theme-doc-markdown markdown"),
    },
    bs_get_text_kwargs={"separator": " | ", "strip": True},
)

docs = []
async for doc in loader.alazy_load():
    docs.append(doc)

assert len(docs) == 1
doc = docs[0]


# In[4]:


print(f"{doc.metadata}\n")
print(doc.page_content[:500])


# In[5]:


print(doc.page_content[-500:])


# Note that this required advance technical knowledge of how the body text is represented in the underlying HTML.
#
# We can parameterize `WebBaseLoader` with a variety of settings, allowing for specification of request headers, rate limits, and parsers and other kwargs for BeautifulSoup. See its [API reference](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.web_base.WebBaseLoader.html) for detail.

# ## Advanced parsing
#
# This method is appropriate if we want more granular control or processing of the page content. Below, instead of generating one `Document` per page and controlling its content via BeautifulSoup, we generate multiple `Document` objects representing distinct structures on a page. These structures can include section titles and their corresponding body texts, lists or enumerations, tables, and more.
#
# Under the hood it uses the `langchain-unstructured` library. See the [integration docs](/docs/integrations/document_loaders/unstructured_file/) for more information about using [Unstructured](https://docs.unstructured.io/welcome) with LangChain.

# In[6]:


from langchain_unstructured import UnstructuredLoader

page_url = "https://python.langchain.com/docs/how_to/chatbots_memory/"
loader = UnstructuredLoader(web_url=page_url)

docs = []
async for doc in loader.alazy_load():
    docs.append(doc)


# Note that with no advance knowledge of the page HTML structure, we recover a natural organization of the body text:

# In[7]:


for doc in docs[:5]:
    print(doc.page_content)


# ### Extracting content from specific sections

# Each `Document` object represents an element of the page. Its metadata contains useful information, such as its category:

# In[8]:


for doc in docs[:5]:
    print(f'{doc.metadata["category"]}: {doc.page_content}')


# Elements may also have parent-child relationships -- for example, a paragraph might belong to a section with a title. If a section is of particular interest (e.g., for indexing) we can isolate the corresponding `Document` objects.
#
# As an example, below we load the content of the "Setup" sections for two web pages:

# In[9]:


from typing import List

from langchain_core.documents import Document


async def _get_setup_docs_from_url(url: str) -> List[Document]:
    loader = UnstructuredLoader(web_url=url)

    setup_docs = []
    parent_id = -1
    async for doc in loader.alazy_load():
        if doc.metadata["category"] == "Title" and doc.page_content.startswith("Setup"):
            parent_id = doc.metadata["element_id"]
        if doc.metadata.get("parent_id") == parent_id:
            setup_docs.append(doc)

    return setup_docs


page_urls = [
    "https://python.langchain.com/docs/how_to/chatbots_memory/",
    "https://python.langchain.com/docs/how_to/chatbots_tools/",
]
setup_docs = []
for url in page_urls:
    page_setup_docs = await _get_setup_docs_from_url(url)
    setup_docs.extend(page_setup_docs)


# In[10]:


from collections import defaultdict

setup_text = defaultdict(str)

for doc in setup_docs:
    url = doc.metadata["url"]
    setup_text[url] += f"{doc.page_content}\n"

dict(setup_text)


# ### Vector search over page content
#
# Once we have loaded the page contents into LangChain `Document` objects, we can index them (e.g., for a RAG application) in the usual way. Below we use OpenAI [embeddings](/docs/concepts/embedding_models), although any LangChain embeddings model will suffice.

# In[ ]:


get_ipython().run_line_magic("pip", "install -qU langchain-openai")


# In[11]:


import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")


# In[12]:


from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

vector_store = InMemoryVectorStore.from_documents(setup_docs, OpenAIEmbeddings())
retrieved_docs = vector_store.similarity_search("Install Tavily", k=2)
for doc in retrieved_docs:
    print(f'Page {doc.metadata["url"]}: {doc.page_content[:300]}\n')


# ## Other web page loaders
#
# For a list of available LangChain web page loaders, please see [this table](/docs/integrations/document_loaders/#webpages).
