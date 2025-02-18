#!/usr/bin/env python
# coding: utf-8

# # LLM Sherpa
#
# This notebook covers how to use `LLM Sherpa` to load files of many types. `LLM Sherpa` supports different file formats including DOCX, PPTX, HTML, TXT, and XML.
#
# `LLMSherpaFileLoader` use LayoutPDFReader, which is part of the LLMSherpa library. This tool is designed to parse PDFs while preserving their layout information, which is often lost when using most PDF to text parsers.
#
# Here are some key features of LayoutPDFReader:
#
# * It can identify and extract sections and subsections along with their levels.
# * It combines lines to form paragraphs.
# * It can identify links between sections and paragraphs.
# * It can extract tables along with the section the tables are found in.
# * It can identify and extract lists and nested lists.
# * It can join content spread across pages.
# * It can remove repeating headers and footers.
# * It can remove watermarks.
#
# check [llmsherpa](https://llmsherpa.readthedocs.io/en/latest/) documentation.
#
# `INFO: this library fail with some pdf files so use it with caution.`

# In[ ]:


# Install package
# !pip install --upgrade --quiet llmsherpa


# ## LLMSherpaFileLoader
#
# Under the hood LLMSherpaFileLoader defined some strategist to load file content: ["sections", "chunks", "html", "text"], setup [nlm-ingestor](https://github.com/nlmatics/nlm-ingestor) to get `llmsherpa_api_url` or use the default.

# ### sections strategy: return the file parsed into sections

# In[5]:


from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader

loader = LLMSherpaFileLoader(
    file_path="https://arxiv.org/pdf/2402.14207.pdf",
    new_indent_parser=True,
    apply_ocr=True,
    strategy="sections",
    llmsherpa_api_url="http://localhost:5010/api/parseDocument?renderFormat=all",
)
docs = loader.load()


# In[6]:


docs[1]


# In[7]:


len(docs)


# ### chunks strategy: return the file parsed into chunks

# In[8]:


from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader

loader = LLMSherpaFileLoader(
    file_path="https://arxiv.org/pdf/2402.14207.pdf",
    new_indent_parser=True,
    apply_ocr=True,
    strategy="chunks",
    llmsherpa_api_url="http://localhost:5010/api/parseDocument?renderFormat=all",
)
docs = loader.load()


# In[9]:


docs[1]


# In[10]:


len(docs)


# ### html strategy: return the file as one html document

# In[10]:


from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader

loader = LLMSherpaFileLoader(
    file_path="https://arxiv.org/pdf/2402.14207.pdf",
    new_indent_parser=True,
    apply_ocr=True,
    strategy="html",
    llmsherpa_api_url="http://localhost:5010/api/parseDocument?renderFormat=all",
)
docs = loader.load()


# In[12]:


docs[0].page_content[:400]


# In[13]:


len(docs)


# ### text strategy: return the file as one text document

# In[1]:


from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader

loader = LLMSherpaFileLoader(
    file_path="https://arxiv.org/pdf/2402.14207.pdf",
    new_indent_parser=True,
    apply_ocr=True,
    strategy="text",
    llmsherpa_api_url="http://localhost:5010/api/parseDocument?renderFormat=all",
)
docs = loader.load()


# In[3]:


docs[0].page_content[:400]


# In[4]:


len(docs)
