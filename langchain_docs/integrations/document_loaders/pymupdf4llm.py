#!/usr/bin/env python
# coding: utf-8

# ---
# sidebar_label: PyMuPDF4LLM
# ---

# # PyMuPDF4LLMLoader
# 
# This notebook provides a quick overview for getting started with PyMuPDF4LLM [document loader](https://python.langchain.com/docs/concepts/#document-loaders). For detailed documentation of all PyMuPDF4LLMLoader features and configurations head to the [GitHub repository](https://github.com/lakinduboteju/langchain-pymupdf4llm).
# 
# ## Overview
# 
# ### Integration details
# 
# | Class | Package | Local | Serializable | JS support |
# | :--- | :--- | :---: | :---: |  :---: |
# | [PyMuPDF4LLMLoader](https://github.com/lakinduboteju/langchain-pymupdf4llm) | [langchain_pymupdf4llm](https://pypi.org/project/langchain-pymupdf4llm) | ✅ | ❌ | ❌ |
# 
# ### Loader features
# 
# | Source | Document Lazy Loading | Native Async Support | Extract Images | Extract Tables |
# | :---: | :---: | :---: | :---: | :---: |
# | PyMuPDF4LLMLoader | ✅ | ❌ | ✅ | ✅ |
# 
# ## Setup
# 
# To access PyMuPDF4LLM document loader you'll need to install the `langchain-pymupdf4llm` integration package.
# 
# ### Credentials
# 
# No credentials are required to use PyMuPDF4LLMLoader.

# To enable automated tracing of your model calls, set your [LangSmith](https://docs.smith.langchain.com/) API key:

# In[ ]:


# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"


# ### Installation
# 
# Install **langchain_community** and **langchain-pymupdf4llm**.

# In[1]:


get_ipython().run_line_magic('pip', 'install -qU langchain_community langchain-pymupdf4llm')


# ## Initialization
# 
# Now we can instantiate our model object and load documents:

# In[3]:


from langchain_pymupdf4llm import PyMuPDF4LLMLoader

file_path = "./example_data/layout-parser-paper.pdf"
loader = PyMuPDF4LLMLoader(file_path)


# ## Load

# In[4]:


docs = loader.load()
docs[0]


# In[5]:


import pprint

pprint.pp(docs[0].metadata)


# ## Lazy Load

# In[6]:


pages = []
for doc in loader.lazy_load():
    pages.append(doc)
    if len(pages) >= 10:
        # do some paged operation, e.g.
        # index.upsert(page)

        pages = []
len(pages)


# In[ ]:


from IPython.display import Markdown, display

part = pages[0].page_content[778:1189]
print(part)
# Markdown rendering
display(Markdown(part))


# In[23]:


pprint.pp(pages[0].metadata)


# The metadata attribute contains at least the following keys:
# - source
# - page (if in mode *page*)
# - total_page
# - creationdate
# - creator
# - producer
# 
# Additional metadata are specific to each parser.
# These pieces of information can be helpful (to categorize your PDFs for example).

# ## Splitting mode & custom pages delimiter

# When loading the PDF file you can split it in two different ways:
# - By page
# - As a single text flow
# 
# By default PyMuPDF4LLMLoader will split the PDF by page.

# ### Extract the PDF by page. Each page is extracted as a langchain Document object:

# In[ ]:


loader = PyMuPDF4LLMLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="page",
)
docs = loader.load()

print(len(docs))
pprint.pp(docs[0].metadata)


# In this mode the pdf is split by pages and the resulting Documents metadata contains the `page` (page number). But in some cases we could want to process the pdf as a single text flow (so we don't cut some paragraphs in half). In this case you can use the *single* mode :

# ### Extract the whole PDF as a single langchain Document object:

# In[ ]:


loader = PyMuPDF4LLMLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="single",
)
docs = loader.load()

print(len(docs))
pprint.pp(docs[0].metadata)


# Logically, in this mode, the `page` (page_number) metadata disappears. Here's how to clearly identify where pages end in the text flow :

# ### Add a custom *pages_delimiter* to identify where are ends of pages in *single* mode:

# In[ ]:


loader = PyMuPDF4LLMLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="single",
    pages_delimiter="\n-------THIS IS A CUSTOM END OF PAGE-------\n\n",
)
docs = loader.load()

part = docs[0].page_content[10663:11317]
print(part)
display(Markdown(part))


# The default `pages_delimiter` is \n-----\n\n.
# But this could simply be \n, or \f to clearly indicate a page change, or \<!-- PAGE BREAK --> for seamless injection in a Markdown viewer without a visual effect.

# # Extract images from the PDF

# You can extract images from your PDFs (in text form) with a choice of three different solutions:
# - rapidOCR (lightweight Optical Character Recognition tool)
# - Tesseract (OCR tool with high precision)
# - Multimodal language model
# 
# The result is inserted at the end of text of the page.

# ### Extract images from the PDF with rapidOCR:

# In[14]:


get_ipython().run_line_magic('pip', 'install -qU rapidocr-onnxruntime pillow')


# In[ ]:


from langchain_community.document_loaders.parsers import RapidOCRBlobParser

loader = PyMuPDF4LLMLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="page",
    extract_images=True,
    images_parser=RapidOCRBlobParser(),
)
docs = loader.load()

part = docs[5].page_content[1863:]
print(part)
display(Markdown(part))


# Be careful, RapidOCR is designed to work with Chinese and English, not other languages.

# ### Extract images from the PDF with Tesseract:

# In[16]:


get_ipython().run_line_magic('pip', 'install -qU pytesseract')


# In[ ]:


from langchain_community.document_loaders.parsers import TesseractBlobParser

loader = PyMuPDF4LLMLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="page",
    extract_images=True,
    images_parser=TesseractBlobParser(),
)
docs = loader.load()

print(docs[5].page_content[1863:])


# ### Extract images from the PDF with multimodal model:

# In[38]:


get_ipython().run_line_magic('pip', 'install -qU langchain_openai')


# In[39]:


import os

from dotenv import load_dotenv

load_dotenv()


# In[40]:


from getpass import getpass

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass("OpenAI API key =")


# In[ ]:


from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_openai import ChatOpenAI

loader = PyMuPDF4LLMLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="page",
    extract_images=True,
    images_parser=LLMImageBlobParser(
        model=ChatOpenAI(model="gpt-4o-mini", max_tokens=1024)
    ),
)
docs = loader.load()

print(docs[5].page_content[1863:])


# # Extract tables from the PDF
# 
# With PyMUPDF4LLM you can extract tables from your PDFs in *markdown* format :

# In[ ]:


loader = PyMuPDF4LLMLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="page",
    # "lines_strict" is the default strategy and
    # is the most accurate for tables with column and row lines,
    # but may not work well with all documents.
    # "lines" is a less strict strategy that may work better with
    # some documents.
    # "text" is the least strict strategy and may work better
    # with documents that do not have tables with lines.
    table_strategy="lines",
)
docs = loader.load()

part = docs[4].page_content[3210:]
print(part)
display(Markdown(part))


# ## Working with Files
# 
# Many document loaders involve parsing files. The difference between such loaders usually stems from how the file is parsed, rather than how the file is loaded. For example, you can use `open` to read the binary content of either a PDF or a markdown file, but you need different parsing logic to convert that binary data into text.
# 
# As a result, it can be helpful to decouple the parsing logic from the loading logic, which makes it easier to re-use a given parser regardless of how the data was loaded.
# You can use this strategy to analyze different files, with the same parsing parameters.

# In[ ]:


from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_pymupdf4llm import PyMuPDF4LLMParser

loader = GenericLoader(
    blob_loader=FileSystemBlobLoader(
        path="./example_data/",
        glob="*.pdf",
    ),
    blob_parser=PyMuPDF4LLMParser(),
)
docs = loader.load()

part = docs[0].page_content[:562]
print(part)
display(Markdown(part))


# ## API reference
# 
# For detailed documentation of all PyMuPDF4LLMLoader features and configurations head to the GitHub repository: https://github.com/lakinduboteju/langchain-pymupdf4llm
