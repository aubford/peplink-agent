#!/usr/bin/env python
# coding: utf-8

# # PyMuPDFLoader
#
# This notebook provides a quick overview for getting started with `PyMuPDF` [document loader](https://python.langchain.com/docs/concepts/document_loaders). For detailed documentation of all __ModuleName__Loader features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.PyMuPDFLoader.html).
#
#
#
# ## Overview
# ### Integration details
#
# | Class | Package | Local | Serializable | JS support|
# | :--- | :--- | :---: | :---: |  :---: |
# | [PyMuPDFLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.PyMuPDFLoader.html) | [langchain_community](https://python.langchain.com/api_reference/community/index.html) | ✅ | ❌ | ❌ |
#
# ---------
#
# ### Loader features
#
# | Source | Document Lazy Loading | Native Async Support | Extract Images | Extract Tables |
# | :---: | :---: | :---: | :---: |:---: |
# | PyMuPDFLoader | ✅ | ❌ | ✅ | ✅ |
#
#
#
# ## Setup
#
# ### Credentials
#
# No credentials are required to use PyMuPDFLoader

# If you want to get automated best in-class tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:

# In[1]:


# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"


# ### Installation
#
# Install **langchain_community** and **pymupdf**.

# In[2]:


get_ipython().run_line_magic("pip", "install -qU langchain_community pymupdf")


# ## Initialization
#
# Now we can instantiate our model object and load documents:

# In[3]:


from langchain_community.document_loaders import PyMuPDFLoader

file_path = "./example_data/layout-parser-paper.pdf"
loader = PyMuPDFLoader(file_path)


# ## Load

# In[4]:


docs = loader.load()
docs[0]


# In[5]:


import pprint

pprint.pp(docs[0].metadata)


# ## Lazy Load
#

# In[6]:


pages = []
for doc in loader.lazy_load():
    pages.append(doc)
    if len(pages) >= 10:
        # do some paged operation, e.g.
        # index.upsert(page)

        pages = []
len(pages)


# In[7]:


print(pages[0].page_content[:100])
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
# By default PDFPlumberLoader will split the PDF by page.

# ### Extract the PDF by page. Each page is extracted as a langchain Document object:

# In[8]:


loader = PyMuPDFLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="page",
)
docs = loader.load()
print(len(docs))
pprint.pp(docs[0].metadata)


# In this mode the pdf is split by pages and the resulting Documents metadata contains the page number. But in some cases we could want to process the pdf as a single text flow (so we don't cut some paragraphs in half). In this case you can use the *single* mode :

# ### Extract the whole PDF as a single langchain Document object:

# In[9]:


loader = PyMuPDFLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="single",
)
docs = loader.load()
print(len(docs))
pprint.pp(docs[0].metadata)


# Logically, in this mode, the ‘page_number’ metadata disappears. Here's how to clearly identify where pages end in the text flow :

# ### Add a custom *pages_delimiter* to identify where are ends of pages in *single* mode:

# In[10]:


loader = PyMuPDFLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="single",
    pages_delimiter="\n-------THIS IS A CUSTOM END OF PAGE-------\n",
)
docs = loader.load()
print(docs[0].page_content[:5780])


# This could simply be \n, or \f to clearly indicate a page change, or \<!-- PAGE BREAK --> for seamless injection in a Markdown viewer without a visual effect.

# # Extract images from the PDF

# You can extract images from your PDFs with a choice of three different solutions:
# - rapidOCR (lightweight Optical Character Recognition tool)
# - Tesseract (OCR tool with high precision)
# - Multimodal language model
#
# You can tune these functions to choose the output format of the extracted images among *html*, *markdown* or *text*
#
# The result is inserted between the last and the second-to-last paragraphs of text of the page.

# ### Extract images from the PDF with rapidOCR:

# In[11]:


get_ipython().run_line_magic("pip", "install -qU rapidocr-onnxruntime")


# In[13]:


from langchain_community.document_loaders.parsers import RapidOCRBlobParser

loader = PyMuPDFLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="page",
    images_inner_format="markdown-img",
    images_parser=RapidOCRBlobParser(),
)
docs = loader.load()

print(docs[5].page_content)


# Be careful, RapidOCR is designed to work with Chinese and English, not other languages.

# ### Extract images from the PDF with Tesseract:

# In[14]:


get_ipython().run_line_magic("pip", "install -qU pytesseract")


# In[15]:


from langchain_community.document_loaders.parsers import TesseractBlobParser

loader = PyMuPDFLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="page",
    images_inner_format="html-img",
    images_parser=TesseractBlobParser(),
)
docs = loader.load()
print(docs[5].page_content)


# ### Extract images from the PDF with multimodal model:

# In[16]:


get_ipython().run_line_magic("pip", "install -qU langchain_openai")


# In[17]:


import os

from dotenv import load_dotenv

load_dotenv()


# In[18]:


from getpass import getpass

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass("OpenAI API key =")


# In[19]:


from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_openai import ChatOpenAI

loader = PyMuPDFLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="page",
    images_inner_format="markdown-img",
    images_parser=LLMImageBlobParser(model=ChatOpenAI(model="gpt-4o", max_tokens=1024)),
)
docs = loader.load()
print(docs[5].page_content)


# # Extract tables from the PDF

# With PyMUPDF you can extract tables from your PDFs in *html*, *markdown* or *csv* format :

# In[49]:


loader = PyMuPDFLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="page",
    extract_tables="markdown",
)
docs = loader.load()
print(docs[4].page_content)


# ## Working with Files
#
# Many document loaders involve parsing files. The difference between such loaders usually stems from how the file is parsed, rather than how the file is loaded. For example, you can use `open` to read the binary content of either a PDF or a markdown file, but you need different parsing logic to convert that binary data into text.
#
# As a result, it can be helpful to decouple the parsing logic from the loading logic, which makes it easier to re-use a given parser regardless of how the data was loaded.
# You can use this strategy to analyze different files, with the same parsing parameters.

# In[21]:


from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import PyMuPDFParser

loader = GenericLoader(
    blob_loader=FileSystemBlobLoader(
        path="./example_data/",
        glob="*.pdf",
    ),
    blob_parser=PyMuPDFParser(),
)
docs = loader.load()
print(docs[0].page_content)
pprint.pp(docs[0].metadata)


# It is possible to work with files from cloud storage.

# In[ ]:


from langchain_community.document_loaders import CloudBlobLoader
from langchain_community.document_loaders.generic import GenericLoader

loader = GenericLoader(
    blob_loader=CloudBlobLoader(
        url="s3:/mybucket",  # Supports s3://, az://, gs://, file:// schemes.
        glob="*.pdf",
    ),
    blob_parser=PyMuPDFParser(),
)
docs = loader.load()
print(docs[0].page_content)
pprint.pp(docs[0].metadata)


# ## API reference
#
# For detailed documentation of all `PyMuPDFLoader` features and configurations head to the API reference: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.PyMuPDFLoader.html
