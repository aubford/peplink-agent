#!/usr/bin/env python
# coding: utf-8

# # Microsoft PowerPoint
# 
# >[Microsoft PowerPoint](https://en.wikipedia.org/wiki/Microsoft_PowerPoint) is a presentation program by Microsoft.
# 
# This covers how to load `Microsoft PowerPoint` documents into a document format that we can use downstream.
# 
# Please see [this guide](/docs/integrations/providers/unstructured/) for more instructions on setting up Unstructured locally, including setting up required system dependencies.

# In[ ]:


# Install packages
get_ipython().run_line_magic('pip', 'install unstructured')
get_ipython().run_line_magic('pip', 'install python-magic')
get_ipython().run_line_magic('pip', 'install python-pptx')


# In[2]:


from langchain_community.document_loaders import UnstructuredPowerPointLoader

loader = UnstructuredPowerPointLoader("./example_data/fake-power-point.pptx")

data = loader.load()

data


# ### Retain Elements
# 
# Under the hood, `Unstructured` creates different "elements" for different chunks of text. By default we combine those together, but you can easily keep that separation by specifying `mode="elements"`.

# In[4]:


loader = UnstructuredPowerPointLoader(
    "./example_data/fake-power-point.pptx", mode="elements"
)

data = loader.load()

data[0]


# ## Using Azure AI Document Intelligence
# 
# >[Azure AI Document Intelligence](https://aka.ms/doc-intelligence) (formerly known as `Azure Form Recognizer`) is machine-learning 
# >based service that extracts texts (including handwriting), tables, document structures (e.g., titles, section headings, etc.) and key-value-pairs from
# >digital or scanned PDFs, images, Office and HTML files.
# >
# >Document Intelligence supports `PDF`, `JPEG/JPG`, `PNG`, `BMP`, `TIFF`, `HEIF`, `DOCX`, `XLSX`, `PPTX` and `HTML`.
# 
# This current implementation of a loader using `Document Intelligence` can incorporate content page-wise and turn it into LangChain documents. The default output format is markdown, which can be easily chained with `MarkdownHeaderTextSplitter` for semantic document chunking. You can also use `mode="single"` or `mode="page"` to return pure texts in a single page or document split by page.
# 

# ## Prerequisite
# 
# An Azure AI Document Intelligence resource in one of the 3 preview regions: **East US**, **West US2**, **West Europe** - follow [this document](https://learn.microsoft.com/azure/ai-services/document-intelligence/create-document-intelligence-resource?view=doc-intel-4.0.0) to create one if you don't have. You will be passing `<endpoint>` and `<key>` as parameters to the loader.

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  langchain langchain-community azure-ai-documentintelligence')


# In[ ]:


from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader

file_path = "<filepath>"
endpoint = "<endpoint>"
key = "<key>"
loader = AzureAIDocumentIntelligenceLoader(
    api_endpoint=endpoint, api_key=key, file_path=file_path, api_model="prebuilt-layout"
)

documents = loader.load()

