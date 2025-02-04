#!/usr/bin/env python
# coding: utf-8

# # PDFMiner
# 
# ## Overview
# ### Integration details
# 
# 
# | Class | Package | Local | Serializable | JS support|
# | :--- | :--- | :---: | :---: |  :---: |
# | [PDFMinerLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.PDFMinerLoader.html) | [langchain_community](https://python.langchain.com/api_reference/community/index.html) | ✅ | ❌ | ❌ | 
# ### Loader features
# | Source | Document Lazy Loading | Native Async Support
# | :---: | :---: | :---: | 
# | PDFMinerLoader | ✅ | ❌ | 
# 
# 
# ## Setup
# 
# ### Credentials
# 
# No credentials are needed for this loader.

# If you want to get automated best in-class tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:

# In[ ]:


# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"


# ### Installation
# 
# Install **langchain_community**.

# In[ ]:


get_ipython().run_line_magic('pip', 'install -qU langchain_community')


# ## Initialization
# 
# Now we can instantiate our model object and load documents:

# In[1]:


from langchain_community.document_loaders import PDFMinerLoader

file_path = "./example_data/layout-parser-paper.pdf"
loader = PDFMinerLoader(file_path)


# ## Load

# In[2]:


docs = loader.load()
docs[0]


# In[3]:


print(docs[0].metadata)


# ## Lazy Load

# In[4]:


page = []
for doc in loader.lazy_load():
    page.append(doc)
    if len(page) >= 10:
        # do some paged operation, e.g.
        # index.upsert(page)

        page = []


# ## Using PDFMiner to generate HTML text
# 
# This can be helpful for chunking texts semantically into sections as the output html content can be parsed via `BeautifulSoup` to get more structured and rich information about font size, page numbers, PDF headers/footers, etc.

# In[ ]:


from langchain_community.document_loaders import PDFMinerPDFasHTMLLoader

file_path = "./example_data/layout-parser-paper.pdf"
loader = PDFMinerPDFasHTMLLoader(file_path)
docs = loader.load()
docs[0]


# In[ ]:


from bs4 import BeautifulSoup

soup = BeautifulSoup(docs[0].page_content, "html.parser")
content = soup.find_all("div")


# In[ ]:


import re

cur_fs = None
cur_text = ""
snippets = []  # first collect all snippets that have the same font size
for c in content:
    sp = c.find("span")
    if not sp:
        continue
    st = sp.get("style")
    if not st:
        continue
    fs = re.findall("font-size:(\d+)px", st)
    if not fs:
        continue
    fs = int(fs[0])
    if not cur_fs:
        cur_fs = fs
    if fs == cur_fs:
        cur_text += c.text
    else:
        snippets.append((cur_text, cur_fs))
        cur_fs = fs
        cur_text = c.text
snippets.append((cur_text, cur_fs))
# Note: The above logic is very straightforward. One can also add more strategies such as removing duplicate snippets (as
# headers/footers in a PDF appear on multiple pages so if we find duplicates it's safe to assume that it is redundant info)


# In[ ]:


from langchain_core.documents import Document

cur_idx = -1
semantic_snippets = []
# Assumption: headings have higher font size than their respective content
for s in snippets:
    # if current snippet's font size > previous section's heading => it is a new heading
    if (
        not semantic_snippets
        or s[1] > semantic_snippets[cur_idx].metadata["heading_font"]
    ):
        metadata = {"heading": s[0], "content_font": 0, "heading_font": s[1]}
        metadata.update(docs[0].metadata)
        semantic_snippets.append(Document(page_content="", metadata=metadata))
        cur_idx += 1
        continue

    # if current snippet's font size <= previous section's content => content belongs to the same section (one can also create
    # a tree like structure for sub sections if needed but that may require some more thinking and may be data specific)
    if (
        not semantic_snippets[cur_idx].metadata["content_font"]
        or s[1] <= semantic_snippets[cur_idx].metadata["content_font"]
    ):
        semantic_snippets[cur_idx].page_content += s[0]
        semantic_snippets[cur_idx].metadata["content_font"] = max(
            s[1], semantic_snippets[cur_idx].metadata["content_font"]
        )
        continue

    # if current snippet's font size > previous section's content but less than previous section's heading than also make a new
    # section (e.g. title of a PDF will have the highest font size but we don't want it to subsume all sections)
    metadata = {"heading": s[0], "content_font": 0, "heading_font": s[1]}
    metadata.update(docs[0].metadata)
    semantic_snippets.append(Document(page_content="", metadata=metadata))
    cur_idx += 1

print(semantic_snippets[4])


# ## API reference
# 
# For detailed documentation of all PDFMinerLoader features and configurations head to the API reference: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.PDFMinerLoader.html
