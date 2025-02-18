#!/usr/bin/env python
# coding: utf-8

# # Vsdx

# > A [visio file](https://fr.wikipedia.org/wiki/Microsoft_Visio) (with extension .vsdx) is associated with Microsoft Visio, a diagram creation software. It stores information about the structure, layout, and graphical elements of a diagram. This format facilitates the creation and sharing of visualizations in areas such as business, engineering, and computer science.

# A Visio file can contain multiple pages. Some of them may serve as the background for others, and this can occur across multiple layers. This **loader** extracts the textual content from each page and its associated pages, enabling the extraction of all visible text from each page, similar to what an OCR algorithm would do.

# **WARNING** : Only Visio files with the **.vsdx** extension are compatible with this loader. Files with extensions such as .vsd, ... are not compatible because they cannot be converted to compressed XML.

# In[ ]:


from langchain_community.document_loaders import VsdxLoader


# In[2]:


loader = VsdxLoader(file_path="./example_data/fake.vsdx")
documents = loader.load()


# **Display loaded documents**

# In[3]:


for i, doc in enumerate(documents):
    print(f"\n------ Page {doc.metadata['page']} ------")
    print(f"Title page : {doc.metadata['page_name']}")
    print(f"Source : {doc.metadata['source']}")
    print("\n==> CONTENT <== ")
    print(doc.page_content)
