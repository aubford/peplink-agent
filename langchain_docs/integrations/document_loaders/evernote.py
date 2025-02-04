#!/usr/bin/env python
# coding: utf-8

# # EverNote
# 
# >[EverNote](https://evernote.com/) is intended for archiving and creating notes in which photos, audio and saved web content can be embedded. Notes are stored in virtual "notebooks" and can be tagged, annotated, edited, searched, and exported.
# 
# This notebook shows how to load an `Evernote` [export](https://help.evernote.com/hc/en-us/articles/209005557-Export-notes-and-notebooks-as-ENEX-or-HTML) file (.enex) from disk.
# 
# A document will be created for each note in the export.

# In[1]:


# lxml and html2text are required to parse EverNote notes
get_ipython().run_line_magic('pip', 'install --upgrade --quiet  lxml')
get_ipython().run_line_magic('pip', 'install --upgrade --quiet  html2text')


# In[2]:


from langchain_community.document_loaders import EverNoteLoader

# By default all notes are combined into a single Document
loader = EverNoteLoader("example_data/testing.enex")
loader.load()


# In[3]:


# It's likely more useful to return a Document for each note
loader = EverNoteLoader("example_data/testing.enex", load_single_document=False)
loader.load()

