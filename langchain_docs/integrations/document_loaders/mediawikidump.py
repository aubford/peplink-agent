#!/usr/bin/env python
# coding: utf-8

# # MediaWiki Dump
# 
# >[MediaWiki XML Dumps](https://www.mediawiki.org/wiki/Manual:Importing_XML_dumps) contain the content of a wiki (wiki pages with all their revisions), without the site-related data. A XML dump does not create a full backup of the wiki database, the dump does not contain user accounts, images, edit logs, etc.
# 
# This covers how to load a MediaWiki XML dump file into a document format that we can use downstream.
# 
# It uses `mwxml` from `mediawiki-utilities` to dump and `mwparserfromhell` from `earwig` to parse MediaWiki wikicode.
# 
# Dump files can be obtained with dumpBackup.php or on the Special:Statistics page of the Wiki.

# In[ ]:


# mediawiki-utilities supports XML schema 0.11 in unmerged branches
get_ipython().run_line_magic('pip', 'install --upgrade --quiet git+https://github.com/mediawiki-utilities/python-mwtypes@updates_schema_0.11')
# mediawiki-utilities mwxml has a bug, fix PR pending
get_ipython().run_line_magic('pip', 'install --upgrade --quiet git+https://github.com/gdedrouas/python-mwxml@xml_format_0.11')
get_ipython().run_line_magic('pip', 'install --upgrade --quiet mwparserfromhell')


# In[3]:


from langchain_community.document_loaders import MWDumpLoader


# In[4]:


loader = MWDumpLoader(
    file_path="example_data/testmw_pages_current.xml",
    encoding="utf8",
    # namespaces = [0,2,3] Optional list to load only specific namespaces. Loads all namespaces by default.
    skip_redirects=True,  # will skip over pages that just redirect to other pages (or not if False)
    stop_on_error=False,  # will skip over pages that cause parsing errors (or not if False)
)
documents = loader.load()
print(f"You have {len(documents)} document(s) in your data ")


# In[7]:


documents[:5]


# In[ ]:




