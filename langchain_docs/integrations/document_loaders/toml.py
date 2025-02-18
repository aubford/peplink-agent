#!/usr/bin/env python
# coding: utf-8

# # TOML
#
# >[TOML](https://en.wikipedia.org/wiki/TOML) is a file format for configuration files. It is intended to be easy to read and write, and is designed to map unambiguously to a dictionary. Its specification is open-source. `TOML` is implemented in many programming languages. The name `TOML` is an acronym for "Tom's Obvious, Minimal Language" referring to its creator, Tom Preston-Werner.
#
# If you need to load `Toml` files, use the `TomlLoader`.

# In[1]:


from langchain_community.document_loaders import TomlLoader


# In[2]:


loader = TomlLoader("example_data/fake_rule.toml")


# In[3]:


rule = loader.load()


# In[4]:


rule


# In[ ]:
