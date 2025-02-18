#!/usr/bin/env python
# coding: utf-8

# # NetworkX
#
# >[NetworkX](https://networkx.org/) is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.
#
# This notebook goes over how to do question answering over a graph data structure.

# ## Setting up
#
# We have to install a Python package.

# In[ ]:


get_ipython().run_line_magic("pip", "install --upgrade --quiet  networkx")


# ## Create the graph
#
# In this section, we construct an example graph. At the moment, this works best for small pieces of text.

# In[1]:


from langchain_community.graphs.index_creator import GraphIndexCreator
from langchain_openai import OpenAI


# In[2]:


index_creator = GraphIndexCreator(llm=OpenAI(temperature=0))


# In[3]:


with open("../../../how_to/state_of_the_union.txt") as f:
    all_text = f.read()


# We will use just a small snippet, because extracting the knowledge triplets is a bit intensive at the moment.

# In[4]:


text = "\n".join(all_text.split("\n\n")[105:108])


# In[5]:


text


# In[6]:


graph = index_creator.from_text(text)


# We can inspect the created graph.

# In[7]:


graph.get_triples()


# ## Querying the graph
# We can now use the graph QA chain to ask question of the graph

# In[8]:


from langchain.chains import GraphQAChain


# In[9]:


chain = GraphQAChain.from_llm(OpenAI(temperature=0), graph=graph, verbose=True)


# In[10]:


chain.run("what is Intel going to build?")


# ## Save the graph
# We can also save and load the graph.

# In[7]:


graph.write_to_gml("graph.gml")


# In[8]:


from langchain_community.graphs import NetworkxEntityGraph


# In[9]:


loaded_graph = NetworkxEntityGraph.from_gml("graph.gml")


# In[10]:


loaded_graph.get_triples()


# In[ ]:


loaded_graph.get_number_of_nodes()


# In[ ]:


loaded_graph.add_node("NewNode")


# In[ ]:


loaded_graph.has_node("NewNode")


# In[ ]:


loaded_graph.remove_node("NewNode")


# In[ ]:


loaded_graph.get_neighbors("Intel")


# In[ ]:


loaded_graph.has_edge("Intel", "Silicon Valley")


# In[ ]:


loaded_graph.remove_edge("Intel", "Silicon Valley")


# In[ ]:


loaded_graph.clear_edges()


# In[ ]:


loaded_graph.clear()
