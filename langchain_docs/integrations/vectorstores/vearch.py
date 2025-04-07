#!/usr/bin/env python
# coding: utf-8

# # Vearch
# 
# >[Vearch](https://vearch.readthedocs.io) is the vector search infrastructure for deeping learning and AI applications.
# 

# ## Setting up
# 
# Follow [instructions](https://vearch.readthedocs.io/en/latest/quick-start-guide.html#).
# 
# You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration

# In[ ]:


get_ipython().run_line_magic('pip', 'install --upgrade --quiet  vearch')

# OR

get_ipython().run_line_magic('pip', 'install --upgrade --quiet  vearch_cluster')


# ## Example

# In[2]:


from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.vearch import Vearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModel, AutoTokenizer

# repalce to your local model path
model_path = "/data/zhx/zhx/langchain-ChatGLM_new/chatglm2-6b"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda(0)


# In[3]:


query = "你好!"
response, history = model.chat(tokenizer, query, history=[])
print(f"Human: {query}\nChatGLM:{response}\n")
query = "你知道凌波微步吗，你知道都有谁学会了吗?"
response, history = model.chat(tokenizer, query, history=history)
print(f"Human: {query}\nChatGLM:{response}\n")


# In[4]:


# Add your local knowledge files
file_path = "/data/zhx/zhx/langchain-ChatGLM_new/knowledge_base/天龙八部/lingboweibu.txt"  # Your local file path"
loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()

# split text into sentences and embedding the sentences
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# replace to your model path
embedding_path = "/data/zhx/zhx/langchain-ChatGLM_new/text2vec/text2vec-large-chinese"
embeddings = HuggingFaceEmbeddings(model_name=embedding_path)


# In[5]:


# first add your document into vearch vectorstore
vearch_standalone = Vearch.from_documents(
    texts,
    embeddings,
    path_or_url="/data/zhx/zhx/langchain-ChatGLM_new/knowledge_base/localdb_new_test",
    table_name="localdb_new_test",
    flag=0,
)

print("***************after is cluster res*****************")

vearch_cluster = Vearch.from_documents(
    texts,
    embeddings,
    path_or_url="http://test-vearch-langchain-router.vectorbase.svc.ht1.n.jd.local",
    db_name="vearch_cluster_langchian",
    table_name="tobenumone",
    flag=1,
)

# The vector data is usually already initialized, so we don’t need the document parameter and can directly create the object.
vearch_cluster_b = Vearch(
    embeddings,
    path_or_url="http://test-vearch-langchain-router.vectorbase.svc.ht1.n.jd.local",
    db_name="vearch_cluster_langchian",
    table_name="tobenumone",
    flag=1,
)


# In[6]:


query = "你知道凌波微步吗，你知道都有谁会凌波微步?"
# The second parameter is the top-n to retrieve, and its default value is 4.
vearch_standalone_res = vearch_standalone.similarity_search(query, 3)
for idx, tmp in enumerate(vearch_standalone_res):
    print(f"{'#'*20}第{idx+1}段相关文档{'#'*20}\n\n{tmp.page_content}\n")

# combine your local knowleadge and query
context = "".join([tmp.page_content for tmp in vearch_standalone_res])
new_query = f"基于以下信息，尽可能准确的来回答用户的问题。背景信息:\n {context} \n 回答用户这个问题:{query}\n\n"
response, history = model.chat(tokenizer, new_query, history=[])
print(f"********ChatGLM:{response}\n")

print("***************************after is cluster res******************************")

query_c = "你知道凌波微步吗，你知道都有谁会凌波微步?"
cluster_res = vearch_cluster.similarity_search(query_c, 3)
for idx, tmp in enumerate(cluster_res):
    print(f"{'#'*20}第{idx+1}段相关文档{'#'*20}\n\n{tmp.page_content}\n")

# In practical applications, we usually limit the boundary value of similarity. The following method can set this value.
cluster_res_with_bound = vearch_cluster.similarity_search_with_score(
    query=query_c, k=3, min_score=0.5
)

# combine your local knowleadge and query
context_c = "".join([tmp.page_content for tmp in cluster_res])
new_query_c = f"基于以下信息，尽可能准确的来回答用户的问题。背景信息:\n {context_c} \n 回答用户这个问题:{query_c}\n\n"
response_c, history_c = model.chat(tokenizer, new_query_c, history=[])
print(f"********ChatGLM:{response_c}\n")


# In[7]:


query = "你知道vearch是什么吗?"
response, history = model.chat(tokenizer, query, history=history)
print(f"Human: {query}\nChatGLM:{response}\n")

vearch_info = [
    "Vearch 是一款存储大语言模型数据的向量数据库，用于存储和快速搜索模型embedding后的向量，可用于基于个人知识库的大模型应用",
    "Vearch 支持OpenAI, Llama, ChatGLM等模型，以及LangChain库",
    "vearch 是基于C语言,go语言开发的，并提供python接口，可以直接通过pip安装",
]
vearch_source = [
    {
        "source": "/data/zhx/zhx/langchain-ChatGLM_new/knowledge_base/tlbb/three_body.txt"
    },
    {
        "source": "/data/zhx/zhx/langchain-ChatGLM_new/knowledge_base/tlbb/three_body.txt"
    },
    {
        "source": "/data/zhx/zhx/langchain-ChatGLM_new/knowledge_base/tlbb/three_body.txt"
    },
]
vearch_standalone.add_texts(vearch_info, vearch_source)

print("*****************after is cluster res********************")

vearch_cluster.add_texts(vearch_info, vearch_source)


# In[8]:


query3 = "你知道vearch是什么吗?"
res1 = vearch_standalone.similarity_search(query3, 3)
for idx, tmp in enumerate(res1):
    print(f"{'#'*20}第{idx+1}段相关文档{'#'*20}\n\n{tmp.page_content}\n")

context1 = "".join([tmp.page_content for tmp in res1])
new_query1 = f"基于以下信息，尽可能准确的来回答用户的问题。背景信息:\n {context1} \n 回答用户这个问题:{query3}\n\n"
response, history = model.chat(tokenizer, new_query1, history=[])
print(f"***************ChatGLM:{response}\n")

print("***************after is cluster res******************")

query3_c = "你知道vearch是什么吗?"
res1_c = vearch_standalone.similarity_search(query3_c, 3)
for idx, tmp in enumerate(res1_c):
    print(f"{'#'*20}第{idx+1}段相关文档{'#'*20}\n\n{tmp.page_content}\n")

context1_C = "".join([tmp.page_content for tmp in res1_c])
new_query1_c = f"基于以下信息，尽可能准确的来回答用户的问题。背景信息:\n {context1_C} \n 回答用户这个问题:{query3_c}\n\n"
response_c, history_c = model.chat(tokenizer, new_query1_c, history=[])

print(f"***************ChatGLM:{response_c}\n")


# In[9]:


##delete and get function need to maintain  docids
##your docid

res_d = vearch_standalone.delete(
    [
        "eee5e7468434427eb49829374c1e8220",
        "2776754da8fc4bb58d3e482006010716",
        "9223acd6d89d4c2c84ff42677ac0d47c",
    ]
)
print("delete vearch standalone docid", res_d)
query = "你知道vearch是什么吗?"
response, history = model.chat(tokenizer, query, history=[])
print(f"Human: {query}\nChatGLM:{response}\n")

res_cluster = vearch_cluster.delete(
    ["-4311783201092343475", "-2899734009733762895", "1342026762029067927"]
)
print("delete vearch cluster docid", res_cluster)
query_c = "你知道vearch是什么吗?"
response_c, history = model.chat(tokenizer, query_c, history=[])
print(f"Human: {query}\nChatGLM:{response_c}\n")


get_delet_doc = vearch_standalone.get(
    [
        "eee5e7468434427eb49829374c1e8220",
        "2776754da8fc4bb58d3e482006010716",
        "9223acd6d89d4c2c84ff42677ac0d47c",
    ]
)
print("after delete docid to query again:", get_delet_doc)
get_id_doc = vearch_standalone.get(
    [
        "18ce6747dca04a2c833e60e8dfd83c04",
        "aafacb0e46574b378a9f433877ab06a8",
        "9776bccfdd8643a8b219ccee0596f370",
        "9223acd6d89d4c2c84ff42677ac0d47c",
    ]
)
print("get existed docid", get_id_doc)

get_delet_doc = vearch_cluster.get(
    ["-4311783201092343475", "-2899734009733762895", "1342026762029067927"]
)
print("after delete docid to query again:", get_delet_doc)
get_id_doc = vearch_cluster.get(
    [
        "1841638988191686991",
        "-4519586577642625749",
        "5028230008472292907",
        "1342026762029067927",
    ]
)
print("get existed docid", get_id_doc)


# In[ ]:





# In[ ]:





# In[ ]:




