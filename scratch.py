from pinecone import Pinecone, ServerlessSpec
from pinecone.data.index import Index
from dotenv import load_dotenv
import os

load_dotenv()


pc = Pinecone(api_key="daffb271-0749-4d8a-a757-2391cbc54e77")
index = pc.Index("pepwave-early-april")

stats = index.describe_index_stats()
print(stats)

for ids in index.list():
    print(ids)
