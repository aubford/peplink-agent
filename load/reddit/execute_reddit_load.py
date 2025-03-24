# %%
from load.reddit.reddit_load import RedditLoad

loader = RedditLoad()
if __name__ == "__main__":
    loader.load()


# %%

batchfile, messages = loader.batch_manager.get_batchfile()
print(messages[0][0])
print(messages[0][1])

# %%

loader.batch_manager.create_batch_job()

# %%

# Uncomment to load to vector store
# loader.staging_to_vector_store()
