# %%
from load.reddit.reddit_load import RedditLoad

r_loader = RedditLoad()
# %%
r_loader.batch_manager.check_batch_and_get_results()
r_loader.create_synth_data_from_batch_results()

# %%
r_loader.apply_synth_data_to_staging()

# %% Uncomment to load to vector store
# loader.staging_to_vector_store()
