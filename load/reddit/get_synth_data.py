# %%
from load.reddit.reddit_load import RedditLoad

loader = RedditLoad()

# %% Create batch job
loader.batch_manager.create_batch_job()

# %% Check batch job status
loader.batch_manager.check_batch_and_get_results()

# %% Create synth data and apply to staging
loader.batch_manager.create_synth_data_from_batch_results()

# %%
loader.apply_synth_data_to_staging()

# %% Uncomment to load to vector store
# loader.staging_to_vector_store()
