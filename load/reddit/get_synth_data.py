# %%
from load.reddit.reddit_load import RedditLoad

loader = RedditLoad()

# %%  Inspact batchfile
batchfile, messages = loader.batch_manager.get_batchfile()

# %% Create batch job
loader.batch_manager.create_batch_job()

# %% Check batch job status
loader.batch_manager.check_batch_and_get_results()

# %% Create synth data and apply to staging
loader.batch_manager.create_synth_data()
loader.apply_generated_data_to_staging()

# %% Uncomment to load to vector store
# loader.staging_to_vector_store()
