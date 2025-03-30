# %%
from load.mongo.mongo_load import MongoLoad

loader = MongoLoad()
output_file = loader.batch_manager.output_file_name


def merge_batch_results() -> None:
    """Merge all batch_results_*.jsonl files into a single JSONL file."""
    batch_files = sorted(loader.batch_manager.batch_path.glob("batch_results_*.jsonl"))

    with open(output_file, 'w') as outfile:
        outfile.write('[')
        is_first = True
        for batch_file in batch_files:
            with open(batch_file) as infile:
                for line in infile:
                    if not is_first:
                        outfile.write(',')
                    else:
                        is_first = False
                    outfile.write('\n' + line.strip())
        outfile.write('\n]')

merge_batch_results()

# %%
loader.create_synth_data_from_batch_results()
loader.apply_synth_data_to_staging()

# %% Uncomment to load to vector store
# loader.staging_to_vector_store()
