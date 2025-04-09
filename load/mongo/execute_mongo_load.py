from load.mongo.mongo_load import MongoLoad

loader = MongoLoad()
if __name__ == "__main__":
    # loader.load()
    # loader.batch_manager.create_batch_job()
    # loader.append_primary_content_embeddings_to_staging_file()
    loader.staging_to_vector_store()
