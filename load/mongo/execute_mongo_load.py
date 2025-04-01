from load.mongo.mongo_load import MongoLoad

loader = MongoLoad()
if __name__ == "__main__":
    loader.load()
    # loader.batch_manager.create_batch_job()
