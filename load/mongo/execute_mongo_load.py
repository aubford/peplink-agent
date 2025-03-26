from load.mongo.mongo_load import MongoLoad

if __name__ == "__main__":
    loader = MongoLoad()
    loader.load()