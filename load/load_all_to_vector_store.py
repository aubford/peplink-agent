from load.mongo.mongo_load import MongoLoad
from load.youtube.youtube_load import YoutubeLoad
from load.html.html_load import HtmlLoad
from load.reddit.reddit_load import RedditLoad
from load.reddit_general.reddit_general_load import RedditGeneralLoad

mongo_loader = MongoLoad()
youtube_loader = YoutubeLoad()
html_loader = HtmlLoad()
reddit_loader = RedditLoad()
reddit_general_loader = RedditGeneralLoad()


def load_all_to_vector_store(alt_column: str | None = None):
    print("Loading MongoDB data to vector store...")
    mongo_loader.staging_to_vector_store(alt_column=alt_column)

    print("Loading YouTube data to vector store...")
    youtube_loader.staging_to_vector_store(alt_column=alt_column)

    print("Loading HTML data to vector store...")
    html_loader.staging_to_vector_store(alt_column=alt_column)

    print("Loading Reddit data to vector store...")
    reddit_loader.staging_to_vector_store(alt_column=alt_column)

    print("Loading Reddit General data to vector store...")
    reddit_general_loader.staging_to_vector_store(alt_column=alt_column)


if __name__ == "__main__":
    load_all_to_vector_store(alt_column="technical_summary_embedding")
