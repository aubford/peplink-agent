# %%

from pathlib import Path
from transform.reddit.reddit_transform import RedditTransform
from transform.mongo.mongo_pepwave_transform import MongoPepwaveTransform
import pandas as pd
from IPython.display import display

pd.set_option("display.max_columns", None)


mongo_artifacts = MongoPepwaveTransform.get_artifacts()
df = mongo_artifacts[0]

df.info()

# Get max length of page_content
max_length = df["page_content"].str.len().max()
print(f"Max length of page_content: {max_length}")

# max length in words
max_length_words = df["page_content"].str.split().str.len().max()
print(f"Max length of page_content in words: {max_length_words}")


# %%
# Get the row with the longest page_content
longest_doc = df.loc[df["page_content"].str.len() == max_length]
print(f"ID: {longest_doc['id'].iloc[0]}")

print("\nLongest document:")
print("-" * 80)
print(f"Length in chars: {len(longest_doc['page_content'].iloc[0])}")
print(f"Length in words: {len(longest_doc['page_content'].iloc[0].split())}")
print("\nContent:")
print(longest_doc["page_content"].iloc[0])

# %%
import json

transformer = MongoPepwaveTransform()
reply_tree = transformer.build_reply_tree(longest_doc["id"].iloc[0])

# Save reply tree to JSON file in current directory
current_dir = Path(__file__).parent
output_path = current_dir / "analyze_output.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(reply_tree, f, indent=2, ensure_ascii=False)

print(f"\nReply tree saved to {output_path}")


# %% RANDOM ROW ###################


# Get a random row
random_doc = df.sample(n=1)
random_doc_id = random_doc["id"]
print(f"ID: {random_doc_id.squeeze()}")

transformer = MongoPepwaveTransform()
reply_tree = transformer.build_reply_tree(random_doc_id.squeeze())

current_dir = Path(__file__).parent
output_path = current_dir / "analyze_output.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(reply_tree, f, indent=2, ensure_ascii=False)

print(f"\nReply tree saved to {output_path}")


# %% API CALLS ###################
# api_url = "https://api.peplink.com/t/peplinks-social-media-post-2023/42561"

product_discussion_topic_id = "65df8fbb6f9b0a6e516202da"
api_url = f"https://forum.peplink.com/api/v1/post/{product_discussion_topic_id}/reply?limit=1000&sort_by=createdAt&order_by=asc&replyNo=1"


import requests

response = requests.get(api_url)
print(response.text)


# %%
from pymongo import MongoClient


def find_duplicate_replies():
    # Initialize MongoDB client
    client = MongoClient("mongodb://localhost:27017")
    db = client["pepwave"]
    posts_collection = db["posts"]

    # Aggregation pipeline to find posts with shared replyTo values
    pipeline = [
        # Only include replies that aren't direct replies to topics
        {
            "$match": {
                "replyTo": {"$exists": True},
                "$expr": {"$ne": ["$replyTo", "$topicId"]},
            }
        },
        # Group by replyTo and count occurrences
        {
            "$group": {
                "_id": "$replyTo",
                "count": {"$sum": 1},
                "posts": {
                    "$push": {
                        "id": "$_id",
                        "content": "$postContent",
                        "creator": "$creator.name",
                    }
                },
            }
        },
        # Filter for groups with more than 1 post
        {"$match": {"count": {"$gt": 2}}},
        # Sort by count descending
        {"$sort": {"count": -1}},
    ]

    results = list(posts_collection.aggregate(pipeline))
    return results


# Execute query and print results
duplicate_replies = find_duplicate_replies()
print(f"Found {len(duplicate_replies)} multi-replies")
for result in duplicate_replies:
    print(f"\nReplyTo: {result['_id']}")
    print(f"Number of replies: {result['count']}")
    for post in result["posts"]:
        print(f"- Post ID: {post['id']} by {post['creator']}")
