# %%
import json
from pydantic import BaseModel, Field, ValidationError
import pandas as pd
from typing import List
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from googleapiclient.discovery import build
from langchain_community.document_loaders.youtube import YoutubeLoader
from VideoItem import VideoItem

load_dotenv()  # take environment variables from .env

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["USER_AGENT"] = os.getenv("USER_AGENT")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini")


# %%
def get_channel_id(api_key, username):
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Search for the channel by username
    request = youtube.search().list(
        part='snippet',
        q=username,
        type='channel',
        maxResults=1
    )
    response = request.execute()

    # Extract the channel ID from the response
    if response['items']:
        channel_id = response['items'][0]['snippet']['channelId']
        return channel_id
    else:
        return None


channel_id = get_channel_id(YOUTUBE_API_KEY, "@peplink")
print(channel_id)


# %%debug
def fetch_videos_with_transcripts(api_key, channel_id) -> List[VideoItem]:
    youtube_client = build("youtube", "v3", developerKey=api_key)
    output = []

    # Get uploads playlist ID for the channel
    request = youtube_client.channels().list(part="contentDetails", id=channel_id)
    response = request.execute()
    uploads_playlist_id = response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

    # Fetch all video IDs from the uploads playlist
    next_page_token = None
    while True:
        playlist_request = youtube_client.playlistItems().list(
            part="contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=50,
            pageToken=next_page_token,
        )
        playlist_response = playlist_request.execute()
        for item in playlist_response["items"]:
            video_id = item["contentDetails"]["videoId"]

            # Check video privacy and caption status
            video_request = youtube_client.videos().list(
                part="status,contentDetails,snippet,statistics,topicDetails,localizations,player,recordingDetails",
                id=video_id
            )
            video_response = video_request.execute()
            if video_response["items"]:
                video_item = video_response["items"][0]
                is_public = video_item["status"]["privacyStatus"] == "public"

                if is_public:
                    try:
                        loader = YoutubeLoader(video_id=video_id)
                        docs = loader.load()
                        # Add transcript content to video item
                        video_item["transcript"] = docs[0].page_content

                        video = VideoItem.model_validate(video_item)
                        output.append(video)

                    except Exception as e:
                        print(f"Could not load transcript for video {video_id}")
                        continue


        next_page_token = playlist_response.get("nextPageToken")
        if not next_page_token:
            break
    return output


videos_with_transcripts = fetch_videos_with_transcripts(YOUTUBE_API_KEY, channel_id)
dicts = [video.model_dump() for video in videos_with_transcripts]

# Write videos to JSON file
with open("youtube_videos.json", "w", encoding="utf-8") as f:
    json.dump(dicts, f, ensure_ascii=False, indent=2)

# Write videos to parquet file
df = pd.DataFrame(dicts)
df.to_parquet("youtube_videos.parquet")
