from pydantic import BaseModel
from typing import Optional

class Thumbnail(BaseModel):
    url: str
    width: int
    height: int

class Snippet(BaseModel):
    publishedAt: str
    channelId: str
    title: str
    description: str
    channelTitle: str
    categoryId: str
    liveBroadcastContent: str

class ContentDetails(BaseModel):
    duration: str
    dimension: str
    definition: str
    caption: str
    licensedContent: bool
    projection: str

class Status(BaseModel):
    uploadStatus: str
    privacyStatus: str
    license: str
    embeddable: bool
    publicStatsViewable: bool
    madeForKids: bool

class Statistics(BaseModel):
    viewCount: str
    favoriteCount: str
    likeCount: Optional[str] = None
    commentCount: Optional[str] = None

class Player(BaseModel):
    embedHtml: str

class VideoItemMetadata(BaseModel):
    id: str
    kind: str
    etag: str
    snippet: Snippet
    contentDetails: ContentDetails
    status: Status
    statistics: Statistics

class VideoItem(BaseModel):
    page_content: str
    metadata: VideoItemMetadata
