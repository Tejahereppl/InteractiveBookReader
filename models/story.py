from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict



class Character(BaseModel):
    """Model representing a character in a story"""
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str
    traits: List[str] = Field(default_factory=list)
    relationships: Dict[str, str] = Field(default_factory=dict)

class StoryMetadata(BaseModel):
    """Model representing story metadata"""
    title: str
    author: str
    publication_date: Optional[str] = None
    genre: Optional[str] = None
    summary: Optional[str] = None

class Story(BaseModel):
    """Model representing a complete story"""
    id: UUID = Field(default_factory=uuid4)
    metadata: Optional[StoryMetadata] = None

    characters: List[Character] = Field(default_factory=list)
    setting: Optional[str] = None
    main_plot: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    source_pdf: str
    vector_store_id: Optional[str] = None

class StoryChunk(BaseModel):
    """Model representing a chunk of text from a story"""
    id: UUID = Field(default_factory=uuid4)
    story_id: UUID
    page_content: str
    page_number: Optional[int] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
    

class ChatMessage(BaseModel):
    """Model representing a message in a chat"""
    role: str  # "user", "assistant", or "system"
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

class ChatSession(BaseModel):
    """Model representing a chat session"""
    id: UUID = Field(default_factory=uuid4)
    story_id: UUID
    character_id: UUID
    user_name: str
    messages: List[ChatMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class StoryScenario(BaseModel):
    """Model representing a generated scenario for the story"""
    id: UUID = Field(default_factory=uuid4)
    story_id: UUID
    title: str
    description: str
    character_prompts: Dict[str, str] = Field(default_factory=dict)
    system_prompt: str
    created_at: datetime = Field(default_factory=datetime.now)

# Request and Response Models
class UploadResponse(BaseModel):
    """Response after uploading a PDF"""
    story_id: UUID
    title: str
    message: str
    characters: List[Character] = Field(default_factory=list)

class ChatRequest(BaseModel):
    """Request to chat with a character"""
    story_id: UUID
    character_id: UUID
    user_name: str
    message: str
    session_id: Optional[UUID] = None

class ChatResponse(BaseModel):
    """Response from a chat"""
    session_id: UUID
    message: str
    character_name: str