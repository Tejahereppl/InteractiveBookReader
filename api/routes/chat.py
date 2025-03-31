import logging
from typing import List, Dict, Optional
from uuid import UUID
from fastapi import APIRouter, HTTPException, Depends

from models.story import ChatRequest, ChatResponse, ChatSession
from services.vector_db import VectorDBService
from services.llm_service import LLMService
from api.routes.upload import get_vector_db_service, get_llm_service, stories, characters

logger = logging.getLogger(__name__)

router = APIRouter()

chat_sessions = {}

@router.post("/chat", response_model=ChatResponse)
async def chat_with_character(
    chat_request: ChatRequest,
    vector_db_service: VectorDBService = Depends(get_vector_db_service),
    llm_service: LLMService = Depends(get_llm_service)
):
    """Chat with a character from a story"""
    try:
        # Validate story exists
        if chat_request.story_id not in stories:
            raise HTTPException(status_code=404, detail="Story not found")
        
        story = stories[chat_request.story_id]
        
        # Validate character exists
        if chat_request.character_id not in characters:
            raise HTTPException(status_code=404, detail="Character not found")
        
        character = characters[chat_request.character_id]
        
        # Get or create chat session
        session = None
        if chat_request.session_id:
            session = chat_sessions.get(chat_request.session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Chat session not found")
        
        # Generate response
        session = await llm_service.chat_as_character(
            story=story,
            character=character,
            user_name=chat_request.user_name,
            message=chat_request.message,
            session=session
        )
        
        # Store or update the session
        chat_sessions[session.id] = session
        
        # Get the last message from the character
        last_message = session.messages[-1].content if session.messages else ""
        
        return ChatResponse(
            session_id=session.id,
            message=last_message,
            character_name=character.name
        )
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat/sessions/{session_id}", response_model=ChatSession)
async def get_chat_session(session_id: UUID):
    """Get a specific chat session by ID"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return chat_sessions[session_id]

@router.get("/chat/history/{session_id}", response_model=List[Dict])
async def get_chat_history(session_id: UUID):
    """Get the chat history for a specific session"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    session = chat_sessions[session_id]
    
    # Format messages for the response
    history = []
    for msg in session.messages:
        history.append({
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp
        })
    
    return history