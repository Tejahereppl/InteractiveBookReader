import logging
from typing import List
from uuid import UUID
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends

from models.story import Story, Character, UploadResponse
from services.pdf_service import PDFService
from services.vector_db import VectorDBService
from services.llm_service import LLMService
from langchain.docstore.document import Document

logger = logging.getLogger(__name__)

router = APIRouter()

stories = {}
characters = {}
scenarios = {}

def get_pdf_service():
    return PDFService()

def get_vector_db_service():
    return VectorDBService()

def get_llm_service(vector_db_service: VectorDBService = Depends(get_vector_db_service)):
    return LLMService(vector_db_service)

@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    pdf_service: PDFService = Depends(get_pdf_service),
    vector_db_service: VectorDBService = Depends(get_vector_db_service),
    llm_service: LLMService = Depends(get_llm_service)
):
    """Upload a PDF book to create an interactive story"""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Read file content
        file_content = await file.read()
        
        # Process the PDF
        story, story_chunks = await pdf_service.process_pdf(file_content, file.filename)
        
        # Store in vector database
        collection_name = await vector_db_service.store_story_chunks(story, story_chunks)
        
        # Update story with vector store ID
        story.vector_store_id = collection_name
        
        # Store in our in-memory storage
        stories[story.id] = story
        
        # Basic response to return immediately
        response = UploadResponse(
            story_id=story.id,
            title=story.metadata.title,
            message=f"Book '{story.metadata.title}' uploaded successfully. Processing content...",
            characters=[]
        )
        
        # Add background task to analyze content and extract characters
        background_tasks.add_task(
            process_story_content,
            story,
            story_chunks,
            llm_service
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error uploading PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_story_content(story: Story, story_chunks, llm_service: LLMService):
    """Background task to process story content"""
    try:
        # Convert story chunks to documents for LLM processing
        
        documents = [
            Document(
                page_content=chunk.page_content,
                metadata={
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index
                }
            )
            for chunk in story_chunks
        ]
        
        # Analyze content to extract characters, setting, plot
        analysis = await llm_service.analyze_story_content(story, documents)
        
        # Update story with extracted information
        story_obj = stories.get(story.id)
        if story_obj:
            story_obj.characters = analysis.get("characters", [])
            story_obj.setting = analysis.get("setting", "")
            story_obj.main_plot = analysis.get("plot", "")
            
            # Store characters for easy lookup
            for character in story_obj.characters:
                characters[character.id] = character
            
            # Generate scenarios
            story_scenarios = await llm_service.generate_story_scenarios(
                story_obj, 
                story_obj.characters
            )
            
            # Store scenarios
            for scenario in story_scenarios:
                scenarios[scenario.id] = scenario
        
        logger.info(f"Completed background processing for story {story.id}")
        
    except Exception as e:
        logger.error(f"Error in background processing for story {story.id}: {e}")

@router.get("/stories", response_model=List[Story])
async def list_stories():
    """List all uploaded stories"""
    return list(stories.values())

@router.get("/stories/{story_id}", response_model=Story)
async def get_story(story_id: UUID):
    """Get a specific story by ID"""
    if story_id not in stories:
        raise HTTPException(status_code=404, detail="Story not found")
    return stories[story_id]

@router.get("/stories/{story_id}/characters", response_model=List[Character])
async def get_story_characters(story_id: UUID):
    """Get characters for a specific story"""
    if story_id not in stories:
        raise HTTPException(status_code=404, detail="Story not found")
    return stories[story_id].characters