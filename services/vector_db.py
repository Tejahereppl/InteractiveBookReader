import logging
from typing import List, Dict, Any, Optional
from uuid import UUID




from langchain.docstore.document import Document
from pinecone import Pinecone, ServerlessSpec
import pinecone
import os
from config.settings import settings_obj
from models.story import Story, StoryChunk

logger = logging.getLogger(__name__)

class VectorDBService:
    """Service for interacting with vector databases"""
    
    def __init__(self):
        self.db_type = settings_obj.VECTOR_DB_TYPE
        self.pc = self._initialize_vector_db()
        self.index_name = "interactive-story"  # Define index name
        self.index = self._get_index()
    
    def _initialize_vector_db(self):
        """Initialize the vector database based on configuration"""

        pinecone.init(api_key=settings_obj.PINECONE_API_KEY, environment=settings_obj.PINECONE_ENVIRONMENT)

    def _get_index(self):
        """Retrieve an existing Pinecone index"""
        return pinecone.Index(self.index_name) 

    def get_vector_store(self, namespace: str):
        """Get a vector store with a specified namespace"""
        return {
            "index": self.index,  
            "namespace": namespace  
        }
    
    def _chunks_to_documents(self, story_id: UUID, chunks: List[StoryChunk]) -> List[Document]:
        """Convert StoryChunk models to LangChain Document objects"""
        documents = []
        
        for chunk in chunks:
            
            doc = Document(
                page_content=chunk.page_content,
                metadata={
                    "story_id": str(story_id),
                    "chunk_id": str(chunk.id),
                    "page_number": chunk.page_number,
                    
                    
                }
            )
            documents.append(doc)
        
        return documents
    
    async def store_story_chunks(self, story: Story, chunks: List[StoryChunk]) -> str:
        """Store story chunks in the vector database"""
        try:
            # Convert chunks to documents
            documents = self._chunks_to_documents(story.id, chunks)
            
            # Create a collection name based on the story ID
            collection_name = f"story_{story.id}"
            
            # Get vector store for this collection
            vector_store = self._get_vector_store(collection_name)
            
            # Add documents to the vector store
            vector_store.add_documents(documents)
            
            
            
            logger.info(f"Stored {len(documents)} chunks for story {story.id} in vector DB")
            
            return collection_name
            
        except Exception as e:
            logger.error(f"Error storing story chunks in vector DB: {e}")
            raise
    
    async def similarity_search(
        self, 
        story_id: UUID, 
        query: str, 
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search for similar documents in the vector database"""
        try:
            # Get the collection for this story
            collection_name = f"story_{story_id}"
            vector_store = self._get_vector_store(collection_name)
            
            # Add default filter for story_id if not provided
            if filter_metadata is None:
                filter_metadata = {}
            if "story_id" not in filter_metadata:
                filter_metadata["story_id"] = str(story_id)
            
            # Search with filter
            
            results = vector_store.similarity_search(
                    query, 
                    k=k,
                    filter=filter_metadata
                )
            
            
            logger.info(f"Found {len(results)} similar chunks for query in story {story_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching in vector DB: {e}")
            raise
    
    async def delete_story(self, story_id: UUID) -> bool:
        """Delete a story's chunks from the vector database"""
        try:
            collection_name = f"story_{story_id}"
            
            if self.db_type == "pinecone":
                index = self.pc.Index("interactive-story")
                index.delete(delete_all=True, namespace=collection_name)
            else:  
                # Get the collection and delete it
                vector_store = self._get_vector_store(collection_name)
                vector_store.delete_collection()
                vector_store.persist()
            
            logger.info(f"Deleted vector DB collection for story {story_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting story from vector DB: {e}")
            return False