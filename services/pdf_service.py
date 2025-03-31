import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from uuid import UUID, uuid4
import os
from pathlib import Path



import fitz  
from langchain.text_splitter import RecursiveCharacterTextSplitter


from models.story import Story, StoryMetadata, StoryChunk, Character

logger = logging.getLogger(__name__)


class PDFService:
    def __init__(self, upload_dir: str = "./uploads", chunk_size: int = 500, chunk_overlap: int = 100):
        self.upload_dir = Path(upload_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        os.makedirs(self.upload_dir, exist_ok=True)

    async def save_uploaded_file(self, file_content: bytes, filename: str) -> Path:
        """Save an uploaded PDF file to disk"""
        file_path = self.upload_dir / filename
        with open(file_path, "wb") as f:
            f.write(file_content)
        logger.info(f"Saved file: {file_path}")
        return file_path

    def extract_text_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract text from PDF using PyMuPDF (fitz)"""
        documents = []
        try:
            doc = fitz.open(str(pdf_path))
            for page_num, page in enumerate(doc):
                text = page.get_text()
                print(text)
                documents.append({
                    "page_content": text,
                    "page_number": page_num + 1,
                    
                })
            logger.info(f"Extracted text from {len(documents)} pages in {pdf_path}")
            return documents
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return []

    


    



    def create_story_model(self, pdf_path: Path, chunks: List[Dict[str, Any]]) -> Tuple[Story, List[StoryChunk]]:
        """Create a Story model and StoryChunks from the extracted content"""
        story_id = uuid4()
        
        story = Story(
            id=story_id,
            source_pdf=str(pdf_path),
        )
        
        story_chunks = []
        for chunk in chunks:
            story_chunk = StoryChunk(
                story_id=story_id,
                page_content=chunk["page_content"],
                page_number=chunk["page_number"],
               
              
            )
            story_chunks.append(story_chunk)
        
        return story, story_chunks
    
    async def process_pdf(self, file_content: bytes, filename: str) -> Tuple[Story, List[StoryChunk]]:
        """Process a PDF file and return a Story model with its chunks"""
    
        pdf_path = await self.save_uploaded_file(file_content, filename)
        text_pages = self.extract_text_from_pdf(pdf_path)
        
        chunks = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        for page in text_pages:
            page_chunks = text_splitter.split_text(page["page_content"])  # 🔹 Chunking happens here
            for i, chunk_text in enumerate(page_chunks):
                chunks.append({
                    "page_content": chunk_text,
                    "page_number": page["page_number"],
                   
                })  

        story, story_chunks = self.create_story_model(pdf_path, chunks)
        
        return story, story_chunks
