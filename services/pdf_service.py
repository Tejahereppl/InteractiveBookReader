import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from uuid import UUID, uuid4
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import logging


import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config.settings import settings_obj
from models.story import Story, StoryMetadata, StoryChunk, Character

logger = logging.getLogger(__name__)


class PDFProcessor:
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

    def load_pdf(self, pdf_path: Path) -> List:

        """Load PDF and extract text with metadata"""
        try:
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
            logger.info(f"Extracted text from {len(documents)} pages in {pdf_path}")
            return documents
        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            return []

    def chunk_text(self, documents: List) -> List:
        """Chunk extracted text for efficient vector search"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} pages")
        return chunks

    def extract_metadata(self, documents: List) -> Dict[str, Any]:
        """Extract basic metadata like title and author"""
        first_page_text = documents[0].page_content if documents else "Unknown"

        title, author = "Unknown Title", "Unknown Author"
        lines = first_page_text.split("\n")
        for i, line in enumerate(lines):
            line = line.strip()
            if line and len(line) < 100:  # Simple heuristic for title
                if i < len(lines) - 1 and "by" in lines[i+1].lower():
                    title = line
                    author = lines[i+1].replace("by", "").strip()
                    break

        metadata = {
            "title": title,
            "author": author,
            "total_pages": len(documents)
        }
        logger.info(f"Extracted metadata: {metadata}")
        return metadata



    def create_story_model(self, pdf_path: Path, chunks: List[Dict[str, Any]], metadata: StoryMetadata) -> Tuple[Story, List[StoryChunk]]:
        """Create a Story model and StoryChunks from the extracted content"""
        story_id = uuid4()
        
        story = Story(
            id=story_id,
            metadata=metadata,
            source_pdf=str(pdf_path),
        )
        
        story_chunks = []
        for chunk in chunks:
            story_chunk = StoryChunk(
                story_id=story_id,
                content=chunk["content"],
                page_number=chunk["page_number"],
                chunk_index=chunk["chunk_index"],
                metadata=chunk["metadata"]
            )
            story_chunks.append(story_chunk)
        
        return story, story_chunks
    
    async def process_pdf(self, file_content: bytes, filename: str) -> Tuple[Story, List[StoryChunk]]:
        """Process a PDF file and return a Story model with its chunks"""
        
        pdf_path = await self.save_uploaded_file(file_content, filename)
        
        # Extract text
        text_pages = self.extract_text_from_pdf(pdf_path)
        
        # Extract metadata
        metadata = self.extract_metadata(text_pages)
        
        # Chunk text
        chunks = self.chunk_text(text_pages)
        
        # Create Story model
        story, story_chunks = self.create_story_model(pdf_path, chunks, metadata)
        
        return story, story_chunks
pdf_processor = PDFProcessor()
pdf_path = r"C:\Users\tejas\Downloads\_OceanofPDF.com_The_Body_Keeps_the_Score_Brain_Mind_and_Body_in_the_Healing_of_Trauma_-_Bessel_van_der_Kolk.pdf" 
documents = pdf_processor.load_pdf(pdf_path)
'''chunks = pdf_processor.chunk_text(documents)
metadata = pdf_processor.extract_metadata(documents)'''