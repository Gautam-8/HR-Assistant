import os
import hashlib
from typing import List, Dict, Any
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document

from config import config

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Ensure documents directory exists
        os.makedirs(config.DOCUMENTS_PATH, exist_ok=True)
    
    def save_uploaded_file(self, file_content: bytes, filename: str) -> str:
        """Save uploaded file and return file path"""
        # Generate unique filename to avoid conflicts
        file_hash = hashlib.md5(file_content).hexdigest()[:8]
        name, ext = os.path.splitext(filename)
        unique_filename = f"{name}_{file_hash}{ext}"
        
        file_path = os.path.join(config.DOCUMENTS_PATH, unique_filename)
        
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        return file_path
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load document based on file type"""
        file_extension = Path(file_path).suffix.lower()
        
        try:
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            elif file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            documents = loader.load()
            return documents
        
        except Exception as e:
            raise Exception(f"Error loading document {file_path}: {str(e)}")
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        chunks = self.text_splitter.split_documents(documents)
        
        # Add metadata to chunks
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'chunk_id': i,
                'total_chunks': len(chunks),
                'chunk_size': len(chunk.page_content)
            })
        
        return chunks
    
    def process_document(self, file_content: bytes, filename: str, 
                        document_type: str = "policy") -> Dict[str, Any]:
        """Complete document processing pipeline"""
        try:
            # Save the file
            file_path = self.save_uploaded_file(file_content, filename)
            
            # Load and process document
            documents = self.load_document(file_path)
            chunks = self.chunk_documents(documents)
            
            # Add document-level metadata
            for chunk in chunks:
                chunk.metadata.update({
                    'source_file': filename,
                    'file_path': file_path,
                    'document_type': document_type,
                    'total_pages': len(documents) if hasattr(documents[0], 'metadata') else 1
                })
            
            return {
                'success': True,
                'file_path': file_path,
                'total_chunks': len(chunks),
                'chunks': chunks,
                'message': f"Successfully processed {filename} into {len(chunks)} chunks"
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"Failed to process {filename}: {str(e)}"
            }
    
    def get_document_info(self, file_path: str) -> Dict[str, Any]:
        """Get information about a processed document"""
        try:
            documents = self.load_document(file_path)
            
            return {
                'filename': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path),
                'total_pages': len(documents),
                'file_type': Path(file_path).suffix.lower(),
                'last_modified': os.path.getmtime(file_path)
            }
        
        except Exception as e:
            return {'error': str(e)} 