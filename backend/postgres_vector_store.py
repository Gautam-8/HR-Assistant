import os
import uuid
import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
from sqlalchemy import create_engine, text, Column, String, Integer, DateTime, Text, ARRAY, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import psycopg2
from sentence_transformers import SentenceTransformer

from langchain_core.documents import Document
from config import config

Base = declarative_base()

class DocumentChunk(Base):
    __tablename__ = 'document_chunks'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content = Column(Text, nullable=False)
    embedding = Column(ARRAY(Float), nullable=False)
    metadata = Column(Text)  # JSON string
    source_file = Column(String(255))
    document_type = Column(String(50))
    chunk_id = Column(Integer)
    total_chunks = Column(Integer)
    chunk_size = Column(Integer)
    page_number = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

class PostgreSQLVectorStore:
    def __init__(self):
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Create database connection
        self.engine = create_engine(config.POSTGRES_URL) # type: ignore
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize database with pgvector extension and tables"""
        try:
            # Create pgvector extension
            with self.engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                conn.commit()
            
            # Create tables
            Base.metadata.create_all(bind=self.engine)
            
            print("Database initialized successfully")
            
        except Exception as e:
            print(f"Database initialization error: {e}")
            raise
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist() # type: ignore
        except Exception as e:
            print(f"Embedding generation error: {e}")
            return [0.0] * self.embedding_dimension # type: ignore
    
    def add_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """Add documents to vector store"""
        try:
            db = self.SessionLocal()
            document_ids = []
            
            for doc in documents:
                # Generate embedding
                embedding = self._get_embedding(doc.page_content)
                
                # Create document chunk
                chunk = DocumentChunk(
                    content=doc.page_content,
                    embedding=embedding,
                    metadata=str(doc.metadata),
                    source_file=doc.metadata.get('source_file', ''),
                    document_type=doc.metadata.get('document_type', 'general'),
                    chunk_id=doc.metadata.get('chunk_id', 0),
                    total_chunks=doc.metadata.get('total_chunks', 1),
                    chunk_size=doc.metadata.get('chunk_size', len(doc.page_content)),
                    page_number=doc.metadata.get('page', 1)
                )
                
                db.add(chunk)
                document_ids.append(str(chunk.id))
            
            db.commit()
            db.close()
            
            return {
                'success': True,
                'added_count': len(documents),
                'document_ids': document_ids,
                'message': f"Successfully added {len(documents)} documents to vector store"
            }
            
        except Exception as e:
            if 'db' in locals():
                db.rollback()
                db.close()
            return {
                'success': False,
                'error': str(e),
                'message': f"Failed to add documents: {str(e)}"
            }
    
    def similarity_search(self, query: str, k: int = 5, 
                         filter_dict: Optional[Dict] = None) -> List[Document]:
        """Search for similar documents"""
        try:
            # Generate query embedding
            query_embedding = self._get_embedding(query)
            
            db = self.SessionLocal()
            
            # Build query with vector similarity
            query_sql = text(f"""
                SELECT id, content, metadata, source_file, document_type, chunk_id, 
                       total_chunks, chunk_size, page_number,
                       embedding <-> CAST(:query_embedding AS vector) AS distance
                FROM document_chunks
                {"WHERE document_type = :doc_type" if filter_dict and 'document_type' in filter_dict else ""}
                ORDER BY embedding <-> CAST(:query_embedding AS vector)
                LIMIT :limit
            """)
            
            params = {
                'query_embedding': query_embedding,
                'limit': k
            }
            
            if filter_dict and 'document_type' in filter_dict:
                params['doc_type'] = filter_dict['document_type']
            
            result = db.execute(query_sql, params)
            rows = result.fetchall()
            
            # Convert to Document objects
            documents = []
            for row in rows:
                metadata = eval(row.metadata) if row.metadata else {}
                metadata.update({
                    'source_file': row.source_file,
                    'document_type': row.document_type,
                    'chunk_id': row.chunk_id,
                    'total_chunks': row.total_chunks,
                    'chunk_size': row.chunk_size,
                    'page': row.page_number
                })
                
                doc = Document(
                    page_content=row.content,
                    metadata=metadata
                )
                documents.append(doc)
            
            db.close()
            return documents
            
        except Exception as e:
            if 'db' in locals():
                db.close()
            print(f"Error in similarity search: {str(e)}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 5,
                                   filter_dict: Optional[Dict] = None) -> List[tuple]:
        """Search for similar documents with relevance scores"""
        try:
            # Generate query embedding
            query_embedding = self._get_embedding(query)
            
            db = self.SessionLocal()
            
            # Build query with vector similarity
            query_sql = text(f"""
                SELECT id, content, metadata, source_file, document_type, chunk_id, 
                       total_chunks, chunk_size, page_number,
                       embedding <-> CAST(:query_embedding AS vector) AS distance
                FROM document_chunks
                {"WHERE document_type = :doc_type" if filter_dict and 'document_type' in filter_dict else ""}
                ORDER BY embedding <-> CAST(:query_embedding AS vector)
                LIMIT :limit
            """)
            
            params = {
                'query_embedding': query_embedding,
                'limit': k
            }
            
            if filter_dict and 'document_type' in filter_dict:
                params['doc_type'] = filter_dict['document_type']
            
            result = db.execute(query_sql, params)
            rows = result.fetchall()
            
            # Convert to Document objects with scores
            documents_with_scores = []
            for row in rows:
                metadata = eval(row.metadata) if row.metadata else {}
                metadata.update({
                    'source_file': row.source_file,
                    'document_type': row.document_type,
                    'chunk_id': row.chunk_id,
                    'total_chunks': row.total_chunks,
                    'chunk_size': row.chunk_size,
                    'page': row.page_number
                })
                
                doc = Document(
                    page_content=row.content,
                    metadata=metadata
                )
                
                # Convert distance to similarity score (lower distance = higher similarity)
                score = 1.0 / (1.0 + row.distance)
                documents_with_scores.append((doc, score))
            
            db.close()
            return documents_with_scores
            
        except Exception as e:
            if 'db' in locals():
                db.close()
            print(f"Error in similarity search with score: {str(e)}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the vector store collection"""
        try:
            db = self.SessionLocal()
            
            # Count total documents
            count_result = db.execute(text("SELECT COUNT(*) FROM document_chunks"))
            document_count = count_result.scalar()
            
            # Get document types
            types_result = db.execute(text("SELECT DISTINCT document_type FROM document_chunks"))
            document_types = [row[0] for row in types_result.fetchall()]
            
            db.close()
            
            return {
                'collection_name': 'hr_documents',
                'document_count': document_count,
                'document_types': document_types
            }
            
        except Exception as e:
            if 'db' in locals():
                db.close()
            return {
                'error': str(e),
                'message': "Failed to get collection info"
            }
    
    def delete_documents(self, document_ids: List[str]) -> Dict[str, Any]:
        """Delete documents from vector store"""
        try:
            db = self.SessionLocal()
            
            # Delete documents by IDs
            for doc_id in document_ids:
                db.execute(text("DELETE FROM document_chunks WHERE id = :id"), {'id': doc_id})
            
            db.commit()
            db.close()
            
            return {
                'success': True,
                'deleted_count': len(document_ids),
                'message': f"Successfully deleted {len(document_ids)} documents"
            }
            
        except Exception as e:
            if 'db' in locals():
                db.rollback()
                db.close()
            return {
                'success': False,
                'error': str(e),
                'message': f"Failed to delete documents: {str(e)}"
            }
    
    def search_by_metadata(self, metadata_filter: Dict[str, Any], 
                          limit: int = 10) -> List[Document]:
        """Search documents by metadata"""
        try:
            db = self.SessionLocal()
            
            # Build WHERE clause based on metadata filter
            where_conditions = []
            params = {'limit': limit}
            
            if 'document_type' in metadata_filter:
                where_conditions.append("document_type = :doc_type")
                params['doc_type'] = metadata_filter['document_type']
            
            if 'source_file' in metadata_filter:
                where_conditions.append("source_file = :source_file")
                params['source_file'] = metadata_filter['source_file']
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            query_sql = text(f"""
                SELECT content, metadata, source_file, document_type, chunk_id, 
                       total_chunks, chunk_size, page_number
                FROM document_chunks
                WHERE {where_clause}
                LIMIT :limit
            """)
            
            result = db.execute(query_sql, params)
            rows = result.fetchall()
            
            # Convert to Document objects
            documents = []
            for row in rows:
                metadata = eval(row.metadata) if row.metadata else {}
                metadata.update({
                    'source_file': row.source_file,
                    'document_type': row.document_type,
                    'chunk_id': row.chunk_id,
                    'total_chunks': row.total_chunks,
                    'chunk_size': row.chunk_size,
                    'page': row.page_number
                })
                
                doc = Document(
                    page_content=row.content,
                    metadata=metadata
                )
                documents.append(doc)
            
            db.close()
            return documents
            
        except Exception as e:
            if 'db' in locals():
                db.close()
            print(f"Error in metadata search: {str(e)}")
            return []
    
    def categorize_query(self, query: str) -> str:
        """Categorize HR query into different types"""
        query_lower = query.lower()
        
        # Define categories and keywords
        categories = {
            'benefits': ['benefit', 'insurance', 'health', 'dental', 'vision', 'retirement', '401k', 'pension'],
            'leave': ['leave', 'vacation', 'sick', 'maternity', 'paternity', 'holiday', 'pto', 'time off'],
            'conduct': ['conduct', 'behavior', 'harassment', 'discrimination', 'ethics', 'policy', 'violation'],
            'compensation': ['salary', 'pay', 'wage', 'bonus', 'raise', 'promotion', 'compensation'],
            'remote_work': ['remote', 'work from home', 'telecommute', 'flexible', 'hybrid'],
            'onboarding': ['onboarding', 'orientation', 'new hire', 'first day', 'training'],
            'general': []
        }
        
        # Check for category keywords
        for category, keywords in categories.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        
        return 'general' 