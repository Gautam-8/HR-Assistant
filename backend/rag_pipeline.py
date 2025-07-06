import os
from typing import List, Dict, Any, Optional
try:
    import google.generativeai as genai
except ImportError:
    genai = None
from langchain_core.documents import Document

from config import config
from backend.document_processor import DocumentProcessor
from backend.postgres_vector_store import PostgreSQLVectorStore

class RAGPipeline:
    def __init__(self):
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.vector_store = PostgreSQLVectorStore()
        
        # Configure Gemini
        if genai:
            genai.configure(api_key=config.GOOGLE_API_KEY) # type: ignore
            self.model = genai.GenerativeModel('gemini-1.5-flash') # type: ignore
        else:
            raise ImportError("google-generativeai package not installed")
        
    def upload_and_process_document(self, file_content: bytes, filename: str, 
                                  document_type: str = "policy") -> Dict[str, Any]:
        """Upload and process a document into the knowledge base"""
        try:
            # Process document
            process_result = self.document_processor.process_document(
                file_content, filename, document_type
            )
            
            if not process_result['success']:
                return process_result
            
            # Add to vector store
            vector_result = self.vector_store.add_documents(process_result['chunks'])
            
            if not vector_result['success']:
                return {
                    'success': False,
                    'error': vector_result['error'],
                    'message': f"Document processed but failed to add to vector store: {vector_result['error']}"
                }
            
            return {
                'success': True,
                'filename': filename,
                'total_chunks': process_result['total_chunks'],
                'document_type': document_type,
                'message': f"Successfully uploaded and processed {filename}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"Failed to upload and process document: {str(e)}"
            }
    
    def query_knowledge_base(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Query the knowledge base and generate response"""
        try:
            # Categorize the query
            category = self.vector_store.categorize_query(query)
            
            # Search for relevant documents
            relevant_docs = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            
            if not relevant_docs:
                return {
                    'success': True,
                    'answer': "I couldn't find relevant information in the knowledge base to answer your question. Please try rephrasing your query or contact HR directly.",
                    'sources': [],
                    'category': category
                }
            
            # Prepare context from relevant documents
            context = self._prepare_context(relevant_docs)
            
            # Generate response using Gemini
            response = self._generate_response(query, context, category)
            
            # Extract sources
            sources = self._extract_sources(relevant_docs)
            
            return {
                'success': True,
                'answer': response,
                'sources': sources,
                'category': category,
                'relevant_chunks': len(relevant_docs)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"Failed to query knowledge base: {str(e)}"
            }
    
    def _prepare_context(self, relevant_docs: List[tuple]) -> str:
        """Prepare context from relevant documents"""
        context_parts = []
        
        for i, (doc, score) in enumerate(relevant_docs):
            # Include document content and metadata
            source_info = f"Source: {doc.metadata.get('source_file', 'Unknown')}"
            if 'page' in doc.metadata:
                source_info += f", Page: {doc.metadata['page']}"
            
            context_parts.append(f"Document {i+1}:\n{source_info}\nContent: {doc.page_content}\n")
        
        return "\n".join(context_parts)
    
    def _generate_response(self, query: str, context: str, category: str) -> str:
        """Generate response using Gemini"""
        # Create HR-specific prompt
        prompt = f"""You are an HR Knowledge Assistant. Answer the employee's question based on the provided company documents.

Instructions:
1. Answer based ONLY on the provided context
2. Be helpful, professional, and accurate
3. If the context doesn't contain enough information, say so
4. Include relevant policy details and procedures
5. For leave/benefits questions, mention any deadlines or requirements
6. Always cite the source document when providing specific information

Query Category: {category}
Employee Question: {query}

Context from Company Documents:
{context}

HR Assistant Response:"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"I apologize, but I encountered an error while generating a response: {str(e)}. Please try again or contact HR directly."
    
    def _extract_sources(self, relevant_docs: List[tuple]) -> List[Dict[str, Any]]:
        """Extract source information from relevant documents"""
        sources = []
        
        for doc, score in relevant_docs:
            source = {
                'filename': doc.metadata.get('source_file', 'Unknown'),
                'document_type': doc.metadata.get('document_type', 'policy'),
                'relevance_score': float(score),
                'page': doc.metadata.get('page', 'N/A'),
                'chunk_id': doc.metadata.get('chunk_id', 'N/A')
            }
            sources.append(source)
        
        return sources
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        try:
            collection_info = self.vector_store.get_collection_info()
            
            # Get document files info
            documents_path = config.DOCUMENTS_PATH
            document_files = []
            
            if os.path.exists(documents_path):
                for file in os.listdir(documents_path):
                    if file.endswith(('.pdf', '.docx', '.txt')):
                        file_path = os.path.join(documents_path, file)
                        file_info = self.document_processor.get_document_info(file_path)
                        document_files.append(file_info)
            
            return {
                'success': True,
                'vector_store_info': collection_info,
                'document_files': document_files,
                'total_files': len(document_files)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"Failed to get knowledge base stats: {str(e)}"
            }
    
    def search_by_category(self, category: str, limit: int = 10) -> List[Document]:
        """Search documents by category"""
        try:
            metadata_filter = {'document_type': category}
            return self.vector_store.search_by_metadata(metadata_filter, limit)
        except Exception as e:
            print(f"Error searching by category: {str(e)}")
            return [] 