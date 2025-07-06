from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import os

from config import config
from backend.rag_pipeline import RAGPipeline

# Initialize FastAPI app
app = FastAPI(
    title=config.APP_NAME,
    version=config.APP_VERSION,
    description="HR Knowledge Assistant API for document management and employee queries"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5

class QueryResponse(BaseModel):
    success: bool
    answer: Optional[str] = None
    sources: Optional[List[dict]] = None
    category: Optional[str] = None
    relevant_chunks: Optional[int] = None
    error: Optional[str] = None
    message: Optional[str] = None

class DocumentUploadResponse(BaseModel):
    success: bool
    filename: Optional[str] = None
    total_chunks: Optional[int] = None
    document_type: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None

# API Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "HR Knowledge Assistant API",
        "version": config.APP_VERSION,
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if vector store is accessible
        stats = rag_pipeline.get_knowledge_base_stats()
        return {
            "status": "healthy",
            "vector_store": "accessible" if stats['success'] else "error",
            "timestamp": "now"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "now"
        }

@app.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    document_type: str = Form(default="policy")
):
    """Upload and process HR document"""
    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.docx', '.doc', '.txt'}
        
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
            
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        # Validate file size
        file_content = await file.read()
        file_size_mb = len(file_content) / (1024 * 1024)
        
        if file_size_mb > config.MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"File size ({file_size_mb:.1f}MB) exceeds maximum allowed size ({config.MAX_FILE_SIZE_MB}MB)"
            )
        
        # Process document
        result = rag_pipeline.upload_and_process_document(
            file_content=file_content,
            filename=file.filename,
            document_type=document_type
        )
        
        if not result['success']:
            raise HTTPException(status_code=500, detail=result['message'])
        
        return DocumentUploadResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """Query the HR knowledge base"""
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Process query
        result = rag_pipeline.query_knowledge_base(
            query=request.query,
            k=request.max_results or 5
        )
        
        return QueryResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/stats")
async def get_knowledge_base_stats():
    """Get knowledge base statistics"""
    try:
        stats = rag_pipeline.get_knowledge_base_stats()
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    try:
        documents_path = config.DOCUMENTS_PATH
        documents = []
        
        if os.path.exists(documents_path):
            for file in os.listdir(documents_path):
                if file.endswith(('.pdf', '.docx', '.doc', '.txt')):
                    file_path = os.path.join(documents_path, file)
                    file_info = rag_pipeline.document_processor.get_document_info(file_path)
                    documents.append(file_info)
        
        return {
            "success": True,
            "documents": documents,
            "total_count": len(documents)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/categories")
async def get_query_categories():
    """Get available query categories"""
    categories = {
        'benefits': 'Employee benefits, insurance, retirement plans',
        'leave': 'Vacation, sick leave, maternity/paternity leave',
        'conduct': 'Code of conduct, harassment policies, ethics',
        'compensation': 'Salary, bonuses, promotions, pay structure',
        'remote_work': 'Remote work policies, flexible arrangements',
        'onboarding': 'New employee orientation and training',
        'general': 'General HR policies and procedures'
    }
    
    return {
        "success": True,
        "categories": categories
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=config.DEBUG) 