"""
RAG Workflows Package
Contains LangGraph workflows for document lifecycle management

Workflows:
- doc_ingestion: Batch document processing workflow (Download → Chunk → Embed → Store)
- doc_query: Query → Retrieve → Generate workflow  
- doc_deleting: Filter → Delete workflow
"""

from .doc_ingestion import batch_doc_ingestion_workflow
from .doc_deleting import doc_deleting_workflow
from .doc_query import rag_query_workflow, rag_answer

__all__ = [
    "batch_doc_ingestion_workflow",
    "doc_deleting_workflow",
    "rag_query_workflow",
    "rag_answer"
] 