"""
Defines the document listing workflow.
This module is designed to be exposed as an API endpoint via `langgraph dev`.
Lists all unique source files in the knowledge base with their chunk counts.
"""
from typing import Dict, Any, List
from typing_extensions import TypedDict
from collections import Counter
from langgraph.graph import StateGraph, END
# 为了兼容 Python 3.11 版本添加
from typing_extensions import TypedDict

from rag import milvus_client

class GraphStateInput(TypedDict):
    """Input state for the document listing workflow."""
    pass  # No input needed for listing

class GraphState(GraphStateInput):
    """
    Represents the state of our graph.

    Attributes:
        files: List of files with their chunk counts
        error: A string to hold any error messages that occur
    """
    files: List[Dict[str, Any]]
    error: str

def list_docs_node(state: GraphState) -> Dict[str, Any]:
    """
    Lists all unique source files in the vector store with their chunk counts.
    
    Args:
        state: The current graph state.
        
    Returns:
        A dictionary containing the list of files and their statistics.
    """
    try:
        print("📋 开始查询知识库文件列表...")
        
        # 1. 查询所有文档的source字段
        results = milvus_client.query(
            collection_name="company_info_primary_key",
            filter="",  # 空字符串表示不过滤
            output_fields=["source"],
            limit=10000  # 设置较大的限制以获取所有文档
        )
        
        if not results:
            print("⚠️ 知识库中暂无文档")
            return {
                "files": [],
                "error": None
            }
        
        # 2. 统计每个source的出现次数
        source_counter = Counter(doc["source"] for doc in results)
        
        # 3. 构建文件列表
        files = [
            {
                "filename": source,
                "chunk_count": count,
            }
            for source, count in source_counter.items()
        ]
        
        # 4. 按文件名排序
        files.sort(key=lambda x: x["filename"])
        
        total_files = len(files)
        total_chunks = sum(f["chunk_count"] for f in files)
        print(f"✅ 查询完成，共有 {total_files} 个文件，{total_chunks} 个文档片段")
        
        return {
            "files": files,
            "total_files": total_files,
            "total_chunks": total_chunks,
            "error": None
        }
        
    except Exception as e:
        error_msg = f"❌ 查询文件列表失败: {str(e)}"
        print(error_msg)
        return {
            "files": [],
            "total_files": 0,
            "total_chunks": 0,
            "error": error_msg
        }

# Build and compile the graph
workflow = StateGraph(GraphState, input=GraphStateInput)
workflow.add_node("list_docs", list_docs_node)
workflow.set_entry_point("list_docs")
workflow.add_edge("list_docs", END)

# Compile the graph
doc_listing_workflow = workflow.compile()
print("✅ 文档列表查询工作流编译完成") 