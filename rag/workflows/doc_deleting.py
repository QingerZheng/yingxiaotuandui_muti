"""
Defines the document deletion workflow.
This module is designed to be exposed as an API endpoint via `langgraph dev`.
"""
from typing import List, TypedDict, Dict, Any
# 为了兼容 Python 3.11 版本添加
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

from rag import milvus_client

class GraphStateInput(TypedDict):
    """Input state for the document deletion workflow."""
    filter: str

class GraphState(GraphStateInput):
    """
    Represents the state of our graph.

    Attributes:
        filter: The filter expression to identify documents to delete.
        deleted_count: Number of documents deleted.
        error: A string to hold any error messages that occur.
    """
    filter: str
    deleted_count: int
    error: str

def delete_docs_node(state: GraphState) -> Dict[str, Any]:
    """
    Deletes documents from the vector store based on the filter expression.
    
    Args:
        state: The current graph state containing the filter expression.
        
    Returns:
        A dictionary with the deletion result or an error.
    """
    try:
        filter_expr = state["filter"]
        print(f"🗑️  开始删除文档，过滤条件: {filter_expr}")
        
        # 1. 查询要删除的文档数量
        results = milvus_client.query(
            collection_name="company_info_primary_key",
            filter=filter_expr,
            output_fields=["source"],
            limit=1000  # 限制查询数量
        )
        
        if not results:
            print("⚠️  未找到匹配的文档")
            return {
                "deleted_count": 0,
                "error": None
            }
            
        # 2. 执行删除操作
        milvus_client.delete(
            collection_name="company_info_primary_key",
            filter=filter_expr
        )
        
        deleted_count = len(results)
        print(f"✅ 成功删除 {deleted_count} 个文档片段")
        
        return {
            "deleted_count": deleted_count,
            "error": None
        }
        
    except Exception as e:
        error_msg = f"❌ 删除文档失败: {str(e)}"
        print(error_msg)
        return {
            "deleted_count": 0,
            "error": error_msg
        }

# Build and compile the graph
workflow = StateGraph(GraphState, input=GraphStateInput)
workflow.add_node("delete_docs", delete_docs_node)
workflow.set_entry_point("delete_docs")
workflow.add_edge("delete_docs", END)

# Compile the graph
doc_deleting_workflow = workflow.compile()
print("✅ 文档删除工作流编译完成") 