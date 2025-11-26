"""
Milvus wrapper for RAG functionality.
提供检索器和相关工具函数。
"""

import os
from typing import List, Any, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from . import milvus_client
from .embedding import embedding_query  # Re-import the query embedding function


class MilvusRetriever(BaseRetriever):
    """
    A custom retriever for Milvus that is compatible with LangChain Expression Language (LCEL).
    """

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        """
        Embeds the query, retrieves documents from Milvus, and formats them.
        """
        try:
            # 1. Embed the user's query into a vector
            print(f"Embedding query: '{query}'...")
            query_vector = embedding_query(query)
            
            # 2. Perform a similarity search in Milvus using the vector
            current_file = kwargs.get("current_file", "")  # 当前正在查询的文件名
            
            # 如果指定了文件，使用元数据过滤
            search_params = {
                "collection_name": "company_info_primary_key",
                "data": query_vector,
                "limit": 15,  # 增加检索数量以获取更多候选结果
                "output_fields": ["text", "source"],
            }
            
            if current_file:
                # 使用 Milvus 的过滤功能直接在数据库层面过滤
                search_params["filter"] = f"source like '%{current_file}%'"
                print(f"🎯 使用数据库级过滤: source like '%{current_file}%'")
            
            search_res = milvus_client.search(**search_params)
            
            # 3. Process and format the search results
            documents = []
            
            print(f"🔍 检索到 {len(search_res[0]) if search_res and search_res[0] else 0} 个原始结果")
            
            if search_res and search_res[0]:
                for i, hit in enumerate(search_res[0]):
                    similarity = hit.get("distance", 0.0)
                    source = hit.get("entity", {}).get("source", "Unknown")
                    
                    print(f"  结果 {i+1}: source='{source}', similarity={similarity}")
                    
                    # 相似度过滤（距离越小越相似）
                    # 大幅放宽相似度阈值，让更多内容通过
                    if similarity > 1.2:  # 只过滤掉完全不相关的内容
                        print(f"    ❌ 相似度过滤: {similarity} > 1.2")
                        continue
                        
                    print(f"    ✅ 通过过滤器")
                        
                    doc = Document(
                        page_content=hit.get("entity", {}).get("text", ""),
                        metadata={
                            "source": source,
                            "similarity": similarity,
                        },
                    )
                    documents.append(doc)

            # 4. 按相似度排序并限制返回数量
            documents.sort(key=lambda x: x.metadata["similarity"])
            documents = documents[:8]  # 增加返回的文档数量，提供更多上下文
            
            print(f"Retrieved {len(documents)} relevant documents.")
            return documents
            
        except Exception as e:
            print(f"An error occurred during Milvus retrieval: {e}")
            return []

# Singleton instance of the retriever
_retriever_instance: Optional[MilvusRetriever] = None

def get_retriever() -> MilvusRetriever:
    """
    Returns a singleton instance of the MilvusRetriever.
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = MilvusRetriever()
    return _retriever_instance


def pre_process_doc(file_url: str) -> str:
    """
    预处理文档的占位符函数
    
    Args:
        file_url: 文档URL
        
    Returns:
        处理结果
    """
    print(f"开始处理文档: {file_url}")
    # 这里可以调用实际的文档处理逻辑
    return f"文档预处理完成: {file_url}"
