"""
文档注入知识库API
支持格式: PDF、Word、文本、图片、PPT、Excel等

使用方式:
1. 启动服务: langgraph dev
2. 调用API: 
   POST /doc_ingestion_agent
   {
       "file_urls": ["file_path_or_url"]
   }
"""
import os
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
# 为了兼容 Python 3.11 版本添加
from typing_extensions import TypedDict

from rag.embedding import embedding_docs
from rag.utils.rag_utils import download_doc, load_and_chunk_document
from rag import milvus_client

class BatchGraphStateInput(TypedDict):
    """
    API输入格式
    
    Example:
        {
            "file_urls": [
                "https://example.com/doc.pdf",
                "C:/Users/docs/file.docx",
                "/absolute/path/to/file.pdf",
                "relative/path/to/file.txt"
            ]
        }
    """
    file_urls: List[str]

class BatchGraphState(BatchGraphStateInput):
    """处理状态"""
    file_urls: List[str]
    processed_count: int
    failed_count: int
    error: str
    messages: List[str]

def check_document_exists(filename: str) -> bool:
    """检查文档是否已存在"""
    try:
        # 使用简单的字符串匹配，避免JSON格式问题
        results = milvus_client.query(
            collection_name="company_info_primary_key",
            filter=f"source like '%{filename}%'",
            output_fields=["source"],
            limit=1
        )
        return len(results) > 0
    except Exception as e:
        print(f"⚠️  检查文档存在性失败: {e}")
        return False

def delete_existing_document(filename: str) -> int:
    """删除已存在的文档"""
    try:
        results = milvus_client.query(
            collection_name="company_info_primary_key",
            filter=f"source like '%{filename}%'",
            output_fields=["source"],
            limit=1000
        )
        
        delete_count = len(results)
        if delete_count > 0:
            milvus_client.delete(
                collection_name="company_info_primary_key",
                filter=f"source like '%{filename}%'"
            )
        return delete_count
    except Exception as e:
        print(f"⚠️  删除文档失败: {e}")
        return 0

def batch_ingest_docs_node(state: BatchGraphState):
    """
    处理文档并注入知识库
    
    支持:
    1. 网络URL
    2. 本地绝对路径
    3. 本地相对路径
    4. 自动文档去重
    5. 多种文档格式
    6. 自动清理网络下载的临时文件
    """
    file_urls = state.get('file_urls', [])
    processed_count = 0
    failed_count = 0
    all_messages = []
    
    for i, file_url in enumerate(file_urls):
        local_path = None
        is_temp_file = False
        
        try:
            # 1. 获取本地文件路径
            # 检查输入是否为JSON格式的字符串
            if file_url.startswith('{') and file_url.endswith('}'):
                print(f"⚠️  检测到JSON格式输入，尝试解析: {file_url[:100]}...")
                import json
                try:
                    parsed_data = json.loads(file_url)
                    if 'file_urls' in parsed_data and len(parsed_data['file_urls']) > 0:
                        file_url = parsed_data['file_urls'][0]
                        print(f"✅ 从JSON中提取文件路径: {file_url}")
                except json.JSONDecodeError:
                    print(f"❌ JSON解析失败: {file_url}")
                    continue
            
            # 判断是否为网络文件
            is_temp_file = file_url.startswith(('http://', 'https://'))
            local_path = download_doc(file_url) if is_temp_file else file_url
            filename = os.path.basename(local_path)
            
            # 2. 检查并删除重复文档
            if check_document_exists(filename):
                deleted_count = delete_existing_document(filename)
                all_messages.append(f"文档已存在，已删除 {deleted_count} 个重复片段: {filename}")
            
            # 3. 加载并分块文档
            chunked_docs = load_and_chunk_document(local_path)
            
            # 4. 生成向量
            doc_vectors = embedding_docs([doc.page_content for doc in chunked_docs])
            
            # 5. 准备数据
            data = [
                {
                    "vector": doc_vectors[j],
                    "text": doc.page_content,
                    "source": doc.metadata.get("source", f"{filename}_chunk_{j}"),
                }
                for j, doc in enumerate(chunked_docs)
            ]

            # 6. 插入到Milvus
            milvus_client.insert(
                collection_name="company_info_primary_key",
                data=data,      
            )
            
            processed_count += 1
            all_messages.append(f"✅ 成功处理文档: {filename} ({len(data)} 个片段)")
            
        except Exception as e:
            failed_count += 1
            error_msg = f"处理失败 {file_url}: {str(e)}"
            all_messages.append(error_msg)
        
        finally:
            # 7. 清理网络下载的临时文件
            if is_temp_file and local_path and os.path.exists(local_path):
                try:
                    os.remove(local_path)
                    print(f"🗑️  已删除临时文件: {local_path}")
                except Exception as e:
                    print(f"⚠️  删除临时文件失败: {local_path}, 错误: {e}")
    
    return {
        "processed_count": processed_count,
        "failed_count": failed_count,
        "error": None if failed_count == 0 else "部分文档处理失败",
        "messages": all_messages
    }

# 构建工作流
workflow = StateGraph(BatchGraphState, input=BatchGraphStateInput)
workflow.add_node("batch_ingest_docs_node", batch_ingest_docs_node)
workflow.set_entry_point("batch_ingest_docs_node")
batch_doc_ingestion_workflow = workflow.compile()
