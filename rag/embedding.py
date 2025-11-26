"""
向量生成模块
提供文本向量化功能
"""
from typing import List
import os
import time
import numpy as np
from dashscope import TextEmbedding
from dotenv import load_dotenv
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from Configurations import Configuration

# 加载.env文件
load_dotenv()

# 初始化API密钥
def _init_dashscope_api():
    """初始化阿里云DashScope API"""
    try:
        # 使用运行时配置
        from agents.persona_config.config_manager import config_manager
        cfg = config_manager.get_config() or {}
        # 获取API密钥
        api_key = cfg.get("dashscope_api_key") or os.getenv("DASHSCOPE_API_KEY")
        
        if not api_key:
            raise ValueError("未找到阿里云API密钥，请在.env文件中设置DASHSCOPE_API_KEY")
        
        print(f"✅ 阿里云DashScope API已初始化")
        class _Cfg:
            embedding_model = cfg.get("embedding_model", "text-embedding-v4")
            embedding_dimension = int(cfg.get("embedding_dimension", 768))
        return _Cfg()
        
    except Exception as e:
        print(f"❌ 初始化阿里云API失败: {e}")
        raise

def embedding_docs(documents: List[str], batch_size: int = 10) -> List[List[float]]:
    """
    使用阿里云text-embedding-v4生成文档向量
    
    Args:
        documents: 文档列表
        batch_size: 批处理大小，避免API限制
        
    Returns:
        向量列表
    """
    try:
        config = _init_dashscope_api()
        print(f"🧮 正在为 {len(documents)} 个文档生成向量...")
        print(f"📋 使用模型: {config.embedding_model}")
        print(f"📏 向量维度: {config.embedding_dimension}")
        
        embeddings = []
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            print(f"  处理批次 {batch_num}/{total_batches} ({len(batch)} 个文档)")
            
            # 调用阿里云API
            response = TextEmbedding.call(
                model=config.embedding_model,
                input=batch,
                text_type='document',
                dimension=config.embedding_dimension
            )
            
            if response.status_code == 200:
                batch_embeddings = response.output['embeddings']
                for embedding_item in batch_embeddings:
                    embeddings.append(embedding_item['embedding'])
                    
                print(f"  ✅ 批次 {batch_num} 处理完成")
            else:
                print(f"  ❌ 批次 {batch_num} 处理失败: {response.message}")
                raise Exception(f"API调用失败: {response.message}")
            
            # 添加延时避免API限制
            if i + batch_size < len(documents):
                time.sleep(0.1)
        
        print(f"✅ 完成 {len(documents)} 个文档的向量化")
        return embeddings
        
    except Exception as e:
        print(f"❌ 生成文档向量失败: {e}")
        raise

def embedding_query(query: str) -> List[List[float]]:
    """
    使用阿里云text-embedding-v4生成查询向量
    
    Args:
        query: 查询文本
        
    Returns:
        向量列表（兼容格式）
    """
    try:
        config = _init_dashscope_api()
        print(f"🔍 正在为查询生成向量: {query[:50]}...")
        
        # 调用阿里云API
        response = TextEmbedding.call(
            model=config.embedding_model,
            input=[query],
            text_type='query',
            dimension=config.embedding_dimension
        )
        
        if response.status_code == 200:
            embeddings = response.output['embeddings']
            query_vector = embeddings[0]['embedding']
            print(f"✅ 查询向量生成完成")
            return [query_vector]  # 返回列表格式保持兼容性
        else:
            print(f"❌ 查询向量生成失败: {response.message}")
            raise Exception(f"API调用失败: {response.message}")
        
    except Exception as e:
        print(f"❌ 生成查询向量失败: {e}")
        raise

def calculate_similarity(vector1: List[float], vector2: List[float]) -> float:
    """
    计算两个向量的余弦相似度
    
    Args:
        vector1: 第一个向量
        vector2: 第二个向量
        
    Returns:
        相似度分数 (-1 到 1)
    """
    try:
        # 转换为numpy数组
        v1 = np.array(vector1)
        v2 = np.array(vector2)
        
        # 计算余弦相似度
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        similarity = dot_product / (norm_v1 * norm_v2)
        return float(similarity)
        
    except Exception as e:
        print(f"❌ 计算相似度失败: {e}")
        return 0.0
