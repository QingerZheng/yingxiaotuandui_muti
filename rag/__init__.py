"""
RAG (Retrieval Augmented Generation) System for MAS
Migrated from {{}}

Provides complete document lifecycle management:
- Document ingestion from URLs with chunking and embedding
- Document query with semantic search and retrieval
- Document deletion with filter-based removal
- Milvus cloud (Zilliz) vector database integration
- Gemini embeddings for high-quality vector representations
"""

import os
from pymilvus import MilvusClient
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 检查必要的环境变量
required_env_vars = {
    "ZILLIZ_BASE_URL": "Milvus/Zilliz 服务地址",
    "ZILLIZ_API_KEY": "Milvus/Zilliz API密钥",
}

optional_env_vars = {
    "NODE_MODEL": ("语言模型名称", "gpt-4o-mini"),
    "DASHSCOPE_API_KEY": ("阿里云 DashScope API密钥", None),
}

# 检查必需的环境变量
missing_vars = []
for var, desc in required_env_vars.items():
    if not os.getenv(var):
        missing_vars.append(f"{var} ({desc})")

if missing_vars:
    raise ValueError(f"缺少必要的环境变量:\n" + "\n".join(missing_vars))

# 设置可选的环境变量默认值
for var, (desc, default) in optional_env_vars.items():
    if not os.getenv(var) and default is not None:
        os.environ[var] = default
        print(f"⚠️ 环境变量 {var} ({desc}) 未设置，使用默认值: {default}")

# Milvus Cloud (Zilliz) Configuration
ZILLIZ_BASE_URL = os.environ["ZILLIZ_BASE_URL"]
ZILLIZ_TOKEN = os.environ["ZILLIZ_API_KEY"]

# Milvus Client Singleton
try:
    milvus_client = MilvusClient(
        uri=ZILLIZ_BASE_URL,
        token=ZILLIZ_TOKEN,
    )
    print("✅ Milvus 客户端初始化成功")
except Exception as e:
    print(f"❌ Milvus 客户端初始化失败: {e}")
    raise
