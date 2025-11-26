"""
测试RAG文档处理
使用方式: python -m rag.entry <file_path>
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rag.workflows.doc_ingestion import batch_doc_ingestion_workflow

def test_doc_process(file_path: str):
    """测试处理单个文档"""
    try:
        result = batch_doc_ingestion_workflow.invoke({
            "file_urls": [file_path]
        })
        print(f"\n处理结果:")
        print(f"成功: {result.get('processed_count', 0)}")
        print(f"失败: {result.get('failed_count', 0)}")
        if result.get('messages'):
            print("\n处理日志:")
            for msg in result['messages']:
                print(f"- {msg}")
    except Exception as e:
        print(f"处理出错: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方式: python -m rag.entry <file_path>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    print(f"测试文件: {file_path}")
    test_doc_process(file_path)
