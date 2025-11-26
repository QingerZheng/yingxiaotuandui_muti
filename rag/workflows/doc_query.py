"""
Defines the RAG (Retrieval-Augmented Generation) question-answering workflow.
This module is designed to be exposed as an API endpoint via `langgraph dev`.
"""
from typing import List, TypedDict
# 为了兼容 Python 3.11 版本添加
from typing_extensions import TypedDict
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 添加项目根目录到 Python 路径
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langgraph.graph import StateGraph, END

# Import core RAG components using absolute paths
from prompts.loader import load_prompt
from rag.milvus_wrapper import get_retriever
from rag.utils.rag_utils import load_chat_model
from rag.workflows.doc_listing import list_docs_node # 导入文档列表节点

# 加载环境变量
load_dotenv()

# 1. Define the state for the graph
class RAGState(TypedDict):
    """Represents the state of the RAG graph."""
    question: str
    answer: str

# 2. Load the prompt template and initialize components
try:
    # 加载提示词模板
    prompt_template_string = load_prompt("rag_prompt.txt", include_base_context=False)
    if not prompt_template_string:
        raise ValueError("RAG提示词模板加载失败")
    RAG_PROMPT = PromptTemplate.from_template(prompt_template_string)
    print("✅ RAG提示词模板加载成功")

    # 初始化检索器
    retriever = get_retriever()
    if not retriever:
        raise ValueError("检索器初始化失败")
    print("✅ 检索器初始化成功")

    # 初始化语言模型（默认使用 OpenRouter 可用模型）
    model_name = os.getenv("NODE_MODEL", "x-ai/grok-3")
    llm = load_chat_model(model_name)
    if not llm:
        raise ValueError(f"语言模型 {model_name} 初始化失败")
    print(f"✅ 语言模型 {model_name} 初始化成功")

except Exception as e:
    print(f"❌ RAG初始化失败: {str(e)}")
    raise

# 3. Define a helper function to format retrieved documents
def format_docs(docs: List[Document]) -> str:
    """Formats a list of retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# 定义一个文件特定的检索器类
class FileSpecificRetriever:
    """A wrapper that filters retrieval results to a specific file."""
    
    def __init__(self, base_retriever, target_file):
        self.base_retriever = base_retriever
        self.target_file = target_file
    
    def __call__(self, query):
        """Make the class callable like a function."""
        return self.base_retriever._get_relevant_documents(
            query, 
            run_manager=None, 
            current_file=self.target_file
        )

# 4. Construct the core RAG chain as a reusable function
def build_rag_chain(retriever):
    """Builds a RAG chain with a given retriever."""
    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

# 5. Define the single node for our graph
def generate_answer_node(state: RAGState):
    """
    智能RAG问答节点
    
    功能：
    1. 从状态中获取用户问题
    2. 智能检测问题中是否包含知识库中的文件名
    3. 如果检测到文件名，创建文件特定的检索器进行精准搜索
    4. 如果未检测到文件名，使用全局检索器进行常规搜索
    5. 构建并执行RAG链，生成答案并更新状态
    """
    try:
        question = state["question"]
        print(f"📝 处理问题: {question}")
        
        # 默认使用全局的retriever
        current_retriever = retriever
        
        # 智能检测：检查问题中是否包含知识库中的文件名
        print("🔍 正在检查问题中是否包含文件名...")
        docs_info = list_docs_node({})
        
        # 获取知识库中所有文件名列表
        if docs_info and docs_info.get("files"):
            knowledge_base_files = [item['filename'] for item in docs_info['files']]
            found_filename = None
            
            # 遍历所有文件名，查找问题中是否包含任何一个
            for filename in knowledge_base_files:
                if filename in question:
                    found_filename = filename
                    break
            
            if found_filename:
                print(f"✅ 在问题中检测到文件名: {found_filename}")
                print(f"🎯 将在文件 '{found_filename}' 中进行定向搜索...")
                # 创建文件特定的检索器
                file_retriever = FileSpecificRetriever(retriever, found_filename)
                current_retriever = RunnableLambda(file_retriever)
            else:
                print("➡️ 未检测到特定文件名，将在整个知识库中搜索。")
        else:
            print("⚠️ 无法获取知识库文件列表，将在整个知识库中搜索。")

        # 使用选定的retriever构建并调用RAG链
        dynamic_rag_chain = build_rag_chain(current_retriever)
        answer = dynamic_rag_chain.invoke(question)
        print("✅ 回答生成成功")
        
        return {"answer": answer}
    except Exception as e:
        print(f"❌ 回答生成失败: {str(e)}")
        return {"answer": f"抱歉，处理您的问题时出现错误: {str(e)}"}

# 6. Build and compile the graph
workflow = StateGraph(RAGState)
workflow.add_node("generate_answer", generate_answer_node)
workflow.set_entry_point("generate_answer")
workflow.add_edge("generate_answer", END)

# 7. Compile the graph
rag_query_workflow = workflow.compile()
print("✅ RAG工作流编译完成")

def rag_answer(query: str) -> str:
    """
    简化的RAG问答接口
    
    Args:
        query: 用户的查询问题
        
    Returns:
        str: 基于知识库的回答
    """
    try:
        result = rag_query_workflow.invoke({
            "question": query,
            "answer": ""  # 初始化为空
        })
        
        if not result or not result.get("answer"):
            raise ValueError("未能生成有效回答")
            
        return result["answer"]
        
    except Exception as e:
        error_msg = f"RAG查询失败: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg
