"""
RAG系统工具函数
提供文档处理、模型加载等通用功能

支持功能:
1. 多格式文档加载和分块
2. 文档下载和缓存
3. 语言模型管理
4. 文件格式识别
"""
import os
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredFileLoader
import requests

# 导入多模态处理功能
from .multimodal_processor import (
    get_multimodal_processor, 
    get_multimodal_supported_formats,
    create_multimodal_documents
)


def load_and_chunk_document(
    file_path: str,
    chunk_size: int = 200,
    chunk_overlap: int = 0,
    **loader_kwargs,
) -> List[Document]:
    """
    通用文档加载和分块函数，支持多种文档格式（包括多模态）
    
    支持的格式：
    - Word文档: .docx, .doc (文本+图片OCR+表格)
    - PDF文档: .pdf (文本+表格提取+图片OCR)
    - 文本文件: .txt, .md
    - 图片文件: .jpg, .jpeg, .png, .bmp (OCR文字识别)
    - PPT文档: .pptx, .ppt (文本+图片OCR+表格)
    - Excel文档: .xlsx, .xls (表格数据)
    - 其他格式: 使用UnstructuredFileLoader尝试处理
    
    Args:
        file_path (str): 文档文件路径
        chunk_size (int): 每个文本块的最大字符数
        chunk_overlap (int): 文本块之间的重叠字符数
        **loader_kwargs: 传递给文档加载器的额外参数
        
    Returns:
        List[Document]: 分块后的文档列表

    Example:
        # 处理PDF文件
        pdf_chunks = load_and_chunk_document("document.pdf")
        
        # 处理图片文件（OCR）
        image_chunks = load_and_chunk_document("image.jpg")
    """
    
    # 获取文件扩展名
    file_extension = os.path.splitext(file_path)[1].lower()
    
    print(f"📄 检测到文件格式: {file_extension}")
    
    # 检查是否为多模态格式
    multimodal_formats = get_multimodal_supported_formats()
    
    if file_extension in multimodal_formats:
        print(f"🎯 使用多模态处理器: {multimodal_formats[file_extension]}")
        try:
            processor = get_multimodal_processor()
            multimodal_docs = create_multimodal_documents(file_path, processor)
            
            if multimodal_docs:
                # 对多模态文档进行分块
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_overlap
                )
                chunked_docs = text_splitter.split_documents(multimodal_docs)
                print(f"✅ 多模态文档处理完成，共 {len(chunked_docs)} 个片段")
                return chunked_docs
            else:
                print("⚠️  多模态处理未提取到内容，尝试传统方法")
        except Exception as e:
            print(f"⚠️  多模态处理失败，回退到传统方法: {str(e)}")
         
    # 传统文档处理方式
    try:
        # 根据文件格式选择合适的加载器
        if file_extension in ['.docx', '.doc']:
            print("📝 使用Word文档加载器")
            loader = UnstructuredWordDocumentLoader(file_path, **loader_kwargs)
        elif file_extension == '.pdf':
            print("📑 使用PDF文档加载器")
            loader = PyPDFLoader(file_path)
            
            # 尝试额外的PDF表格提取
            try:
                processor = get_multimodal_processor()
                pdf_tables = processor.extract_tables_from_pdf(file_path)
                if pdf_tables:
                    print(f"📊 额外提取了 {len(pdf_tables)} 个PDF表格")
            except:
                pass  # 忽略表格提取错误
                
        elif file_extension in ['.txt', '.md']:
            print("📃 使用文本文件加载器")
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            print(f"🔧 使用通用文档加载器处理 {file_extension} 格式")
            loader = UnstructuredFileLoader(file_path, **loader_kwargs)
        
        # 加载文档
        docs = loader.load()
        print(f"✅ 文档加载成功，共 {len(docs)} 页/段")
        
        # 文本分块
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        chunked_docs = text_splitter.split_documents(docs)
        
        print(f"✅ 文档分块完成，共 {len(chunked_docs)} 个片段")
        return chunked_docs
         
    except Exception as e:
        print(f"❌ 文档处理失败: {str(e)}")
        raise

  
def download_doc(file_url_or_path: str) -> str:
    """
    处理文档文件（支持本地文件和URL下载）
    
    Args:
        file_url_or_path (str): 文档文件的URL或本地文件路径
        
    Returns:
        str: 文件路径（本地文件直接返回，URL下载后返回本地路径）

    支持的输入格式：
        - 本地文件路径: ./document.docx, /path/to/file.pdf
        - 网络URL: https://example.com/document.docx
        
    支持的文件格式：
        - Word文档: .docx, .doc
        - PDF文档: .pdf
        - 文本文件: .txt, .md
        - 图片文件: .jpg, .png, .bmp (多模态)
        - PPT文档: .pptx, .ppt (多模态)
        - Excel文档: .xlsx, .xls (多模态)
        - 其他文档格式
    """
    # 检查是否为本地文件路径
    if not file_url_or_path.startswith(('http://', 'https://', 'ftp://')):
        print(f"📁 检测到本地文件路径: {file_url_or_path}")
        
        # 尝试多个可能的路径
        possible_paths = [
            file_url_or_path,  # 原始路径
            os.path.join(os.getcwd(), file_url_or_path),  # 相对于当前目录
            os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), file_url_or_path)  # 相对于上级目录
        ]
        
        # 尝试每个路径
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            print(f"📂 尝试路径: {abs_path}")
            if os.path.exists(abs_path):
                file_extension = os.path.splitext(abs_path)[1].lower()
                print(f"✅ 找到文件，格式: {file_extension}")
                return abs_path
        
        # 如果所有路径都失败，打印尝试过的路径并抛出异常
        error_msg = "本地文件不存在，尝试过以下路径:\n" + "\n".join(f"- {p}" for p in possible_paths)
        raise FileNotFoundError(error_msg)
            
    # URL下载逻辑保持不变
    try:
        print(f"📥 开始下载文档: {file_url_or_path}")
        
        # 首先尝试简单下载（适用于直接文件链接）
        try:
            doc_response = requests.get(file_url_or_path, timeout=30)
            doc_response.raise_for_status()
            
            # 检查是否获取到了实际文件
            content_type = doc_response.headers.get('content-type', '').lower()
            if 'text/html' in content_type:
                raise ValueError("获取到HTML页面")
            
            print(f"✅ 简单下载成功，内容类型: {content_type}")
            
        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"⚠️  简单下载失败: {e}")
            print("🔄 尝试使用浏览器模拟下载...")
            
            # 使用浏览器模拟下载（适用于需要重定向的链接）
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            # 允许重定向，设置超时
            doc_response = requests.get(file_url_or_path, headers=headers, allow_redirects=True, timeout=30)
            doc_response.raise_for_status()
        
            # 再次检查响应内容类型
            content_type = doc_response.headers.get('content-type', '').lower()
            print(f"📄 响应内容类型: {content_type}")
            
            # 如果还是HTML页面，说明这个链接需要特殊处理
            if 'text/html' in content_type:
                print("⚠️  仍然检测到HTML响应，可能需要手动下载或使用不同的URL")
                raise ValueError(f"无法从此URL获取文件，可能需要直接下载链接: {file_url_or_path}")
        
        # 从URL获取文件名
        filename = file_url_or_path.split("/")[-1]
        
        # 检查文件格式
        file_extension = os.path.splitext(filename)[1].lower()
        
        # 获取所有支持的格式（包括多模态格式）
        multimodal_formats = get_multimodal_supported_formats()
        base_formats = ['.pdf', '.docx', '.doc', '.txt', '.md']
        all_supported_formats = base_formats + list(multimodal_formats.keys())
        
        if file_extension in all_supported_formats:
            format_desc = multimodal_formats.get(file_extension, "文档文件")
            print(f"✅ 支持的文件格式: {file_extension} ({format_desc})")
        else:
            print(f"⚠️  未明确支持的格式 {file_extension}，将尝试通用处理")
        
        with open(filename, "wb") as f:
            f.write(doc_response.content)
            
        print(f"✅ 文档下载完成: {filename}")
        return filename
        
    except requests.exceptions.RequestException as e:
        print(f"❌ 下载文件失败 {file_url_or_path}: {e}")
        raise
    except FileNotFoundError as e:
        print(f"❌ 文件不存在: {e}")
        raise
    except Exception as e:
        print(f"❌ 处理文件时出现意外错误: {e}")
        raise


def get_supported_formats():
    """
    返回支持的文档格式列表（包括多模态格式）
    
    Returns:
        dict: 支持的格式和对应的描述
    """
    # 基础格式（纯文本）
    base_formats = {
        '.txt': '纯文本文件',
        '.md': 'Markdown文件',
    }
    
    # 获取多模态格式（包括PDF、Word、PPT、Excel、图片等）
    multimodal_formats = get_multimodal_supported_formats()
    base_formats.update(multimodal_formats)
    
    # 添加其他格式说明
    base_formats['others'] = '其他格式 (通过UnstructuredFileLoader处理)'
    
    return base_formats


def load_chat_model(model_name: str, temperature: float = 0, **kwargs):
    """
    加载聊天模型，优先使用LangChain原生模型
    
    Args:
        model_name: 模型名称
        temperature: 温度参数
        **kwargs: 其他模型参数
        
    Returns:
        LangChain兼容的LLM实例
    """
    try:
        # 解析模型名称
        if '/' in model_name:
            provider, model = model_name.split('/', 1)
        else:
            provider = 'openai'
            model = model_name
        
        # 统一通过项目工厂创建（默认 OpenRouter）
        if provider in ['openai', 'openrouter'] or model.startswith('gpt-'):
            from llm import create_llm
            model_provider = 'openrouter' if provider != 'openai' else 'openai'
            return create_llm(
                model_provider=model_provider,
                model_name=model,
                temperature=temperature,
                **kwargs
            )
        elif provider == 'anthropic' or model.startswith('claude-'):
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model,
                temperature=temperature,
                **kwargs
            )
        elif provider == 'google' or model.startswith('gemini-'):
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                **kwargs
            )
        else:
            # 回退到项目的采样器工厂（需要包装）
            from sampler.factory import SamplerFactory
            sampler, _ = SamplerFactory.get_sampler_and_cost(model_name)
            
            # 创建LangChain兼容的包装器
            from langchain_core.language_models.base import BaseLanguageModel
            from langchain_core.outputs import LLMResult, Generation
            from langchain_core.callbacks.manager import CallbackManagerForLLMRun
            from typing import List, Optional, Any
            
            class SamplerLLMWrapper(BaseLanguageModel):
                def __init__(self, sampler, temperature=0):
                    super().__init__()
                    self.sampler = sampler
                    self.temperature = temperature
                
                def _generate(
                    self,
                    messages: List,
                    stop: Optional[List[str]] = None,
                    run_manager: Optional[CallbackManagerForLLMRun] = None,
                    **kwargs: Any,
                ) -> LLMResult:
                    # 转换消息格式
                    text = "\n".join([msg.content if hasattr(msg, 'content') else str(msg) for msg in messages])
                    response = self.sampler.sample(text, temperature=self.temperature)
                    return LLMResult(generations=[[Generation(text=response)]])
                
                @property
                def _llm_type(self) -> str:
                    return "sampler_wrapper"
            
            return SamplerLLMWrapper(sampler, temperature)
            
    except Exception as e:
        print(f"❌ 加载模型失败 {model_name}: {e}")
        # 最后的回退选项：走工厂并使用 openrouter 默认模型
        from llm import create_llm
        return create_llm(
            model_provider='openrouter',
            model_name='x-ai/grok-3',
            temperature=temperature,
            **kwargs
        )
