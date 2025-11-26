"""网络搜索工具模块 - 修复版"""

import os
from langchain_core.tools import tool

try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    HAS_TAVILY = True
except ImportError:
    HAS_TAVILY = False

@tool
def web_search(query: str, max_results: int = 3) -> str:
    """
    网络搜索工具：使用Tavily从互联网获取最新信息。
    
    适用场景：
    - 最新新闻、实时信息
    - 知识库中没有的信息
    - 当前时事、股价、天气等实时数据
    
    Args:
        query (str): 搜索查询内容
        max_results (int): 最大返回结果数，默认为3
        
    Returns:
        str: 网络搜索结果摘要
    """
    print(f"🔍 开始网络搜索: {query}")
    
    # 检查Tavily是否可用
    if not HAS_TAVILY:
        error_msg = "❌ Tavily搜索功能未安装。请运行: pip install tavily-python"
        print(error_msg)
        return error_msg
    
    # 检查API密钥
    if not os.getenv("TAVILY_API_KEY"):
        error_msg = "❌ 未设置TAVILY_API_KEY环境变量。请在.env文件中添加您的Tavily API密钥。"
        print(error_msg)
        return error_msg
    
    try:
        print(f"🌐 使用Tavily搜索: {query}")
        
        # 创建Tavily搜索实例
        tavily_search = TavilySearchResults(
            max_results=max_results,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False,
        )
        
        # 执行搜索（使用同步方法）
        search_results = tavily_search.invoke({"query": query})
        
        if not search_results:
            result = "未找到相关的网络信息。"
            print(f"⚠️ {result}")
            return result

        # 直接返回原始搜索结果，不进行LLM总结处理
        formatted_results = "根据网络搜索结果：\n\n"
        for i, result in enumerate(search_results[:max_results], 1):
            content = result.get("content", "")
            url = result.get("url", "")
            title = result.get("title", "")
            
            formatted_results += f"结果{i}:\n"
            if title:
                formatted_results += f"标题: {title}\n"
            formatted_results += f"内容: {content}\n"
            formatted_results += f"来源: {url}\n\n"
        
        print(f"✅ Tavily搜索成功，返回 {len(search_results)} 个结果（原始数据）")
        return formatted_results
        
    except Exception as e:
        error_msg = f"❌ 网络搜索出现错误：{str(e)}"
        print(error_msg)
        return error_msg
