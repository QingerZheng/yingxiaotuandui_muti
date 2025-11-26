"""网络搜索工具模块

提供联网搜索功能，获取最新信息和热点
"""

import os
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults


@tool
def web_search_tool(query: str) -> str:
    """联网搜索工具，获取最新信息和热点"""
    # 检查Tavily是否可用
    if not os.getenv("TAVILY_API_KEY"):
        return "网络搜索功能需要设置 TAVILY_API_KEY 环境变量"
    
    try:
        # 创建Tavily搜索实例
        tavily_search = TavilySearchResults(
            max_results=3,
            search_depth="basic",
            include_answer=True
        )
        
        # 执行搜索
        search_results = tavily_search.invoke({"query": query})
        
        if not search_results:
            return "未找到相关的网络信息"

        # 格式化搜索结果
        formatted_results = "网络搜索结果：\n\n"
        for i, result in enumerate(search_results, 1):
            content = result.get("content", "")
            url = result.get("url", "")
            title = result.get("title", "")
            
            formatted_results += f"{i}. {title}\n{content}\n来源: {url}\n\n"
        
        return formatted_results
        
    except Exception as e:
        return f"网络搜索出现错误：{str(e)}"