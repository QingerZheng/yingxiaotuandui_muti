from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

# 导入我们的配置模型
from Configurations import Configuration
# 导入主图
from parallel_tools_agent_workflow import create_parallel_tools_graph
# 导入上下文更新工作流
from nodes import create_context_update_workflow
# 导入配置管理器
from agents.persona_config.config_manager import config_manager
from agents.persona_config.multi_assistant_config_manager import multi_assistant_config_manager

def get_graph(config: RunnableConfig) -> CompiledStateGraph:
    """
    LangGraph Cloud 热更新入口函数。
    
    这个函数会在每次请求时被调用，根据 assistant_id 加载对应配置
    动态地构建和编译一个新的图实例，实现真正的热更新。
    
    Args:
        config: RunnableConfig 对象，包含运行时配置信息
    
    Returns:
        编译后的 LangGraph 实例
    """
    print(f"[DEBUG] get_graph 接收到的 config: {config}")
    
    # 提取 assistant_id（如果存在）
    assistant_id = None
    if config and "configurable" in config:
        assistant_id = config["configurable"].get("assistant_id")
    
    print(f"[DEBUG] 提取到的 assistant_id: {assistant_id}")
    
    # 根据是否有 assistant_id 来决定配置加载策略
    if assistant_id:
        # 加载指定 assistant 的配置
        hot_config = multi_assistant_config_manager.get_assistant_config(assistant_id)
        print(f"[DEBUG] 使用 assistant {assistant_id} 的配置: {len(hot_config)} 个字段")
        
        # 如果 assistant 配置为空，尝试从默认配置创建
        if not hot_config:
            print(f"[DEBUG] assistant {assistant_id} 配置不存在，尝试从默认配置创建")
            multi_assistant_config_manager.create_assistant_from_default(assistant_id)
            hot_config = multi_assistant_config_manager.get_assistant_config(assistant_id)
    else:
        # 使用全局配置（向后兼容）
        hot_config = config_manager.get_config() or {}
        print(f"[DEBUG] 使用全局配置: {len(hot_config)} 个字段")
    
    # 构建并编译图，传入热更新配置
    workflow = create_parallel_tools_graph(hot_config)
    
    # 编译并返回图实例
    # 注意：每次调用都会创建新的图实例，确保使用最新配置
    return workflow

def get_context_update_graph(config: RunnableConfig) -> CompiledStateGraph:
    """
    上下文更新工作流的入口函数。

    这个函数专门用于处理向现有thread注入上下文信息的请求。
    每次请求时动态构建上下文更新工作流实例。

    Args:
        config: RunnableConfig 对象，包含运行时配置信息

    Returns:
        编译后的上下文更新工作流实例
    """
    print(f"[DEBUG] get_context_update_graph 接收到的 config: {config}")

    # 提取 assistant_id（如果存在）
    assistant_id = None
    if config and "configurable" in config:
        assistant_id = config["configurable"].get("assistant_id")

    print(f"[DEBUG] 上下文更新 - 提取到的 assistant_id: {assistant_id}")

    # 构建上下文更新工作流
    # 注意：这个工作流相对简单，不需要复杂的配置管理
    workflow = create_context_update_workflow()

    # 编译并返回图实例
    return workflow