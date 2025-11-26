from langgraph.graph import StateGraph, START, END
from nodes import update_state_memory_node, multi_subgraph_parallel_node, send_or_response_node

from states import AgentInput,AgentState,AgentOutput
from Configurations import Configuration
from typing import Dict, Any

def create_parallel_tools_graph(hot_config: Dict[str, Any] = None):
    """
    创建并编译主图，支持热更新配置
    
    Args:
        hot_config: 热更新配置字典，包含所有可配置参数
    
    Returns:
        编译后的StateGraph实例
    """
    
    # 创建配置schema实例，用于运行时配置传递
    # 注意：不在这里使用hot_config，而是依赖runtime config
    config_schema = Configuration
    
    # 创建主图
    workflow = StateGraph(
        state_schema=AgentState,
        input=AgentInput,
        config_schema=config_schema,
        output=AgentOutput
    )
    
    # 添加节点
    workflow.add_node("update_state_memory", update_state_memory_node)
    workflow.add_node("multi_subgraph_parallel", multi_subgraph_parallel_node)
    workflow.add_node("send_or_response", send_or_response_node)
    
    # 添加边
    workflow.add_edge(START, "update_state_memory")
    workflow.add_edge("update_state_memory", "multi_subgraph_parallel")
    workflow.add_edge("multi_subgraph_parallel", "send_or_response")
    workflow.add_edge("send_or_response", END)
    
    # 编译图，启用checkpointer以支持状态持久化
    # 这样可以确保上下文消息在多次调用之间被保留
    try:
        # 尝试导入MemorySaver用于内存状态持久化
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
        print("[DEBUG] 使用MemorySaver作为checkpointer")
    except ImportError:
        # 如果MemorySaver不可用，使用None（但会警告）
        checkpointer = None
        print("[WARNING] MemorySaver不可用，状态将不会持久化")

    return workflow.compile(
        checkpointer=checkpointer,  # 启用checkpointer以支持状态持久化
        interrupt_before=None,
        interrupt_after=None,
        debug=False
    )

# 保持向后兼容性
parallel_tools_graph = create_parallel_tools_graph()
