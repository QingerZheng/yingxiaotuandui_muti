import os
from typing import List, Dict, Any


class BaseBlock:
    """
    所有销售能力模块的基础类 (Base Class for all Sales Skill Blocks)。

    这个基类定义了所有具体能力模块（如问候、需求澄清等）都必须遵循的通用接口和核心属性。
    它确保了每个模块都能被元智能体（Conversation Strategist）以统一的方式调用和管理。

    Attributes:
        block_name (str): 模块的名称，用于日志记录和调试。
        model_sampler_map (Dict): 一个映射，存储了可用的不同语言模型采样器实例。
        node_model (str): 当前模块默认使用的语言模型名称。
    """
    def __init__(self, block_name: str, sampler: Any, node_model: str):
        """
        初始化一个能力模块。

        Args:
            block_name (str): 模块的名称，通常用于加载对应的 Prompt。
            sampler (any): 一个已经初始化好的模型采样器实例。
            node_model (str): 当前节点将要使用的模型名称。
        """
        self.block_name = block_name
        self.sampler = sampler
        self.node_model = node_model

    def forward(self, state: Any) -> str:
        """
        模块的核心执行方法。

        每个子类都需要实现这个方法，以定义其独特的对话能力。
        这个方法接收当前的对话历史，并生成下一步的回复。

        Args:
            conversation_history (List[Dict[str, str]]): 一个包含对话历史的列表。
            temperature (float): 用于本次回复生成的动态温度值。

        Returns:
            str: 由模块生成的回复字符串。
        
        Raises:
            NotImplementedError: 如果子类没有实现此方法，则会抛出此异常。
        """
        raise NotImplementedError("每个能力模块都必须实现 forward 方法。")

    def __repr__(self) -> str:
        """
        返回模块的字符串表示，方便调试。
        """
        return f"{self.block_name}(model={self.node_model})" 