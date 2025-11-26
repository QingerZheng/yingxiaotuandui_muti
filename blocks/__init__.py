from typing import Dict, Type
from .base import BaseBlock
# 使得人工接管状态能够恢复
# 移除重复的预约管理模块，active_close已覆盖相关功能
from .state_evaluator import evaluate_state
from .intent_analyzer import analyze_customer_intent, update_appointment_info

# 占位注册表与工厂：当前主流程未使用具体 Block 类，保留空实现以兼容历史导入
BLOCK_REGISTRY: Dict[str, Type[BaseBlock]] = {}

def create_block(action: str, sampler: any, node_model: str) -> BaseBlock:
    """
    兼容性工厂：当前未注册任何具体 Block，始终返回 None。
    """
    block_class = BLOCK_REGISTRY.get(action)
    if block_class:
        return block_class(sampler, node_model)
    return None