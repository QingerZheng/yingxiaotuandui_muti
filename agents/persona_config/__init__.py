"""
人设配置模块
用于管理AI助手的人设相关配置
"""

from .config_manager import config_manager
from .persona_update import persona_config_graph

__all__ = ["config_manager", "persona_config_graph"]