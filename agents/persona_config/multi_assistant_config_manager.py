"""多助手配置管理器 - 支持按 assistant_id 独立管理配置.

每个 assistant_id 都有自己独立的配置文件，存储在 assistants_config/{assistant_id}.json
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from threading import RLock
import uuid
from datetime import datetime

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from Configurations import Configuration


class MultiAssistantConfigManager:
    """多助手配置管理器，支持按 assistant_id 独立配置管理."""
    
    _instance = None
    _lock = RLock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            # 配置文件目录
            self._config_dir = os.path.join(os.path.dirname(__file__), "assistants_config")
            self._ensure_config_dir()
            
            # 默认配置文件
            self._default_config_file = os.path.join(self._config_dir, "default_assistant.json")
            
            self._initialized = True
    
    def _ensure_config_dir(self):
        """确保配置目录存在."""
        os.makedirs(self._config_dir, exist_ok=True)
    
    def _get_config_file_path(self, assistant_id: str) -> str:
        """获取指定 assistant_id 的配置文件路径."""
        return os.path.join(self._config_dir, f"{assistant_id}.json")
    
    def _load_config_from_file(self, config_file: str) -> Dict[str, Any]:
        """从指定文件加载配置."""
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"[MultiAssistantConfigManager] 从 {config_file} 加载配置: {len(config)} 个字段")
                return config
            else:
                print(f"[MultiAssistantConfigManager] 配置文件 {config_file} 不存在")
                return {}
        except Exception as e:
            print(f"[MultiAssistantConfigManager] 加载配置失败 {config_file}: {e}")
            return {}
    
    def _save_config_to_file(self, config: Dict[str, Any], config_file: str) -> bool:
        """保存配置到指定文件."""
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            print(f"[MultiAssistantConfigManager] 保存配置到 {config_file}: {len(config)} 个字段")
            return True
        except Exception as e:
            print(f"[MultiAssistantConfigManager] 保存配置失败 {config_file}: {e}")
            return False
    
    def get_assistant_config(self, assistant_id: str) -> Dict[str, Any]:
        """获取指定 assistant 的配置."""
        with self._lock:
            config_file = self._get_config_file_path(assistant_id)
            config = self._load_config_from_file(config_file)
            
            # 如果助手配置不存在，尝试从默认配置复制
            if not config:
                default_config = self._load_config_from_file(self._default_config_file)
                if default_config:
                    print(f"[MultiAssistantConfigManager] 使用默认配置初始化 assistant {assistant_id}")
                    config = default_config.copy()
            
            return config
    
    def update_assistant_config(self, assistant_id: str, config_updates: Dict[str, Any]) -> bool:
        """更新指定 assistant 的配置."""
        try:
            with self._lock:
                # 验证配置字段
                default_config = Configuration()
                valid_updates = {}
                
                for field_name, field_value in config_updates.items():
                    if hasattr(default_config, field_name):
                        valid_updates[field_name] = field_value
                    else:
                        print(f"[MultiAssistantConfigManager] 忽略无效字段: {field_name}")
                
                # 获取现有配置
                current_config = self.get_assistant_config(assistant_id)
                
                # 合并更新
                current_config.update(valid_updates)
                
                # 保存配置
                config_file = self._get_config_file_path(assistant_id)
                success = self._save_config_to_file(current_config, config_file)
                
                if success:
                    print(f"[MultiAssistantConfigManager] 成功更新 assistant {assistant_id} 的 {len(valid_updates)} 个配置项")
                
                return success
                
        except Exception as e:
            print(f"[MultiAssistantConfigManager] 更新配置失败: {e}")
            return False
    
    def get_merged_config(self, assistant_id: Optional[str] = None, base_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """获取合并后的配置（assistant配置优先）."""
        with self._lock:
            if base_config is None:
                base_config = {}
            
            merged = base_config.copy()
            
            # 如果指定了 assistant_id，加载其配置
            if assistant_id:
                assistant_config = self.get_assistant_config(assistant_id)
                merged.update(assistant_config)
            
            return merged
    
    def list_assistants(self) -> List[str]:
        """列出所有已配置的 assistant_id."""
        try:
            assistant_ids = []
            for filename in os.listdir(self._config_dir):
                if filename.endswith('.json') and filename != 'default_assistant.json':
                    assistant_id = filename[:-5]  # 移除 .json 后缀
                    assistant_ids.append(assistant_id)
            return assistant_ids
        except Exception as e:
            print(f"[MultiAssistantConfigManager] 列出助手失败: {e}")
            return []
    
    def delete_assistant_config(self, assistant_id: str) -> bool:
        """删除指定 assistant 的配置."""
        try:
            with self._lock:
                config_file = self._get_config_file_path(assistant_id)
                if os.path.exists(config_file):
                    # 删除文件
                    os.remove(config_file)
                    print(f"[MultiAssistantConfigManager] 删除 assistant {assistant_id} 的配置")
                    return True
                else:
                    print(f"[MultiAssistantConfigManager] assistant {assistant_id} 的配置不存在")
                    return False
        except Exception as e:
            print(f"[MultiAssistantConfigManager] 删除配置失败: {e}")
            return False
    
    def has_assistant_config(self, assistant_id: str) -> bool:
        """检查指定 assistant 是否有配置."""
        config_file = self._get_config_file_path(assistant_id)
        return os.path.exists(config_file)
    
    def create_assistant_from_default(self, assistant_id: str) -> bool:
        """从默认配置创建新的 assistant 配置."""
        try:
            with self._lock:
                if self.has_assistant_config(assistant_id):
                    print(f"[MultiAssistantConfigManager] assistant {assistant_id} 配置已存在")
                    return True
                
                default_config = self._load_config_from_file(self._default_config_file)
                if default_config:
                    config_file = self._get_config_file_path(assistant_id)
                    success = self._save_config_to_file(default_config, config_file)
                    if success:
                        print(f"[MultiAssistantConfigManager] 从默认配置创建 assistant {assistant_id}")
                    return success
                else:
                    print(f"[MultiAssistantConfigManager] 默认配置不存在，无法创建 assistant {assistant_id}")
                    return False
        except Exception as e:
            print(f"[MultiAssistantConfigManager] 创建 assistant 配置失败: {e}")
            return False


# 全局多助手配置管理器实例
multi_assistant_config_manager = MultiAssistantConfigManager()