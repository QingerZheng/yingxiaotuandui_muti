"""配置管理器 - 用于运行时动态配置管理.
负责读取和写入persona_config.json文件，并提供配置更新功能。


"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from threading import Lock

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from Configurations import Configuration


class ConfigManager:
    """运行时配置管理器，支持动态配置更新."""
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._runtime_config = {}
            # 使用相对路径，配置文件保存在当前模块目录下
            self._config_file = os.path.join(os.path.dirname(__file__), "runtime_config.json")
            self._load_config()
            self._initialized = True
    
    def _load_config(self):
        """从文件加载运行时配置."""
        try:
            if os.path.exists(self._config_file):
                with open(self._config_file, 'r', encoding='utf-8') as f:
                    self._runtime_config = json.load(f)
                print(f"[ConfigManager] 从 {self._config_file} 加载运行时配置: {len(self._runtime_config)} 个字段")
            else:
                print(f"[ConfigManager] 运行时配置文件 {self._config_file} 不存在，使用空配置")
        except Exception as e:
            print(f"[ConfigManager] 加载配置失败: {e}")
            self._runtime_config = {}
    
    def _save_config(self):
        """保存运行时配置到文件."""
        try:
            with open(self._config_file, 'w', encoding='utf-8') as f:
                json.dump(self._runtime_config, f, ensure_ascii=False, indent=2)
            print(f"[ConfigManager] 保存运行时配置: {len(self._runtime_config)} 个字段")
        except Exception as e:
            print(f"[ConfigManager] 保存配置失败: {e}")
    
    def update_config(self, config_updates: Dict[str, Any]) -> bool:
        """更新运行时配置."""
        try:
            with self._lock:
                # 验证配置字段
                default_config = Configuration()
                valid_updates = {}
                
                for field_name, field_value in config_updates.items():
                    if hasattr(default_config, field_name):
                        valid_updates[field_name] = field_value
                    else:
                        print(f"[ConfigManager] 忽略无效字段: {field_name}")
                
                # 更新配置
                self._runtime_config.update(valid_updates)
                
                # 保存到文件
                self._save_config()
                
                print(f"[ConfigManager] 成功更新 {len(valid_updates)} 个配置项")
                return True
                
        except Exception as e:
            print(f"[ConfigManager] 更新配置失败: {e}")
            return False
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前运行时配置."""
        with self._lock:
            return self._runtime_config.copy()
    
    def get_merged_config(self, base_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """获取合并后的配置（运行时配置优先）."""
        with self._lock:
            if base_config is None:
                base_config = {}
            
            merged = base_config.copy()
            merged.update(self._runtime_config)
            return merged
    
    def clear_config(self):
        """清空运行时配置."""
        with self._lock:
            self._runtime_config = {}
            self._save_config()
            print("[ConfigManager] 已清空运行时配置")
    
    def has_runtime_config(self) -> bool:
        """检查是否有运行时配置."""
        with self._lock:
            return bool(self._runtime_config)


# 全局配置管理器实例
config_manager = ConfigManager()