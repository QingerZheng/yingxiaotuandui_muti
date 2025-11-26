"""人设配置更新 Agent - 
定义了一个 LangGraph 图，可以通过 API 调用来安全地修改 runtime_config.json，从而实现配置的动态更新
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Dict, Any, Optional
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
import json
from Configurations import Configuration

# 尝试多种导入方式
try:
    from agents.persona_config.config_manager import config_manager
    from agents.persona_config.multi_assistant_config_manager import multi_assistant_config_manager
except ImportError:
    try:
        from .config_manager import config_manager
        from .multi_assistant_config_manager import multi_assistant_config_manager
    except ImportError:
        # 如果都失败，直接导入同目录下的文件
        import importlib.util
        config_manager_path = current_dir / "config_manager.py"
        spec = importlib.util.spec_from_file_location("config_manager", config_manager_path)
        config_manager_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_manager_module)
        config_manager = config_manager_module.config_manager
        
        multi_config_manager_path = current_dir / "multi_assistant_config_manager.py"
        spec2 = importlib.util.spec_from_file_location("multi_assistant_config_manager", multi_config_manager_path)
        multi_config_manager_module = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(multi_config_manager_module)
        multi_assistant_config_manager = multi_config_manager_module.multi_assistant_config_manager


class PersonaConfigInput(TypedDict):
    """人设配置输入."""
    config: Dict[str, Any]  # 标准 LangGraph config 格式


class PersonaConfigState(TypedDict):
    """人设配置状态."""
    config: Dict[str, Any]  # 输入的 config 数据





def validate_persona_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """验证人设配置数据."""
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # 定义必需字段和可选字段
    required_fields = []  # 没有必需字段，都是可选的
    optional_fields = [
        "assistant_id",  # 新增：支持指定要更新的 assistant_id
        "agent_name", "agent_nickname", "agent_gender", "agent_age", "agent_birthday",
        "agent_address", "agent_role", "company_name", "industry", "industry_knowledge", "agent_personality",
        "agent_goal", "agent_side_goal", "company_address", "service_time",
        "service_price", "extra_infomation", "base_context_prompt",
        "model_provider", "model_name", "agent_temperature"
    ]
    
    # 检查未知字段
    all_valid_fields = set(required_fields + optional_fields)
    for field_name in config_data.keys():
        if field_name not in all_valid_fields:
            validation_result["warnings"].append(f"未知字段: {field_name}")
    
    # 验证特定字段的值
    if "agent_gender" in config_data:
        valid_genders = ["男", "女", "其他"]
        if config_data["agent_gender"] not in valid_genders:
            validation_result["warnings"].append(f"agent_gender 建议使用: {valid_genders}")
    
    if "agent_age" in config_data:
        age_value = config_data["agent_age"]
        if isinstance(age_value, str):
            try:
                int(age_value)
            except ValueError:
                # 检查是否是中文数字
                chinese_numbers = ["十八", "十九", "二十", "二十一", "二十二", "二十三", "二十四", "二十五", "二十六", "二十七", "二十八", "二十九", "三十"]
                if age_value not in chinese_numbers:
                    validation_result["warnings"].append(f"agent_age 格式可能不正确: {age_value}")
    
    if "agent_temperature" in config_data:
        temp = config_data["agent_temperature"]
        if not isinstance(temp, (int, float)) or temp < 0 or temp > 1:
            validation_result["errors"].append("agent_temperature 必须是 0-1 之间的数字")
            validation_result["valid"] = False
    
    if "model_provider" in config_data:
        valid_providers = ["openai", "openrouter", "dashscope", "anthropic"]
        if config_data["model_provider"] not in valid_providers:
            validation_result["warnings"].append(f"model_provider 建议使用: {valid_providers}")

    # 验证新添加的字段
    if "agent_birthday" in config_data:
        import re
        birthday = config_data["agent_birthday"]
        # 检查日期格式 YYYY-MM-DD
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', str(birthday)):
            validation_result["warnings"].append("agent_birthday 建议使用 YYYY-MM-DD 格式")

    if "agent_role" in config_data:
        role = config_data["agent_role"]
        if len(str(role)) > 50:
            validation_result["warnings"].append("agent_role 长度建议不超过50个字符")

    return validation_result


def process_persona_config(state: PersonaConfigState, config: RunnableConfig) -> Dict[str, Any]:
    """处理人设配置 - 提取、验证和更新配置."""
    try:
        # 调试信息：打印接收到的数据
        print(f"[DEBUG] 接收到的 state: {state}")
        print(f"[DEBUG] 接收到的 config: {config}")
        
        # 步骤1: 提取配置数据 - 支持多种输入格式
        print(f"[DEBUG] 完整的 state: {state}")
        
        configurable = {}
        
        # 格式1: 直接的 configurable 字段
        if "configurable" in state:
            configurable = state["configurable"]
            print(f"[DEBUG] 使用格式1 - 直接 configurable: {configurable}")
        
        # 格式2: config.configurable 结构
        elif "config" in state and isinstance(state["config"], dict):
            input_config = state["config"]
            if "configurable" in input_config:
                configurable = input_config["configurable"]
                print(f"[DEBUG] 使用格式2 - config.configurable: {configurable}")
            # 格式3: config.config.configurable 结构（LangGraph Studio嵌套）
            elif "config" in input_config and "configurable" in input_config["config"]:
                configurable = input_config["config"]["configurable"]
                print(f"[DEBUG] 使用格式3 - config.config.configurable: {configurable}")
        
        # 如果都没找到，返回错误
        if not configurable:
            return {
                "success": False,
                "message": "没有找到配置数据。请使用 'configurable' 字段或 'config.configurable' 格式"
            }
        
        # 步骤2: 验证配置数据
        validation_result = validate_persona_config(configurable)
        
        if not validation_result["valid"]:
            return {
                "validation_result": validation_result,
                "success": False,
                "message": "配置验证失败"
            }
        
        # 步骤3: 提取 assistant_id（如果存在）
        assistant_id = configurable.get("assistant_id")
        print(f"[DEBUG] 提取到的 assistant_id: {assistant_id}")
        
        # 步骤4: 处理和标准化配置
        # 创建默认配置实例用于验证字段
        default_config = Configuration()
        
        # 更新配置，只更新有效字段（排除 assistant_id）
        processed_config = {}
        for field_name, field_value in configurable.items():
            if field_name == "assistant_id":
                continue  # assistant_id 不保存到配置文件中
            if hasattr(default_config, field_name):
                processed_config[field_name] = field_value
        
        # 添加一些处理逻辑
        if "agent_age" in processed_config and isinstance(processed_config["agent_age"], str):
            # 处理中文数字转换
            chinese_to_num = {
                "十八": 18, "十九": 19, "二十": 20, "二十一": 21, "二十二": 22,
                "二十三": 23, "二十四": 24, "二十五": 25, "二十六": 26, "二十七": 27,
                "二十八": 28, "二十九": 29, "三十": 30
            }
            if processed_config["agent_age"] in chinese_to_num:
                processed_config["agent_age"] = chinese_to_num[processed_config["agent_age"]]
        
        # 步骤5: 保存配置
        if assistant_id:
            # 保存到指定 assistant 的配置文件
            config_saved = multi_assistant_config_manager.update_assistant_config(assistant_id, processed_config)
            save_target = f"assistant {assistant_id}"
        else:
            # 保存到全局配置（向后兼容）
            config_saved = config_manager.update_config(processed_config)
            save_target = "全局配置"
        
        if not config_saved:
            return {
                "updated_config": processed_config,
                "validation_result": validation_result,
                "success": False,
                "message": f"配置验证成功但保存到{save_target}失败，处理了 {len(processed_config)} 个配置项"
            }
        
        # 获取完整的更新后配置用于返回
        if assistant_id:
            updated_config = multi_assistant_config_manager.get_assistant_config(assistant_id)
        else:
            current_configurable = config.get("configurable", {})
            updated_config = current_configurable.copy()
            updated_config.update(processed_config)
        
        return {
            "updated_config": updated_config,  # 返回完整的更新后配置
            "validation_result": validation_result,
            "success": True,
            "message": f"成功更新并保存 {len(processed_config)} 个配置项到{save_target}",
            "assistant_id": assistant_id,  # 返回处理的 assistant_id
            "save_target": save_target
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"处理配置时出错: {str(e)}"
        }


def create_persona_config_graph():
    """创建人设配置更新工作流."""
    graph = StateGraph(PersonaConfigState)
    
    # 添加单个处理节点
    graph.add_node("process_config", process_persona_config)
    
    # 添加边
    graph.add_edge(START, "process_config")
    graph.add_edge("process_config", END)
    
    return graph.compile()


# 导出主要接口
persona_config_graph = create_persona_config_graph()


if __name__ == "__main__":
    print("人设配置更新模块加载完成")
    
    # 测试示例
    test_config = {
        "config": {
            "configurable": {
                "agent_name": "",
                "company_name": "{{}}",
                "agent_gender": "女",
                "agent_age": "25",
                "agent_personality": "温柔"
            }
        }
    }
    
    print(f"测试配置: {test_config}")
    # 测试配置提取
    configurable = test_config["config"]["configurable"]
    validation = validate_persona_config(configurable)
    print(f"验证结果: {validation}")