from typing import Dict, List, Any
from states import AgentState, EmotionalState, DebugInfo
from prompts.loader import load_prompt
from langchain_core.messages import HumanMessage
import json
import re
from langchain_core.tools import tool
from Configurations import Configuration
from dataclasses import asdict
from json_parser_utils import robust_json_parse, create_fallback_dict
@tool
def evaluate_state(state_dict: dict = None):
    """
    评估当前对话状态，包括情感和客户意图。
    现在直接使用 SamplerFactory 获取所需采样器。
    """

    # 如果传入的是包装的字典，提取实际的state_dict
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # 确保 state_dict 是字典
    if not isinstance(state_dict, dict):
        print("错误: state_dict 不是字典类型")
        return {}

    # 格式化对话历史
    messages = state_dict.get("long_term_messages", [])
    if messages is None:
        messages = []

    history = "\n".join(
        [f"{getattr(msg, 'type', 'unknown')}: {getattr(msg, 'content', '')}" for msg in messages]
    )

    # 加载 Prompt - 使用正确的 load_prompt 函数
    try:
        prompt_template = load_prompt("state_evaluator", include_base_context=False)
    except FileNotFoundError:
        print("错误: 找不到 state_evaluator.txt，无法进行状态评估。")
        return {} # 评估失败
    debug_info = state_dict.get("debug_info", DebugInfo())

    # 使用状态中的助手配置，如果没有则使用默认配置
    cfg = state_dict.get("assistant_config", {})
    if not cfg:
        from agents.persona_config.config_manager import config_manager
        cfg = config_manager.get_config() or {}
    
    # 字段兼容与别名回填，避免 KeyError
    try:
        if "industry_konwledge" in cfg and "industry_knowledge" not in cfg:
            cfg["industry_knowledge"] = cfg.get("industry_konwledge")
        if "industry" not in cfg and "industry_knowledge" in cfg:
            cfg["industry"] = cfg["industry_knowledge"]
        if "industry_knowledge" not in cfg and "industry" in cfg:
            cfg["industry_knowledge"] = cfg["industry"]
    except Exception:
        pass

    # 获取debug_info，如果不存在则使用默认值
    debug_info = state_dict.get("debug_info")
    
    prompt = prompt_template.format(
        message_history=history,
        current_stage=debug_info.current_stage if debug_info else "initial_contact",
        user_profile={},
        **cfg
    )
    
    # 使用配置创建LLM
    try:
        from llm import create_llm
        llm = create_llm(
            model_provider=cfg.get("model_provider", "openrouter"),
            model_name=cfg.get("evaluation_model", cfg.get("model_name", "x-ai/grok-code-fast-1")),
            temperature=0.5
        )
    except Exception as e:
        print(f"错误：无法创建评估模型: {e}")
        return {} # 返回空字典表示失败

    # 调用 LLM 进行评估
    try:
        # 创建消息对象
        message = HumanMessage(content=prompt)
        
        # 调用 LLM，要求返回 JSON 格式
        response = llm.invoke(
            [message],
            response_format={"type": "json_object"}
        )
        response_text = response.content
        
        # 使用鲁棒的JSON解析工具
        print(f"[DEBUG-状态评估] 原始模型响应: {response_text}")
        
        fallback_dict = create_fallback_dict("状态评估")
        llm_output = robust_json_parse(
            response_text, 
            context="状态评估", 
            fallback_dict=fallback_dict,
            debug=True
        )
        
        print(f"[DEBUG-状态评估] 解析结果: {llm_output}")
            
    except Exception as e:
        print(f"[ERROR-状态评估] 模型调用或解析失败: {e}")
        if 'response_text' in locals() and response_text is not None:
            print(f"[ERROR-状态评估] 原始响应: {response_text[:300]}...")
        # 使用兜底字典
        llm_output = create_fallback_dict("状态评估")

    # 从LLM的输出中提取并构建状态
    emotional_state_data = llm_output.get("emotional_state", {})
    customer_info = llm_output.get("customer_info", {})

    # 安全地创建EmotionalState实例
    from json_parser_utils import safe_create_emotional_state
    emotional_state = safe_create_emotional_state(emotional_state_data)

    result = {
        "emotional_state": emotional_state,
        "customer_intent_level": llm_output.get("customer_intent_level", "low"),
        "customer_info": customer_info
    }
    
    return result