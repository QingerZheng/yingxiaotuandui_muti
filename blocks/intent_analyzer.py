"""
客户行为意图识别模块
专门识别客户的具体行为意图，如预约请求、价格咨询、顾虑表达等
"""

from typing import Dict, List, Any, Optional

from langchain_core.messages import HumanMessage
from Configurations import Configuration

from states import AgentState, CustomerIntent, AppointmentInfo
import json
import re
from langchain_core.tools import tool
from json_parser_utils import robust_json_parse, create_fallback_dict
@tool
def analyze_customer_intent(state_dict: dict = None):
    """
    分析客户的行为意图，识别具体的行为信号
    """
    # 如果传入的是包装的字典，提取实际的state_dict
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
        
    messages = state_dict.get("long_term_messages", [])
    if not messages:
        return {}
    
    # 获取最新的客户消息
    last_customer_msg = None
    for msg in reversed(messages):
        if msg.type == "human":
            last_customer_msg = msg.content
            break
    
    if not last_customer_msg:
        return {}
    

    
    from agents.persona_config.config_manager import config_manager
    cfg = config_manager.get_config() or {}
    try:
        from llm import create_llm
        llm = create_llm(
            model_provider=cfg.get("model_provider", "openrouter"),
            model_name=cfg.get("intent_model", cfg.get("model_name", "x-ai/grok-code-fast-1")),
            temperature=0.5
        )
    except Exception as e:
        print(f"错误：无法创建意图分析模型: {e}")
        return {}
    
    # 构建对话历史
    dialog_history = "\n".join([f"{msg.type}: {msg.content}" for msg in messages[-5:]])  # 只取最近5轮
    
    prompt = f"""
你是一个客户行为意图识别专家。分析客户的最新消息，识别其具体的行为意图。

**对话历史（最近5轮）:**
{dialog_history}

**客户最新消息:** "{last_customer_msg}"

**意图类型定义:**
1. "appointment_request" - 客户明确表达预约意图（如确认时间、询问预约等）
2. "time_confirmation" - 客户确认或询问具体时间
3. "price_inquiry" - 询问价格、费用相关
4. "concern_raised" - 表达顾虑、担心、疑问（如担心效果、风险等）
5. "general_chat" - 一般聊天、寒暄
6. "ready_to_book" - 准备下单、预约
7. "info_providing" - 提供个人信息（姓名、电话等）
8. "info_seeking" - 寻求信息，如询问可用项目、服务细节等

**分析任务:**
1. 识别客户的主要意图类型
2. 评估意图的置信度（0.0-1.0）
3. 提取关键信息（时间、价格、服务类型等）
4. 确定需要的后续动作

**输出格式:**
```json
{{
    "intent_type": "具体的意图类型",
    "confidence": 0.9,
    "extracted_info": {{
        "time": "3点",
        "service": "光子嫩肤",
        "price_range": "1000-2000"
    }},
    "requires_action": ["confirm_time", "collect_contact", "provide_address"]
}}
```
"""

    try:
        message = HumanMessage(content=prompt)
        response_result = llm.invoke(
            [message],
            response_format={'type': 'json_object'}
        )
        response_text = response_result.content if hasattr(response_result, 'content') else str(response_result)
        
        print(f"[DEBUG-意图分析] 原始模型响应: {response_text}")
        
        # JSON解析工具
        fallback_dict = create_fallback_dict("意图分析")
        intent_data = robust_json_parse(
            response_text, 
            context="意图分析", 
            fallback_dict=fallback_dict,
            debug=True
        )
        
        print(f"[DEBUG-意图分析] 解析结果: {intent_data}")
        
        # 如果解析成功且包含客户意图信息
        if intent_data and "intent_type" in intent_data:
            return {
                "customer_intent": CustomerIntent(
                    intent_type=intent_data.get("intent_type", "general_chat"),
                    confidence=float(intent_data.get("confidence", 0.5)),
                    extracted_info=intent_data.get("extracted_info", {}),
                    requires_action=intent_data.get("requires_action", [])
                )
            }
        else:
            print(f"[DEBUG-意图分析] 解析结果不包含有效意图信息，返回空结果")
            return {}
            
    except Exception as e:
        print(f"[ERROR-意图分析] 模型调用或解析失败: {e}")
        if 'response_text' in locals() and response_text is not None:
            print(f"[ERROR-意图分析] 原始响应: {response_text[:300]}...")
        return {}

def update_appointment_info(state: AgentState, customer_intent: CustomerIntent) -> Dict[str, Any]:
    """
    根据客户意图更新预约信息
    """
    current_appointment = state.get("appointment_info") or AppointmentInfo()
    extracted_info = customer_intent.extracted_info
    
    # 更新时间信息
    if "time" in extracted_info and not current_appointment.has_time:
        current_appointment.has_time = True
        current_appointment.preferred_time = extracted_info["time"]
    
    # 更新姓名信息
    if "name" in extracted_info and not current_appointment.has_name:
        current_appointment.has_name = True
        current_appointment.customer_name = extracted_info["name"]
    
    # 更新电话信息
    if "phone" in extracted_info and not current_appointment.has_phone:
        current_appointment.has_phone = True
        current_appointment.customer_phone = extracted_info["phone"]
    
    # 更新服务类型
    if "service" in extracted_info:
        current_appointment.preferred_service = extracted_info["service"]
    
    # 更新预约状态
    if customer_intent.intent_type in ["appointment_request", "time_confirmation", "ready_to_book"]:
        if current_appointment.appointment_status == "pending":
            current_appointment.appointment_status = "info_collecting"
    
    return {"appointment_info": current_appointment}