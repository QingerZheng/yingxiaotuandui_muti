"""
通用JSON解析工具模块
专门处理国内模型（如GLM）的JSON输出不稳定问题，提供多层兜底机制
"""

import json
import re
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

def robust_json_parse(
    response_text: str, 
    context: str = "未知", 
    fallback_dict: Optional[Dict[str, Any]] = None,
    debug: bool = True,
    validate: bool = True
) -> Dict[str, Any]:
    """
   JSON解析函数，专门处理国内模型的输出格式问题
    
    Args:
        response_text: 模型的原始响应文本
        context: 上下文描述，用于调试
        fallback_dict: 解析失败时的兜底字典
        debug: 是否输出调试信息
        
    Returns:
        解析后的字典对象
    """
    if fallback_dict is None:
        fallback_dict = {}
        
    if not response_text or response_text.strip() == "":
        if debug:
            print(f"[JSON解析-{context}] 空响应，使用兜底字典")
        return fallback_dict
    
    original_text = response_text
    
    # 第一步：直接尝试解析
    try:
        result = json.loads(response_text.strip())
        if debug:
            print(f"[JSON解析-{context}] 直接解析成功")
        
        # 应用验证
        if validate and result:
            try:
                from response_validator import validate_and_fix_response
                result = validate_and_fix_response(result, context)
                if debug:
                    print(f"[JSON解析-{context}] 响应验证完成")
            except ImportError:
                if debug:
                    print(f"[JSON解析-{context}] 响应验证模块未找到，跳过验证")
            except Exception as e:
                if debug:
                    print(f"[JSON解析-{context}] 响应验证失败: {e}")
        
        return result
    except json.JSONDecodeError as e:
        if debug:
            print(f"[JSON解析-{context}] 直接解析失败: {e}")
    
    # 第二步：清理常见的格式问题
    cleaned_text = clean_response_text(response_text)
    if cleaned_text != response_text:
        try:
            result = json.loads(cleaned_text)
            if debug:
                print(f"[JSON解析-{context}] 清理后解析成功")
            return result
        except json.JSONDecodeError as e:
            if debug:
                print(f"[JSON解析-{context}] 清理后解析失败: {e}")
    
    # 第三步：正则提取JSON部分
    json_extracted = extract_json_from_text(cleaned_text)
    if json_extracted:
        try:
            result = json.loads(json_extracted)
            if debug:
                print(f"[JSON解析-{context}] 正则提取后解析成功")
            return result
        except json.JSONDecodeError as e:
            if debug:
                print(f"[JSON解析-{context}] 正则提取后解析失败: {e}")
    
    # 第四步：尝试修复常见的JSON语法错误
    fixed_json = fix_common_json_errors(json_extracted or cleaned_text)
    if fixed_json:
        try:
            result = json.loads(fixed_json)
            if debug:
                print(f"[JSON解析-{context}] 语法修复后解析成功")
            return result
        except json.JSONDecodeError as e:
            if debug:
                print(f"[JSON解析-{context}] 语法修复后解析失败: {e}")
    
    # 第五步：尝试从文本中提取关键信息（针对特定场景）
    extracted_info = extract_info_from_text(original_text, context)
    if extracted_info:
        if debug:
            print(f"[JSON解析-{context}] 文本信息提取成功: {extracted_info}")
        return extracted_info
    
    # 最终兜底
    if debug:
        print(f"[JSON解析-{context}] 所有解析方法失败，使用兜底字典")
        print(f"[JSON解析-{context}] 原始响应: {original_text[:200]}...")
    
    result = fallback_dict
    
    # 如果启用验证，对结果进行验证和修复
    if validate and result:
        try:
            from response_validator import validate_and_fix_response
            result = validate_and_fix_response(result, context)
            if debug:
                print(f"[JSON解析-{context}] 响应验证完成")
        except ImportError:
            if debug:
                print(f"[JSON解析-{context}] 响应验证模块未找到，跳过验证")
        except Exception as e:
            if debug:
                print(f"[JSON解析-{context}] 响应验证失败: {e}")
        
    return result

def clean_response_text(text: str) -> str:
    """清理响应文本中的常见格式问题"""
    # 移除markdown代码块标记
    text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
    
    # 移除可能的前缀说明文字 - 只有当文本不是以{开头时才处理
    if not text.lstrip().startswith('{'):
        text = re.sub(r'^.*?(?=\{)', '', text, flags=re.DOTALL)
    
    # 移除可能的后缀说明文字（但要保持JSON的完整性）
    # 只有当最后一个}后面还有非空白字符时才进行清理
    if re.search(r'\}\s*[^\s]', text):
        # 使用更安全的方式：找到最后一个}的位置
        last_brace_pos = text.rfind('}')
        if last_brace_pos != -1:
            # 检查}后面是否有非空白字符
            after_brace = text[last_brace_pos + 1:].strip()
            if after_brace:
                text = text[:last_brace_pos + 1]
    
    # 清理空白字符
    text = text.strip()
    
    return text

def extract_json_from_text(text: str) -> Optional[str]:
    """从文本中提取JSON部分"""
    # 首先尝试手动解析括号匹配（更准确）
    bracket_result = extract_json_by_brackets(text)
    if bracket_result:
        return bracket_result
    
    # 如果手动解析失败，尝试正则表达式
    patterns = [
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # 简单嵌套
        r'\{.*?\}',  # 最简单的匹配
    ]
    
    for pattern in patterns:
        try:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                # 返回最长的匹配
                return max(matches, key=len)
        except Exception:
            continue
    
    return None

def extract_json_by_brackets(text: str) -> Optional[str]:
    """通过括号匹配提取JSON"""
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    bracket_count = 0
    for i, char in enumerate(text[start_idx:], start_idx):
        if char == '{':
            bracket_count += 1
        elif char == '}':
            bracket_count -= 1
            if bracket_count == 0:
                return text[start_idx:i+1]
    
    return None

def fix_common_json_errors(text: str) -> Optional[str]:
    """修复常见的JSON语法错误"""
    if not text:
        return None
    
    # 修复常见问题
    fixes = [
        # 修复单引号
        (r"'([^']*)':", r'"\1":'),
        # 修复尾随逗号
        (r',\s*}', '}'),
        (r',\s*]', ']'),
        # 修复未引用的键
        (r'(\w+):', r'"\1":'),
        # 修复True/False/None
        (r'\bTrue\b', 'true'),
        (r'\bFalse\b', 'false'),
        (r'\bNone\b', 'null'),
    ]
    
    fixed_text = text
    for pattern, replacement in fixes:
        try:
            fixed_text = re.sub(pattern, replacement, fixed_text)
        except Exception:
            continue
    
    return fixed_text if fixed_text != text else None

def extract_info_from_text(text: str, context: str) -> Optional[Dict[str, Any]]:
    """从文本中提取关键信息（针对特定场景的兜底策略）"""
    result = {}
    
    # 根据不同上下文提取不同信息
    if "状态评估" in context or "emotion" in context.lower():
        # 尝试从文本中提取各种情感指标
        emotional_state = {
            "security_level": 0.0,
            "familiarity_level": 0.0,
            "comfort_level": 0.0,
            "intimacy_level": 0.0,
            "gain_level": 0.0,
            "recognition_level": 0.0,
            "trust_level": 0.0
        }
        
        # 提取信任度
        trust_patterns = [
            r'信任.*?(\d+\.?\d*)',
            r'trust.*?(\d+\.?\d*)',
            r'"trust_level".*?(\d+\.?\d*)'
        ]
        for pattern in trust_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                trust_value = float(match.group(1))
                if trust_value > 1:
                    trust_value = trust_value / 100
                emotional_state["trust_level"] = min(max(trust_value, 0.0), 1.0)
                break
        
        # 提取舒适度
        comfort_patterns = [
            r'舒适.*?(\d+\.?\d*)',
            r'comfort.*?(\d+\.?\d*)',
            r'"comfort_level".*?(\d+\.?\d*)'
        ]
        for pattern in comfort_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                comfort_value = float(match.group(1))
                if comfort_value > 1:
                    comfort_value = comfort_value / 100
                emotional_state["comfort_level"] = min(max(comfort_value, 0.0), 1.0)
                break
        
        # 提取熟悉度
        familiarity_patterns = [
            r'熟悉.*?(\d+\.?\d*)',
            r'familiar.*?(\d+\.?\d*)',
            r'"familiarity_level".*?(\d+\.?\d*)'
        ]
        for pattern in familiarity_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                familiarity_value = float(match.group(1))
                if familiarity_value > 1:
                    familiarity_value = familiarity_value / 100
                emotional_state["familiarity_level"] = min(max(familiarity_value, 0.0), 1.0)
                break
        
        result["emotional_state"] = emotional_state
        
        # 提取客户意向等级
        intent_patterns = [
            r'意向.*?(high|medium|low|fake_high)',
            r'intent.*?(high|medium|low|fake_high)',
            r'"customer_intent_level".*?"(high|medium|low|fake_high)"'
        ]
        for pattern in intent_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["customer_intent_level"] = match.group(1).lower()
                break
        
        if "customer_intent_level" not in result:
            result["customer_intent_level"] = "low"
        
        # 提取客户信息
        result["customer_info"] = {}
    
    elif "意图分析" in context or "intent" in context.lower():
        # 意图分析相关信息提取
        intent_patterns = [
            r'appointment_request', r'time_confirmation', r'price_inquiry',
            r'concern_raised', r'general_chat', r'ready_to_book',
            r'info_providing', r'info_seeking'
        ]
        for pattern in intent_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                result["intent_type"] = pattern
                result["confidence"] = 0.7  # 默认置信度
                break
    
    elif "邀约" in context or "invitation" in context.lower():
        # 邀约状态相关信息提取
        if re.search(r'同意|确认|好的|可以', text):
            result["invitation_status"] = 1
        elif re.search(r'拒绝|不行|不可以|取消', text):
            result["invitation_status"] = 0
        elif re.search(r'推迟|延期|改时间', text):
            result["invitation_status"] = 2
        else:
            result["invitation_status"] = 0
        
        result["invitation_time"] = None
        result["invitation_project"] = None
    
    return result if result else None

def create_fallback_dict(context: str) -> Dict[str, Any]:
    """根据上下文创建合适的兜底字典"""
    if "状态评估" in context or "emotion" in context.lower():
        # 返回字典格式的EmotionalState数据，而不是实例
        return {
            "emotional_state": {
                "security_level": 0.0,
                "familiarity_level": 0.0,
                "comfort_level": 0.0,
                "intimacy_level": 0.0,
                "gain_level": 0.0,
                "recognition_level": 0.0,
                "trust_level": 0.0
            },
            "customer_intent_level": "low",
            "customer_info": {}
        }
    elif "意图分析" in context or "intent" in context.lower():
        return {
            "customer_intent": None
        }
    elif "邀约" in context or "invitation" in context.lower():
        return {
            "invitation_status": 0,
            "invitation_time": None,
            "invitation_project": None
        }
    else:
        return {}

# 装饰器函数，用于包装需要JSON解析的函数
def json_parse_wrapper(context: str, debug: bool = True):
    """装饰器，为函数添加鲁棒的JSON解析功能"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (json.JSONDecodeError, ValueError) as e:
                if debug:
                    print(f"[{context}] JSON解析异常，使用兜底机制: {e}")
                return create_fallback_dict(context)
            except Exception as e:
                if debug:
                    print(f"[{context}] 其他异常: {e}")
                return create_fallback_dict(context)
        return wrapper
    return decorator

def safe_create_emotional_state(data: Any):
    """安全地创建EmotionalState实例"""
    from states import EmotionalState
    
    if isinstance(data, EmotionalState):
        return data
    elif isinstance(data, dict):
        try:
            # 确保所有必需的字段都存在，并且是正确的类型
            safe_data = {}
            fields = ['security_level', 'familiarity_level', 'comfort_level', 
                     'intimacy_level', 'gain_level', 'recognition_level', 'trust_level']
            
            for field in fields:
                value = data.get(field, 0.0)
                if isinstance(value, (int, float)):
                    safe_data[field] = float(value)
                else:
                    safe_data[field] = 0.0
            
            return EmotionalState(**safe_data)
        except Exception as e:
            print(f"[安全创建情感状态] 使用字典创建失败: {e}")
            return EmotionalState()
    else:
        print(f"[安全创建情感状态] 不支持的数据类型: {type(data)}")
        return EmotionalState()