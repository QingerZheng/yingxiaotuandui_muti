"""
响应格式验证和自动修复模块
专门用于验证和修复模型响应的格式问题，确保系统稳定性
"""

from typing import Dict, Any, Optional, Union, List
from states import EmotionalState, CustomerIntent, AppointmentInfo
import re
import logging

logger = logging.getLogger(__name__)

class ResponseValidator:
    """响应格式验证器"""
    
    @staticmethod
    def validate_emotional_state(data: Dict[str, Any]) -> EmotionalState:
        """验证和修复情感状态数据"""
        if not isinstance(data, dict):
            logger.warning(f"情感状态数据不是字典格式: {type(data)}")
            return EmotionalState()
        
        # 确保所有必需字段都存在且在有效范围内
        validated_data = {}
        
        # 情感状态的所有字段
        emotional_fields = [
            'security_level', 'familiarity_level', 'comfort_level', 
            'intimacy_level', 'gain_level', 'recognition_level', 'trust_level'
        ]
        
        for field in emotional_fields:
            value = data.get(field, 0.0)
            try:
                # 转换为浮点数并限制在0-1范围内
                validated_value = float(value)
                validated_value = max(0.0, min(1.0, validated_value))
                validated_data[field] = validated_value
            except (TypeError, ValueError):
                logger.warning(f"情感状态字段 {field} 值无效: {value}，使用默认值0.0")
                validated_data[field] = 0.0
        
        return EmotionalState(**validated_data)
    
    @staticmethod
    def validate_customer_intent_level(level: Any) -> str:
        """验证和修复客户意向等级"""
        if not isinstance(level, str):
            level = str(level).lower()
        
        valid_levels = ["low", "medium", "high", "fake_high"]
        
        if level in valid_levels:
            return level
        
        # 智能匹配
        level_mapping = {
            "低": "low", "低级": "low", "低等": "low",
            "中": "medium", "中等": "medium", "中级": "medium", "中度": "medium",
            "高": "high", "高级": "high", "高等": "high", "高度": "high",
            "假": "fake_high", "虚假": "fake_high", "假高": "fake_high",
            "0": "low", "1": "medium", "2": "high", "3": "fake_high"
        }
        
        for key, value in level_mapping.items():
            if key in str(level):
                logger.info(f"客户意向等级自动修复: {level} -> {value}")
                return value
        
        logger.warning(f"未知的客户意向等级: {level}，使用默认值 'low'")
        return "low"
    
    @staticmethod
    def validate_customer_intent(data: Dict[str, Any]) -> Optional[CustomerIntent]:
        """验证和修复客户意图数据"""
        if not isinstance(data, dict):
            return None
        
        # 验证意图类型
        intent_type = data.get("intent_type", "general_chat")
        valid_types = [
            "appointment_request", "time_confirmation", "price_inquiry",
            "concern_raised", "general_chat", "ready_to_book",
            "info_providing", "info_seeking"
        ]
        
        if intent_type not in valid_types:
            # 尝试智能匹配
            type_mapping = {
                "预约": "appointment_request", "约时间": "appointment_request",
                "时间": "time_confirmation", "确认时间": "time_confirmation",
                "价格": "price_inquiry", "费用": "price_inquiry", "多少钱": "price_inquiry",
                "担心": "concern_raised", "顾虑": "concern_raised", "疑问": "concern_raised",
                "聊天": "general_chat", "闲聊": "general_chat",
                "预定": "ready_to_book", "下单": "ready_to_book",
                "提供": "info_providing", "告诉": "info_providing",
                "询问": "info_seeking", "了解": "info_seeking", "咨询": "info_seeking"
            }
            
            for key, value in type_mapping.items():
                if key in str(intent_type):
                    intent_type = value
                    logger.info(f"意图类型自动修复: {data.get('intent_type')} -> {intent_type}")
                    break
            else:
                intent_type = "general_chat"
                logger.warning(f"未知的意图类型: {data.get('intent_type')}，使用默认值")
        
        # 验证置信度
        confidence = data.get("confidence", 0.5)
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.5
            logger.warning(f"置信度值无效: {data.get('confidence')}，使用默认值0.5")
        
        # 验证提取信息
        extracted_info = data.get("extracted_info", {})
        if not isinstance(extracted_info, dict):
            extracted_info = {}
            logger.warning("提取信息不是字典格式，使用空字典")
        
        # 验证所需动作
        requires_action = data.get("requires_action", [])
        if not isinstance(requires_action, list):
            requires_action = []
            logger.warning("所需动作不是列表格式，使用空列表")
        
        return CustomerIntent(
            intent_type=intent_type,
            confidence=confidence,
            extracted_info=extracted_info,
            requires_action=requires_action
        )
    
    @staticmethod
    def validate_score(score: Any) -> float:
        """验证和修复评分"""
        try:
            score = float(score)
            
            # 如果分数超出0-1范围，尝试智能调整
            if score > 1.0:
                if score <= 10.0:
                    # 可能是10分制，转换为1分制
                    score = score / 10.0
                    logger.info(f"分数从10分制转换为1分制: {score * 10} -> {score}")
                elif score <= 100.0:
                    # 可能是100分制，转换为1分制
                    score = score / 100.0
                    logger.info(f"分数从100分制转换为1分制: {score * 100} -> {score}")
                else:
                    # 超出合理范围，使用默认值
                    score = 0.5
                    logger.warning(f"分数超出合理范围，使用默认值0.5")
            
            # 确保在0-1范围内
            score = max(0.0, min(1.0, score))
            return score
            
        except (TypeError, ValueError):
            logger.warning(f"无效的分数值: {score}，使用默认值0.5")
            return 0.5
    
    @staticmethod
    def validate_invitation_status(status: Any) -> int:
        """验证和修复邀约状态"""
        try:
            status = int(status)
            if status in [0, 1, 2]:
                return status
            else:
                # 尝试智能映射
                if status > 2:
                    return 1  # 大于2的值可能表示确认
                else:
                    return 0  # 其他值默认为未邀约
        except (TypeError, ValueError):
            # 尝试从字符串中提取含义
            status_str = str(status).lower()
            if any(word in status_str for word in ["同意", "确认", "好的", "可以", "是的", "yes", "ok"]):
                return 1
            elif any(word in status_str for word in ["推迟", "延期", "改时间", "postpone", "delay"]):
                return 2
            else:
                return 0
    
    @staticmethod
    def validate_response_text(text: Any) -> str:
        """验证和修复响应文本"""
        if text is None:
            return ""
        
        text = str(text).strip()
        
        # 移除可能的格式标记
        text = re.sub(r'^```.*?\n', '', text)
        text = re.sub(r'\n```$', '', text)
        
        # 限制长度（避免过长的响应）
        max_length = 500
        if len(text) > max_length:
            text = text[:max_length] + "..."
            logger.warning(f"响应文本过长，已截断到{max_length}字符")
        
        return text
    
    @staticmethod
    def validate_timestamp(timestamp: Any) -> Optional[int]:
        """验证和修复时间戳"""
        if timestamp is None:
            return None
        
        try:
            timestamp = int(timestamp)
            
            # 检查时间戳格式（10位或13位）
            if len(str(timestamp)) == 10:
                # 10位时间戳，转换为13位
                timestamp = timestamp * 1000
                logger.info("时间戳从10位转换为13位")
            elif len(str(timestamp)) != 13:
                # 不是有效的时间戳格式
                logger.warning(f"无效的时间戳格式: {timestamp}")
                return None
            
            # 检查时间戳是否在合理范围内（2020-2030年）
            min_timestamp = 1577836800000  # 2020-01-01
            max_timestamp = 1893456000000  # 2030-01-01
            
            if timestamp < min_timestamp or timestamp > max_timestamp:
                logger.warning(f"时间戳超出合理范围: {timestamp}")
                return None
            
            return timestamp
            
        except (TypeError, ValueError):
            logger.warning(f"无法转换为时间戳: {timestamp}")
            return None

def validate_and_fix_response(response_data: Dict[str, Any], context: str) -> Dict[str, Any]:
    """
    统一的响应验证和修复函数
    
    Args:
        response_data: 需要验证的响应数据
        context: 上下文（如"状态评估"、"意图分析"等）
        
    Returns:
        验证和修复后的响应数据
    """
    validator = ResponseValidator()
    validated_data = {}
    
    try:
        if "状态评估" in context:
            # 验证情感状态
            if "emotional_state" in response_data:
                validated_data["emotional_state"] = validator.validate_emotional_state(
                    response_data["emotional_state"]
                )
            
            # 验证客户意向等级
            if "customer_intent_level" in response_data:
                validated_data["customer_intent_level"] = validator.validate_customer_intent_level(
                    response_data["customer_intent_level"]
                )
            
            # 验证客户信息
            if "customer_info" in response_data:
                customer_info = response_data["customer_info"]
                if isinstance(customer_info, dict):
                    validated_data["customer_info"] = customer_info
                else:
                    validated_data["customer_info"] = {}
        
        elif "意图分析" in context:
            # 验证客户意图
            validated_intent = validator.validate_customer_intent(response_data)
            if validated_intent:
                validated_data["customer_intent"] = validated_intent
        
        elif "邀约" in context:
            # 验证邀约状态
            if "invitation_status" in response_data:
                validated_data["invitation_status"] = validator.validate_invitation_status(
                    response_data["invitation_status"]
                )
            
            # 验证时间戳
            if "invitation_time" in response_data:
                validated_data["invitation_time"] = validator.validate_timestamp(
                    response_data["invitation_time"]
                )
            
            # 验证项目名称
            if "invitation_project" in response_data:
                project = response_data["invitation_project"]
                validated_data["invitation_project"] = str(project) if project else None
        
        elif "评估" in context:
            # 验证分数
            if "score" in response_data:
                validated_data["score"] = validator.validate_score(response_data["score"])
            
            # 验证推理
            if "reasoning" in response_data:
                validated_data["reasoning"] = validator.validate_response_text(
                    response_data["reasoning"]
                )
        
        elif "生成响应" in context:
            # 验证响应文本
            if "response" in response_data:
                validated_data["response"] = validator.validate_response_text(
                    response_data["response"]
                )
        
        # 保留其他未处理的字段
        for key, value in response_data.items():
            if key not in validated_data:
                validated_data[key] = value
        
        logger.info(f"响应验证完成 - {context}: {len(validated_data)} 个字段")
        return validated_data
        
    except Exception as e:
        logger.error(f"响应验证失败 - {context}: {e}")
        return response_data  # 验证失败时返回原始数据