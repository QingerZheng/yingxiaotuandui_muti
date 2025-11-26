"""
MAS Cloud Agent - 通用类型定义和数据结构
"""
from enum import Enum
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict, Annotated
from dataclasses import dataclass, field, asdict
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
class EventType(str, Enum):
    """事件类型枚举"""
    OPENING_GREETING = "opening_greeting"      # 开场白事件
    CUSTOMER_FOLLOWUP = "customer_followup"    # 客户回访事件
    APPOINTMENT_REMINDER = "appointment_reminder"  # 邀约提醒事件
    PENDING_ACTIVATION = "pending_activation"  # 待唤醒事件
    CONNECTION_ATTEMPT = "connection_attempt"  # 连接尝试事件
class EventInstance(BaseModel):
    """事件实例"""
    event_type: EventType = Field(..., description="事件类型")
    event_time: str = Field(..., description="事件时间，始终为字符串")

    def dict(self, *args, **kwargs):
        d = super().dict(*args, **kwargs)
        return d

@dataclass
class Emotionalstate:#其实完全可以考虑将情感状态的数据传给meta，结合一些数学技巧作判断。目前完全是基于写死的搜索空间去做回复。
    """用户情感状态数据类"""
    security_level: float = field(default=0.0)      # 安全感等级 (0-1)
    familiarity_level: float = field(default=0.0)   # 熟悉感等级 (0-1)
    comfort_level: float = field(default=0.0)       # 舒适感等级 (0-1)
    intimacy_level: float = field(default=0.0)      # 亲密感等级 (0-1)
    gain_level: float = field(default=0.0)          # 获得感等级 (0-1)
    recognition_level: float = field(default=0.0)   # 认同感等级 (0-1)
    trust_level: float = field(default=0.0)         # 信任感等级 (0-1)

    # 为向后兼容 pydantic BaseModel 的接口，补充两个辅助方法
    def model_dump(self) -> dict:
        """返回 dataclass 字典表示，以兼容 pydantic 的 model_dump。"""
        return asdict(self)

    def model_dump_json(self) -> str:
        """返回 JSON 字符串，以兼容 model_dump_json 调用。"""
        import json
        return json.dumps(asdict(self), ensure_ascii=False)

class CustomerIntent(BaseModel):
    """客户行为意图分析结果"""
    intent_type: str  # "appointment_request", "price_inquiry", "concern_raised", "general_chat", "ready_to_book"
    confidence: float  # 0.0-1.0 置信度
    extracted_info: Dict[str, Any] = {}  # 提取的结构化信息
    requires_action: List[str] = []  # 需要的后续动作

class AppointmentInfo(BaseModel):
    """预约信息管理"""
    has_time: bool = False
    preferred_time: Optional[str] = None
    has_name: bool = False
    customer_name: Optional[str] = None
    has_phone: bool = False
    customer_phone: Optional[str] = None
    has_address_confirmed: bool = False
    preferred_service: Optional[str] = None
    appointment_status: str = "pending"  # "pending", "confirmed", "info_collecting"

class Agentstate(TypedDict):
    """LangGraph 状态定义"""
    # 核心消息处理
    messages: Annotated[List[BaseMessage], add_messages]
    user_input: Optional[str]  # 专门用于接收单次用户输入

    thread_id: str  # 用于标识对话线程，必要
    assistant_id: str  # 用于助手识别，必要
    
    # 对话状态
    current_stage: str  # 当前对话阶段
    emotional_state: Emotionalstate  # 用户情感状态
    user_profile: dict  # 用户信息，如痛点、兴趣等
    turn_count: int  # 对话轮次计数
    customer_intent_level: Optional[str]  # 客户意向：low, medium, high, fake_high，后期应该会改到五级，但是在graph的meta_design_node里会很难办。
    
    # 运行时控制字段
    internal_monologue: Optional[List[str]]  # 内部独白，用于调试，简化版
    candidate_actions: Optional[List[str]]   # 候选行动
    evaluated_responses: Optional[List[Dict[str, Any]]]  # 评估的回复
    final_response: Optional[str]  # 最终回复
    last_message: Optional[str]  # API输出的最后消息


    # 模型配置
    agent_temperature: Optional[float]  # 生成温度
    node_model: Optional[str]  # 节点使用的模型
    feedback_model: Optional[str]  # 反馈模型
    verbose: Optional[bool]  # 调试开关

    # 新增：行为意图和预约管理
    customer_intent: Optional[CustomerIntent]  # 客户行为意图
    appointment_info: Optional[AppointmentInfo]  # 预约信息
    
    # 人工接管控制
    need_tools: Optional[bool]  # 是否需要调用工具

    # 新增：事件系统相关字段
    event_instance: Optional[EventInstance]  # 当前事件实例
    appointment_time: Optional[str]  # 预约时间
    user_last_reply_time: Optional[str]  # 用户最后回复时间
    last_active_send_time: Optional[str]  # 最后主动发送时间
    event_info: Optional[bool]  # 仅仅做事件请求通知发送用，不参与决策，不用传入，本地调试打印专用。
    user_treatment_completion_info: Optional[str]
    event_happens:Optional[bool]#是否有事件发生了

    # 额外参数
    send_response_yes_or_no: Optional[bool]  # 是否需要给用户发消息
    user_requires_message: Optional[bool]  # 用户需要我发消息给用户
    sales_requires_message: Optional[bool]  # 销售需要我发消息给用户






@dataclass
class DebugInfo:
    """用于API输出的调试信息"""
    current_stage: Optional[str] = None
    emotional_state: Optional[Dict[str, Any]] = None
    internal_monologue: Optional[List[str]] = None

@dataclass
class MasAgentOutput:
    """MAS Agent的API输出状态 - 只包含必要字段"""
    last_message: str = field(default="")
    debug_info: Optional[DebugInfo] = field(default=None)
    event_instance: Optional[EventInstance] = field(default=None)  # 当前事件实例
    appointment_time: Optional[str] = field(default=None)  # 预约时间
    user_last_reply_time: Optional[str] = field(default=None)  # 用户最后回复时间
    last_active_send_time: Optional[str] = field(default=None)  # 最后主动发送时间
    customer_info: Optional[Dict[str, str]] = field(default=None)  # Extracted customer information