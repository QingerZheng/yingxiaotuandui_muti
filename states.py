from dataclasses import field, asdict, dataclass
from enum import Enum
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
class EventType(str, Enum):
    """事件类型枚举，后期可以自己添加新事件，比如增加主动结束聊天到下次唤醒的事件，增加生日事件等等"""
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
class EmotionalState:#其实完全可以考虑将情感状态的数据传给meta，结合一些数学技巧作判断。目前完全是基于写死的搜索空间去做回复。
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

@dataclass
class DebugInfo:
    """用于API输出的调试信息"""
    current_stage: Optional[str] = None
    emotional_state: Optional[Dict[str, Any]] = None
    internal_monologue: Optional[List[str]] = None

class AgentInput(TypedDict):
    messages: Optional[List[BaseMessage]]#新消息，用户发消息时必要，用来输入
    thread_id: Optional[str]  # 用于标识对话线程，必要
    assistant_id: Optional[str]   # 用于助手识别，必要
    event_instance: Optional[EventInstance]  # 当前事件实例
    appointment_time: Optional[str]  # 预约时间
    user_last_reply_time: Optional[str]  # 用户最后回复时间
    last_active_send_time: Optional[str]  # 最后主动发送时间

    agent_temperature: Optional[float] # 生成温度
    # 是否在本轮生成语音回复
    audio_reply: Optional[bool]


class AgentState(TypedDict):
    messages: Optional[List[BaseMessage]]  # 新消息原始格式，用户发消息时必要，用来输入
    processed_messages: Optional[List[BaseMessage]]# 新消息预处理格式，是messages的文本转化格式
    long_term_messages: Optional[List[BaseMessage]]
    thread_id: Optional[str]  # 用于标识对话线程，必要，用来输入，绑定线程
    assistant_id: Optional[str]   # 用于助手识别，必要，用来输入，绑定人格，助手
    last_message: Optional[str]   # ai生成的消息，必要，用来输出
    last_message_audio_url: Optional[str]  # 若启用TTS，生成的语音MP3公网URL
    final_response: Optional[str] #最终回复，本地调试使用
    evaluated_responses: List[Dict[str, Any]]  # 评估的回复

    # 子图1：可选工具对应返回参数
    used_tools: Optional[List[Dict[str, str]]]  # 工具调用列表，如 [{"tool": "rag", "reason": "..."}, {"tool": "search", "reason": "..."}]
    tool_results: Optional[List[Dict[str, str]]]  # 新增：通用工具结果

    # 子图2：必选工具变量参数
    emotional_state:Optional[EmotionalState]
    customer_intent:Optional[CustomerIntent]
    appointment_info:Optional[AppointmentInfo]
    debug_info: Optional[DebugInfo]
    agent_temperature: Optional[float]  # 生成温度
    verbose: Optional[bool]  # 调试开关
    turn_count: Optional[int]# 对话轮次计数
    customer_intent_level: Optional[str]# 客户意向：low, medium, high, fake_high，后期应该会改到五级，但是在graph的meta_design_node里会很难办。
    candidate_actions: List[str]  # 候选行动
    # 运行时控制：是否在本轮生成语音回复
    audio_reply: Optional[bool]

    # 子图3：对应事件返回参数
    event_instance: Optional[EventInstance]  # 当前事件实例
    appointment_time: Optional[str]  # 预约时间
    user_last_reply_time: Optional[str]  # 用户最后回复时间
    last_active_send_time: Optional[str]  # 最后主动发送时间
    event_info: Optional[bool]  # 仅仅做事件请求通知发送用，不参与决策，不用传入，本地调试打印专用。
    user_treatment_completion_info: Optional[str]
    customer_info: Optional[Dict[str, str]]


    #额外判断参数
    event_happens: Optional[bool]  # 是否有事件发生了
    send_response_yes_or_no: Optional[bool]#是否需要给用户发消息
    user_requires_message: Optional[bool]# 用户需要我发消息给用户
    sales_requires_message: Optional[bool]# 销售需要我发消息给用户

    #增加邀约状态、邀约时间、邀约项目的字段
    invitation_status: Optional[int]  # 是否邀约状态，False 或者True
    invitation_time: Optional[str]  # 邀约的具体时间，如："2025-08-07 15:00"
    invitation_project: Optional[str]  # 邀约的项目名称，如：皮肤管理、光子嫩肤等

    # 语音识别相关字段
    custom_audio_text: Optional[List[str]]  # 用户语音识别出的文字内容数组，与用户消息数量对应

    # 图片发送相关字段
    selected_image: Optional[Dict[str, str]]  # 选中的图片信息 {"id": str, "name": str}
    image_request_detected: Optional[bool]  # 是否检测到图片请求

@dataclass
class AgentOutput:
    """Agent的API输出状态 - 只包含必要字段"""
    # last_message: Optional[str] = field(default=None)
    messages: Optional[List[Dict[str, Any]]] = field(default=None)
    custom_audio_text: Optional[List[str]] = field(default=None)  # 用户语音识别出的文字内容数组，与用户消息数量对应
    custom_status: Optional[Dict[str, Any]] = field(default=None)
    token_usage: Optional[Dict[str, Any]] = field(default=None)