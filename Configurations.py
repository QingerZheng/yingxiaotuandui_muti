"""Define the configurable parameters for the agent."""

from __future__ import annotations
from pydantic import BaseModel, Field
from langchain_core.runnables import ensure_config
from langgraph.config import get_config
from typing import Optional, Union
import os
from dotenv import load_dotenv


class Configuration(BaseModel):
    # Agent配置
    agent_name: str = Field(default="{{}}", description="智能体名字")
    agent_nickname: str = Field(default="小{{}}", description="智能体昵称")
    agent_gender: str = Field(default="女", description="限定性别，避免任意值")
    agent_age: Union[int, str] = Field(default=26, description="年龄可以是整数或字符串")
    agent_birthday: str = Field(default="", description="生日日期")
    agent_address: str = Field(default="", description="个人地址")
    agent_role: str = Field(default="顾问", description="角色/职位")
    company_name: str = Field(default="{{}}", description="公司名称")
    industry_knowledge: str = Field(default="医美", description="所在行业的潜在知识，如医美行业：价格表不会直接透露等")
    # 兼容旧字段：industry（模板与部分prompt仍使用）
    industry: str = Field(default="医美", description="所在行业（兼容旧字段）")
    agent_personality: str = Field(default="热情", description="个性描述（如'热情')")
    agent_goal: str = Field(default="提供专业咨询服务", description="主要目标")
    agent_side_goal: str = Field(default="建立客户关系", description="次要目标")
    company_address: str = Field(default="")
    service_time: str = Field(default="")
    service_price: str = Field(default="特色优势：{{}}\n核心服务：")
    extra_infomation: str = Field(default="")
    base_context_prompt: str = Field(default="角色与身份\n你是{agent_name}，你是一个专业顾问\n你以亲切、包容、专业的语气为客户提供{industry_knowledge}咨询和到店服务...........")
    model_provider: str = Field(default="openrouter")
    model_name: str = Field(default="deepseek/deepseek-chat-v3.1")
    agent_temperature: float = Field(default=0.6)  

    # API Keys (从环境变量加载，不暴露具体值)
    openai_api_key: Optional[str] = Field(default=None, exclude=True)
    dashscope_api_key: Optional[str] = Field(default=None, exclude=True)
    cloud_kb_api_key: Optional[str] = Field(default=None, exclude=True)
    tavily_api_key: Optional[str] = Field(default=None, exclude=True)
    
    # Endpoints
    cloud_kb_api_endpoint: Optional[str] = Field(default=None, exclude=True)
    
    # Model Settings
    # 核心对话生成模型
    generation_model: str = Field(default="openai/gpt-5-chat")
    # 状态评估模型
    evaluation_model: str = Field(default="deepseek/deepseek-chat-v3.1")
    # 意图分析模型
    intent_model: str = Field(default="deepseek/deepseek-chat-v3.1")
    # 工具规划模型
    planning_model: str = Field(default="deepseek/deepseek-chat-v3.1")
    # 事件决策模型
    decision_model: str = Field(default="deepseek/deepseek-chat-v3.1")
    # 自我验证模型
    verification_model: str = Field(default="deepseek/deepseek-chat-v3.1")
    # 视觉模型（默认使用 OpenRouter 的 z-ai/glm-4.5v，支持多模态图片输入）
    vision_model: str = Field(default="z-ai/glm-4.5v")
    # 语音识别模型
    whisper_model: str = Field(default="whisper-1")
    # 嵌入模型
    embedding_model: str = Field(default="text-embedding-v4")
    embedding_dimension: int = Field(default=768)
    # ==== TTS 配置 ====
    tts_enabled: bool = Field(default=False, description="是否启用语音回复（全局默认，单轮可由输入覆盖）")
    tts_provider: str = Field(default="stepfun", description="TTS提供商：stepfun")
    tts_model: str = Field(default="step-tts-vivid", description="TTS模型名称（按提供商文档）")
    tts_voice: str = Field(default="huolinvsheng", description="默认中文女声，按提供商可用列表设置")
    tts_format: str = Field(default="mp3", description="输出格式：mp3/wav等")
    tts_speed: float = Field(default=1.0, description="语速倍速，1.0为标准")
    tts_pitch: float = Field(default=0.0, description="音高调整，0为不变")
    
    def model_post_init(self, __context):
        """Pydantic v2 post-init method for handling environment variables and field validation"""
        # 延迟加载环境变量，避免在异步上下文中阻塞
        try:
            import asyncio
            # 检查是否在异步上下文中
            if asyncio.get_event_loop().is_running():
                # 在异步上下文中，跳过 load_dotenv 避免阻塞
                pass
            else:
                # 在同步上下文中，正常加载
                load_dotenv()
        except (RuntimeError, AttributeError):
            # 如果没有事件循环或无法获取，正常加载
            load_dotenv()
        
        # 处理agent_age的类型转换
        if isinstance(self.agent_age, str):
            # 尝试从中文数字转换为阿拉伯数字
            chinese_numbers = {
                '零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
                '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
                '十一': 11, '十二': 12, '十三': 13, '十四': 14, '十五': 15,
                '十六': 16, '十七': 17, '十八': 18, '十九': 19, '二十': 20,
                '二十一': 21, '二十二': 22, '二十三': 23, '二十四': 24, '二十五': 25,
                '二十六': 26, '二十七': 27, '二十八': 28, '二十九': 29, '三十': 30,
                '三十一': 31, '三十二': 32, '三十三': 33, '三十四': 34, '三十五': 35,
                '三十六': 36, '三十七': 37, '三十八': 38, '三十九': 39, '四十': 40,
                '四十一': 41, '四十二': 42, '四十三': 43, '四十四': 44, '四十五': 45,
                '四十六': 46, '四十七': 47, '四十八': 48, '四十九': 49, '五十': 50
            }
            
            if self.agent_age in chinese_numbers:
                self.agent_age = chinese_numbers[self.agent_age]
            else:
                # 尝试直接转换数字字符串
                try:
                    self.agent_age = int(self.agent_age)
                except ValueError:
                    # 如果无法转换，使用默认值
                    print(f"警告：无法转换agent_age '{self.agent_age}' 为整数，使用默认值30")
                    self.agent_age = 30
        
        # 确保agent_temperature在合理范围内
        if isinstance(self.agent_temperature, (int, float)):
            self.agent_temperature = max(0.0, min(1.0, float(self.agent_temperature)))
        else:
            self.agent_temperature = 0.6

        # 兼容字段回填：industry 与 industry_knowledge 互相同步
        try:
            if (not getattr(self, "industry", None)) and getattr(self, "industry_knowledge", None):
                self.industry = self.industry_knowledge
            if (not getattr(self, "industry_knowledge", None)) and getattr(self, "industry", None):
                self.industry_knowledge = self.industry
        except Exception:
            pass
        
        # 从环境变量加载API密钥（如果未设置）
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.dashscope_api_key is None:
            self.dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
        if self.cloud_kb_api_key is None:
            self.cloud_kb_api_key = os.getenv("CLOUD_KB_API_KEY")
        if self.tavily_api_key is None:
            self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        if self.cloud_kb_api_endpoint is None:
            self.cloud_kb_api_endpoint = os.getenv("CLOUD_KB_API_ENDPOINT")
        # StepFun API Key
        if os.getenv("STEPFUN_API_KEY") and not hasattr(self, "stepfun_api_key"):
            # 动态注入，避免破坏 schema；在调用处通过 os.getenv 读取
            pass
        
        # 从环境变量加载模型设置（如果未设置）
        if self.embedding_model == "text-embedding-v4":
            self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
        if self.embedding_dimension == 768:
            self.embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "768"))

    @classmethod
    def from_context(cls) -> 'Configuration':
        """Create a Configuration instance from a RunnableConfig object."""
        try:
            config = get_config()
        except RuntimeError:
            # 如果没有 LangGraph 运行时上下文，返回默认配置
            print("警告：没有 LangGraph 运行时上下文，使用默认配置")
            return cls()
        
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {field_name for field_name in cls.model_fields.keys()}
        
        # 如果 configurable 为空，返回默认配置
        if not configurable:
            print("警告：configurable 为空，使用默认配置")
            return cls()
            
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
