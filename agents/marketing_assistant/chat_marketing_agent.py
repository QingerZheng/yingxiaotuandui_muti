"""聊天营销智能体工作流
采用LangGraph的Plan-and-Execute模式实现智能营销助手：
1. 计划阶段 - 分析用户需求，制定多步骤执行计划
2. 执行阶段 - 按计划逐步执行，包括：
   - 聊天对话与需求理解
   - 时间信息获取
   - 联网搜索最新趋势
   - 营销文案生成
3. 重新规划 - 根据执行结果调整计划或完成任务

输入格式:
{
  "messages": [
    {
      "type": "human",
      "content": "用户输入内容"
    }
  ]
}
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Literal, Union
from typing_extensions import TypedDict, Annotated
import asyncio

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from llm import create_llm
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

# 导入营销工具
from agents.marketing_assistant.marketing_tool import get_current_time, web_search_tool, marketing_copy_generator
from agents.marketing_assistant.marketing_tool.time_tool import TimeInfo
# 导入人设提示词模板
from agents.marketing_assistant.persona_prompt_template import PersonaPromptTemplate

# ==================== 全局变量 ====================

# 存储计划阶段决定使用的工具名称列表
planned_tools = []

# 当前轮次数据字典 - 替代inter_node_memory状态传递机制
current_round_data = {
    "round_id": 0,
    "user_input": "",
    "plan_steps": [],
    "execution_results": [],
    "tool_outputs": {},
    "marketing_copies": None,
    "search_results": None,
    "time_info": None
}

# 独立的对话记忆全局字典 - 存储近两轮对话
recent_conversation_memory = {
    "messages": [],  # 存储近两轮对话消息
    "last_updated": None  # 最后更新时间
}

# ==================== 全局字典管理函数 ====================

def init_round_data(round_id: int, user_input: str) -> None:
    """初始化当前轮次数据"""
    global current_round_data
    current_round_data = {
        "round_id": round_id,
        "user_input": user_input,
        "plan_steps": [],
        "execution_results": [],
        "tool_outputs": {},
        "marketing_copies": None,
        "search_results": None,
        "time_info": None
    }
    # print(f"🔄 初始化轮次 {round_id} 数据: {user_input}")

def update_recent_conversation_memory(conversation_memory: list) -> None:
    """更新近两轮对话记忆到全局字典"""
    global recent_conversation_memory
    from datetime import datetime
    
    # 只保留最近两轮对话（最后4条消息：用户-AI-用户-AI）
    recent_messages = conversation_memory[-4:] if len(conversation_memory) > 4 else conversation_memory
    
    recent_conversation_memory = {
        "messages": recent_messages,
        "last_updated": datetime.now().isoformat()
    }
    # print(f"📝 更新近两轮对话记忆: {len(recent_messages)}条消息")

def get_recent_conversation_memory() -> list:
    """获取近两轮对话记忆"""
    global recent_conversation_memory
    return recent_conversation_memory.get("messages", [])

def get_round_data(key: str = None):
    """获取当前轮次数据"""
    global current_round_data
    if key:
        return current_round_data.get(key)
    return current_round_data.copy()

def set_round_data(key: str, value) -> None:
    """设置当前轮次数据"""
    global current_round_data
    current_round_data[key] = value
    # print(f"📝 设置轮次数据 {key}: {value}")

def clear_round_data() -> None:
    """清理当前轮次数据"""
    global current_round_data
    current_round_data = {
        "round_id": 0,
        "user_input": "",
        "plan_steps": [],
        "execution_results": [],
        "tool_outputs": {},
        "marketing_copies": None,
        "search_results": None,
        "time_info": None
    }
    # print("🧹 清理轮次数据完成")

# ==================== 配置和枚举 ====================

class ChatMarketingStep(Enum):
    """聊天营销工作流步骤枚举"""
    PLAN = "plan"
    EXECUTE = "execute"
    REPLAN = "replan"
    COMPLETE = "complete"
    CHAT = "chat"
# ==================== 数据模型 ====================

class Plan(BaseModel):
    """执行计划模型"""
    conversation_type: str = Field(
        description="对话类型：chat、discussion或generation"
    )
    execution_plan: str = Field(
        description="详细的执行步骤说明"
    )
    steps: List[str] = Field(
        description="按顺序执行的步骤列表，每个步骤应该是具体可执行的任务",
        default_factory=list
    )

class Response(BaseModel):
    """最终回复模型"""
    response: str = Field(description="给用户的最终回复")
    marketing_copies: Optional[str] = Field(description="营销文案内容，JSON格式", default=None)

class Act(BaseModel):
    """重新规划动作模型"""
    action: Plan | Response = Field(
        description="下一步动作：继续执行新计划或给出最终回复"
    )

class ChatMarketingState(TypedDict):
    """聊天营销工作流状态"""
    # 输入消息字段 - 与ChatMarketingInput保持一致
    messages: Optional[Sequence[BaseMessage]]  # 输入消息
    
    # 会话记忆字段
    conversation_memory: Optional[List[BaseMessage]]  # 会话记忆，存储用户与AI的聊天记录
    
    # 当前用户输入字段 - 专门存储当前轮次的用户输入
    current_user_input: Optional[str]  # 当前用户输入，避免从全局字典中解析
    
    # 轮次ID字段 - 用于区分不同对话轮次，避免跨轮污染
    round_id: Optional[int]  # 当前对话轮次ID，从1开始计数
    
    # Plan-and-Execute 状态字段
    plan: List[str]
    past_steps: List[tuple[str, str]]  # (步骤, 执行结果)
    response: Optional[str]
    
    # 其他状态字段
    current_step: Optional[ChatMarketingStep]
    time_info: Optional[TimeInfo]
    search_results: Optional[str]
    marketing_copies: Optional[str]  # 营销文案工具返回结果
    error_message: Optional[str]

class ChatMarketingInput(TypedDict):
    """聊天营销工作流输入"""
    messages: Sequence[BaseMessage]  # 标准LangGraph消息格式

class ChatMarketingOutput(TypedDict):
    """聊天营销工作流输出"""
    response: str
    marketing_copies: Optional[str]

# ==================== 工作流节点 ====================

class ChatMarketingNodes:
    """聊天营销工作流节点集合"""
    
    def __init__(self):
        """初始化聊天营销工作流节点"""
        # print(f"🚀 开始初始化聊天营销工作流节点")
        
# ==================== 提示词配置 ====================
# 这里需要配置营销助手的基础提示词和系统指令
# 包含:
# 1. 营销专家角色定位
# 2. 专业知识范围界定
# 3. 回复风格和语气设置
# 4. 工具使用规范
# 5. 安全限制和边界
  # ==================== 提示词配置 ====================     
        # 初始化人设提示词模板
        self.persona_template = PersonaPromptTemplate()
        # 按需组合基础提示词
        persona_parts = [
            self.persona_template.FOUNDATION_PROMPTS["role_identity"],
            self.persona_template.FOUNDATION_PROMPTS["core_principles"],
            self.persona_template.INTERACTION_PROMPTS["communication_style"],
            self.persona_template.INTERACTION_PROMPTS["output_format"]
        ]
        self.persona_prompt = "\n\n".join(persona_parts)
        
        # 创建工具和LLM
        self.tools = [get_current_time, web_search_tool, marketing_copy_generator]
        # print(f"📦 工具列表初始化完成，共{len(self.tools)}个工具")
        
        try:
            self.planner_llm = self._create_llm(temperature=0.1)  # 计划需要更精确
            self.executor_llm = self._create_llm(temperature=0.6)  # 执行可以更有创意
            
            
            self.planner = self._create_planner()
            self.replanner = self._create_replanner()
            
            # 注意：执行器Agent将在execute_step中动态创建，以便传入对话记忆
            
            # print(f"✅ 聊天营销工作流节点初始化成功")
        except Exception as e:
            # print(f"❌ 初始化失败: {e}")
            # 设置为None以便后续检查
            self.planner = None
            self.replanner = None
            raise Exception(f"聊天营销工作流节点初始化失败: {e}")
    
    def _create_planner(self):
        """创建计划器"""
        if not self.planner_llm:
            return None
        
        return self.planner_llm.with_structured_output(Plan)
    
    def _create_replanner(self):
        """创建重新规划器"""
        if not self.planner_llm:
            return None
            
        # 为replanner添加系统提示词
        replanner_system_prompt = self.persona_template.REPLANNER_PROMPTS["system_prompt"]
        
        # 创建带有系统提示词的LLM
        from langchain_core.prompts import ChatPromptTemplate
        
        # 创建提示词模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", replanner_system_prompt),
            ("human", "{input}")
        ])
        
        # 创建链式调用：prompt -> llm -> structured_output
        chain = prompt | self.planner_llm.with_structured_output(Act)
        return chain
    
    def _create_executor_agent(self, conversation_memory_text: str = ""):
        """创建执行器Agent
        
        Args:
            conversation_memory_text: 格式化的对话记忆文本，用于文案修改场景
        """
        try:
            if not self.executor_llm:
                # print(f"❌ executor_llm为None，无法创建执行器Agent")
                raise Exception("executor_llm未正确初始化")
                
                
            # print(f"📝 开始创建执行器Agent，工具数量: {len(self.tools)}")
            # print(f"📝 可用工具: {[tool.name for tool in self.tools]}")
          

             # 添加执行器特定的指导
            executor_instructions = f"""

执行指导：
你正在执行一个多步骤计划中的特定步骤。请专注于当前任务，使用合适的工具完成目标。

{self.persona_template.EXECUTOR_PROMPTS["classification_rules"]}

{self.persona_template.EXECUTOR_PROMPTS["tool_usage_requirements"]}

{self.persona_template.EXECUTOR_PROMPTS["important_reminders"]}

## 当前可用的对话记忆：
{conversation_memory_text}

请根据给定的步骤描述，选择合适的工具完成任务。仅在文案修改场景（use_previous_copy=True）时传递conversation_memory参数。"""
            
            system_prompt = self.persona_prompt + executor_instructions
            #==================== 提示词配置结束 ====================
            
            agent = create_react_agent(self.executor_llm, self.tools, prompt=system_prompt)
            # print(f"✅ 执行器Agent创建成功")
            return agent
        except Exception as e:
            # print(f"❌ 创建执行器Agent失败: {e}")
            raise Exception(f"无法创建执行器Agent: {e}")
    

    
    def _create_llm(self, temperature: float = 0.1):
        """创建LLM实例"""
        try:
            from agents.persona_config.config_manager import config_manager
            cfg = config_manager.get_config() or {}
            
            llm = create_llm(
                model_provider=cfg.get("model_provider", "openrouter"),
                model_name=cfg.get("model_name", "openai/gpt-4o"),
                temperature=temperature
            )
            # print(f"✅ LLM创建成功: {cfg.get('model_provider', 'openrouter')}/{cfg.get('model_name', 'openai/gpt-5-chat')}")
            return llm
        except Exception as e:
            # print(f"❌ 创建LLM失败: {e}")
            raise Exception(f"无法创建LLM: {e}")
    
    async def input_step(self, state: ChatMarketingState) -> Dict[str, Any]:
        """输入处理步骤 - 处理和转递输入消息，并初始化当前轮次数据"""
        try:
            # 清空上一轮的全局字典数据
            clear_round_data()
            # print("🧹 已清空上一轮的全局字典数据")
            
            # 生成新的轮次ID
            current_round_id = state.get("round_id", 0) + 1
            # print(f"🔄 开始新一轮对话，轮次ID: {current_round_id}")
            
            # 获取输入消息
            input_messages = state.get("messages", [])
            
            # print(f"[DEBUG] input_step - 原始输入消息: {input_messages}")
            # print(f"[DEBUG] input_step - 原始输入消息类型: {type(input_messages)}")
            
            # 提取用户输入内容
            user_input = ""
            for msg in input_messages:
                if isinstance(msg, dict):
                    user_input = msg.get("content", "")
                    break
                elif hasattr(msg, 'content'):
                    # 处理BaseMessage对象
                    user_input = msg.content
                    break
            
            # 初始化当前轮次的全局数据
            init_round_data(current_round_id, user_input)
            
            # 获取历史会话记忆
            conversation_memory = state.get("conversation_memory", [])
            # print(f"📚 获取到历史会话记忆: {len(conversation_memory)}条消息")
            # print(f"🔍 conversation_memory详细内容:")
            # for i, msg in enumerate(conversation_memory):
            #     print(f"  消息{i}: 类型={type(msg)}, 内容={getattr(msg, 'content', 'No content')[:50]}...")
            
            # print(f"🔍 调用update_recent_conversation_memory之前，recent_conversation_memory状态:")
            # print(f"  current recent_conversation_memory: {recent_conversation_memory}")
            
            # 更新近两轮对话记忆到独立的全局字典
            update_recent_conversation_memory(conversation_memory)
            
            # print(f"🔍 调用update_recent_conversation_memory之后，recent_conversation_memory状态:")
            # print(f"  updated recent_conversation_memory: {recent_conversation_memory}")
            
            # 验证get_recent_conversation_memory函数
            test_messages = get_recent_conversation_memory()
            print(f"通过get_recent_conversation_memory()获取的消息: {test_messages}")
            print(f"get_recent_conversation_memory()返回类型: {type(test_messages)}")
            print(f"get_recent_conversation_memory()返回长度: {len(test_messages) if test_messages else 0}")
            
            # print(f"用户输入: {user_input}")
            # print(f"历史记忆数量: {len(conversation_memory)}")
            # print(f"当前轮次ID: {current_round_id}")
            # print(f"[DEBUG] input_step - 即将返回的current_user_input: {user_input}")
            
            return {
                "current_user_input": user_input,  # 存储当前用户输入
                "round_id": current_round_id,  # 设置当前轮次ID
                "current_step": ChatMarketingStep.PLAN
            }
        
        except Exception as e:
            # print(f"输入处理步骤失败: {e}")
            raise e
    
    async def plan_step(self, state: ChatMarketingState) -> Dict[str, Any]:
        """计划步骤 - 分析用户需求并制定执行计划"""
        try:
            # 如果没有计划器就报错
            if not self.planner:
                raise Exception("计划器未正确初始化，无法生成执行计划")
            
            # 从全局字典获取当前轮次数据
            round_data = get_round_data()
            if not round_data:
                raise Exception("无法获取当前轮次数据")
            
            user_input = round_data.get('user_input', '')
            round_id = round_data.get('round_id', 0)
            
            # print(f"[DEBUG] 从全局字典获取用户输入: {user_input}")
            # print(f"[DEBUG] 当前轮次ID: {round_id}")
            
            if not user_input:
                plan_steps = ["用户没有输入哦，问问是怎么回事，需要什么需求"]
                set_round_data('plan', plan_steps)
                return {
                    "plan": plan_steps,
                    "current_step": ChatMarketingStep.PLAN
                }
            
            # 获取历史会话记忆用于构建对话上下文
            conversation_memory = list(state.get("conversation_memory", []))
            
            # 构建简化的对话上下文（包含历史记忆和当前输入）
            conversation_context = ""
            if conversation_memory and len(conversation_memory) > 0:
                # 只取最近4条历史消息，并智能处理消息长度
                recent_messages = conversation_memory[-4:]
                conversation_context = "\n\n## 简要对话历史\n"
                for msg in recent_messages:
                    if hasattr(msg, 'content'):
                        # 保留所有消息的完整内容，以便准确引用历史对话
                        if hasattr(msg, 'type') and msg.type == 'human':
                            conversation_context += f"用户: {msg.content}\n"
                        elif hasattr(msg, 'type') and msg.type == 'ai':
                            conversation_context += f"助手: {msg.content}\n"
            
            conversation_context += f"\n**当前用户输入：{user_input}**\n"
              

            
            # 构建计划器的系统提示词
            planner_system_prompt = f"""

你是一个专业的营销任务规划器。你的职责是分析用户需求，确定合适的对话类型并制定执行计划。

{conversation_context}

## 你的任务
分析用户的输入，将其分类为三种对话类型之一，然后提供具体的执行计划。

## 对话类型定义
### 类型1：chat（闲聊）
**使用场景：** 问候、闲聊、营销知识问答、一般性咨询、时间查询、信息搜索
**特征：**
- 用户进行日常对话
- 关于营销概念或一般建议的问题
- 问候和社交互动
- 询问当前时间、日期等时间信息
- 需要搜索最新信息或趋势的问题
**执行方式：** 根据用户需求灵活使用工具，可能包括get_current_time获取时间信息、web_search_tool搜索相关内容，或直接回复
**重要规则：** 如果需要使用web_search_tool进行网络搜索，必须先调用get_current_time获取当前时间，以确保搜索结果的时效性和准确性

### 类型2：discussion（讨论）
**使用场景：** 用户需要澄清或对现有文案提供反馈，或者文案需求信息不足
**特征：**
- 用户的文案需求不完整或模糊（如"帮我生成一个营销文案"、"写个文案"等没有具体产品或受众或场景信息的请求）
- 用户对之前生成的文案提供反馈（"第一个不错"、"这个文案很好"、"不错"、"很棒"）
- 需要收集更多关于目标受众、产品特点或营销目标的信息
- 用户讨论现有文案但不要求生成新文案
- 首次提及文案生成但缺乏具体信息的情况
**重要判断规则：** 如果对话历史中助手已经询问过文案需求，且用户在当前回复中提供了产品信息（即使简单如"护手霜"），或者用户表示"都行"、"随便"、"可以"等同意词汇，应该转为generation类型
**执行方式：** 通过对话澄清需求或确认反馈，不使用工具

### 类型3：generation（生成）
**使用场景：** 用户提供了足够信息的文案创作或修改请求
**关键词：** "写文案"、"生成文案"、"创作文案"、"给XX写文案"、"帮我写XX文案"、"帮我生成XX文案"
**特征：**
- 用户明确要求生成新文案且提供了具体的产品、受众或场景信息
- 包含具体产品名称的文案请求（如"帮我生成护手霜文案"、"写个朋友圈文案"）
- 包含发布渠道信息的请求（如"朋友圈文案"、"群发文案"）
- 包含热销、趋势等搜索需求的文案请求（如"今年什么热销，帮我生成朋友圈文案"）
- 用户想要修改、优化、重写或润色现有文案
- 要求缩短、加长或风格变化的请求
- 用户使用"第一种"、"第一条"、"第一个"等表述指代特定文案进行修改
- 在discussion环节后，用户提供了足够信息的文案生成请求
- **重要：** 当对话历史显示助手已询问文案需求，且用户提供了产品名称（如"护手霜"、"面膜"等）或表示"都行"、"随便"、"可以"等同意词汇时，应立即生成文案而非继续询问
**执行方式：** 使用适当的工具 - 可能包括 get_current_time、web_search_tool、marketing_copy_generator
**重要规则：** 如果需要使用web_search_tool进行网络搜索，必须先调用get_current_time获取当前时间，以确保搜索结果的时效性和准确性

## 输出要求
你必须输出结构化的JSON格式，包含以下字段：
- conversation_type: 对话类型（chat/discussion/generation）
- execution_plan: 详细的执行步骤说明
- steps: 具体执行步骤列表（可选，如果没有具体步骤可以为空）

## 分类要求
1. 始终分类为三种类型中的一种
2. **优先级原则：** 对于模糊的文案生成请求（如"帮我生成一个营销文案"、"写个文案"等缺乏具体信息的请求），优先选择discussion类型进行需求澄清
3. **关键判断：** 仔细分析对话历史，如果助手之前已经询问过文案需求信息，且用户在当前回复中：
   - 提供了具体产品名称（如"护手霜"、"面膜"、"口红"等）
   - 或表示同意/随意的词汇（如"都行"、"随便"、"可以"、"行"等）
   则应该选择generation类型，立即生成文案，而不是继续discussion
4. 对于generation类型，在execution_plan中指定要使用的工具：get_current_time、web_search_tool、marketing_copy_generator
5. 对于chat类型，根据用户需求在execution_plan中指定要使用的工具（如时间查询使用get_current_time，信息搜索使用web_search_tool）或说明"直接回复，不使用工具"
6. 对于discussion类型，主要通过对话澄清需求，必要时可在execution_plan中指定使用相关工具（如时间查询、信息搜索等）
7. execution_plan要具体明确

## 输出格式
输出结构化JSON，包含conversation_type、execution_plan和steps字段。对于generation类型，需在execution_plan中说明工具使用和字数设置策略。

      """
            
            # 调用计划器生成计划
            user_prompt = f"当前用户需求：{user_input}"
            plan_result = await self.planner.ainvoke([
                ("system", planner_system_prompt),
                ("user", user_prompt)
            ])
            
            # 从structured output中直接获取结果
            conversation_type = plan_result.conversation_type
            execution_plan = plan_result.execution_plan
            plan_steps = plan_result.steps if plan_result.steps else [execution_plan]
            
            # print(f"[DEBUG] LLM结构化输出 - 对话类型: {conversation_type}")
            # print(f"[DEBUG] LLM结构化输出 - 执行计划: {execution_plan}")
            # print(f"[DEBUG] LLM结构化输出 - 步骤列表: {plan_steps}")
            
            # 验证对话类型是否有效
            available_conversation_types = ["chat", "discussion", "generation"]
            if conversation_type not in available_conversation_types:
                raise ValueError(f"无效的对话类型: {conversation_type}，有效类型: {available_conversation_types}")
            
            # 提取工具名称
            available_tools = ["get_current_time", "web_search_tool", "marketing_copy_generator"]
            planned_tools = []
            execution_plan_lower = execution_plan.lower()
            for tool in available_tools:
                if tool in execution_plan_lower:
                    planned_tools.append(tool)
            
            # print(f"[DEBUG] 大模型计划使用的工具: {planned_tools}")
          
            # 存储到全局字典
            set_round_data('plan', plan_steps)
            set_round_data('planned_tools', planned_tools)
            set_round_data('conversation_type', conversation_type)
            
            # print(f"📋 本次对话类型: {conversation_type}")
            # print(f"🔧 计划使用的工具: {planned_tools}")
            # print(f"📝 执行计划: {plan_steps}")
            
            # 如果没有工具需要使用，直接回复
            if not planned_tools:
                # print("🗣️ 规划器决定直接回复")
                # 即使直接回复，也要在plan中体现出执行计划
                direct_reply_plan = [f"根据{conversation_type}类型对话直接回复用户: {execution_plan}"]
                return {
                    "current_step": ChatMarketingStep.COMPLETE,
                    "plan": direct_reply_plan,
                    "past_steps": []
                }
            
            return {
                "plan": plan_steps,
                "current_step": ChatMarketingStep.PLAN,
                "past_steps": []
            }
            
        except Exception as e:
            # print(f"❌ 计划生成失败: {str(e)}")
            raise e
    
    async def execute_step(self, state: ChatMarketingState) -> Dict[str, Any]:
        """执行步骤 - 执行计划中的当前步骤"""
        try:
            # 从全局字典和状态获取数据
            round_data = get_round_data()
            if not round_data:
                raise Exception("无法获取当前轮次数据")
            
            plan = state.get("plan", [])
            past_steps = state.get("past_steps", [])
            
            if not plan:
                raise ValueError("没有可执行的计划，程序终止执行")
            
            # 获取当前要执行的步骤
            current_task = plan[0]
            round_id = round_data.get('round_id', 0)
            
            # print(f"[DEBUG] 执行步骤: {current_task}")
            # print(f"[DEBUG] 当前轮次ID: {round_id}")
            
            # 构建执行上下文
            plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
            past_steps_str = "\n".join(f"- {step}: {result}" for step, result in past_steps)
            
            # 获取对话历史上下文
            conversation_memory = state.get("conversation_memory", [])
            conversation_context = ""
            if conversation_memory:
                conversation_context = "\n\n## 对话历史\n"
                for msg in conversation_memory:
                    if hasattr(msg, 'content'):
                        if hasattr(msg, 'type') and msg.type == 'human':
                            conversation_context += f"用户: {msg.content}\n"
                        elif hasattr(msg, 'type') and msg.type == 'ai':
                            conversation_context += f"助手: {msg.content}\n"
            
            # 获取当前用户输入
            current_user_input = state.get("current_user_input", "")
            # print(f"[DEBUG] execute_step - 从state获取的current_user_input: {current_user_input}")
            if current_user_input:
                conversation_context += f"\n**当前用户输入：{current_user_input}**\n"
            
            # 构建对话记忆文本供工具使用
            conversation_memory_text = ""
            if conversation_memory:
                for msg in conversation_memory[-4:]:  # 只取最近4条消息
                    if hasattr(msg, 'content') and hasattr(msg, 'type'):
                        if msg.type == 'human':
                            conversation_memory_text += f"用户: {msg.content}\n"
                        elif msg.type == 'ai':
                            conversation_memory_text += f"助手: {msg.content}\n"
            
            task_prompt = f"""执行计划：
{plan_str}

已完成步骤：
{past_steps_str}

当前任务：执行步骤{len(past_steps) + 1} - {current_task}
{conversation_context}

请专注完成这个具体任务，根据对话历史和当前用户输入来执行。

**字数要求设置**：请根据用户需求智能设置min_word_count参数：
- 标准文案需求：设置为50-80字
- 特定场景：朋友圈/标题(50-80字)、产品介绍(100-150字)、品牌故事(200-300字)"""
            
            # 动态创建执行器Agent，传入对话记忆
            executor_agent = self._create_executor_agent(conversation_memory_text)
            if not executor_agent:
                raise RuntimeError("执行器Agent创建失败，无法执行工具调用任务")
            
            # 调用执行器Agent
            response = await executor_agent.ainvoke({
                "messages": [("ai", task_prompt)]
            })
            
            # 提取执行结果 - 只存储简单的状态信息，避免冗余的文案内容
            ai_messages = [msg for msg in response["messages"] if isinstance(msg, AIMessage)]
            execution_result = ai_messages[-1].content if ai_messages else "执行完成"
            
            # 将执行结果存储到全局字典 - 不存储冗余的result字段
            current_execution_results = round_data.get('execution_results', [])
            current_execution_results.append({
                'task': current_task,
                'response_messages': response["messages"]
            })
            set_round_data('execution_results', current_execution_results)
            
            # print(f"📋 执行结果已存储到全局字典: {current_task}")
            
            result = {
                "past_steps": past_steps + [(current_task, execution_result)],
                "plan": plan[1:],  # 移除已执行的步骤
                "current_step": ChatMarketingStep.EXECUTE
            }

            return result
            
        except Exception as e:
            # print(f"❌ 执行步骤失败: {str(e)}")
            raise e
    
    def _format_past_steps(self, past_steps: List[tuple]) -> str:
        """格式化已完成步骤，保留ToolMessage的原始结构"""
        formatted_steps = []
        for step, result in past_steps:
            # 如果结果包含ToolMessage，保留其原始JSON结构
            if isinstance(result, str) and 'ToolMessage' in result and 'marketing_copies' in result:
                formatted_steps.append(f"- {step}: {result}")
            else:
                formatted_steps.append(f"- {step}: {result}")
        return "\n".join(formatted_steps)
    
    async def replan_step(self, state: ChatMarketingState) -> Dict[str, Any]:
        """重新规划步骤 - 决定继续执行还是完成任务"""
        import json
        try:
            plan = state.get("plan", [])
            past_steps = state.get("past_steps", [])
            
            # 如果没有剩余计划，转到final节点处理回复
            if not plan:
                raise ValueError("没有可执行的计划，程序终止执行")
            
            # 获取重试次数，防止无限循环
            retry_count = state.get("retry_count", 0)
            max_retries = 3
            
            if retry_count >= max_retries:
                # print(f"⚠️ 达到最大重试次数({max_retries})，强制完成任务")
                return {
                    "current_step": ChatMarketingStep.COMPLETE
                }
            
            # 分析执行结果，检测失败的步骤
            round_data = get_round_data()
            execution_results = round_data.get('execution_results', []) if round_data else []
            planned_tools = round_data.get('planned_tools', []) if round_data else []
            conversation_type = round_data.get('conversation_type', 'chat') if round_data else 'chat'
            
            # 检查营销文案生成状态
            marketing_copy_success = False
            if 'marketing_copy_generator' in planned_tools:
                for execution_result in execution_results:
                    response_messages = execution_result.get('response_messages', [])
                    for msg in response_messages:
                        if hasattr(msg, 'name') and msg.name == 'marketing_copy_generator':
                            try:
                                content_data = json.loads(msg.content)
                                if content_data.get('marketing_copies'):
                                    marketing_copy_success = True
                                    break
                            except:
                                pass
                    if marketing_copy_success:
                        break
            
            # 构建详细的重新规划上下文
            remaining_plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
            
            # 分析执行状态
            execution_status = "执行状态分析：\n"
            if conversation_type == 'generation':
                if marketing_copy_success:
                    execution_status += "营销文案生成成功\n"
                else:
                    execution_status += "营销文案生成失败，需要重新尝试\n"
            elif conversation_type in ['chat', 'discussion']:
                if past_steps:
                    execution_status += "对话/讨论步骤已执行\n"
                else:
                    execution_status += "对话/讨论步骤执行异常\n"
            
            replan_context = f"""## 当前状态分析
对话类型: {conversation_type}
重试次数: {retry_count + 1}/{max_retries}

{execution_status}
已完成的步骤：
{self._format_past_steps(past_steps)}

剩余计划：
{remaining_plan_str}

## 决策要求
根据上述分析，请决定：
1. 如果任务已成功完成，返回Response
2. 如果需要重新执行失败的步骤，返回包含具体步骤的Plan
3. 重点关注营销文案生成是否成功（如果是generation类型）

请基于执行状态做出明智决策。"""
            
            replan_result = await self.replanner.ainvoke({"input": replan_context})
            
            if isinstance(replan_result.action, Response):
                return {
                    "current_step": ChatMarketingStep.COMPLETE
                }
            else:
                # 增加重试次数
                return {
                    "plan": replan_result.action.steps,
                    "current_step": ChatMarketingStep.EXECUTE,
                    "retry_count": retry_count + 1
                }
                
        except Exception as e:
            # print(f"❌ 重新规划失败: {str(e)}")
            raise e
    

    
    async def chat_node(self, state: ChatMarketingState) -> Dict[str, Any]:
        """聊天节点 - Plan-and-Execute模式的入口"""
        try:
            current_step = state.get("current_step", ChatMarketingStep.PLAN)
            
            # 根据当前步骤调用相应的处理方法
            if current_step == ChatMarketingStep.PLAN:
                return await self.plan_step(state)
            elif current_step == ChatMarketingStep.EXECUTE:
                return await self.execute_step(state)
            elif current_step == ChatMarketingStep.REPLAN:
                return await self.replan_step(state)
            else:
                # 默认情况或完成状态
                response = state.get("response")
                
                return {
                    "current_step": ChatMarketingStep.COMPLETE
                }
            
        except Exception as e:
            # print(f"❌ 聊天节点处理失败: {str(e)}")
            raise e
    



# ==================== 工作流控制 ====================

async def create_final_output(state: ChatMarketingState) -> Dict[str, Any]:
    """创建最终输出 - 简化版本，直接从工具结果获取文案并让LLM生成自然回复"""
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    import json
    
    try:
        # 从全局字典获取当前轮次数据
        round_data = get_round_data()
        if not round_data:
            raise Exception("无法获取当前轮次数据")
        
        user_input = round_data.get('user_input', '')
        execution_results = round_data.get('execution_results', [])
        if not user_input:
            raise ValueError("用户输入为空")
        
        # print(f"[DEBUG] Final节点 - 用户输入: {user_input}")
        
        # 先检查planned_tools是否包含文案生成工具
        planned_tools = round_data.get('planned_tools', [])
        # print(f"[DEBUG] Final节点 - planned_tools: {planned_tools}")
        
        # 获取 marketing_copies - 使用与print_memory节点相同的提取逻辑
        marketing_copies = None
        try:
            # 遍历查找marketing_copy_generator工具结果
            for execution_result in execution_results:
                response_messages = execution_result.get('response_messages', [])
                for msg in response_messages:
                    if hasattr(msg, 'name') and msg.name == 'marketing_copy_generator':
                        try:
                            content_data = json.loads(msg.content)
                            found_copies = content_data.get('marketing_copies', [])
                            # 只有当找到非空文案时才设置marketing_copies
                            if found_copies:
                                marketing_copies = found_copies
                                # print(f"[DEBUG] Final节点 - 获取到 {len(marketing_copies)} 个文案")
                            break
                        except json.JSONDecodeError as e:
                            # print(f"[DEBUG] Final节点 - JSON解析失败: {e}")
                            continue
                if marketing_copies:  # 找到后跳出外层循环
                    break
            if not marketing_copies:
                marketing_copies = None
        except Exception as e:
            # print(f"[DEBUG] Final节点 - 获取marketing_copies失败: {e}")
            marketing_copies = None
        # print(f"[DEBUG] Final节点 - marketing_copies最终值: {marketing_copies}")
        
        # 创建LLM实例
        from agents.persona_config.config_manager import config_manager
        cfg = config_manager.get_config() or {}
        
        llm = create_llm(
            model_provider=cfg.get("model_provider", "openrouter"),
            model_name=cfg.get("model_name", "openai/gpt-5-chat"),
            temperature=0.7
        )
        
        # 获取人设提示词
        from agents.marketing_assistant.persona_prompt_template import PersonaPromptTemplate
        persona_template = PersonaPromptTemplate()
        
        # 按需组合基础人设提示词
        persona_parts = [
            persona_template.FOUNDATION_PROMPTS["role_identity"],
            persona_template.FOUNDATION_PROMPTS["core_principles"],
            persona_template.COGNITIVE_PROMPTS["analysis"],
            persona_template.TASK_PROMPTS["copywriting"],
            persona_template.INTERACTION_PROMPTS["communication_style"],
            persona_template.INTERACTION_PROMPTS["output_format"]
        ]
        persona_prompt = "\n\n".join(persona_parts)
        
        # 构建完整的系统提示词，包含自然对话和业务引导规则
        system_parts = [
            persona_prompt,
            persona_template.TASK_PROMPTS["consultation"],
            persona_template.TASK_PROMPTS["copywriting"],
            persona_template.INTERACTION_PROMPTS["interaction_rules"],
            persona_template.INTERACTION_PROMPTS["natural_conversation"],
            persona_template.INTERACTION_PROMPTS["business_guidance"]
        ]
        
        # 根据对话类型添加不同的提示词
        conversation_type = round_data.get('conversation_type')
        if conversation_type == "discussion":
            system_parts.append(persona_template.TASK_PROMPTS["discussion_mode"])
        else:
            # 添加针对文案展示的特定要求
            system_parts.append(persona_template.TASK_PROMPTS["copywriting_display"])
        system_prompt = "\n\n".join(system_parts)
        
        # 获取对话历史
        conversation_memory = state.get("conversation_memory", [])
        conversation_text = ""
        is_first_chat = not conversation_memory or len(conversation_memory) == 0
        
        if conversation_memory:
            for msg in conversation_memory[-4:]:  # 只取最近4条消息
                if hasattr(msg, 'content') and hasattr(msg, 'type'):
                    if msg.type == 'human':
                        conversation_text += f"用户: {msg.content}\n"
                    elif msg.type == 'ai':
                        conversation_text += f"助手: {msg.content}\n"
        
        # 获取对话类型
        conversation_type = round_data.get('conversation_type')
        
        # 构建用户消息
        if marketing_copies:
            # 有文案时展示文案
            user_message = f"""用户最新消息: {user_input}

对话历史:
{conversation_text}

对话类型: {conversation_type}

我刚刚根据用户需求生成了以下营销文案:
{json.dumps(marketing_copies, ensure_ascii=False, indent=2)}

现在请作为{{XX}}（营销专家）自然地回复用户：
- 用口语化的方式介绍你刚创作的文案，避免书面化表达
- 不要说"集中在"、"以及"、"同时"等词汇，用"还有"、"另外就是"等自然表达
- 像朋友聊天一样轻松介绍文案特点
- 回复后要判断是否与营销业务相关，如果相关就自然引导到你的专业服务
- 用"对了"、"说到这个"等自然过渡，不要生硬推销
- 让客户感觉你是在分享经验，而不是在推销"""
        
        # 没有文案时根据对话类型回复
        if not marketing_copies:
            # 获取past_steps信息
            past_steps = state.get("past_steps", [])
            past_steps_text = ""
            if past_steps:
                # 格式化past_steps信息
                for i, (step_name, step_result) in enumerate(past_steps):
                    past_steps_text += f"步骤{i+1}: {step_name}\n结果: {step_result}\n\n"
            
            # 根据对话类型构建不同的用户消息
            if conversation_type == "discussion":
                user_message = f"""用户最新消息: {user_input}

对话历史:
{conversation_text}

对话类型: {conversation_type}

现在请作为{{XX}}（营销专家）自然地回复用户：
- 用户的文案需求还不够具体，需要了解更多信息
- 用口语化的方式询问产品信息、目标受众等
- 不要列举式提问，要像朋友聊天一样自然询问
- 避免"第一、第二、第三"的表达方式
- 用"我想了解一下"、"你能跟我说说"等自然表达
- 让用户感觉你是真心想帮助他们，而不是在走流程"""
            else:
                user_message = f"""用户最新消息: {user_input}

对话历史:
{conversation_text}

对话类型: {conversation_type}

执行步骤和结果:
{past_steps_text}

现在请作为{{XX}}（营销专家）自然地回复用户：
- 根据用户需求和执行结果给出口语化的回复
- 避免书面化表达，用"我觉得"、"我发现"等口语化表达
- 回答完问题后要判断是否与营销相关，如果相关就自然引导
- 用"对了"、"说到这个"等过渡词自然引导到业务
- 让客户感觉你是在分享专业见解，而不是推销"""
        
        # 调用LLM生成回复
        # print(f"[DEBUG] Final节点 - System Prompt: {system_prompt[:200]}...")
        # print(f"[DEBUG] Final节点 - User Message: {user_message[:300]}...")
        
        response = await llm.ainvoke([
            ("system", system_prompt),
            ("user", user_message)
        ])
        
        # 处理回复格式，移除多余空行
        final_response = response.content.strip()
        # 将多个连续换行替换为单个换行
        import re
        final_response = re.sub(r'\n\s*\n', '\n', final_response)
        # print(f"[DEBUG] LLM生成回复: {final_response[:100]}...")
        
        # 更新会话记忆
        conversation_memory = state.get("conversation_memory", [])
        updated_memory = list(conversation_memory)
        
        # 添加用户消息
        updated_memory.append(HumanMessage(content=user_input))
        
        # 添加助手回复（如果有文案，包含在记忆中）
        ai_message_content = final_response
        if marketing_copies:
            ai_message_content += f"\n\n[生成的营销文案]\n{json.dumps(marketing_copies, ensure_ascii=False, indent=2)}"
        updated_memory.append(AIMessage(content=ai_message_content))
        
        # 保持最近的对话（最多6轮，即12条消息）
        if len(updated_memory) > 12:
            updated_memory = updated_memory[-12:]
        
        # print(f"✅ Final节点处理完成，文案数量: {len(marketing_copies) if marketing_copies else 0}")
        
        return {
            "response": final_response,
            "marketing_copies": marketing_copies,  # 直接从工具结果赋值
            "conversation_memory": updated_memory,
            "current_step": ChatMarketingStep.COMPLETE
        }
        
    except Exception as e:
        import traceback
        # print(f"❌ Final节点处理失败: {str(e)}")
        # print(f"错误堆栈: {traceback.format_exc()}")
        raise

def print_conversation_memory(state: ChatMarketingState) -> Dict[str, Any]:
    """
    打印会话记忆内容和当前轮次的全局字典信息
    """
    # print("\n🔍🔍🔍 PRINT_MEMORY节点被调用了！🔍🔍🔍")
    conversation_memory = state.get("conversation_memory", [])
    current_round_id = state.get("round_id", 1)
    
    # print("\n=== conversation_memory ===")
    # print(conversation_memory)
    # print("=== conversation_memory 结束 ===\n")
    
    # 打印当前轮次的全局字典信息
    round_data = get_round_data()
    
    print(f"\n=== 当前轮次({current_round_id})的全局字典数据 ===")
    print(round_data)
    print("=== 全局字典数据 结束 ===\n")
    
    # 打印本次对话的类型分类
    conversation_type = round_data.get('conversation_type', '未分类') if round_data else '未分类'
    # print(f"\n📋 本次对话类型: {conversation_type}\n")
    
    # 直接获取 marketing_copies 数据
    print("\n🧪🧪🧪 测试打印 marketing_copies 数据 🧪🧪🧪")
    try:
        if round_data and round_data.get('execution_results'):
            import json
            marketing_copies = []
            # 遍历查找marketing_copy_generator工具结果
            for execution_result in round_data['execution_results']:
                response_messages = execution_result.get('response_messages', [])
                for msg in response_messages:
                    if hasattr(msg, 'name') and msg.name == 'marketing_copy_generator':
                        try:
                            content_data = json.loads(msg.content)
                            marketing_copies = content_data.get('marketing_copies', [])
                            print(f"\n=== Marketing Copies 数据 ===")
                            print(marketing_copies)
                            break
                        except json.JSONDecodeError as e:
                            print(f"JSON解析失败: {e}")
                if marketing_copies:  # 找到后跳出外层循环
                    break
            if not marketing_copies:
                print("未找到marketing_copies数据")
    except Exception as e:
        print(f"获取 marketing_copies 失败: {e}")
    print("🧪🧪🧪 测试结束 🧪🧪🧪\n")
    
    return {}

class ChatMarketingWorkflow:
    """聊天营销工作流 - Plan-and-Execute模式"""
    
    def __init__(self):
        """初始化聊天营销工作流"""
        self.nodes = ChatMarketingNodes()
        self.workflow = self._build_workflow()
    
    def _build_workflow(self):
        """构建Plan-and-Execute工作流图"""
        try:
            from langgraph.graph import StateGraph, START, END
            from langgraph.checkpoint.memory import MemorySaver
            
            # 创建状态图
            builder = StateGraph(ChatMarketingState, input=ChatMarketingInput, output=ChatMarketingOutput)
            
            # 添加节点
            builder.add_node("input", self.nodes.input_step)
            builder.add_node("plan", self.nodes.plan_step)
            builder.add_node("execute", self.nodes.execute_step)
            builder.add_node("replan", self.nodes.replan_step)
            builder.add_node("final_output", create_final_output)
            builder.add_node("print_memory", print_conversation_memory)
            
            # 添加边和条件判断
            # 工作流顺序：START -> input -> plan -> execute -> replan -> final_output -> END
            builder.add_edge(START, "input")
            builder.add_edge("input", "plan")
            
            # 从计划节点的条件判断
            builder.add_conditional_edges(
                "plan",
                lambda state: "final_output" if state.get("current_step") == ChatMarketingStep.COMPLETE else "execute",
                {
                    "execute": "execute",  # 有工具需要执行时进入执行节点
                    "final_output": "final_output"  # 直接回复时跳过执行直接到最终输出
                }
            )
            
            # 从执行节点的条件判断
            builder.add_conditional_edges(
                "execute",
                self._should_continue,
                {
                    "continue": "replan",  # 继续执行需要重新规划
                    "end": "final_output"  # 结束时处理最终输出
                }
            )
            
            # 从重新规划节点回到执行节点
            builder.add_edge("replan", "execute")
            
            # 从最终输出节点到打印记忆节点
            builder.add_edge("final_output", "print_memory")
            
            # 从打印记忆节点到结束
            builder.add_edge("print_memory", END)
            
            # 编译工作流（LangGraph API会自动处理持久化）
            return builder.compile()
            
        except Exception as e:
            print(f"构建工作流失败: {e}")
            # 不返回None，而是抛出异常让上层处理
            raise e
    
    def _should_continue(self, state: ChatMarketingState) -> str:
        """判断是否继续执行"""
        import json
        
        # 如果有response字段，说明任务完成
        if "response" in state and state["response"]:
            return "end"
        
        # 检查重试次数，防止无限循环
        retry_count = state.get("retry_count", 0)
        max_retries = 3
        if retry_count >= max_retries:
            return "end"
        
        # 获取执行结果和计划信息
        plan = state.get("plan", [])
        round_data = get_round_data()
        
        if not round_data:
            return "end"
            
        planned_tools = round_data.get('planned_tools', [])
        conversation_type = round_data.get('conversation_type', 'chat')
        execution_results = round_data.get('execution_results', [])
        
        # 对于生成类型，检查营销文案是否成功生成
        if conversation_type == 'generation' and 'marketing_copy_generator' in planned_tools:
            marketing_copy_success = False
            for execution_result in execution_results:
                response_messages = execution_result.get('response_messages', [])
                for msg in response_messages:
                    if hasattr(msg, 'name') and msg.name == 'marketing_copy_generator':
                        try:
                            content_data = json.loads(msg.content)
                            if content_data.get('marketing_copies'):
                                marketing_copy_success = True
                                break
                        except:
                            pass
                if marketing_copy_success:
                    break
            
            # 如果文案生成成功，可以结束；否则需要重试
            if marketing_copy_success:
                return "end"
            elif plan and len(plan) > 0:
                return "continue"  # 还有计划且文案未成功，继续重试
            else:
                return "end"  # 没有计划了，强制结束
        
        # 对于聊天和讨论类型，检查是否还有计划要执行
        elif conversation_type in ['chat', 'discussion']:
            if plan and len(plan) > 0:
                return "continue"
            else:
                return "end"
        
        # 默认情况：检查是否还有计划要执行
        if plan and len(plan) > 0:
            return "continue"
        
        return "end"

# ==================== 工作流构建 ====================

def create_chat_marketing_workflow():
    """创建聊天营销工作流 - Plan-and-Execute模式"""
    try:
        # 创建工作流实例
        workflow_instance = ChatMarketingWorkflow()
        if workflow_instance.workflow is None:
            raise ValueError("Failed to build workflow graph")
        return workflow_instance.workflow
        
    except Exception as e:
        print(f"创建聊天营销工作流失败: {e}")
        # 创建一个最小的fallback图以避免None
        from langgraph.graph import StateGraph, START, END
        fallback_builder = StateGraph(ChatMarketingState, input=ChatMarketingInput, output=ChatMarketingOutput)
        
        # 添加一个简单的fallback节点
        def fallback_node(state: ChatMarketingState) -> Dict[str, Any]:
            return {
                "response": "系统暂时不可用，请稍后重试。",
                "messages": [AIMessage(content="系统暂时不可用，请稍后重试。")]
            }
        
        fallback_builder.add_node("fallback", fallback_node)
        fallback_builder.add_edge(START, "fallback")
        fallback_builder.add_edge("fallback", END)
        
        return fallback_builder.compile()



# ==================== 导出 ====================

# 编译工作流
chat_marketing_graph = create_chat_marketing_workflow()

# 导出主要组件
__all__ = [
    "ChatMarketingState",
    "ChatMarketingInput",
    "ChatMarketingOutput",
    "ChatMarketingStep",
    "IntentType",
    "TimeInfo",
    "ChatMarketingNodes",
    "chat_marketing_graph"
]