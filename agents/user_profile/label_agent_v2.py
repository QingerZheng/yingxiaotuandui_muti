"""用户画像生成模块 - 分步式优化版本

采用分步式Prompt设计：
1. 第一步：分析对话，提取基础用户信息
2. 第二步：基于基础信息，生成标准化标签
3. 自动验证和修正不合规标签
"""

import asyncio
import json
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from llm import create_llm # 导入新的 LLM 工厂函数
# 默认模型使用 OpenRouter 可用的快速模型
from agents.shared.profile_variables import profile_variables
from dataclasses import dataclass, field
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import trim_messages, RemoveMessage

logger = logging.getLogger(__name__)


class UserProfile(BaseModel):
    """用户画像结构 - 按业务逻辑分组排序"""
    
    # === 社会画像 ===
    occupation: Optional[str] = Field(default=None, description="职业")
    age: Optional[str] = Field(default=None, description="年龄段") 
    region: Optional[str] = Field(default=None, description="地区")
    lifestyle: Optional[str] = Field(default=None, description="生活方式")
    family_status: Optional[str] = Field(default=None, description="家庭状况")
    emotion: Optional[str] = Field(default=None, description="情绪状态")
    
    # === 性格特征 ===
    character: Optional[str] = Field(default=None, description="性格类型")
    values: Optional[str] = Field(default=None, description="价值观")
    aesthetic_style: Optional[str] = Field(default=None, description="审美风格")
    
    # === 消费画像 ===
    ability: Optional[str] = Field(default=None, description="消费能力")
    willingness: Optional[str] = Field(default=None, description="消费意愿")
    preferences: Optional[str] = Field(default=None, description="品牌偏好")
    
    # === 产品意图 ===
    current_use: Optional[str] = Field(default=None, description="当前使用产品")
    potential_needs: Optional[str] = Field(default=None, description="潜在需求")
    decision_factors: Optional[str] = Field(default=None, description="决策因素")
    purchase_intent_score: Optional[str] = Field(default=None, description="购买意向评分")
    
    # === 客户生命周期 ===
    stage: Optional[str] = Field(default=None, description="客户阶段")
    value: Optional[str] = Field(default=None, description="客户价值")
    retention_strategy: Optional[str] = Field(default=None, description="留存策略")

    def get_filled_count(self) -> int:
        """获取已填充字段数量"""
        return sum(1 for v in self.model_dump().values() if v is not None)
    
    def get_total_count(self) -> int:
        """获取总字段数量"""
        return len(self.model_fields)
    
    def get_grouped_data(self) -> Dict[str, Dict[str, Optional[str]]]:
        """获取按分组组织的数据"""
        data = self.model_dump()
        return {
            "社会画像": {
                "occupation": data["occupation"],
                "age": data["age"], 
                "region": data["region"],
                "lifestyle": data["lifestyle"],
                "family_status": data["family_status"],
                "emotion": data["emotion"]
            },
            "性格特征": {
                "character": data["character"],
                "values": data["values"],
                "aesthetic_style": data["aesthetic_style"]
            },
            "消费画像": {
                "ability": data["ability"],
                "willingness": data["willingness"],
                "preferences": data["preferences"]
            },
            "产品意图": {
                "current_use": data["current_use"],
                "potential_needs": data["potential_needs"],
                "decision_factors": data["decision_factors"],
                "purchase_intent_score": data["purchase_intent_score"]
            },
            "客户生命周期": {
                "stage": data["stage"],
                "value": data["value"],
                "retention_strategy": data["retention_strategy"]
            }
        }

# 定义分析结果数据模型 - 必须在ProfileGenerator类之前定义
class AnalysisResult(BaseModel):
    """第一步分析结果结构"""
    basic_info: str = Field(description="用户基本信息摘要")
    personality: str = Field(description="性格特征摘要")
    consumption: str = Field(description="消费行为摘要")
    beauty_needs: str = Field(description="美容需求摘要")
    customer_status: str = Field(description="客户状态摘要")
    dialogue_quality: str = Field(description="对话信息充足度(1-10分)")
    summarize: str = Field(description="综合所有分析内容的一句话总结")


class ProfileGenerator:
    """分步式用户画像生成器"""
    
    def __init__(self, model_provider: str, model_name: str, temperature: float):
        # 创建支持JSON格式输出的模型
        # 统一通过工厂创建，底层已将 openai 路由到 openrouter
        self.model = create_llm(
            model_provider=model_provider,
            model_name=model_name,
            temperature=temperature,
        )
    
    def _build_analysis_prompt(self) -> str:
        """第一步：构建对话分析提示"""
        return """你是专业的用户画像分析专家。请基于上述聊天记录进行用户画像分析。

**重要：你必须严格按照JSON格式返回分析结果，不要包含任何其他文本或格式标记。**

**分析指导：通过用户的表达方式、语言习惯、关注点等信息，进行专业的用户画像分析。**

分析要点：
- 基本信息：从对话内容分析用户的基本特征
- 性格特征：分析用户的表达方式和沟通风格
- 消费行为：了解用户的消费偏好和能力
- 美容需求：分析用户对美容服务的需求
- 客户状态：评估用户的服务满意度和购买意向
- 对话质量：评估本次对话信息的丰富程度

请严格按照以下JSON格式返回分析结果：
{
    "basic_info": "用户基本信息摘要（从对话时间、表达方式、关注点等推断年龄段、职业类型、地区等）",
    "personality": "性格特征摘要（从对话方式、情绪表达、语言风格等分析性格类型、价值观等）", 
    "consumption": "消费行为摘要（从工作状态、对话时间、表达方式等推断消费能力、消费偏好等）",
    "beauty_needs": "美容需求摘要（从对话中的美容话题、关注点、询问方式等分析需求）",
    "customer_status": "客户状态摘要（从对话态度、服务满意度、互动方式等判断客户阶段、购买意向、流失风险等）",
    "dialogue_quality": "对话信息充足度(字符串格式，如\"5\")，1=信息很少，10=信息丰富",
    "summarize": "综合以上分析，形成用户的整体画像描述"
}

输出要求：
1. 必须返回有效的JSON格式
2. 基于对话内容进行专业分析
2. 严格按照上述JSON格式输出
3. 只输出JSON内容，不要添加其他文字"""

    def _normalize_messages(self, messages: List[Any]) -> List[BaseMessage]:
        """将输入的消息列表统一转换为 LangChain 的 BaseMessage 列表，避免 MESSAGE_COERCION_FAILURE。

        支持以下格式：
        - 字典：{"role": "user|human|assistant|ai|system", "content": ...}
        - BaseMessage 实例
        - content 为多模态 list，提取其中的 text 字段
        - 大小写不规范的角色名（如 "Human"）
        未识别角色降级为 human。
        """
        normalized: List[BaseMessage] = []
        if not messages:
            return normalized

        def _extract_text(content: Any) -> str:
            if isinstance(content, list):
                parts: List[str] = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(str(part.get("text", "")))
                return "".join(parts)
            return str(content) if content is not None else ""

        for msg in messages:
            if isinstance(msg, BaseMessage):
                normalized.append(msg)
                continue
            if isinstance(msg, dict):
                role = str(msg.get("role", "")).strip().lower()
                content = _extract_text(msg.get("content", ""))
                if role in ("user", "human"):
                    normalized.append(HumanMessage(content=content))
                elif role in ("assistant", "ai"):
                    normalized.append(AIMessage(content=content))
                elif role == "system":
                    normalized.append(SystemMessage(content=content))
                else:
                    normalized.append(HumanMessage(content=content))
                continue
            normalized.append(HumanMessage(content=str(msg)))
        return normalized

    def _build_labeling_prompt(self, analysis: Dict[str, str]) -> str:
        """第二步：构建标签生成提示"""
        options_str = self._format_options()
        
        dialogue_quality = analysis.get("dialogue_quality", "5")
        
        return f"""基于分析结果，生成标准化用户画像标签。根据对话质量({dialogue_quality}分)调整推理积极度。

**重要：你必须严格按照JSON格式返回标签结果，不要包含任何其他文本或格式标记。**

分析结果：
{json.dumps(analysis, ensure_ascii=False, indent=2)}

可选标签：
{options_str}

标签生成策略：
1. 对话质量≥7分：积极推理，基于线索合理推断标签
2. 对话质量4-6分：适度推理，明确线索才填写标签  
3. 对话质量≤3分：保守策略，只填写非常确定的标签

推理指导：
- 工作繁忙+加班 → lifestyle可能是"熬夜"
- 情绪从正常变消极 → emotion填写最终状态
- 要求转人工+攻击性语言 → stage可能是"流失客户期"
- 消极情绪用户 → retention_strategy建议"客户关怀"
- 周末加班 → 可能是高压职业类型

规则：
1. 只能选择上述预定义选项，不能自创
2. 多个选项用英文逗号分隔：如"程序员,设计师"
3. 无合适选项时填null
4. 不能用"和"、"与"等连接词

请严格按照以下JSON格式返回标签结果，不要添加任何其他文本：
{{
    "occupation": "从预定义选项选择或null",
    "age": "从预定义选项选择或null",
    "region": "从预定义选项选择或null",
    "lifestyle": "从预定义选项选择或null",
    "family_status": "从预定义选项选择或null",
    "emotion": "从预定义选项选择或null",
    "character": "从预定义选项选择或null",
    "values": "从预定义选项选择或null",
    "aesthetic_style": "从预定义选项选择或null",
    "ability": "从预定义选项选择或null",
    "willingness": "从预定义选项选择或null",
    "preferences": "从预定义选项选择或null",
    "current_use": "从预定义选项选择或null",
    "potential_needs": "从预定义选项选择或null",
    "decision_factors": "从预定义选项选择或null",
    "purchase_intent_score": "从预定义选项选择或null",
    "stage": "从预定义选项选择或null",
    "value": "从预定义选项选择或null",
    "retention_strategy": "从预定义选项选择或null"
}}

输出要求：
1. 必须返回有效的JSON格式
2. 只输出JSON内容，不要添加其他文字
3. 严格按照上述JSON格式输出"""

    def _format_options(self) -> str:
        """格式化预定义选项"""
        sections = []
        
        # 社会画像
        social = profile_variables["social_profile"]
        sections.append(f"occupation: {social['occupation']}")
        sections.append(f"age: {social['age']}")
        sections.append(f"region: {social['region']}")
        sections.append(f"lifestyle: {social['lifestyle']}")
        sections.append(f"family_status: {social['family_status']}")
        sections.append(f"emotion: {social['emotion']}")
        
        # 性格特征
        personality = profile_variables["personality_traits"]
        sections.append(f"character: {personality['character']}")
        sections.append(f"values: {personality['values']}")
        sections.append(f"aesthetic_style: {personality['aesthetic_style']}")
        
        # 消费画像
        consumption = profile_variables["consumption_profile"]
        sections.append(f"ability: {consumption['ability']}")
        sections.append(f"willingness: {consumption['willingness']}")
        sections.append(f"preferences: {consumption['preferences']}")
        
        # 产品意图
        product = profile_variables["product_intent"]
        sections.append(f"current_use: {product['current_use']}")
        sections.append(f"potential_needs: {product['potential_needs']}")
        sections.append(f"decision_factors: {product['decision_factors']}")
        sections.append(f"purchase_intent_score: {product['purchase_intent_score']}")
        
        # 客户生命周期
        lifecycle = profile_variables["customer_lifecycle"]
        sections.append(f"stage: {lifecycle['stage']}")
        sections.append(f"value: {lifecycle['value']}")
        sections.append(f"retention_strategy: {lifecycle['retention_strategy']}")
        
        return "\n".join(sections)

    async def analyze_conversation(self, config: Optional[Dict] = None, state: Optional[Dict] = None) -> AnalysisResult:
        """🔍 【核心方法1】执行第一步对话分析 - 自动获取当前会话历史记录
        
        参数:
            config: Optional[Dict] - 配置信息
            state: Optional[Dict] - LangGraph状态，包含消息历史
        返回:
            AnalysisResult - 分析结果（包含用户基本信息、性格特征等）
        """
        try:
            # 🔍 直接从state中获取完整历史消息（LangGraph状态机制）
            messages = []
            if state:
                # 直接使用long_term_messages获取完整历史对话
                messages = state.get("long_term_messages", [])
                if messages:
                    logger.info(f"从long_term_messages获取到{len(messages)}条历史消息")
            
            if not messages:
                raise ValueError("当前会话没有历史聊天记录")
            
            # 构建分析提示词
            analysis_prompt = self._build_analysis_prompt()
            # 🎯 关键步骤：将对话记录与分析提示组合（先规范化历史消息）
            normalized_history = self._normalize_messages(messages)
            analysis_messages = normalized_history + [HumanMessage(content=analysis_prompt)]
            
            logger.info(f"执行第一步对话分析，使用{len(messages)}条对话消息")
            # 🤖 调用AI模型分析历史聊天记录
            analysis_response = await asyncio.to_thread(self.model.invoke, analysis_messages)
            
            # 调试：打印原始响应
            logger.info(f"[DEBUG] AI模型原始响应: {analysis_response.content}")
            
            # 解析JSON响应（使用response_format后应该直接是JSON格式）
            import json
            
            try:
                analysis_data = json.loads(analysis_response.content.strip())
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析失败: {e}")
                logger.error(f"原始响应内容: {repr(analysis_response.content)}")
                # 如果直接解析失败，尝试清理markdown标记
                import re
                content = analysis_response.content.strip()
                content = re.sub(r'^```json\s*', '', content)
                content = re.sub(r'\s*```$', '', content)
                content = content.strip()
                try:
                    analysis_data = json.loads(content)
                    logger.warning("通过清理markdown标记成功解析JSON")
                except json.JSONDecodeError:
                    raise ValueError(f"AI模型返回的不是有效JSON格式: {analysis_response.content[:200]}...")
            
            return AnalysisResult(**analysis_data)
            
        except Exception as e:
            logger.error(f"对话分析失败: {e}")
            raise
    
    async def generate(self, messages: List[BaseMessage]) -> UserProfile:
        """🏷️ 【核心方法2】生成用户画像（完整流程）- 处理历史聊天记录
        
        参数:
            messages: List[BaseMessage] - 历史聊天记录列表
        返回:
            UserProfile - 完整的用户画像标签
        """
        if not messages:
            raise ValueError("聊天记录为空")
        
        try:
            # 🔍 第一步：分析历史对话记录（内部会自动过滤消息）
            config_dict = {"configurable": {"messages": messages}}
            analysis_result = await self.analyze_conversation(config_dict)
            
            # 🏷️ 第二步：基于分析结果生成标准化标签
            profile = await self.generate_labels_from_analysis(analysis_result)
            
            return profile
            
        except Exception as e:
            logger.error(f"生成用户画像失败: {e}")
            raise
    
    async def generate_labels_from_analysis(self, analysis_result: AnalysisResult) -> UserProfile:
        """基于分析结果生成用户画像标签"""
        try:
            analysis_data = analysis_result.model_dump()
            
            # 输出分析结果（调试用）
            print("🔍 基于分析结果生成标签:")
            for key, value in analysis_data.items():
                print(f"  {key}: {value}")
            print()
            
            # 生成标签
            labeling_prompt = self._build_labeling_prompt(analysis_data)
            labeling_messages = [SystemMessage(content="你是标签生成专家"), HumanMessage(content=labeling_prompt)]
            
            logger.info("生成标准化标签")
            labeling_response = await asyncio.to_thread(self.model.invoke, labeling_messages)
            
            # 调试：打印原始响应
            logger.info(f"[DEBUG] 标签生成原始响应: {labeling_response.content}")
            
            # 解析JSON响应（使用response_format后应该直接是JSON格式）
            import json
            try:
                profile_data = json.loads(labeling_response.content.strip())
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析失败: {e}")
                logger.error(f"原始响应内容: {repr(labeling_response.content)}")
                # 如果直接解析失败，尝试清理markdown标记
                import re
                content = labeling_response.content.strip()
                content = re.sub(r'^```json\s*', '', content)
                content = re.sub(r'\s*```$', '', content)
                content = content.strip()
                try:
                    profile_data = json.loads(content)
                    logger.warning("通过清理markdown标记成功解析JSON")
                except json.JSONDecodeError:
                    raise ValueError(f"AI模型返回的不是有效JSON格式: {labeling_response.content[:200]}...")
            
            return UserProfile(**profile_data)
            
        except Exception as e:
            logger.error(f"基于分析结果生成标签失败: {e}")
            raise

# LangGraph 集成
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langchain_core.runnables import RunnableConfig


# 自定义输出类 - 只包含需要的字段
@dataclass
class ProfileLabelOutput:
    """用户画像标签输出 - 只包含必要字段"""
    user_profile_label: Optional[UserProfile] = field(default=None)
    error_message: Optional[str] = field(default=None)

@dataclass
class ProfileAnalysisOutput:
    """用户画像分析输出 - 只包含必要字段"""
    analysis_result: Optional[AnalysisResult] = field(default=None)
    error_message: Optional[str] = field(default=None)

# 导入AgentState
from states import AgentState

async def profile_analysis_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """🔄 【LangGraph节点1】第一步分析节点 - 自动获取当前会话历史聊天记录分析"""
    try:
        # 📝 自动获取当前会话的历史聊天记录
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            return {"error_message": "缺少会话线程ID，无法获取历史聊天记录"}
        
        # 直接从state中获取完整历史消息（LangGraph状态机制）
        # 直接使用long_term_messages获取完整历史对话
        messages = state.get("long_term_messages", [])
        if messages:
            logger.info(f"profile_analysis_node从long_term_messages获取到{len(messages)}条历史消息")
        
        if not messages:
            return {"error_message": "当前会话暂无历史聊天记录"}
        
        # 获取配置
        model_provider = config.get("configurable", {}).get("model_provider", "openrouter")
        model_name = config.get("configurable", {}).get("model_name", "x-ai/grok-3")
        temperature = config.get("configurable", {}).get("temperature", 0.3)
        
        # 🤖 创建生成器并分析历史聊天记录
        generator = ProfileGenerator(model_provider, model_name, temperature)
        analysis_result = await generator.analyze_conversation(config, state)
        
        return {
            "analysis_result": analysis_result,
            "error_message": None
        }
    
    except Exception as e:
        logger.error(f"对话分析失败: {e}")
        return {"error_message": str(e)}

def create_profile_analysis_graph():
    """创建第一步分析工作流 - 专门用于替代profile_agent"""
    # 定义输入模型，只包含必要的输入字段
    class ProfileAnalysisInput(BaseModel):
        """用户画像分析输入"""
        pass  # 空输入，所有数据通过LangGraph的messages自动注入
    
    graph = StateGraph(input=ProfileAnalysisInput, state_schema=AgentState, output=ProfileAnalysisOutput)
    graph.add_node("analysis_generator", profile_analysis_node)
    graph.add_edge(START, "analysis_generator")
    
    compiled_graph = graph.compile()
    return compiled_graph

# 导出第一步分析工作流
profile_analysis_graph = create_profile_analysis_graph()

async def profile_label_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """🔄 【LangGraph节点2】用户画像标签生成节点 - 自动获取当前会话历史聊天记录的完整流程"""
    try:
        # 🔍 调试信息：打印完整的config内容
        logger.info(f"[DEBUG] profile_label_node 接收到的 config: {config}")
        logger.info(f"[DEBUG] profile_label_node 接收到的 state: {state}")
        
        # 获取配置
        model_provider = config.get("configurable", {}).get("model_provider", "openrouter")
        model_name = config.get("configurable", {}).get("model_name", "x-ai/grok-3")
        temperature = config.get("configurable", {}).get("temperature", 0.3)
        
        generator = ProfileGenerator(model_provider, model_name, temperature)
        
        # 检查是否有分析结果
        analysis_result = state.get("analysis_result")
        
        if not analysis_result:
            # 📝 自动获取当前会话的历史聊天记录
            thread_id = config.get("configurable", {}).get("thread_id")
            logger.info(f"[DEBUG] 提取到的 thread_id: {thread_id}")
            
            if not thread_id:
                return {"error_message": "缺少会话线程ID，无法获取历史聊天记录"}
            
            # 直接从state中获取完整历史消息（LangGraph状态机制）
            # 直接使用long_term_messages获取完整历史对话
            messages = state.get("long_term_messages", [])
            if messages:
                logger.info(f"profile_label_node从long_term_messages获取到{len(messages)}条历史消息")
            
            if not messages:
                return {"error_message": "当前会话暂无历史聊天记录，无法生成标签"}
            
            # 🔍 第一步：分析历史对话记录（会自动过滤只保留人类和AI对话）
            analysis_result = await generator.analyze_conversation(config, state)
        
        # 第二步：基于分析结果生成标签
        profile = await generator.generate_labels_from_analysis(analysis_result)
        
        return {
            "user_profile_label": profile,
            "analysis_result": analysis_result,  # 同时返回分析结果
            "error_message": None
        }
    
    except Exception as e:
        logger.error(f"用户画像标签生成失败: {e}")
        return {"error_message": str(e)}

def create_profile_label_graph():
    """创建用户画像标签生成工作流"""
    # 定义输入模型，只包含必要的输入字段
    class ProfileLabelInput(BaseModel):
        """用户画像标签生成输入"""
        pass  # 空输入，所有数据通过LangGraph的messages自动注入
    
    graph = StateGraph(input=ProfileLabelInput, state_schema=AgentState, output=ProfileLabelOutput)
    graph.add_node("profile_generator", profile_label_node)
    graph.add_edge(START, "profile_generator")
    
    compiled_graph = graph.compile()
    return compiled_graph

# 导出主要接口
profile_label_graph = create_profile_label_graph()


if __name__ == "__main__":
    print("用户画像生成模块 - 分步式优化版本加载完成")