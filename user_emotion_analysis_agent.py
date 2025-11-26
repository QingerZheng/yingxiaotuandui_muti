from typing import List, Dict, Optional, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from Configurations import Configuration
from states import AgentState, EmotionalState, CustomerIntent, AppointmentInfo, DebugInfo
from json_parser_utils import robust_json_parse, create_fallback_dict
import asyncio
class Output(TypedDict):
    """子图的输出状态 - 只包含最终回复"""
    agent_temperature:Optional[float]
    emotional_state: Optional[EmotionalState]
    customer_intent: Optional[CustomerIntent]
    appointment_info: Optional[AppointmentInfo]
    customer_info:Optional[Dict[str, str]]
    debug_info: Optional[DebugInfo]
    candidate_actions: List[str]  # 候选行动
    invitation_status: Optional[int]
    invitation_time: Optional[int] # 13位毫秒时间戳
    invitation_project: Optional[str]

def analyze_sentiment_node(state: any, config=None):
    """
    根据当前的情感状态，动态设置助手的温度（创造性）。
    前期其实可以不用这个模块7/3
    """
    user_requires_message = state["user_requires_message"]
    if not user_requires_message:  # 用户没有发消息给销售，不需要回复，直接退出这个节点
        return state

    # 正确访问 internal_monologue，它在 debug_info 对象内部
    debug_info = state.get("debug_info",DebugInfo())
    internal_monologue = debug_info.internal_monologue if debug_info and debug_info.internal_monologue else []
    emotional_state = state.get("emotional_state", EmotionalState())  # 我们从这里获取情感
    verbose = state.get("verbose", False)  # 现在可以直接从 state 获取

    if not emotional_state:
        # 如果没有情感状态，使用默认温度
        return {"agent_temperature": 0.6}

    # 基于舒适度和熟悉度来设定温度
    # 如果用户感到舒适和高兴，我们说话的方式就会更活泼，更像朋友。
    # 但其实这部分，应该在模型敲定后再做评估，因为每个模型的风格并不同。
    comfort = emotional_state.comfort_level
    familiarity = emotional_state.familiarity_level

    agent_temperature = state.get("agent_temperature", 0.5)  # 默认值，qwen使用低温，避免过度活跃
    if comfort > 0.6 and familiarity > 0.5:
        agent_temperature = 0.6  # 更富创造性、更像朋友
    elif comfort < 0.3:
        agent_temperature = 0.6  # 更保守、更谨慎

    new_monologue = internal_monologue + [
        f"温度设定：根据当前情感 (舒适度:{comfort:.2f}, 熟悉度:{familiarity:.2f})，设定温度为 {agent_temperature}。"]

    # 只在verbose模式下输出调试信息
    if verbose:
        print(f"[DEBUG] 情感分析节点: 温度设定为 {agent_temperature}")

    # 更新 debug_info 中的 internal_monologue
    updated_debug_info = DebugInfo(
        current_stage=debug_info.current_stage if debug_info else "initial_contact",
        emotional_state=debug_info.emotional_state if debug_info else EmotionalState(),
        internal_monologue=new_monologue
    )

    return {
        "agent_temperature": agent_temperature,
        "debug_info": updated_debug_info,
    }


async def _design_node(state: AgentState, config=None) -> Dict[str, Any]:
    """
    智能决策节点，重新设计为服务导向而非销售导向。
    允许"正确的缺点"，表现得更加自然和人性化。
    """
    user_requires_message = state["user_requires_message"]
    if not user_requires_message:#用户没有发消息给销售，不需要回复，直接退出这个节点
        return state

    # 正确访问 internal_monologue，它在 debug_info 对象内部
    debug_info = state.get("debug_info")
    internal_monologue = debug_info.internal_monologue if debug_info and debug_info.internal_monologue else []
    verbose = state.get("verbose", False)  # 现在可以直接从 state 获取

    # 1. 调用状态评估器，获取最新的情感状态
    from blocks.state_evaluator import evaluate_state
    from blocks.intent_analyzer import analyze_customer_intent, update_appointment_info
    state_data=dict(state)

    # 异步并行执行两个工具调用
    evaluation_result, intent_result,judge_invitation_result = await asyncio.gather(
                asyncio.to_thread(evaluate_state.invoke, {"state_dict": state_data}),
        asyncio.to_thread(analyze_customer_intent.invoke, {"state_dict": state_data}),
        asyncio.to_thread(judge_invitation_state.invoke, {"state_dict": state_data, "config": config}),
        return_exceptions=True
    )
    
    # 处理异常情况
    if isinstance(evaluation_result, Exception):
        print(f"状态评估失败: {evaluation_result}")
        evaluation_result = {}
    if isinstance(intent_result, Exception):
        print(f"意图分析失败: {intent_result}")
        intent_result = {}
    if isinstance(judge_invitation_result, Exception):
        print(f"邀约状态判断失败: {judge_invitation_result}")
        judge_invitation_result = {}



    # 更新状态。如果评估失败，则使用旧状态
    # 安全地获取情感状态
    from json_parser_utils import safe_create_emotional_state
    
    if "emotional_state" in evaluation_result and evaluation_result["emotional_state"] is not None:
        current_emotional_state = safe_create_emotional_state(evaluation_result["emotional_state"])
    else:
        # 从状态中获取现有的情感状态，如果没有则创建新的
        existing_state = state.get("emotional_state")
        current_emotional_state = safe_create_emotional_state(existing_state)
    
    customer_intent = evaluation_result.get("customer_intent_level", state.get("customer_intent_level", "low"))
    customer_info = evaluation_result.get("customer_info", state.get("customer_info", {}))
    current_customer_intent = intent_result.get("customer_intent")


    # 3. 新增：更新预约信息
    appointment_updates = {}
    if current_customer_intent:
        appointment_updates = update_appointment_info(state, current_customer_intent)

    # 合并预约信息更新
    current_appointment_info = state.get("appointment_info")
    if appointment_updates.get("appointment_info"):
        current_appointment_info = appointment_updates["appointment_info"]

    internal_monologue.append(f"情感评估完成: {current_emotional_state.model_dump_json()}")
    internal_monologue.append(f"客户意向评估: {customer_intent}")
    if current_customer_intent:
        internal_monologue.append(
            f"行为意图识别: {current_customer_intent.intent_type} (置信度: {current_customer_intent.confidence:.2f})")
        if current_customer_intent.extracted_info:
            internal_monologue.append(f"提取信息: {current_customer_intent.extracted_info}")
    if current_appointment_info:
        internal_monologue.append(
            f"预约状态: {current_appointment_info.appointment_status}, 时间: {current_appointment_info.preferred_time or '未定'}")

    if verbose:
        print(f"[DEBUG] 策略设计节点: 客户意向={customer_intent}, 信任度={current_emotional_state.trust_level:.2f}")

    # 2. 改进对话阶段 - 更自然的推进逻辑
    # 从 debug_info 中获取 current_stage
    current_stage = debug_info.current_stage if debug_info and debug_info.current_stage else "initial_contact"
    trust_level = current_emotional_state.trust_level
    comfort_level = current_emotional_state.comfort_level
    familiarity_level = current_emotional_state.familiarity_level
    turn_count = state.get("turn_count", 0)

    new_stage = current_stage  # 默认保持当前阶段

    # 改进的阶段推进逻辑（保持原有阶段名称）
    if current_stage == "initial_contact":
        # 阶段1：初次接触 - 自然问候，建立基础连接
        if turn_count >= 1 and comfort_level > 0.2:
            new_stage = "ice_breaking"
    elif current_stage == "ice_breaking":
        # 阶段2：轻松破冰 - 建立真实连接，允许"缺陷"
        if familiarity_level > 0.3:
            new_stage = "subtle_expertise"
    elif current_stage == "subtle_expertise":
        # 阶段3：展示专业 - 客观展示，非夸大宣传
        if trust_level > 0.4:
            new_stage = "pain_point_mining"
    elif current_stage == "pain_point_mining":
        # 阶段4：了解需求 - 真诚询问，非推销式
        if trust_level > 0.6 and customer_intent in ["medium", "high"]:
            new_stage = "solution_visualization"
    elif current_stage == "solution_visualization":
        # 阶段5：解决方案 - 协助决策，非强制推销
        if trust_level > 0.7 and customer_intent == "high":
            new_stage = "natural_invitation"

    # 自然回退机制：如果客户不舒服，回到更轻松的阶段
    if comfort_level < 0.3 and current_stage not in ["initial_contact", "ice_breaking"]:
        new_stage = "ice_breaking"  # 自然回退到轻松破冰
        internal_monologue.append(f"检测到舒适度过低 ({comfort_level:.2f})，自然回退到轻松破冰")
    elif trust_level < 0.2 and current_stage not in ["initial_contact"]:
        new_stage = "initial_contact"  # 重新开始
        internal_monologue.append(f"检测到信任度过低 ({trust_level:.2f})，重新开始对话")

    if new_stage != current_stage:
        internal_monologue.append(
            f"自然流程推进: '{current_stage}' → '{new_stage}' (信任{trust_level:.2f}/舒适{comfort_level:.2f}/熟悉{familiarity_level:.2f})")

    # 3. 改进动作决策 - 基于现有模块，让行为更自然
    candidate_actions = []

    # 优先级0：处理邀约确认状态（新增）
    invitation_status = judge_invitation_result.get("invitation_status")
    if invitation_status == 1:
        # 客户已确认邀约，应该进行确认和后续安排
        candidate_actions = ["active_close", "value_display"]
        internal_monologue.append(f"检测到邀约确认状态，进行预约确认和后续安排")
    
    # 优先级1：处理明确的预约意图
    elif current_customer_intent and current_customer_intent.intent_type in ["appointment_request", "time_confirmation",
                                                                           "ready_to_book"]:
        if current_customer_intent.confidence > 0.8:
            # 高置信度：使用自然邀约
            candidate_actions = ["active_close", "value_display"]
            internal_monologue.append(f"检测到明确预约需求，进行自然邀约")
        else:
            # 低置信度：先了解需求
            candidate_actions = ["needs_analysis", "value_display"]
            internal_monologue.append(f"预约意图不明确，先了解具体需求")

    # 优先级2：处理信息咨询
    elif current_customer_intent and current_customer_intent.intent_type == "info_seeking":
        # 明确需求时优先提供信息，而不是挖掘需求
        candidate_actions = ["value_display"]
        # 只有在提供基本信息后，才考虑了解细节
        if familiarity_level > 0.4:  # 已经有一定基础时才询问细节
            candidate_actions.append("needs_analysis")
        internal_monologue.append(f"客户寻求信息，优先提供项目介绍")

    # 优先级3：处理价格询问（真实回应而非销售话术）
    elif current_customer_intent and current_customer_intent.intent_type == "price_inquiry":
        candidate_actions = ["value_display", "value_pitch"]
        if trust_level > 0.5:
            candidate_actions.append("active_close")  # 高信任时可以推进
        internal_monologue.append(f"价格咨询，提供真实信息")

    # 优先级4：处理顾虑（理解而非反驳）
    elif current_customer_intent and current_customer_intent.intent_type == "concern_raised":
        candidate_actions = ["stress_response", "rapport_building"]
        if comfort_level < 0.4:
            candidate_actions.append("rapport_building")  # 舒适度低时重建关系
        internal_monologue.append(f"客户有顾虑，给予理解和缓解")

    # 优先级5：基于阶段的自然对话流程
    else:
        # 根据当前阶段决定自然回应策略
        if new_stage == "initial_contact":
            candidate_actions = ["greeting","needs_analysis", "value_display"]
        elif new_stage == "ice_breaking":
            candidate_actions = ["rapport_building", "needs_analysis", "value_display"]
            # 偶尔允许"缺陷"：简短回复
            if turn_count % 4 == 0:  # 偶尔表现得不那么完美
                candidate_actions = ["rapport_building"]  # 保持简洁
        elif new_stage == "subtle_expertise":
            candidate_actions = ["value_display", "needs_analysis"]
            if familiarity_level > 0.4:
                candidate_actions.append("needs_analysis")
        elif new_stage == "pain_point_mining":
            # 根据客户需求明确程度调整策略
            if current_customer_intent and current_customer_intent.intent_type == "info_seeking":
                # 如果客户已经表达明确需求，直接提供信息
                candidate_actions = ["value_display", "needs_analysis"]
            else:
                # 否则才进行需求挖掘
                candidate_actions = ["needs_analysis", "pain_point_test"]
                if trust_level > 0.6:
                    candidate_actions.append("value_display")
        elif new_stage == "solution_visualization":
            candidate_actions = ["value_pitch", "value_display"]
            if customer_intent == "high":
                candidate_actions.append("active_close")
        elif new_stage == "natural_invitation":
            candidate_actions = ["active_close"]
            if customer_intent != "high":
                candidate_actions.append("value_pitch")

    #     # 新增：基于语义的动作建议--7/18凌晨——效果很差
    #     last_user_message = ''
    #     for msg in reversed(state.get("messages", [])):
    #         if msg.type == 'human':
    #             last_user_message = msg.content
    #             break
    #     if last_user_message:
    #         feedback_sampler, _ = SamplerFactory.get_sampler_and_cost(state.get("feedback_model") or "o3")
    #         semantic_prompt = f'''
    # 你是一个对话策略专家。根据用户最后消息"{last_user_message}"，从以下动作中建议1-3个最合适的：
    # 可用动作: greeting, rapport_building, needs_analysis, value_display, stress_response, pain_point_test, value_pitch, active_close, reverse_probe
    # 输出JSON: {{"suggested_actions": ["action1", "action2"]}}
    #         '''
    #         semantic_response, _ = feedback_sampler([{'role': 'user', 'content': semantic_prompt}], temperature=0.1, response_format='json_object')
    #         try:
    #             suggested = json.loads(semantic_response).get('suggested_actions', [])
    #             candidate_actions.extend(suggested)
    #             internal_monologue.append(f'语义建议动作: {suggested}')
    #         except:
    #             pass

    # 智能策略调整：让行为更自然和贴近真人

    # 1. 根据情感状态调整策略
    if trust_level < 0.3:
        # 信任度低时，优先建立关系
        candidate_actions = ["rapport_building", "stress_response", "needs_analysis"]
        internal_monologue.append(f"信任度过低 ({trust_level:.2f})，优先建立关系")
    elif comfort_level < 0.2 and new_stage in ["solution_visualization", "natural_invitation"]:
        # 舒适度低时，回退到缓解压力
        candidate_actions.insert(0, "stress_response")
        internal_monologue.append(f"舒适度过低 ({comfort_level:.2f})，优先缓解压力")

    # 2. 意向等级特殊处理
    if customer_intent == "fake_high" and "reverse_probe" not in candidate_actions:
        candidate_actions.append("reverse_probe")  # 识别虚假高意向
        internal_monologue.append(f"检测到虚假高意向，添加反向试探")
    elif customer_intent == "low" and new_stage in ["solution_visualization", "natural_invitation"]:
        # 低意向客户不应该进入高压销售阶段
        candidate_actions = ["rapport_building", "needs_analysis", "stress_response"]
        internal_monologue.append(f"低意向客户，回退到基础交流")

    # 3. 自然搜索空间管理
    search_space_size = len(candidate_actions)

    if search_space_size == 1:
        # 适当扩展，保持灵活性
        primary_action = candidate_actions[0]

        if primary_action == "active_close":
            if comfort_level < 0.6:
                candidate_actions.append("stress_response")
            if trust_level > 0.7:
                candidate_actions.append("value_display")
        elif primary_action in ["value_display", "value_pitch"]:
            candidate_actions.append("needs_analysis")
            if trust_level > 0.6:
                candidate_actions.append("active_close")
        elif primary_action == "stress_response":
            candidate_actions.append("rapport_building")

        internal_monologue.append(f"扩展搜索空间: {primary_action} → {candidate_actions}")

    elif search_space_size > 3:
        # 保持合理范围
        candidate_actions = candidate_actions[:3]
        internal_monologue.append(f"限制搜索空间为3个选项")

    # 确保至少有基础回应能力
    if not candidate_actions:
        candidate_actions = ["rapport_building"]
        internal_monologue.append(f"兜底策略：使用基础关系建立")

    final_search_space = len(candidate_actions)
    decision_context = f"阶段:{new_stage}, 情感:{customer_intent}, 信任:{trust_level:.2f}"
    if current_customer_intent:
        decision_context += f", 意图:{current_customer_intent.intent_type}"
    internal_monologue.append(f"策略决策 ({decision_context}) -> 候选动作: {candidate_actions}")

    # 构建返回状态
    result = {
        "emotional_state": current_emotional_state,
        "customer_intent_level": customer_intent,
        "candidate_actions": list(set(candidate_actions)),
        "current_stage": new_stage,
    }

    # 更新 debug_info 对象，包含 current_stage 和 internal_monologue
    updated_debug_info = DebugInfo(
        current_stage=new_stage,
        emotional_state=current_emotional_state.model_dump() if current_emotional_state else None,
        internal_monologue=internal_monologue
    )
    result["debug_info"] = updated_debug_info

    # 添加新的状态字段（只添加 AgentState 中存在的字段）
    if current_customer_intent:
        result["customer_intent"] = current_customer_intent
    if current_appointment_info:
        result["appointment_info"] = current_appointment_info
    if customer_info:
        result["customer_info"] = customer_info
    result.update(judge_invitation_result)

    return result

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import json
from datetime import datetime, timezone, timedelta

@tool
def judge_invitation_state(state_dict: dict = None, config=None) -> dict:
    """
    【判断邀约情况工具】
    使用大模型根据聊天记录判断客户是否已明确同意邀约，并提取最新邀约时间和项目。
    返回字段：
    - invitation_status: 是否已邀约（int）
    - invitation_time: 邀约的13位时间戳（如无则为 null）
    - invitation_project: 项目名称（如无则为 null）
    """
    print("=" * 80)
    print("🔍 [DEBUG-邀约判断] 开始执行judge_invitation_state")
    print("=" * 80)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
        print("🔍 [DEBUG-邀约判断] 解包了state_dict包装")

    messages = state_dict.get("long_term_messages", [])
    print(f"🔍 [DEBUG-邀约判断] 获取到 {len(messages)} 条消息")

    role_map = {
        "human": "客户",
        "user": "客户",
        "ai": "销售顾问",
        "assistant": "销售顾问"
    }

    history = []
    for msg in messages:
        # 处理不同类型的消息对象
        if isinstance(msg, dict):
            # 字典格式的消息
            msg_type = msg.get("type", "").lower()
            content = msg.get("content", "")
            # 获取时间戳信息
            timestamp = msg.get("additional_kwargs", {}).get("timestamp", "")
        elif hasattr(msg, 'type') and hasattr(msg, 'content'):
            # HumanMessage/AIMessage 等对象
            msg_type = getattr(msg, 'type', '').lower()
            content = getattr(msg, 'content', '')
            # 获取时间戳信息
            timestamp = getattr(msg, 'additional_kwargs', {}).get("timestamp", "") if hasattr(msg, 'additional_kwargs') else ""
        else:
            # 其他格式，尝试通用属性访问
            msg_type = getattr(msg, 'type', '').lower()
            content = getattr(msg, 'content', str(msg))
            timestamp = ""
        
        role = role_map.get(msg_type, msg_type if msg_type else "未知")
        # 包含时间戳的对话历史
        if timestamp:
            history.append(f"[{timestamp}] {role}: {content}")
        else:
            history.append(f"{role}: {content}")
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone(timedelta(hours=8)))
    current_time_iso = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+08:00"

    #确认邀约的时间生成仍然存在问题：不一定都是准的，可能非常晚，也可能是过去的时间。不过一旦指定了时间，确实是准的。但是下次对话，**可能**会胡乱修改掉时间。需要做个大修改，生成13位毫秒级时间戳不能完全依赖llm，多靠工具--08-15_黄国强

    prompt = f"""
你是一个智能邀约判断助手。请从以下对话中判断客户是否已**明确同意邀约**，并提取**最新有效**的邀约信息（时间和项目）。

**关键要求**：请仔细分析对话中的时间戳，根据对话发生的具体时间来计算相对时间表达（如"明天"、"后天"等）的绝对日期。

- 确认邀约的指标为"invitation_status": 1，你需要分析对话历史，识别该客户是否已经同意到店，此时才能将"invitation_status"设置为1。如果客户从已邀约状态变更，并推迟时间为未知则为2。如果客户初次聊天，并且无到店意向为0
- 如果客户提出变更时间、推迟或取消邀约，则"invitation_status": 2 
- 邀约时间请输出ISO格式时间字符串（如：2025-08-12T15:30:00+08:00），若无有效时间输出 null。
- 项目名称若无也输出 null。
- **重要：如果客户确认邀约，请根据对话中的时间戳来确定相对时间表达的具体日期**
  - 如果客户同意了邀约状态，并且说"明天"，请根据对话时间戳计算下一天的日期
  - 如果客户同意了邀约状态，并且说"后天"，请根据对话时间戳计算下两天的日期
  - 如果客户同意了邀约状态，并且说"下周一"，请根据对话时间戳计算下一个周一的日期
  - **时间计算示例**：
    - 对话时间：2025-08-11T12:06:18+08:00，客户说"明天下午" → 邀约时间：2025-08-12 15:30:00
    - 对话时间：2025-08-11T12:06:18+08:00，客户说"后天上午" → 邀约时间：2025-08-13 10:30:00
- 若客户同意了邀约状态，但却只说了“上午”或“下午”等模糊时间段，而并未明确具体时间点，请根据对话历史生成一个合理的工作时间默认时间



- 晚上不做默认时间，若无明确时间则返回 null。

请严格按照如下格式输出 JSON，且只输出 JSON，不要带其他说明：

{{
  "invitation_status": 0    如果客户无到店意向为0，如果客户**确认邀约，比如“好的，明天上午10点”“恩，我过去看一下”**则为1，否则为0，
  如果客户从已邀约状态变更，并推迟时间为未知则为2 如果客户提出变更时间、推迟或取消邀约，则"invitation_status": 2 
  "schedule_time": 邀约的ISO格式时间字符串（如无则为 null），
  "invitation_project": 项目名称字符串（如无则为 null）
}}

根据对话历史：
{history}

# 以下是示例：

示例1：
对话历史:
[2025-08-03T12:04:18.830868+08:00] user: 你们什么时候有空？
[2025-08-03T12:04:48.830868+08:00] assistant: 我们周六上午有空，可以过来吗？
[2025-08-03T12:05:18.830868+08:00] user: 好的。
输出：
{{
  "invitation_status": 1,
  "schedule_time": "2025-08-09T10:30:00+08:00",  // 2025-08-09 10:30:00（示例时间）
  "invitation_project": null
}}

示例2（客户变更时间）：
对话历史:
[2025-08-03T15:20:15.123+08:00] user: 周六上午可以吗？
[2025-08-03T15:20:45.123+08:00] assistant: 可以的，您周六上午几点方便？
[2025-08-03T15:21:15.123+08:00] user: 我周六上午没空，改成周日上午9点可以吗？
[2025-08-03T15:21:45.123+08:00] assistant: 好的，周日上午9点为您预约。
输出：
{{
  "invitation_status": 1,
  "schedule_time": "2025-08-10T09:00:00+08:00",  // 2025-08-10 09:00:00（示例时间）
  "invitation_project": null
}}

示例3（客户拒绝或无同意）：
对话历史:
[2025-08-03T14:30:45.789+08:00] user: 你们什么时候有空？
[2025-08-03T14:31:15.789+08:00] assistant: 周六上午可以来吗？
[2025-08-03T14:31:45.789+08:00] user: 我先考虑一下。
输出：
{{
  "invitation_status": 0,
  "schedule_time": null,
  "invitation_project": null
}}

示例4（客户明确接受并指定项目）：
对话历史:
[2025-08-03T16:45:22.456+08:00] user: 你们那个水光针最近有活动吗？
[2025-08-03T16:45:52.456+08:00] assistant: 有的，最近水光针做活动，您可以周六上午来体验一下。
[2025-08-03T16:46:22.456+08:00] user: 好的，那就周六上午做水光针。
输出：
{{
  "invitation_status": 1,
  "schedule_time": "2025-08-09T10:30:00+08:00",  // 2025-08-09 10:30:00（示例时间）
  "invitation_project": "水光针"
}}

示例5（相对时间表达）：
对话历史:
[2025-08-11T12:06:18.830868+08:00] user: 我想明天下午去你们店里做水光针，可以吗？
[2025-08-11T12:06:48.830868+08:00] assistant: 可以的，明天下午3:30为您预约水光针。
**时间计算说明**：对话时间是8月11日12:06，客户说"明天下午" = 8月12日下午15:30
输出：
{{
  "invitation_status": 1,
  "schedule_time": "2025-08-12T15:30:00+08:00",  // 2025-08-12 15:30:00（明天下午3:30）
  "invitation_project": "水光针"
}}

请基于以上要求判断并输出结果。
"""

    try:
        print("🔍 [DEBUG-邀约判断] 开始配置模型...")

        # 从config中提取热更新配置
        hot_config = None
        if config and hasattr(config, 'get'):
            configurable = config.get("configurable", {})
            if configurable:
                hot_config = configurable
                print("🔍 [DEBUG-邀约判断] 发现热更新配置")

        agent_temperature = state_dict.get("agent_temperature", 0.5)
        print(f"🔍 [DEBUG-邀约判断] agent_temperature: {agent_temperature}")

        # 优先使用热更新配置，否则使用默认配置
        configuration = Configuration.from_context()
        print(f"🔍 [DEBUG-邀约判断] 默认配置 - provider: {configuration.model_provider}, model: {configuration.evaluation_model}")

        # 默认值设置
        model_provider = "openrouter"
        model_name = "deepseek/deepseek-chat-v3.1"
        print(f"🔍 [DEBUG-邀约判断] 使用默认值 - provider: {model_provider}, model: {model_name}")

        if hot_config:
            model_provider = hot_config.get("model_provider", model_provider)
            # 优先使用热更新配置中的evaluation_model，如果没有则使用model_name
            model_name = hot_config.get("evaluation_model", hot_config.get("model_name", model_name))
            # 使用热更新的temperature，如果没有则使用状态中的
            agent_temperature = hot_config.get("agent_temperature", agent_temperature)
            config_dict = hot_config
            print(f"🔍 [DEBUG-邀约判断] 热更新配置覆盖 - provider: {model_provider}, model: {model_name}, temp: {agent_temperature}")
        else:
            model_provider = configuration.model_provider
            model_name = configuration.evaluation_model
            config_dict = configuration.model_dump()
            print(f"🔍 [DEBUG-邀约判断] 使用默认配置 - provider: {model_provider}, model: {model_name}")

        from llm import create_llm
        print(f"🔍 [DEBUG-邀约判断] 准备创建LLM - provider: {model_provider}, model: {model_name}")

        llm = create_llm(
            model_provider=model_provider,
            model_name=model_name,
            temperature=0.5
        )
        print(f"🔍 [DEBUG-邀约判断] LLM创建成功 - {type(llm)}")
    except Exception as e:
        print(f"🔍 [DEBUG-邀约判断] 错误：无法创建评估模型 '{model_name}' (provider: {model_provider}): {e}")
        import traceback
        print(f"🔍 [DEBUG-邀约判断] 详细错误信息:\n{traceback.format_exc()}")
        return {
            "invitation_status": 0,
            "invitation_time": None,
            "invitation_project": None,
            "error": "无法初始化模型"
        }

    try:
        print(f"🔍 [DEBUG-邀约判断] 准备调用模型，prompt长度: {len(prompt)} 字符")
        print(f"🔍 [DEBUG-邀约判断] prompt预览（前200字符）:\n{prompt[:200]}...")

        message = HumanMessage(content=prompt)
        print(f"🔍 [DEBUG-邀约判断] 创建HumanMessage成功")

        print(f"🔍 [DEBUG-邀约判断] 开始调用LLM...")
        response = llm.invoke(
            [message],
            response_format={"type": "json_object"}
        )
        print(f"🔍 [DEBUG-邀约判断] LLM调用完成，响应类型: {type(response)}")

        response_text = response.content
        print(f"🔍 [DEBUG-邀约判断] 获取响应内容，长度: {len(response_text)} 字符")
        print(f"🔍 [DEBUG-邀约判断] 原始模型响应:\n{response_text}")

        # 检查响应是否为空
        if not response_text or response_text.strip() == "":
            print(f"🔍 [DEBUG-邀约判断] 警告：模型返回空响应")
            return {
                "invitation_status": 0,
                "invitation_time": None,
                "invitation_project": None,
                "error": "模型返回空响应"
            }

        # 使用鲁棒的JSON解析工具
        print(f"🔍 [DEBUG-邀约判断] 开始JSON解析...")
        fallback_dict = create_fallback_dict("邀约判断")
        data = robust_json_parse(
            response_text,
            context="邀约判断",
            fallback_dict=fallback_dict,
            debug=True
        )

        print(f"🔍 [DEBUG-邀约判断] JSON解析完成，结果: {data}")

        # 验证解析结果
        if not isinstance(data, dict):
            print(f"🔍 [DEBUG-邀约判断] 错误：解析结果不是字典类型，实际类型: {type(data)}")
            return {
                "invitation_status": 0,
                "invitation_time": None,
                "invitation_project": None,
                "error": "解析结果格式错误"
            }

        def safe_timestamp(value):
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        # 获取邀约信息
        invitation_status = data.get("invitation_status")
        schedule_time = data.get("schedule_time")
        invitation_project = data.get("invitation_project")
        
        # 将ISO格式时间字符串转换为13位毫秒时间戳
        invitation_time = None
        if schedule_time:
            try:
                from datetime import datetime
                # 解析ISO格式时间字符串
                schedule_datetime = datetime.fromisoformat(schedule_time.replace('Z', '+00:00'))
                # 转换为13位毫秒时间戳
                invitation_time = int(schedule_datetime.timestamp() * 1000)
                print(f"[DEBUG] 转换时间: {schedule_time} -> {invitation_time}")
            except Exception as e:
                print(f"[DEBUG] 时间转换失败: {e}")
                invitation_time = None

        # 判断邀约时间是否已过期
        if invitation_status and invitation_time:
            # 将13位毫秒时间戳转换为datetime对象进行比较
            from datetime import datetime, timezone
            # 将时间戳转换为北京时间（+8小时）
            invitation_datetime = datetime.fromtimestamp(invitation_time / 1000, tz=timezone(timedelta(hours=8)))
            current_datetime = datetime.now(timezone(timedelta(hours=8)))
            
            # 添加调试信息
            print(f"[DEBUG] 邀约时间: {invitation_datetime}")
            print(f"[DEBUG] 当前时间: {current_datetime}")
            print(f"[DEBUG] 邀约状态: {invitation_status}")
            print(f"[DEBUG] 邀约项目: {invitation_project}")
            
            # 如果当前时间已经过了邀约时间超过1天，则邀约失效
            # 给客户1天的缓冲时间
            from datetime import timedelta
            buffer_time = invitation_datetime + timedelta(days=1)
            
            if current_datetime > buffer_time:
                print(f"[DEBUG] 邀约已过期超过1天，自动失效")
                invitation_status = 2
                invitation_time = None
                invitation_project = None
            elif current_datetime > invitation_datetime:
                print(f"[DEBUG] 邀约时间已过，但在1天缓冲期内，保持有效")

        return {
            "invitation_status": invitation_status,
            "invitation_time": invitation_time,
            "invitation_project": invitation_project
        }

    except Exception as e:
        return {
            "invitation_status": 0,
            "invitation_time": None,
            "invitation_project": None,
            "error": f"模型解析失败: {e}"
        }


def user_emotion_analysis_workflow():
    """创建外部信息查询工作流"""
    # 创建主图
    config_schema = Configuration
    user_emotion_analysis_graph = StateGraph(AgentState,config_schema=config_schema, output=Output)

    # 添加节点
    user_emotion_analysis_graph.add_node("analyze_sentiment", analyze_sentiment_node)
    user_emotion_analysis_graph.add_node("analyze_decision", _design_node)  # 并行执行工具，获得tool_results

    # 添加边
    user_emotion_analysis_graph.add_edge(START, "analyze_decision")
    user_emotion_analysis_graph.add_edge("analyze_decision", "analyze_sentiment")
    user_emotion_analysis_graph.add_edge("analyze_sentiment", END)  # 直接结束

    # 编译并返回
    return user_emotion_analysis_graph.compile()