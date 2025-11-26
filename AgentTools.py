import json
from typing import List, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from states import AgentState, DebugInfo, EmotionalState
from langchain_core.tools import tool
from concurrent.futures import ThreadPoolExecutor, as_completed
from Configurations import Configuration
from dataclasses import asdict
from json_parser_utils import robust_json_parse, create_fallback_dict


def _extract_llm_usage(output_obj: Any) -> dict:
    """
    从 LangChain LLM 输出对象中尽可能提取 token 用量。
    优先读取 usage_metadata，其次读取 response_metadata.token_usage。
    返回字典包含 input、output、total 三个字段，均为 int。
    """
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    try:
        # LangChain >=0.2: usage_metadata 通常包含 input_tokens/output_tokens/total_tokens
        usage_meta = getattr(output_obj, "usage_metadata", None) or {}
        if isinstance(usage_meta, dict):
            input_tokens = int(usage_meta.get("input_tokens", 0) or 0)
            output_tokens = int(usage_meta.get("output_tokens", 0) or 0)
            total_tokens = int(usage_meta.get("total_tokens", input_tokens + output_tokens) or 0)
        # 兼容部分驱动把 token_usage 放在 response_metadata 里
        if (input_tokens + output_tokens) == 0:
            resp_meta = getattr(output_obj, "response_metadata", None) or {}
            token_usage = {}
            if isinstance(resp_meta, dict):
                token_usage = resp_meta.get("token_usage") or resp_meta.get("usage") or {}
            if isinstance(token_usage, dict):
                # 兼容不同字段命名
                input_tokens = int(token_usage.get("input_tokens", token_usage.get("prompt_tokens", 0)) or 0)
                output_tokens = int(token_usage.get("output_tokens", token_usage.get("completion_tokens", 0)) or 0)
                total_tokens = int(token_usage.get("total_tokens", input_tokens + output_tokens) or 0)
    except Exception:
        # 静默失败，返回0
        pass
    return {"input": input_tokens, "output": output_tokens, "total": total_tokens}

def _fallback_evaluation(action: str, response: str, current_stage: str, emotional_state,
                         customer_intent_level: str) -> float:
    """
    基于规则的兜底评估机制，当评估模型失败时使用
    """
    score = 0.5  # 默认中等分数

    # 检查回复是否过短或过长
    if len(response.strip()) < 3:
        return 0.2  # 过短回复
    if len(response) > 500:
        return 0.4  # 过长回复

    # 根据阶段调整基础分数
    stage_scores = {
        "initial_contact": {"greeting": 0.8, "rapport_building": 0.7},
        "ice_breaking": {"rapport_building": 0.8, "needs_analysis": 0.6},
        "subtle_expertise": {"value_display": 0.8, "needs_analysis": 0.7},
        "pain_point_mining": {"needs_analysis": 0.8, "pain_point_test": 0.7},
        "solution_visualization": {"value_pitch": 0.8, "value_display": 0.7},
        "natural_invitation": {"active_close": 0.8, "value_pitch": 0.6}
    }

    if current_stage in stage_scores and action in stage_scores[current_stage]:
        score = stage_scores[current_stage][action]

    # 根据情感状态调整
    trust_level = emotional_state.trust_level if emotional_state else 0.5
    comfort_level = emotional_state.comfort_level if emotional_state else 0.5

    # 信任度低时，优先关系建立
    if trust_level < 0.3:
        if action in ["rapport_building", "greeting"]:
            score += 0.1
        elif action in ["active_close", "value_pitch"]:
            score -= 0.2

    # 舒适度低时，避免压力过大的动作
    if comfort_level < 0.3:
        if action in ["stress_response", "rapport_building"]:
            score += 0.1
        elif action in ["active_close"]:
            score -= 0.1

    # 根据客户意向调整
    if customer_intent_level == "high" and action == "active_close":
        score += 0.1
    elif customer_intent_level == "low" and action == "active_close":
        score -= 0.2


    # 检查回复是否直接提供了信息而不是继续提问
    if action == "value_display":
        if any(keyword in response for keyword in ["项目", "方法", "价格", "效果", "可以"]):
            score += 0.15  # 奖励提供信息的回复
    elif action == "needs_analysis":
        if any(keyword in response for keyword in ["什么", "怎么", "哪种", "为什么"]):
            score -= 0.1  # 降低继续提问的回复分数

    return max(0.1, min(1.0, score))  # 确保在合理范围内

def _generate_and_evaluate_action(
        action: str,
        state_data: dict,  # 传递整个状态以获取更丰富的上下文
):
    """
    为单个动作生成并评估回复。这是一个辅助函数，用于并行执行。
    """

    # 从 state 中解构所需变量
    messages = state_data.get("long_term_messages", [])
    agent_temperature = state_data.get("agent_temperature", 0.5)
    debug_info = state_data.get("debug_info", DebugInfo())
    current_stage = debug_info.current_stage if debug_info and debug_info.current_stage else ""
    emotional_state = state_data.get("emotional_state", EmotionalState())
    customer_intent_level = state_data.get("customer_intent_level", "low")
    
    print(f"[DEBUG] [{action}] 解构变量完成:")
    print(f"[DEBUG] [{action}] - messages 数量: {len(messages)}")
    print(f"[DEBUG] [{action}] - agent_temperature: {agent_temperature}")
    print(f"[DEBUG] [{action}] - current_stage: {current_stage}")
    print(f"[DEBUG] [{action}] - customer_intent_level: {customer_intent_level}")

    # 检查最后一条用户消息
    user_messages = []
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_messages.insert(0, msg.content)  # 保持原有顺序
        elif isinstance(msg, AIMessage):
            break  # 遇到AI消息就停止
    last_user_message = "".join(user_messages) if user_messages else ""

    # 首先生成回复
    try:
        print(f"[DEBUG] [{action}] 创建 LLM 实例...")
        
        # 仅使用运行时配置（优先 assistant 级别，再回退全局）
        from agents.persona_config.config_manager import config_manager
        try:
            from agents.persona_config.multi_assistant_config_manager import (
                multi_assistant_config_manager,
            )
        except Exception:
            multi_assistant_config_manager = None

        assistant_id_in_state = state_data.get("assistant_id")
        config_dict = {}
        if multi_assistant_config_manager and assistant_id_in_state:
            try:
                config_dict = (
                    multi_assistant_config_manager.get_assistant_config(
                        assistant_id_in_state
                    )
                    or {}
                )
            except Exception:
                config_dict = {}
        if not config_dict:
            config_dict = config_manager.get_config() or {}
        # 字段兼容与别名回填，避免模板format时 KeyError
        try:
            # 纠正常见拼写
            if "industry_konwledge" in config_dict and "industry_knowledge" not in config_dict:
                config_dict["industry_knowledge"] = config_dict.get("industry_konwledge")
            # 别名同步
            if "industry" not in config_dict and "industry_knowledge" in config_dict:
                config_dict["industry"] = config_dict["industry_knowledge"]
            if "industry_knowledge" not in config_dict and "industry" in config_dict:
                config_dict["industry_knowledge"] = config_dict["industry"]
            # 补全默认助手模板里常用字段，避免 KeyError
            defaults = {
                "agent_nickname": config_dict.get("agent_name", "{{}}"),
                "agent_birthday": config_dict.get("agent_birthday", "1998-01-01"),
                "agent_goal": "邀约到店",
                "agent_side_goal": "收集客户反馈",
            }
            for k, v in defaults.items():
                config_dict.setdefault(k, v)
        except Exception:
            pass

        model_provider = config_dict.get("model_provider", "openrouter")
        model_name = config_dict.get(
            "model_name", config_dict.get("generation_model", "openai/gpt-5-chat")
        )
        agent_temperature = float(
            config_dict.get("agent_temperature", agent_temperature)
        )
        
        from llm import create_llm
        response_sampler = create_llm(
            model_provider=model_provider,
            model_name=model_name,
            temperature=agent_temperature
        )
        print(f"[DEBUG] [{action}] LLM 实例创建成功: {type(response_sampler)}")

        # 使用与之前相同的逻辑：加载 prompt 模板并格式化
        from prompts.loader import load_prompt

        # 加载对应的 prompt 模板
        print(f"[DEBUG] [{action}] 加载 prompt 模板...")
        # 使用配置中的base_context_prompt
        custom_base_context = config_dict.get("base_context_prompt") or state_data.get("base_context")
        prompt_template = load_prompt(action, custom_base_context=custom_base_context)
        print(f"[DEBUG] [{action}] prompt 模板加载成功，长度: {len(prompt_template)}")

        # 格式化对话历史
        # 格式化对话历史
        def _format_messages(messages: List[Any]) -> str:
            """将 LangChain BaseMessage 对象的列表格式化为单个字符串。"""
            if not messages:
                return "（无历史记录）"

            formatted_string = ""
            for message in messages:
                # 修复：使用 __class__.__name__ 来判断消息类型
                message_type = message.__class__.__name__
                if message_type == "HumanMessage":
                    role = "客户"
                elif message_type == "AIMessage":
                    role = "AI"
                elif message_type == "SystemMessage":
                    role = "系统"
                else:
                    role = "其他"
                
                # 获取消息内容
                content = message.content
                
                # 安全提取send_style字段，提供默认值
                send_style = message.additional_kwargs.get('send_style', 'text')
                content = f"[{send_style}] {content}"
                
                formatted_string += f"{role}: {content}\n"
            return formatted_string.strip()

        # 格式化 prompt，将对话历史插入模板
        
        # 调试输出：打印配置信息
        print(f"[DEBUG] [{action}] 配置信息:")
        print(f"[DEBUG] [{action}] - agent_name: {config_dict.get('agent_name', 'N/A')}")
        print(f"[DEBUG] [{action}] - company_name: {config_dict.get('company_name', 'N/A')}")
        print(f"[DEBUG] [{action}] - service_time: {config_dict.get('service_time', 'N/A')}")
        base_context = config_dict.get('base_context_prompt', '')
        print(f"[DEBUG] [{action}] - base_context_prompt: {base_context[:100]}..." if base_context else "[DEBUG] - base_context_prompt: None")
        
        # 生成时间相关的轻微提示（只在特定动作下）
        import datetime
        from zoneinfo import ZoneInfo
        current_time_info = datetime.datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
        now = datetime.datetime.now(ZoneInfo("Asia/Shanghai"))
        current_hour = now.hour
        
        # 只在特定动作下添加时间相关指令，并作为轻微的补充信息
        time_context = ""
        if action in ["greeting", "rapport_building"]:
            if 6 <= current_hour < 11:
                time_context = "\n\n偶尔提及即可，不要一直提及时间概念：现在是早上时间，可以稍微有活力一些。"
            elif 11 <= current_hour < 17:
                time_context = "\n\n偶尔提及即可，不要一直提及时间概念：现在是下午时间，可以适当关心对方。"
            elif 17 <= current_hour < 22:
                time_context = "\n\n偶尔提及即可，不要一直提及时间概念：现在是傍晚时间，语气可以稍微轻松一些。"
            else:  # 22:00 - 06:00
                time_context = "\n\n偶尔提及即可，不要一直提及时间概念：现在是深夜时间，注意关心对方休息。另外到晚上的时候，你会稍微诱人一些，你会稍微引诱一下客人的亲密关系，但是很挑逗，并且不经意，假装矜持的挑逗"
        
        # 调试输出：打印时间相关指令
        print(f"[DEBUG] [{action}] 时间补充提示: {time_context}")
        
        # 将时间提示整合到主prompt中，而不是作为独立消息
        base_prompt = prompt_template.format(
            message_history=_format_messages(messages),
            **config_dict
        )

        # 将时间提示作为主prompt的补充部分
        prompt = base_prompt + time_context

        # ===== 新增：多媒体内容感知 =====
        # 检查是否有即将发送的多媒体内容，告知AI以便生成协调的回复
        multimedia_context = ""

        # 检查是否有选中的素材即将发送
        selected_image = state_data.get("selected_image")
        if selected_image and isinstance(selected_image, dict):
            material_name = selected_image.get("name", "素材")
            material_type = selected_image.get("materialType", 2)  # 使用新的materialType字段

            # 根据素材类型生成不同的提示词
            material_type_names = {
                2: "图片", 3: "视频", 4: "卡片链接", 5: "卡片", 6: "语音", 7: "文件"
            }
            type_name = material_type_names.get(material_type, "素材")

            if material_type == 2:  # 图片
                multimedia_context += f"\n\n【系统提示】你将同时发送一张图片给用户，图片名称为：{material_name}。请在回复中自然地提及这张图片，比如可以说'我发了一张{material_name}给你看看'或'这是我们的{material_name}'，让回复内容与图片形成良好的配合。"
            elif material_type == 3:  # 视频
                multimedia_context += f"\n\n【系统提示】你将同时发送一个视频给用户，视频名称为：{material_name}。请在回复中自然地引导用户观看视频，比如可以说'我发了一个{material_name}的视频给你'或'你可以看看这个{material_name}的演示视频'，让回复内容与视频内容协调一致。"
            elif material_type == 4:  # 卡片链接
                multimedia_context += f"\n\n【系统提示】你将同时发送一个卡片链接给用户，卡片名称为：{material_name}。请在回复中自然地提及这个链接，比如可以说'我发了一个{material_name}的详细介绍给你'或'你可以点击查看{material_name}的详细信息'，引导用户点击链接。"
            elif material_type == 5:  # 卡片
                multimedia_context += f"\n\n【系统提示】你将同时发送一个卡片给用户，卡片名称为：{material_name}。请在回复中自然地配合这个卡片，比如可以说'我整理了一个{material_name}给你'或'这是{material_name}的相关信息'，让回复与卡片内容形成互补,但是不要提及“卡片”二字本身。"
            elif material_type == 6:  # 语音
                multimedia_context += f"\n\n【系统提示】你将同时发送一个语音文件给用户，语音名称为：{material_name}。请在回复中提及这个语音，比如可以说'我录制了一个{material_name}的语音给你'或'你可以听听这个{material_name}的语音介绍'。"
            elif material_type == 7:  # 文件
                multimedia_context += f"\n\n【系统提示】你将同时发送一个文件给用户，文件名称为：{material_name}。请在回复中提及这个文件，比如可以说'我发了一个{material_name}的文件给你'或'你可以下载查看{material_name}的详细资料'。"
            else:
                multimedia_context += f"\n\n【系统提示】你将同时发送一个{type_name}给用户，名称为：{material_name}。请在回复中自然地提及这个{type_name}，让回复内容与之协调一致。"

        # 检查是否有语音回复的意图
        audio_reply = state_data.get("audio_reply")
        if audio_reply:
            multimedia_context += f"\n\n【系统提示】你当前的场景是在微信上的语音回复。你将要以语音形式回复用户，而不是文字。请确保回复内容适合语音播放，语气自然、口语化。避免出现“我现在没法发送语音”“我等一下再给你发语音”这样的回复。"

        # 如果有语音识别的文字内容，也要考虑
        custom_audio_text = state_data.get("custom_audio_text", [])
        if custom_audio_text:
            # 过滤掉空字符串，合并所有有效的语音识别内容
            valid_audio_texts = [text for text in custom_audio_text if text and text.strip()]
            if valid_audio_texts:
                combined_audio_text = "\n".join(valid_audio_texts)
                multimedia_context += f"\n\n【系统提示】用户通过语音发送了消息，语音识别内容为：{combined_audio_text[:100]}... 请结合语音识别内容进行回复。"

        # 检查是否有素材选择失败的情况
        material_failure = state_data.get("need_material_failure_response")
        if material_failure:
            failure_reason = state_data.get("material_selection_failure_reason", "unknown")
            if failure_reason == "no_suitable_material":
                multimedia_context += "\n\n【系统提示】用户请求发送附件，但当前没有找到合适的素材。请在回复中礼貌地说明暂时无法提供相应的附件，并可以建议用户用其他方式描述需求，或者表示会尽快补充相关材料。"
            else:
                multimedia_context += "\n\n【系统提示】附件处理过程中出现异常。请在回复中安慰用户，并表示暂时还不能发送。正在处理这个问题。"

        # 将多媒体上下文添加到主prompt
        if multimedia_context:
            prompt += multimedia_context
            print(f"[MULTIMEDIA] [{action}] 添加多媒体上下文到prompt: {multimedia_context.strip()}")
        
        # 调试输出：打印最终生成的prompt
        print(f"[DEBUG] [{action}] base + time (前200字符):")
        print(f"[DEBUG] [{action}] {prompt[:200]}...")
        
        messages_for_sampler = [SystemMessage(content=prompt)]
        
        # 添加当前时间信息（作为单独的系统消息，但优先级较低）
        messages_for_sampler.append(SystemMessage(content="当前时间是:"+current_time_info))
        
        TOOL_NAME_MAP={"search": "联网搜索"}
        tool_results=state_data.get("tool_results", {})
        used_tools=state_data.get("used_tools", {})
        
        for tool_info, result_info in zip(used_tools, tool_results):
            tool_name = tool_info.get("tool")
            reason = tool_info.get("reason", "")
            result = result_info.get("result")

            if result:
                tool_label = TOOL_NAME_MAP.get(tool_name, tool_name)
                system_msg = (
                    f"客户可能关心：{reason}。\n"
                    f"调用工具【{tool_label}】得到的结果是：\n{result}\n\n"
                    f"根据工具调用的结果，用一句连续的话来回复客户：\n"
                    f"记住：无论之前调用了什么工具，你现在要继续用原有的口吻回答用户"
                    f"- 可以只回复很短的话，也可以稍长一点\n"
                    f"- 记住你的身份和工作，别跑偏了，你要让人感觉不到地将话题引导回你的业务范围\n"
                )
                messages_for_sampler.append(SystemMessage(content=system_msg))
        
        # 调试输出：打印所有发送给模型的消息
        print(f"[DEBUG] [{action}] 发送给模型的所有消息:")
        for i, msg in enumerate(messages_for_sampler):
            # 确保content是字符串类型再进行切片
            content = msg.content
            if isinstance(content, str):
                content_preview = content[:100]
            else:
                content_preview = str(content)[:100]
            print(f"[DEBUG] [{action}] 消息 {i+1}: {content_preview}...")
        
        # 确保有一条明确的人类指令，避免部分模型在仅有系统消息时返回空串
        messages_for_sampler.append(HumanMessage(content='请根据上述要求，严格仅输出一个 JSON 对象：{"response": "你的自然回应"}。不要输出其他说明。'))
        
        # 根据 action 提供对应的 fallback 回复 - 移到这里，这是万不得已的回复，只不过多样性一点而已
        fallback_responses = {
            "greeting": "您好，有什么可以帮您？",
            "rapport_building": "我们聊点别的吧！",
            "needs_analysis": "关于您的情况，能再多说一点吗？",
            "value_display": "针对您的情况，我们有很多专业的解决方案。",
            "stress_response": "抱歉，我们换个话题吧。",
            "pain_point_test": "我们聊聊您遇到的具体情况吧？",
            "value_pitch": "关于我们的方案，您最关心哪个方面？",
            "active_close": "我们直接进入下一步吧！",
            "reverse_probe": "可以多告诉我一些您的具体情况吗？"
        }
        fallback_response = fallback_responses.get(action, "嗯嗯，好的")

        # 调用模型生成回复
        try:
            # 明确要求返回 JSON，降低空响应概率
            response_result = response_sampler.invoke(
                messages_for_sampler,
                response_format={"type": "json_object"}
            )
            print(f"[DEBUG] [{action}] 生成模型调用成功，返回类型: {type(response_result)}")
            
            response_text = response_result.content if hasattr(response_result, 'content') else str(response_result)
            print(f"[DEBUG] [{action}] 提取的response_text类型: {type(response_text)}, 长度: {len(str(response_text))}")
            # 若仍为空，构造规范JSON以便后续解析兜底
            if not isinstance(response_text, str) or not response_text.strip():
                response_text = json.dumps({"response": fallback_response}, ensure_ascii=False)
            
        except Exception as e:
            print(f"[DEBUG] [{action}] 生成模型调用失败，错误类型: {type(e)}, 错误信息: {e}")
            print(f"[DEBUG] [{action}] 错误详情: {str(e)}")
            import traceback
            print(f"[DEBUG] [{action}] 错误堆栈: {traceback.format_exc()}")
            response_text = fallback_response

        # 统计生成阶段 token
        generation_usage = _extract_llm_usage(locals().get("response_result")) if 'response_result' in locals() else {"input": 0, "output": 0, "total": 0}

        # 安全解析 JSON 响应
        def _safe_json_parse(response_text: str, fallback_response: str) -> str:
            """安全地解析API返回的JSON响应，处理各种异常情况"""
            if response_text is None:
                return fallback_response
            
            print(f"[DEBUG-生成解析-{action}] 原始模型响应: {response_text}")
            
            # 使用鲁棒的JSON解析工具
            fallback_dict = {"response": fallback_response}
            parsed_data = robust_json_parse(
                response_text, 
                context=f"生成响应解析-{action}", 
                fallback_dict=fallback_dict,
                debug=True
            )
            
            result = parsed_data.get("response", fallback_response)
            print(f"[DEBUG-生成解析-{action}] 解析结果: {result}")
            return result

        response = _safe_json_parse(response_text, fallback_response)

        # 然后评估回复
        print(f"[DEBUG] [{action}] 创建评估模型实例...")
        # 评估模型也来自运行时配置（若缺失则回退到生成模型）
        verification_model = config_dict.get("verification_model") or config_dict.get("model_name", "x-ai/grok-code-fast-1")

        # 确保使用openrouter支持的模型，避免使用不存在的模型
        if verification_model.startswith("openai/gpt-5") or verification_model == "openai/gpt-5-chat":
            verification_model = "x-ai/grok-code-fast-1"
        elif verification_model.startswith("gpt-5") and "/" not in verification_model:
            verification_model = "x-ai/grok-code-fast-1"

        feedback_sampler = create_llm(
            model_provider=model_provider,
            model_name=verification_model,
            temperature=agent_temperature
        )
        print(f"[DEBUG] [{action}] 评估模型实例创建成功: {type(feedback_sampler)}")

        # 🎯 改进的评估prompt：重点关注需求满足

        feedback_prompt = f"""
你是对话质量评估专家。评估这个回复是否合适。

**关键原则：当客户表达明确需求时，优先满足需求而不是继续挖掘**

**用户最后说：** "{last_user_message}"

**候选回复 (策略: {action}):**
"{response}"

**评估要点：**
1. 如果用户说"我想美白"、"想了解XX"等明确需求，优先给分高的回复应该是：
- 直接提供相关信息/项目介绍 (高分)
- 而不是继续问"您想改善什么问题" (低分)

2. 如果用户已经选择了具体项目，优先给分高的回复应该是：
- 进入预约流程/提供案例 (高分)
- 而不是继续了解需求 (低分)
- 不贴合口语化、回复内容超出业务范围 (低分)

**评分标准:**
- 0.8-1.0: 回复直接满足了用户需求
- 0.6-0.7: 回复基本合适，略有偏离但可接受
- 0.4-0.5: 回复一般，没有很好满足需求
- 0.2-0.3: 回复偏离了用户意图
- 0.0-0.1: 回复完全不合适




JSON格式: {{"score": 数值, "reasoning": "简短理由"}}
"""
        try:
            raw_feedback = feedback_sampler.invoke(
                [HumanMessage(content=feedback_prompt)],
                response_format={"type": "json_object"}
            )
            print(f"[DEBUG] [{action}] 评估模型调用成功，返回类型: {type(raw_feedback)}")
            
        except Exception as e:
            print(f"[DEBUG] [{action}] 评估模型调用失败，错误类型: {type(e)}, 错误信息: {e}")
            print(f"[DEBUG] [{action}] 错误详情: {str(e)}")
            import traceback
            print(f"[DEBUG] [{action}] 错误堆栈: {traceback.format_exc()}")
            
            # 使用规则评估作为兜底
            score = _fallback_evaluation(action, response, current_stage, emotional_state, customer_intent_level or "low")
            reasoning = f"评估模型调用失败，使用规则评估: {e}"
            evaluated_response = {
                "action": action,
                "response": response,
                "score": score,
                "reasoning": reasoning
            }
            monologue_entry = f"  - [{action}] 生成回复: '{response[:30]}...' -> 评估得分: {score} (原因: {reasoning})"
            # 评估失败时，本阶段 token 记为0，仅返回生成阶段的用量
            round_usage = {
                "input_tokens": generation_usage.get("input", 0),
                "output_tokens": generation_usage.get("output", 0),
                "total_tokens": generation_usage.get("total", 0),
            }
            return evaluated_response, monologue_entry, round_usage

        # 评估阶段 token 统计
        evaluation_usage = _extract_llm_usage(locals().get("raw_feedback")) if 'raw_feedback' in locals() else {"input": 0, "output": 0, "total": 0}

        # 使用鲁棒的JSON解析工具
        raw_feedback_str = raw_feedback.content if hasattr(raw_feedback, 'content') else str(raw_feedback)
        
        print(f"[DEBUG-评估-{action}] 原始模型响应: {raw_feedback_str}")
        
        fallback_dict = {"score": 0.5, "reasoning": "解析失败，使用默认评分"}
        feedback_data = robust_json_parse(
            raw_feedback_str, 
            context=f"评估-{action}", 
            fallback_dict=fallback_dict,
            debug=True
        )
        
        print(f"[DEBUG-评估-{action}] 解析结果: {feedback_data}")

        score = float(feedback_data.get("score", 0.5))  # 默认给中等分
        reasoning = feedback_data.get("reasoning", "评估成功")

        # 确保分数在合理范围内
        score = max(0.0, min(1.0, score))

    except Exception as eval_error:
        # 🛡️ 强化兜底策略：基于规则的快速评估
        if 'response' not in locals():
            # 如果生成回复失败，使用默认回复
            response = ""
        score = _fallback_evaluation(action, response, current_stage, emotional_state, customer_intent_level or "low")
        reasoning = f"生成或评估失败，使用规则评估: {eval_error}"
    evaluated_response = {
        "action": action,
        "response": response,
        "score": score,
        "reasoning": reasoning
    }
    monologue_entry = f"  - [{action}] 生成回复: '{response[:30]}...' -> 评估得分: {score} (原因: {reasoning})"
    # 汇总当次（生成 + 评估）token 用量
    try:
        round_usage = {
            "input_tokens": int((generation_usage.get("input", 0)) + (locals().get("evaluation_usage", {}).get("input", 0))),
            "output_tokens": int((generation_usage.get("output", 0)) + (locals().get("evaluation_usage", {}).get("output", 0))),
            "total_tokens": int((generation_usage.get("total", 0)) + (locals().get("evaluation_usage", {}).get("total", 0))),
        }
    except Exception:
        round_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    return evaluated_response, monologue_entry, round_usage

@tool
def generate_and_evaluate_node(state_data: dict):
    """
    并行地为每个候选动作生成回复并获取反馈。
    """
    debug_info = state_data.get("debug_info", DebugInfo())
    internal_monologue = debug_info.internal_monologue if debug_info and debug_info.internal_monologue else []
    candidate_actions = state_data.get("candidate_actions", [])

    print(f"[DEBUG] candidate_actions: {candidate_actions}")
    print(f"[DEBUG] candidate_actions 数量: {len(candidate_actions)}")

    evaluated_responses = []
    new_monologue = list(internal_monologue)
    # 累计当轮所有候选动作调用的 token
    round_input_tokens = 0
    round_output_tokens = 0
    round_total_tokens = 0

    max_concurrent_requests = min(3, len(candidate_actions) or 1)

    # 使用 ThreadPoolExecutor 来处理异步函数
    print(f"[DEBUG] 创建 ThreadPoolExecutor...")
    with ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
        print(f"[DEBUG] 开始提交任务...")
        # 提交所有任务
        future_to_action = {}
        for action in candidate_actions:
            print(f"[DEBUG] 提交任务: {action}")
            future = executor.submit(_generate_and_evaluate_action, action, state_data)
            future_to_action[future] = action
        print(f"[DEBUG] 所有任务提交完成，共 {len(future_to_action)} 个任务")

        # 收集结果
        results = []
        for future in as_completed(future_to_action):
            try:
                print(f"[DEBUG] 等待任务完成...")
                result = future.result()
                print(f"[DEBUG] 任务完成，结果类型: {type(result)}")
                results.append(result)
            except Exception as e:
                action = future_to_action[future]
                print(f"[DEBUG] 任务 {action} 执行失败，错误类型: {type(e)}")
                print(f"[DEBUG] 错误信息: {e}")
                import traceback
                print(f"[DEBUG] 完整错误堆栈: {traceback.format_exc()}")
                results.append(e)



    for i, result in enumerate(results):
        action = candidate_actions[i]
        if isinstance(result, Exception):
            new_monologue.append(f"  - [{action}] 在并行执行中捕获到致命错误: {result}")
        else:
            # 兼容返回 2 元组或 3 元组
            if isinstance(result, tuple) and len(result) == 3:
                evaluated_response, monologue_entry, usage_info = result
                try:
                    round_input_tokens += int(usage_info.get("input_tokens", 0) or 0)
                    round_output_tokens += int(usage_info.get("output_tokens", 0) or 0)
                    round_total_tokens += int(usage_info.get("total_tokens", 0) or 0)
                except Exception:
                    pass
            else:
                evaluated_response, monologue_entry = result
            if evaluated_response:
                evaluated_responses.append(evaluated_response)
            if monologue_entry:
                new_monologue.append(monologue_entry)

    verbose = state_data.get("verbose", False)
    if verbose:
        print(f"[DEBUG] 生成评估节点: 评估了 {len(evaluated_responses)} 个候选回复")

    if not evaluated_responses:
        new_monologue.append("所有模块都执行失败了，使用紧急兜底回复")
        # 🛡️ 多级兜底机制
        try:
            # 尝试人工转接 - 此处为紧急情况，直接返回固定回复
            final_response = "抱歉，我现在有点忙\n 晚点再联系您"
        except Exception as e:
            new_monologue.append(f"生成紧急回复时也失败了: {e}")
            # 最终兜底：固定回复
            final_response = "稍等哈 有点忙"
        debug_info = DebugInfo(
            current_stage=state_data.get("current_stage"),
            emotional_state=state_data.get("emotional_state").model_dump() if state_data.get(
                "emotional_state") else None,
            internal_monologue=new_monologue,
        )

        return {
            "final_response": final_response,
            "last_message": final_response,
            "debug_info": debug_info,
            # 即使兜底，也返回累计的当轮 token
            "round_token_used": int(round_total_tokens)
        }
    debug_info = DebugInfo(
        current_stage=state_data.get("current_stage"),
        emotional_state=state_data.get("emotional_state").model_dump() if state_data.get(
            "emotional_state") else None,
        internal_monologue=new_monologue,
    )

    return {
        "evaluated_responses": evaluated_responses,
        "debug_info": debug_info,
        # 对话中本轮所有候选动作（生成+评估）的 token 总和
        "round_token_used": int(round_total_tokens)
    }

@tool
def self_verification_node(state_data: dict):
    """
    从评估过的候选项中选择最佳响应。
    """
    evaluated_responses = state_data.get("evaluated_responses", [])
    debug_info = state_data.get("debug_info", DebugInfo())
    internal_monologue = debug_info.internal_monologue if debug_info and debug_info.internal_monologue else []

    # 不再需要从这里获取采样器，因为评分已在 generate_and_evaluate_node 完成
    # sampler = ...

    # 🔧 优化选择逻辑：降低质量门槛，确保总是有回复

    # 先尝试0.3以上的回复
    high_quality_responses = [r for r in evaluated_responses if r.get('score', 0.0) > 0.3]

    # 如果没有0.3以上的，尝试0.2以上的
    if not high_quality_responses:
        high_quality_responses = [r for r in evaluated_responses if r.get('score', 0.0) > 0.2]

    # 如果还是没有，选择所有回复中得分最高的
    if not high_quality_responses and evaluated_responses:
        high_quality_responses = sorted(evaluated_responses, key=lambda x: x.get('score', 0.0), reverse=True)

    # 极端情况：没有任何回复
    if not high_quality_responses:
        new_monologue = internal_monologue + ["自我验证失败：没有可供选择的候选回复，使用紧急回复"]
        fallback_response = "嗯嗯，好的"  # 简单自然的兜底回复
        debug_info = DebugInfo(
            current_stage=state_data.get("current_stage"),
            emotional_state=state_data.get("emotional_state").model_dump() if state_data.get(
                "emotional_state") else None,
            internal_monologue=new_monologue,
        )
        return {
            "last_message": fallback_response,
            "debug_info": debug_info
        }

    if len(high_quality_responses) == 1:
        final_response = high_quality_responses[0]['response']
        new_monologue = internal_monologue + [
            f"自我验证：只有1个高质量选项，直接选择 '{high_quality_responses[0]['action']}'。"]
    else:
        # 直接按得分排序选择最高分的回复
        best_response = sorted(high_quality_responses, key=lambda x: x['score'], reverse=True)[0]
        final_response = best_response['response']
        new_monologue = internal_monologue + [
            f"自我验证：从 {len(high_quality_responses)} 个选项中选择得分最高的回复 (模块: {best_response['action']}, 得分: {best_response['score']:.2f})。"
        ]

    # --- 关键改动：将AI的最终回复更新回消息历史中 ---
    # With `add_messages`, we just need to return the new message(s) in a list.
    # LangGraph will handle appending it to the state.
    new_monologue.append("将AI的最新回复返回，由LangGraph自动更新历史。")



    verbose = state_data.get("verbose", False)


    if verbose:
        print(f"[DEBUG] 最终回复: {final_response}")

    # --- 新增：如果 verbose 模式开启，则准备调试信息 ---
    debug_info = DebugInfo(
            current_stage=state_data.get("current_stage"),
            emotional_state=state_data.get("emotional_state").model_dump() if state_data.get("emotional_state") else None,
            internal_monologue=new_monologue,
        )
    return {
        "last_message": final_response,  # 新增：用于API输出
        "debug_info": debug_info,  # 将调试信息添加到返回字典中
    }
