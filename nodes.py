import asyncio
import json
import logging
import os
import re
import threading
import traceback
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import requests
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import START, END, StateGraph
from typing_extensions import TypedDict

from AgentTools import generate_and_evaluate_node, self_verification_node

# 语音关键词配置 - 统一管理，减少重复
AUDIO_KEYWORDS_BASE = [
    "语音", "说出来", "读一下", "念一下", "播报", "播放语音", "配音",
    "唱歌", "唱首歌", "唱个歌", "voice", "tts", "audio"
]

AUDIO_KEYWORDS_SALES_EXTRA = [
    "说话", "真人", "AI", "好假", "你好喔",
    "宝贝", "想你了", "好的", "讲话", "语气", "嗯嗯", "下次吧",
    "呃", "额", "讲讲", "声音"
]
from Configurations import Configuration
from outside_info_aegnt import create_outside_info_workflow
from prompts.prompts_event import (
    get_event_action_mapping,
    get_whoareyou_prompt,
    get_event_decision_prompt_triggered,
    get_event_decision_prompt_untriggered,
)
from states import (
    AgentInput,
    AgentOutput,
    AgentState,
    AppointmentInfo,
    CustomerIntent,
    DebugInfo,
    EmotionalState,
    EventInstance,
    EventType,
)
from user_emotion_analysis_agent import user_emotion_analysis_workflow
from utils import describe_image_urls, parse_event_decision, transcribe_audio_urls, describe_video_urls, describe_webpage_urls
from utils import synthesize_tts_stepfun
from utils import transcribe_audio_urls_with_emotion
from utils import get_audio_duration_ms

logger = logging.getLogger(__name__)
BEIJING_TZ = timezone(timedelta(hours=8))
DISABLE_EVENT_SYSTEM = True  # 临时禁用事件系统开关（最小改动断开事件相关逻辑）
def state_memory_node(state: AgentState):#示例，如何传递获取传递的参数，可以给到提示词等等
    # 仅使用运行时配置
    from agents.persona_config.config_manager import config_manager
    cfg = config_manager.get_config() or {}
    agent_name = cfg.get("agent_name", "")


async def update_state_memory_node(state: AgentState, config=None):
    """
    更新状态记忆节点 - 核心状态管理函数
    
    该函数负责更新代理的状态和长期对话历史，处理各种场景下的消息同步：
    - 人工接管转AI托管的场景
    - 用户发送新消息的场景  
    - 主动聊天事件触发的场景
    - 多媒体内容（图片、音频、视频）的识别和处理
    
    主要功能包括：
    1. 注入assistant_id和assistant_config到状态中
    2. 检测并处理消息中的多媒体URL（图片、音频、视频）
    3. 异步处理多媒体内容（描述图片、转录音频、描述视频）
    4. 更新长期记忆和当前处理的消息
    5. 为所有消息添加时间戳信息
    
    Args:
        state (AgentState): 代理状态对象，包含当前对话状态和历史消息
        config (dict, optional): 配置信息，包含assistant_id等元数据
        
    Returns:
        AgentState: 更新后的状态对象，包含处理后的消息和多媒体内容
        
    Note:
        - 该函数会异步处理多媒体内容，提高性能
        - 支持多种消息格式（dict、HumanMessage、AIMessage等）
        - 自动处理时区转换，使用北京时间
        - 多媒体处理失败时会记录日志但不影响主流程
    """
    print(f"[DEBUG] === update_state_memory_node 开始执行 ===")
    print(f"[DEBUG] 输入消息数量: {len(state.get('messages', []))}")
    print(f"[DEBUG] 长期消息数量: {len(state.get('long_term_messages', []))}")
    
    # 1) 注入 assistant_id 与 assistant_config 到状态，供后续节点使用
    try:
        assistant_id = None
        if isinstance(config, dict):
            # 优先从 configurable 读取（调用方可显式传入）
            assistant_id = (
                config.get("configurable", {}) or {}
            ).get("assistant_id")
            # 其次从元数据读取（LangGraph Cloud 会在 metadata 放平台 assistant_id）
            if not assistant_id:
                assistant_id = (
                    config.get("metadata", {}) or {}
                ).get("assistant_id")
        if assistant_id:
            state["assistant_id"] = assistant_id
            try:
                from agents.persona_config.multi_assistant_config_manager import (
                    multi_assistant_config_manager,
                )
                assistant_cfg = (
                    multi_assistant_config_manager.get_assistant_config(assistant_id)
                    or {}
                )
                if assistant_cfg:
                    state["assistant_config"] = assistant_cfg
            except Exception:
                # 忽略个别环境下的导入/读取失败，保持回退逻辑
                pass
    except Exception:
        pass
    msgs = state.get("messages") or []
    long_term_messages = state.get("long_term_messages") or []

    # 调试：打印当前状态中的long_term_messages
    print(f"[DEBUG] 输入的long_term_messages数量: {len(long_term_messages)}")
    for i, msg in enumerate(long_term_messages):
        if isinstance(msg, dict):
            msg_type = msg.get("type", "unknown")
            raw_content = msg.get("content", "")
            # 确保content是字符串类型再进行切片
            if isinstance(raw_content, str):
                content = raw_content[:100]
            else:
                content = str(raw_content)[:100]
            additional_kwargs = msg.get("additional_kwargs", {})
        elif isinstance(msg, HumanMessage) or isinstance(msg, AIMessage):
            msg_type = "Human" if isinstance(msg, HumanMessage) else "AI"
            # 确保content是字符串类型再进行切片
            raw_content = msg.content
            if isinstance(raw_content, str):
                content = raw_content[:100]
            else:
                content = str(raw_content)[:100]
            additional_kwargs = getattr(msg, 'additional_kwargs', {})
        else:
            msg_type = "Unknown"
            content = str(msg)[:100]
            additional_kwargs = {}

        print(f"[DEBUG]  long_term_messages[{i}] ({msg_type}): {content}...")
        if additional_kwargs:
            print(f"[DEBUG]    additional_kwargs: {additional_kwargs}")

    image_url_pattern = re.compile(r'https?://\S+(?:\.(?:png|jpg|jpeg|gif|webp)|/wechat/image/[^?\s]*|/image/[^?\s]*)', re.IGNORECASE)
    audio_url_pattern = re.compile(r'https?://\S+\.mp3', re.IGNORECASE)
    generic_url_pattern = re.compile(r'https?://[^\s]+', re.IGNORECASE)
    
    # 视频格式模式 - 支持zhuanhuan.py中的所有格式
    video_formats = [
        'wmv', 'asf', 'asx', 'rm', 'rmvb', 'mp4', 'mpeg', 'mpg', '3gp', 
        'mov', 'm4v', 'avi', 'dat', 'mkv', 'flv', 'vob', 'ogv', 'webm', 
        'ts', 'mts', 'm2ts', 'divx', 'xvid', 'swf', 'f4v', 'f4p', 'f4a', 'f4b'
    ]
    # 修复正则表达式：匹配包含视频格式的完整URL，不要求格式在末尾
    video_pattern = re.compile(r'https?://[^\s]+\.(' + '|'.join(video_formats) + ')(?:\?[^\s]*)?', re.IGNORECASE)

    # 收集图片、语音、视频、网页URL及其对应位置
    image_entries = []  # (msg_idx, url)
    audio_entries = []  # (msg_idx, url)
    video_entries = []  # (msg_idx, url)
    webpage_entries = []  # (msg_idx, url)
    clean_texts = []    # 原消息的文字内容（去除URL）

    for i, msg in enumerate(msgs):
        # 获取消息内容和类型
        content = ""
        msg_type = "unknown"
        
        if isinstance(msg, dict):
            content = msg.get("content", "")
            raw_type = msg.get("type", "unknown").lower()
            # 支持多种角色类型映射
            if raw_type in ["human", "user"]:
                msg_type = "human"
            elif raw_type in ["ai", "assistant"]:
                msg_type = "ai"
            else:
                msg_type = "unknown"
        elif isinstance(msg, HumanMessage):
            content = msg.content
            msg_type = "human"
        elif isinstance(msg, AIMessage):
            content = msg.content
            msg_type = "ai"
        elif hasattr(msg, 'content'):
            content = msg.content
            msg_type = "unknown"
        else:
            content = str(msg)
            msg_type = "unknown"
        
        # 检查是否为有效消息且内容为字符串
        is_valid_message = (
            isinstance(content, str) and 
            msg_type in ["human", "ai"]
        )
        
        if not is_valid_message:
            clean_texts.append(content)
            continue

        # 检测图片、语音和视频URL（无论Human还是AI消息）
        images = image_url_pattern.findall(content)
        audios = audio_url_pattern.findall(content)
        # 对于视频，我们需要完整的URL，而不是只匹配的格式
        video_matches = video_pattern.finditer(content)
        videos = [match.group(0) for match in video_matches]
        # 通用网页链接
        generic_urls = generic_url_pattern.findall(content)
        # 过滤掉图片/音频/视频URL，保留纯网页URL（如公众号链接等）
        filtered_web_urls = []
        for u in generic_urls:
            if image_url_pattern.search(u) or audio_url_pattern.search(u) or video_pattern.search(u):
                continue
            filtered_web_urls.append(u)
        
        # 添加详细的调试信息

        for url in images:
            image_entries.append((i, url))
        for url in audios:
            audio_entries.append((i, url))
        for url in videos:
            video_entries.append((i, url))
        for url in filtered_web_urls:
            webpage_entries.append((i, url))

        # 移除URL，保留纯文本
        text_without_urls = image_url_pattern.sub('', content)
        text_without_urls = audio_url_pattern.sub('', text_without_urls)
        text_without_urls = video_pattern.sub('', text_without_urls)
        text_without_urls = generic_url_pattern.sub('', text_without_urls).strip()
        clean_texts.append(text_without_urls)
    
    # 统一异步处理
    image_urls = [e[1] for e in image_entries]
    audio_urls = [e[1] for e in audio_entries]
    video_urls = [e[1] for e in video_entries]
    webpage_urls = [e[1] for e in webpage_entries]
    

    # 异步处理多媒体内容
    if image_urls:
        image_descs = await describe_image_urls(image_urls)
        print(f"[DEBUG] 图片处理完成: {len(image_descs)} 个描述")
    else:
        image_descs = []
        
    if audio_urls:
        # 使用带情感的转写
        audio_results = await transcribe_audio_urls_with_emotion(audio_urls)
        # 兼容旧变量名
        audio_texts = [r.get("text", "") for r in audio_results]
        print(f"[DEBUG] 音频处理完成: {len(audio_texts)} 个转录")
    else:
        audio_results = []
        audio_texts = []

    # 构建与用户消息数量对应的语音识别文字数组
    custom_audio_text = []
    human_message_count = 0

    # 统计用户消息数量（type为human的消息）
    for msg in msgs:
        msg_type = "unknown"
        if isinstance(msg, dict):
            raw_type = msg.get("type", "unknown").lower()
            if raw_type in ["human", "user"]:
                msg_type = "human"
        elif isinstance(msg, HumanMessage):
            msg_type = "human"

        if msg_type == "human":
            human_message_count += 1

    print(f"[DEBUG] 用户消息数量: {human_message_count}")

    # 为每个用户消息构建对应的语音识别结果
    # audio_entries是(msg_idx, url)的列表，包含所有检测到的音频URL及其在消息中的绝对位置
    audio_map = {msg_idx: text for (msg_idx, _), text in zip(audio_entries, audio_texts)}

    print(f"[DEBUG] audio_map构建完成: {audio_map}")

    for i, msg in enumerate(msgs):
        msg_type = "unknown"
        if isinstance(msg, dict):
            raw_type = msg.get("type", "unknown").lower()
            if raw_type in ["human", "user"]:
                msg_type = "human"
        elif isinstance(msg, HumanMessage):
            msg_type = "human"

        if msg_type == "human":
            # 使用消息的绝对索引来查找语音识别结果
            if i in audio_map:
                audio_text = audio_map[i]
                # 如果语音识别成功且有内容，返回识别结果；否则返回空字符串
                if audio_text and audio_text.strip() and not audio_text.startswith("[SenseVoice子任务失败"):
                    custom_audio_text.append(audio_text.strip())
                    print(f"[DEBUG] 消息索引 {i} 语音识别成功: {audio_text.strip()}")
                else:
                    custom_audio_text.append("")
                    print(f"[DEBUG] 消息索引 {i} 语音识别失败或无内容，返回空字符串")
            else:
                # 非语音消息，返回空字符串
                custom_audio_text.append("")
                print(f"[DEBUG] 消息索引 {i} 非语音消息，返回空字符串")

    # 存储到状态中
    state["custom_audio_text"] = custom_audio_text
    print(f"[DEBUG] 语音识别文字数组已存储: {state['custom_audio_text']}")
        
    if video_urls:
        video_descs = await describe_video_urls(video_urls)
        print(f"[DEBUG] 视频处理完成: {len(video_descs)} 个描述")
    else:
        video_descs = []
    
    if webpage_urls:
        webpage_descs = await describe_webpage_urls(webpage_urls)
        print(f"[DEBUG] 网页处理完成: {len(webpage_descs)} 个摘要")
    else:
        webpage_descs = []

    # 将处理结果插回到原消息
    msg_map = {}  # msg_idx -> list of parts

    for idx, text in enumerate(clean_texts):
        msg_map[idx] = [text] if text else []

    for (msg_idx, _), desc in zip(image_entries, image_descs):
        msg_map[msg_idx].append(f"[该消息是图片，图片内容为]: {desc}")

    # 将情感拼入语音描述
    for idx, ((msg_idx, _), text) in enumerate(zip(audio_entries, audio_texts)):
        emotion = "未知"
        if idx < len(audio_results):
            emotion = audio_results[idx].get("emotion", "未知")
        if emotion and emotion != "未知":
            msg_map[msg_idx].append(f"[该消息是语音（情感：{emotion}），语音内容为]: {text}")
        else:
            msg_map[msg_idx].append(f"[该消息是语音，语音内容为]: {text}")
    
    # 记录哪些消息包含语音内容
    voice_message_indices = set(msg_idx for msg_idx, _ in audio_entries)

    for (msg_idx, _), desc in zip(video_entries, video_descs):
        msg_map[msg_idx].append(f"[该消息是视频，视频内容为]: {desc}")
    
    for (msg_idx, _), desc in zip(webpage_entries, webpage_descs):
        msg_map[msg_idx].append(f"[该消息是网页链接，网页主要内容为]: {desc}")

    # 生成最终处理后的新消息
    processed_messages = []
    for i, msg in enumerate(msgs):
        # 获取消息类型
        msg_type = "unknown"
        if isinstance(msg, dict):
            raw_type = msg.get("type", "unknown").lower()
            # 支持多种角色类型映射
            if raw_type in ["human", "user"]:
                msg_type = "human"
            elif raw_type in ["ai", "assistant"]:
                msg_type = "ai"
            else:
                msg_type = "unknown"
        elif isinstance(msg, HumanMessage):
            msg_type = "human"
        elif isinstance(msg, AIMessage):
            msg_type = "ai"
        
        # 提取时间戳信息
        timestamp = None
        if isinstance(msg, dict):
            # 从字典格式的消息中提取时间戳
            additional_kwargs = msg.get("additional_kwargs", {})
            timestamp = additional_kwargs.get("timestamp")
        elif hasattr(msg, 'additional_kwargs'):
            # 从消息对象中提取时间戳
            timestamp = getattr(msg, 'additional_kwargs', {}).get("timestamp")
        
        # 如果没有时间戳，使用当前时间
        if not timestamp:
            from datetime import datetime, timezone, timedelta
            timestamp = datetime.now(timezone(timedelta(hours=8))).isoformat()
        
        if i not in msg_map:
            # 处理没有URL的消息
            if isinstance(msg, dict):
                content = msg.get("content", "")
                # 从消息的additional_kwargs中获取原始send_style
                existing_kwargs = msg.get("additional_kwargs", {})
                original_send_style = existing_kwargs.get("send_style", "text")  # 提供默认值避免KeyError
                
                if msg_type == "human":
                    processed_messages.append(HumanMessage(
                        content=content, 
                        additional_kwargs={"timestamp": timestamp, "send_style": original_send_style}
                    ))
                elif msg_type == "ai":
                    processed_messages.append(AIMessage(
                        content=content, 
                        additional_kwargs={"timestamp": timestamp, "send_style": original_send_style}
                    ))
                else:
                    processed_messages.append(HumanMessage(
                        content=content, 
                        additional_kwargs={"timestamp": timestamp, "send_style": original_send_style}
                    ))
            else:
                # 对于已经是消息对象的情况，保留原有时间戳或添加新时间戳
                if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs.get("timestamp"):
                    processed_messages.append(msg)
                else:
                    # 创建新的消息对象，添加时间戳
                    # 从消息的additional_kwargs中获取原始send_style
                    existing_kwargs = getattr(msg, 'additional_kwargs', {}) or {}
                    original_send_style = existing_kwargs.get("send_style", "text")  # 提供默认值避免KeyError
                    
                    if isinstance(msg, HumanMessage):
                        processed_messages.append(HumanMessage(
                            content=msg.content,
                            id=getattr(msg, 'id', None),
                            additional_kwargs={"timestamp": timestamp, "send_style": original_send_style}
                        ))
                    elif isinstance(msg, AIMessage):
                        processed_messages.append(AIMessage(
                            content=msg.content,
                            id=getattr(msg, 'id', None),
                            additional_kwargs={"timestamp": timestamp, "send_style": original_send_style}
                        ))
                    else:
                        processed_messages.append(msg)
        else:
            # 处理有URL的消息
            full_text = "\n".join(msg_map[i])
            # 安全地获取消息ID
            msg_id = None
            if hasattr(msg, "id"):
                msg_id = msg.id
            elif isinstance(msg, dict):
                msg_id = msg.get("id")
            
            # 获取原始send_style
            if isinstance(msg, dict):
                existing_kwargs = msg.get("additional_kwargs", {})
            else:
                existing_kwargs = getattr(msg, 'additional_kwargs', {}) or {}
            original_send_style = existing_kwargs.get("send_style", "text")  # 提供默认值避免KeyError
            
            # 动态设置send_style：如果消息包含音频内容，则设置为"audio"
            if i in voice_message_indices:
                final_send_style = "audio"  # 如果包含语音内容，设置为audio
            else:
                final_send_style = original_send_style  # 否则使用原有值
            
            # 根据消息类型创建正确的消息对象，保留时间戳
            if msg_type == "human":
                processed_messages.append(HumanMessage(
                    content=full_text, 
                    id=msg_id,
                    additional_kwargs={"timestamp": timestamp, "send_style": final_send_style}
                ))
            elif msg_type == "ai":
                processed_messages.append(AIMessage(
                    content=full_text, 
                    id=msg_id,
                    additional_kwargs={"timestamp": timestamp, "send_style": final_send_style}
                ))
            else:
                processed_messages.append(HumanMessage(
                    content=full_text, 
                    id=msg_id,
                    additional_kwargs={"timestamp": timestamp, "send_style": final_send_style}
                ))

    # 更新历史
    print("[DEBUG] processed_messages内容是：",processed_messages)

    # 确保long_term_messages中的字典格式消息被正确转换为Message对象
    converted_long_term_messages = []
    structured_context_found = False

    for msg in long_term_messages:
        if isinstance(msg, dict):
            # 特殊处理结构化上下文数据
            msg_type = msg.get("type", "").lower()
            content = msg.get("content", "")

            # 支持的结构化字段（包含 report_update_time 修正）
            if msg_type in [
                "name", "sex", "age", "phone", "address", "birthday",
                "job", "doctor", "project", "is_deal", "is_deal_price",
                "not_deal", "not_deal_reason", "intent_project", "extra_info",
                "report_update_time"
            ]:
                structured_context_found = True

                field_name_map = {
                    "name": "姓名",
                    "sex": "性别",
                    "age": "年龄",
                    "phone": "电话",
                    "birthday": "生日",
                    "address": "住址",
                    "job": "职业",
                    "doctor": "面诊咨询师",
                    "project": "项目",
                    "is_deal": "已成交",
                    "is_deal_price": "项目价格",
                    "not_deal": "未成交",
                    "not_deal_reason": "未成交原因",
                    "intent_project": "感兴趣项目",
                    "extra_info": "补充说明信息",
                    "report_update_time": "上次到店面诊日期"
                }

                field_name = field_name_map.get(msg_type, msg_type)
                human_content = f"{field_name}：{content}"

                converted_msg = HumanMessage(
                    content=human_content,
                    additional_kwargs={
                        "context_update": True,
                        "update_type": "user_profile",
                        "field_type": msg_type,
                        "send_style": "text"
                    }
                )
                converted_long_term_messages.append(converted_msg)

            elif msg_type == "additional_kwargs" and isinstance(content, dict):
                # 这是一个上下文标记，不需要转换为消息
                # 标记信息已经包含在之前的消息中
                structured_context_found = True
                continue

            else:
                # 处理普通消息格式 + 兜底：未知字典也转成HumanMessage，避免后续 .type 访问报错
                additional_kwargs = msg.get("additional_kwargs", {})

                if msg_type in ["human", "user"]:
                    additional_kwargs["send_style"] = additional_kwargs.get("send_style", "text")
                    converted_msg = HumanMessage(
                        content=str(content),
                        additional_kwargs=additional_kwargs
                    )
                elif msg_type in ["ai", "assistant"]:
                    additional_kwargs["send_style"] = additional_kwargs.get("send_style", "text")
                    converted_msg = AIMessage(
                        content=str(content),
                        additional_kwargs=additional_kwargs
                    )
                else:
                    converted_msg = HumanMessage(
                        content=str(content),
                        additional_kwargs={
                            **({} if not isinstance(additional_kwargs, dict) else additional_kwargs),
                            "context_update": True,
                            "update_type": "user_profile",
                            "field_type": msg_type or "unknown",
                            "send_style": "text"
                        }
                    )
                converted_long_term_messages.append(converted_msg)
        else:
            converted_long_term_messages.append(msg)

    # 如果检测到结构化上下文，添加一个统一的上下文标记
    if structured_context_found:
        context_marker = HumanMessage(
            content="[系统提示] 以上是用户的个人资料信息，请在对话中适当使用这些信息。",
            additional_kwargs={
                "context_update": True,
                "update_type": "system_context",
                "context_marker": True
            }
        )
        converted_long_term_messages.append(context_marker)

    # 更新长期记忆，确保包含上下文消息
    state["long_term_messages"] = converted_long_term_messages + processed_messages
    print(f"[DEBUG] long_term_messages 更新后总数量: {len(state['long_term_messages'])}")

    # 调试输出更新后的long_term_messages
    print("[DEBUG] 更新后的long_term_messages内容:")
    for i, msg in enumerate(state["long_term_messages"]):
        if isinstance(msg, HumanMessage):
            msg_type = "Human"
            content = msg.content
        elif isinstance(msg, AIMessage):
            msg_type = "AI"
            content = msg.content
        else:
            msg_type = "Unknown"
            content = str(msg)
        print(f"  消息 {i} ({msg_type}): {content[:100]}...")

    state["processed_messages"] = processed_messages#更新新传输的消息为文本格式
    state["last_message"]=""#初始化ai生成的消息为空
    return state


async def multi_subgraph_parallel_node(state: AgentState, config=None):
    """
    多子图并行执行节点 - 核心业务逻辑处理函数
    
    该函数负责异步并行执行多个子图，实现高效的业务逻辑处理：
    - 外部信息查询子图：获取用户相关的业务信息
    - 用户情绪分析子图：分析用户的情绪状态和意图
    - 事件生成和调度子图：处理主动聊天事件和定时任务
    
    执行逻辑：
    1. 判断是否需要给用户发送消息（用户主动发消息 vs 主动事件触发）
    2. 根据判断结果选择性地执行相应的子图
    3. 使用asyncio.gather实现真正的异步并行执行
    4. 合并所有子图的输出结果到主状态中
    
    Args:
        state (AgentState): 代理状态对象，包含当前对话状态和事件信息
        config (dict, optional): 配置信息，用于子图执行
        
    Returns:
        AgentState: 更新后的状态对象，包含所有子图的处理结果
        
    Note:
        - 使用异步并行执行，显著提升性能
        - 智能判断执行场景，避免不必要的子图调用
        - 包含完整的错误处理和日志记录
        - 支持事件驱动和用户驱动的两种执行模式
    """
    print(f"\n🔄 === 异步并行子图执行开始 ===")
    # 创建子图实例
    outside_info_subgraph = create_outside_info_workflow()
    user_emotion_analysis_subgraph = user_emotion_analysis_workflow()
    # 事件相关子图在禁用时不创建
    if not DISABLE_EVENT_SYSTEM:
        event_generation_and_scheduling_subgraph = create_event_generation_and_scheduling_workflow()
    print(f"\n === 先判断 要/不要给用户发送消息 ===")
    state.update({"send_response_yes_or_no": False, "user_requires_message": False,
                  "sales_requires_message": False, "event_happens": False})
    
    # 初始化result变量
    result = []
    
    # 检查processed_messages是否为空或None
    processed_messages = state.get("processed_messages", [])
    if processed_messages and len(processed_messages) > 0:#是后端上传了消息
        # 检查新消息中是否包含人类消息
        if any(isinstance(msg, HumanMessage) for msg in processed_messages):
            print("[DEBUG] 后端上传了最新消息到记忆，其中包含人类消息，需要回复")
            state.update({"send_response_yes_or_no": True,"user_requires_message": True})

            # 检测图片请求
            await detect_and_select_image(state)

            try:
                print(f"🚀 开始异步并行执行子图...")
                tasks = [
                    outside_info_subgraph.ainvoke(state),
                    user_emotion_analysis_subgraph.ainvoke(state),
                ]
                if not DISABLE_EVENT_SYSTEM:
                    tasks.append(event_generation_and_scheduling_subgraph.ainvoke(state))
                result = await asyncio.gather(*tasks)
            except Exception as e:
                print(f"❌ 异步并行执行出错: {e}")
                import traceback
                print(f"🔍 错误详情:\n{traceback.format_exc()}")
                # 返回原始状态
                return state
        else:
            print("[DEBUG] 可能是人工接管状态转换成了ai托管状态，后端上传了最新消息到记忆，其中不包含人类回复，消息已同步到记忆中，无需回复")
            return state
    else:#后端没有上传任何消息
        # 禁用事件系统时，直接返回状态，不进行任何事件相关判断与生成
        if DISABLE_EVENT_SYSTEM:
            print("[DEBUG] 事件系统已禁用，未收到新消息时直接返回状态")
            return state
        try:
            event_instance = state.get("event_instance",None)
            if not event_instance:
                return state
            # 检查事件时间是否到达
            from datetime import datetime, timezone, timedelta
            current_time = datetime.now(timezone(timedelta(hours=8)))
            # 兼容 dict、对象、None
            if isinstance(event_instance, dict):
                event_time_str = event_instance.get("event_time")
                event_type = event_instance.get("event_type")
            else:
                event_time_str = getattr(event_instance, "event_time", None)
                event_type = getattr(event_instance, "event_type", None)
            if not event_time_str or not event_type:
                print(f"[DEBUG] 没有获取到事件类型或者事件触发时间，不产生主动回复")
                return state
            if not state.get("thread_id"):
                print(f"[DEBUG] 没有获取到线程号，不产生主动回复")
                return  state
            if not state.get("assistant_id"):
                print(f"[DEBUG] 没有获取到助手号，不产生主动回复")
                return state
            event_time = datetime.fromisoformat(event_time_str.replace('Z', '+00:00')).astimezone(
                timezone(timedelta(hours=8)))
            if current_time >= event_time:
                # 检查是否为有效的主动事件类型
                if event_type in [e.value for e in EventType]:
                    print(f"[DEBUG] 主动事件发生了，正在生成主动回复中")
                    state.update({"send_response_yes_or_no": True, "sales_requires_message": True, "event_happens": True})
                    try:
                        print(f"🚀 开始异步并行执行子图...")
                        result = await asyncio.gather(
                            # user_emotion_analysis_subgraph.ainvoke(state),
                            event_generation_and_scheduling_subgraph.ainvoke(state)
                        )
                    except Exception as e:
                        print(f"❌ 异步并行执行出错: {e}")
                        import traceback
                        print(f"🔍 错误详情:\n{traceback.format_exc()}")
                        # 返回原始状态
                        return state
                else:
                    print(f"[WARNING] 获得到了未知的事件类型,类型名字是: {event_type}")
                    return state
            else:
                print(f"[DEBUG] 未到事件发生时间")
                return state
        except:
            return state
    #用户需要ai回复 或者 销售需要发ai发送消息 均会过这里
    if state["event_happens"]:
        print("[DEBUG] 主动事件发生了！已执行具体分析")
    else:
        print("[DEBUG] 可选工具已调用，情绪状态分析已执行")
    print("[DEBUG] 更新子图输出到状态中")
    merged_state = {}
    for item in result:#result=[{子图1的output字典},{子图2的output字典},{子图3的output字典}]
        merged_state.update(item)
    state.update(merged_state)
    return state


async def send_or_response_node(state: AgentState, config=None):
    """
    发送或回复节点 - AI消息生成和发送的核心函数
    
    该函数负责根据不同的业务场景，生成AI回复消息并处理消息发送逻辑：
    - 用户需要AI回复：调用生成和评估节点，生成合适的回复内容
    - 销售需要发送消息：处理主动事件触发的消息发送
    - 消息验证：通过自验证节点确保回复质量
    
    主要功能：
    1. 判断是否需要生成和发送消息
    2. 调用AI生成和评估节点生成回复内容
    3. 通过自验证节点验证回复质量
    4. 为AI回复添加时间戳并更新长期记忆
    5. 处理不同类型的消息发送需求
    
    Args:
        state (AgentState): 代理状态对象，包含对话状态和业务需求标识
        config (dict, optional): 配置信息，用于消息生成
        
    Returns:
        AgentState: 更新后的状态对象，包含生成的AI回复消息
        
    Note:
        - 支持用户主动回复和销售主动发送两种模式
        - 包含完整的消息质量验证流程
        - 自动管理消息时间戳和长期记忆
        - 空消息会被过滤，不加入长期记忆
    """
    if not state["send_response_yes_or_no"]:
        # 正确处理邀请时间，支持整数和字符串格式
        invitation_time_value = state.get("invitation_time")
        invitation_time_ms = None
        if invitation_time_value is not None:
            try:
                if isinstance(invitation_time_value, int):
                    invitation_time_ms = invitation_time_value
                elif isinstance(invitation_time_value, str) and invitation_time_value:
                    dt = datetime.fromisoformat(invitation_time_value.replace('Z', '+00:00'))
                    invitation_time_ms = int(dt.timestamp() * 1000)
            except Exception:
                invitation_time_ms = None
        return {"last_message": "", "messages": [], "custom_status": {"invitation_status": state.get("invitation_status", 0), "invitation_time": invitation_time_ms, "invitation_project": state.get("invitation_project")}, "token_usage": {"current_used": 0, "total_used": state.get("token_total_used", 0)}, "custom_audio_text": state.get("custom_audio_text", []), "selected_image": state.get("selected_image")}
    elif state["user_requires_message"]:
        state_data=dict(state)

        # 在调用生成节点之前，先判断是否需要语音回复，并设置相关状态
        async def _should_audio_reply() -> bool:
            # 使用大模型智能判断是否需要语音回复（基于近3轮用户消息）
            try:
                from langchain_core.tools import tool
                from langchain_core.messages import HumanMessage
                import json
                from llm import create_llm
                
                # 获取最近3轮用户消息进行上下文分析
                msgs = state.get("processed_messages") or []
                human_texts = [m.content for m in msgs if isinstance(m, HumanMessage) and isinstance(m.content, str) and m.content.strip()]
                
                # 获取最近3轮用户消息，如果不足3轮则取全部
                recent_messages = human_texts[-3:] if len(human_texts) >= 3 else human_texts
                latest = recent_messages[-1] if recent_messages else ""
                
                if not latest:
                    raise Exception("没有找到用户消息，无法进行语音判断")
                
                # 构建多轮对话上下文
                context_messages = "\n".join([f"第{i+1}轮: {msg}" for i, msg in enumerate(recent_messages)])
                
                # 构建基于多轮对话上下文的判断提示词
                prompt = f"""
你是一个智能语音回复判断助手。请基于用户的多轮对话上下文，智能判断是否需要生成语音回复。

分析维度：
1. 直接语音请求：用户明确要求语音回复（如："语音"、"说出来"、"读一下"、"念一下"、"播报"、"播放语音"、"配音"等）
2. 音频内容需求：用户要求唱歌、朗诵、配音等音频形式内容
3. 语音技术询问：用户提到语音相关词汇（如："voice"、"tts"、"audio"、"声音"、"语气"等）
4. 交互体验评价：用户询问或评价AI的声音、语气、说话方式
5. 情感亲密表达：用户使用亲昵称呼或情感表达，暗示希望更亲密的语音交流
6. 对话连续性：结合前几轮对话，判断用户是否在延续语音相关的话题
7. 情境适配性：根据对话情境判断语音回复是否更合适（如讲故事、解释复杂概念等）

用户近期对话上下文：
{context_messages}

用户的最新输入："{latest}"

请综合分析多轮对话的上下文信息，判断是否需要语音回复。严格按照以下JSON格式输出，只输出JSON：
{{
  "need_audio_reply": true/false,
  "reason": "基于多轮对话分析的判断理由",
  "context_analysis": "对话上下文分析"
}}
"""
                
                # 创建LLM - 直接从环境变量获取配置
                import os
                model_provider = os.getenv("MODEL_PROVIDER", "openai")
                model_name = os.getenv("EVALUATION_MODEL", "gpt-4o-mini")
                llm = create_llm(
                    model_provider=model_provider,
                    model_name=model_name,
                    temperature=0.3
                )
                
                # 调用大模型判断
                message = HumanMessage(content=prompt)
                response = await llm.ainvoke(
                    [message],
                    response_format={"type": "json_object"}
                )
                
                response_text = response.content
                if response_text:
                    data = json.loads(response_text)
                    decision = data.get("need_audio_reply", False)
                    reason = data.get("reason", "")
                    context_analysis = data.get("context_analysis", "")
                    print(f"[TTS] 多轮对话分析 - 消息数量={len(recent_messages)}, 最新消息='{latest[:50]}', 决策={decision}")
                    print(f"[TTS] 判断理由: {reason}")
                    print(f"[TTS] 上下文分析: {context_analysis}")
                    return decision
                else:
                    raise Exception("大模型返回为空，无法进行语音判断")
                    
            except Exception as e:
                print(f"[TTS] 大模型判断失败: {e}")
                raise e

        # 设置语音回复状态，让生成模型知道是否要生成语音回复
        should_audio = await _should_audio_reply()
        state_data["audio_reply"] = should_audio
        print(f"[AUDIO_PROMPT] 设置audio_reply状态为: {should_audio}")

        result=await asyncio.to_thread(generate_and_evaluate_node.invoke, {
            "state_data": state_data
        })
        if result:
            state_data.update(result)
            state.update(result)
        result=await asyncio.to_thread(self_verification_node.invoke, {"state_data": state_data})
        if result:
            state.update(result)
        last_message = state.get("last_message", "")
        if last_message.strip():
            # 使用前面已经设置的audio_reply状态
            use_audio = should_audio
            # 每轮先清理上一轮的音频URL，避免残留到本轮
            try:
                state.pop("last_message_audio_url", None)
            except Exception:
                pass
            # 若需要语音，合成音频并仅记录URL，不再覆盖last_message文本
            if use_audio:
                try:
                    audio_url = await synthesize_tts_stepfun(state["last_message"])  # 使用默认voice/format
                    if audio_url:
                        state["last_message_audio_url"] = audio_url
                        print(f"[TTS] 合成成功，生成音频URL: {audio_url}")
                    else:
                        print(f"[TTS] 合成失败或未返回URL，保持文字回复")
                except Exception:
                    print(f"[TTS] 合成异常，保持文字回复")

            # 为AI回复添加时间戳（此时last_message可能已被音频URL替换）
            from datetime import datetime, timezone, timedelta
            current_timestamp = datetime.now(timezone(timedelta(hours=8))).isoformat()
            # 根据是否有音频URL来设置send_style
            send_style = "audio" if state.get("last_message_audio_url") else "text"
            ai_respond_message = AIMessage(
                content=state["last_message"],
                additional_kwargs={"timestamp": current_timestamp, "send_style": send_style}
            )
            state["long_term_messages"].append(ai_respond_message)

            # 组装输出payload
            async def _build_messages_payload(text_content: str, audio_url: Optional[str]) -> list:
                items = []
                # 文本
                if isinstance(text_content, str) and text_content.strip():
                    items.append({"type": "text", "content": text_content})
                    # 提取内含URL/图片/文件
                    url_pattern = re.compile(r"https?://[^\s]+", re.IGNORECASE)
                    image_pattern = re.compile(r"https?://\S+(?:\.(?:png|jpg|jpeg|gif|webp)|/wechat/image/[^?\s]*|/image/[^?\s]*)", re.IGNORECASE)
                    file_pattern = re.compile(r"(?:https?://\S+\.(?:pdf|docx?|xlsx?|pptx?)|/(?:[^\s]+\.(?:pdf|docx?|xlsx?|pptx?)))", re.IGNORECASE)
                    # 图片
                    for u in image_pattern.findall(text_content):
                        items.append({"type": "image", "content": u, "title": None})
                    # 文件
                    for u in file_pattern.findall(text_content):
                        items.append({"type": "file", "content": u})
                    # 纯URL（去掉已作为图片/文件的）
                    for u in url_pattern.findall(text_content):
                        if image_pattern.search(u) or file_pattern.search(u):
                            continue
                        items.append({"type": "url", "content": u})

                # 检查是否有选中的素材需要发送
                selected_image = state.get("selected_image")
                if selected_image and isinstance(selected_image, dict):
                    material_id = selected_image.get("id")
                    material_name = selected_image.get("name", "")
                    material_type = selected_image.get("materialType", 2)  # 使用materialType字段

                    if material_id:
                        print(f"[MATERIAL] 添加选中的素材到输出: {material_name} (ID: {material_id}, 类型: {material_type})")

                        # 统一的素材格式，包含materialType字段
                        items.append({
                            "type": "material",
                            "content": material_id,
                            "title": material_name,
                            "materialType": material_type
                        })

                        # 素材发送后立即清除状态，确保只发送一次
                        print(f"[MATERIAL] 清除selected_image状态，防止重复发送")
                        state["selected_image"] = None
                        state["image_request_detected"] = False

                # 音频
                if audio_url:
                    # 获取音频时长
                    duration_ms = await get_audio_duration_ms(audio_url)
                    items.append({"type": "audio", "content": audio_url, "duration": duration_ms})
                return items

            # 自定义状态区
            def _build_custom_status_payload(s: AgentState) -> dict:
                invitation_status = s.get("invitation_status", 0)
                invitation_time_value = s.get("invitation_time")
                invitation_project = s.get("invitation_project")
                # 转毫秒时间戳或None，支持字符串和整数两种格式
                invitation_time_ms = None
                if invitation_time_value is not None:
                    try:
                        if isinstance(invitation_time_value, int):
                            # 如果已经是整数毫秒时间戳，直接使用
                            invitation_time_ms = invitation_time_value
                        elif isinstance(invitation_time_value, str) and invitation_time_value:
                            # 如果是字符串，尝试解析ISO格式
                            dt = datetime.fromisoformat(invitation_time_value.replace('Z', '+00:00'))
                            invitation_time_ms = int(dt.timestamp() * 1000)
                        else:
                            # 其他情况保持None
                            invitation_time_ms = None
                    except Exception:
                        invitation_time_ms = None
                return {
                    "invitation_status": invitation_status,
                    "invitation_time": invitation_time_ms,
                    "invitation_project": invitation_project,
                }

            state["messages"] = await _build_messages_payload(state.get("last_message", ""), state.get("last_message_audio_url"))
            state["custom_status"] = _build_custom_status_payload(state)
            # 统计并输出 token 用量
            try:
                current_used = int(state.get("round_token_used") or 0)
            except Exception:
                current_used = 0
            try:
                total_prev = int(state.get("token_total_used") or 0)
            except Exception:
                total_prev = 0
            total_used = total_prev + current_used
            state["token_total_used"] = total_used
            state["token_usage"] = {"current_used": current_used, "total_used": total_used}
        return state

    elif state["sales_requires_message"]:
        if state.get("last_message", "").strip():#对于空字符串、只包含空格、只包含空白字符的消息不加入
            # 将AI生成的消息添加到长期记忆中，添加时间戳
            # 同样遵循语音替代文本逻辑
            def _should_audio_reply_sales() -> bool:
                if state.get("audio_reply") is not None:
                    return bool(state.get("audio_reply"))
                try:
                    msgs = state.get("processed_messages") or []
                    human_texts = [m.content for m in msgs if isinstance(m, HumanMessage)]
                    latest = human_texts[-1] if human_texts else ""
                    # 使用统一的语音关键词配置（销售场景 + 基础关键词）
                    audio_kw = AUDIO_KEYWORDS_BASE + AUDIO_KEYWORDS_SALES_EXTRA
                    decision = any(k in latest for k in audio_kw)
                    print(f"[TTS] (主动事件) latest_human='{str(latest)[:50]}', decision={decision}, audio_reply_flag={state.get('audio_reply')}")
                    return decision
                except Exception:
                    return False

            # 每轮先清理上一轮的音频URL，避免残留到本轮
            try:
                state.pop("last_message_audio_url", None)
            except Exception:
                pass
            if _should_audio_reply_sales():
                try:
                    audio_url = await synthesize_tts_stepfun(state["last_message"])  # 使用默认voice/format
                    if audio_url:
                        state["last_message_audio_url"] = audio_url
                        print(f"[TTS] (主动事件) 合成成功，生成音频URL: {audio_url}")
                    else:
                        print(f"[TTS] (主动事件) 合成失败或未返回URL，保持文字回复")
                except Exception:
                    print(f"[TTS] (主动事件) 合成异常，保持文字回复")

            from datetime import datetime, timezone, timedelta
            current_timestamp = datetime.now(timezone(timedelta(hours=8))).isoformat()
            # 根据是否有音频URL来设置send_style
            send_style = "audio" if state.get("last_message_audio_url") else "text"
            ai_send_message = AIMessage(
                content=state["last_message"],
                additional_kwargs={
                    "timestamp": current_timestamp,
                    "send_style": send_style
                }
            )
            state["long_term_messages"].append(ai_send_message)
            # 同步组装输出结构
            async def _build_messages_payload(text_content: str, audio_url: Optional[str]) -> list:
                items = []
                if isinstance(text_content, str) and text_content.strip():
                    items.append({"type": "text", "content": text_content})
                    url_pattern = re.compile(r"https?://[^\s]+", re.IGNORECASE)
                    image_pattern = re.compile(r"https?://\S+(?:\.(?:png|jpg|jpeg|gif|webp)|/wechat/image/[^?\s]*|/image/[^?\s]*)", re.IGNORECASE)
                    file_pattern = re.compile(r"(?:https?://\S+\.(?:pdf|docx?|xlsx?|pptx?)|/(?:[^\s]+\.(?:pdf|docx?|xlsx?|pptx?)))", re.IGNORECASE)
                    for u in image_pattern.findall(text_content):
                        items.append({"type": "image", "content": u, "title": None})
                    for u in file_pattern.findall(text_content):
                        items.append({"type": "file", "content": u})
                    for u in url_pattern.findall(text_content):
                        if image_pattern.search(u) or file_pattern.search(u):
                            continue
                        items.append({"type": "url", "content": u})

                # 检查是否有选中的素材需要发送
                selected_image = state.get("selected_image")
                if selected_image and isinstance(selected_image, dict):
                    material_id = selected_image.get("id")
                    material_name = selected_image.get("name", "")
                    material_type = selected_image.get("materialType", 2)  # 使用materialType字段

                    if material_id:
                        print(f"[MATERIAL] (销售消息) 添加选中的素材到输出: {material_name} (ID: {material_id}, 类型: {material_type})")

                        # 统一的素材格式，包含materialType字段
                        items.append({
                            "type": "material",
                            "content": material_id,
                            "title": material_name,
                            "materialType": material_type
                        })

                        # 素材发送后立即清除状态，确保只发送一次
                        print(f"[MATERIAL] (销售消息) 清除selected_image状态，防止重复发送")
                        state["selected_image"] = None
                        state["image_request_detected"] = False

                if audio_url:
                    # 获取音频时长
                    duration_ms = await get_audio_duration_ms(audio_url)
                    items.append({"type": "audio", "content": audio_url, "duration": duration_ms})
                return items

            def _build_custom_status_payload(s: AgentState) -> dict:
                invitation_status = s.get("invitation_status", 0)
                invitation_time_value = s.get("invitation_time")
                invitation_project = s.get("invitation_project")
                # 转毫秒时间戳或None，支持字符串和整数两种格式
                invitation_time_ms = None
                if invitation_time_value is not None:
                    try:
                        if isinstance(invitation_time_value, int):
                            # 如果已经是整数毫秒时间戳，直接使用
                            invitation_time_ms = invitation_time_value
                        elif isinstance(invitation_time_value, str) and invitation_time_value:
                            # 如果是字符串，尝试解析ISO格式
                            dt = datetime.fromisoformat(invitation_time_value.replace('Z', '+00:00'))
                            invitation_time_ms = int(dt.timestamp() * 1000)
                        else:
                            # 其他情况保持None
                            invitation_time_ms = None
                    except Exception:
                        invitation_time_ms = None
                return {
                    "invitation_status": invitation_status,
                    "invitation_time": invitation_time_ms,
                    "invitation_project": invitation_project,
                }

            state["messages"] = await _build_messages_payload(state.get("last_message", ""), state.get("last_message_audio_url"))
            state["custom_status"] = _build_custom_status_payload(state)
            # 统计并输出 token 用量（主动事件通道同样累计）
            try:
                current_used = int(state.get("round_token_used") or 0)
            except Exception:
                current_used = 0
            try:
                total_prev = int(state.get("token_total_used") or 0)
            except Exception:
                total_prev = 0
            total_used = total_prev + current_used
            state["token_total_used"] = total_used
            state["token_usage"] = {"current_used": current_used, "total_used": total_used}
            return state #因为在multi_subgraph_parallel_node中子图3的event_generation_and_scheduling_graph已经给出了回复
        else:
            # 正确处理邀请时间，支持整数和字符串格式
            invitation_time_value = state.get("invitation_time")
            invitation_time_ms = None
            if invitation_time_value is not None:
                try:
                    if isinstance(invitation_time_value, int):
                        invitation_time_ms = invitation_time_value
                    elif isinstance(invitation_time_value, str) and invitation_time_value:
                        dt = datetime.fromisoformat(invitation_time_value.replace('Z', '+00:00'))
                        invitation_time_ms = int(dt.timestamp() * 1000)
                except Exception:
                    invitation_time_ms = None
            return {"last_message": "", "messages": [], "custom_status": {"invitation_status": state.get("invitation_status", 0), "invitation_time": invitation_time_ms, "invitation_project": state.get("invitation_project")}, "token_usage": {"current_used": 0, "total_used": state.get("token_total_used", 0)}, "custom_audio_text": state.get("custom_audio_text", []), "selected_image": state.get("selected_image")}
    else:
        # 正确处理邀请时间，支持整数和字符串格式
        invitation_time_value = state.get("invitation_time")
        invitation_time_ms = None
        if invitation_time_value is not None:
            try:
                if isinstance(invitation_time_value, int):
                    invitation_time_ms = invitation_time_value
                elif isinstance(invitation_time_value, str) and invitation_time_value:
                    dt = datetime.fromisoformat(invitation_time_value.replace('Z', '+00:00'))
                    invitation_time_ms = int(dt.timestamp() * 1000)
            except Exception:
                invitation_time_ms = None
        return {"last_message": "", "messages": [], "custom_status": {"invitation_status": state.get("invitation_status", 0), "invitation_time": invitation_time_ms, "invitation_project": state.get("invitation_project")}, "token_usage": {"current_used": 0, "total_used": state.get("token_total_used", 0)}, "custom_audio_text": state.get("custom_audio_text", []), "selected_image": state.get("selected_image")}

class Output(TypedDict):
    """子图的输出状态 - 只包含最终回复"""
    last_message: Optional[str]
    event_instance: Optional[EventInstance] # 当前事件实例
    appointment_time: Optional[str]  # 预约时间
    user_last_reply_time: Optional[str]  # 用户最后回复时间
    last_active_send_time: Optional[str]  # 最后主动发送时间

class ContextUpdateRequest(TypedDict):
    """上下文更新请求结构"""
    thread_id: str
    context_messages: List[Dict[str, Any]]  # 要注入的上下文消息
    metadata: Optional[Dict[str, Any]]  # 额外的元数据
    update_type: str  # "background_info", "system_context", "user_profile"

# 导入LLM创建函数，但不在模块级别创建实例
from llm import create_llm

async def context_update_node(state: AgentState, context_request: ContextUpdateRequest = None):
    """
    上下文更新节点 - 专门处理向现有thread注入上下文信息

    该函数允许通过API调用向正在进行的对话线程注入新的上下文信息，
    这些信息会被无缝集成到对话历史中，影响后续的AI回复。

    主要功能：
    1. 验证上下文更新请求的合法性
    2. 将新的上下文消息转换为标准的消息格式
    3. 将上下文信息注入到长期记忆中
    4. 添加时间戳和元数据标记
    5. 确保上下文信息不影响正常的对话流程

    Args:
        state (AgentState): 当前代理状态
        context_request (ContextUpdateRequest): 上下文更新请求

    Returns:
        AgentState: 更新后的状态对象

    Note:
        - 上下文消息会被标记为系统消息，不影响用户体验
        - 支持多种类型的上下文更新：背景信息、系统上下文、用户画像
        - 自动添加时间戳，确保消息时序正确
        - 上下文更新不会触发AI回复，只更新状态
    """
    if not context_request:
        print("[DEBUG] 没有上下文更新请求，直接返回原状态")
        return state

    print("[DEBUG] === 开始处理上下文更新请求 ===")
    print(f"[DEBUG] 更新类型: {context_request.get('update_type', 'unknown')}")
    print(f"[DEBUG] 上下文消息数量: {len(context_request.get('context_messages', []))}")

    # 获取现有的长期记忆
    long_term_messages = state.get("long_term_messages", [])
    new_context_messages = []

    # 处理每个上下文消息
    for i, context_msg in enumerate(context_request.get("context_messages", [])):
        try:
            # 标准化消息格式
            msg_type = context_msg.get("type", "human").lower()
            content = context_msg.get("content", "")

            if not content.strip():
                continue

            # 根据更新类型添加不同的前缀
            update_type = context_request.get("update_type", "background_info")
            type_prefixes = {
                "background_info": "[背景信息] ",
                "system_context": "[系统上下文] ",
                "user_profile": "[用户画像] "
            }
            prefix = type_prefixes.get(update_type, "[上下文信息] ")
            enhanced_content = prefix + content

            # 创建消息对象
            timestamp = datetime.now(BEIJING_TZ).isoformat()

            if msg_type == "human":
                new_msg = HumanMessage(
                    content=enhanced_content,
                    additional_kwargs={
                        "timestamp": timestamp,
                        "context_update": True,
                        "update_type": update_type,
        
                    }
                )
            else:
                new_msg = AIMessage(
                    content=enhanced_content,
                    additional_kwargs={
                        "timestamp": timestamp,
                        "context_update": True,
                        "update_type": update_type,
                        "send_style": "text"
                    }
                )

            new_context_messages.append(new_msg)
            print(f"[DEBUG] 创建上下文消息 {i+1}: {enhanced_content[:50]}...")

        except Exception as e:
            print(f"[ERROR] 处理上下文消息 {i+1} 时出错: {e}")
            continue

    # 将新的上下文消息添加到长期记忆的开头（作为背景信息）
    if new_context_messages:
        # 在现有消息前插入上下文信息
        updated_long_term_messages = new_context_messages + long_term_messages
        state["long_term_messages"] = updated_long_term_messages

        print(f"[DEBUG] 上下文更新完成，共添加 {len(new_context_messages)} 条消息")
        print(f"[DEBUG] 更新后的长期记忆总数量: {len(updated_long_term_messages)}")

        # 添加更新标记到状态中
        state["context_updated"] = True
        state["last_context_update"] = datetime.now(BEIJING_TZ).isoformat()
        state["context_update_type"] = context_request.get("update_type")

        # 如果有额外的元数据，也添加到状态中
        if context_request.get("metadata"):
            state["context_metadata"] = context_request["metadata"]
    else:
        print("[DEBUG] 没有有效的上下文消息需要添加")

    return state

@tool
def inject_structured_context_to_thread(thread_id: str, user_profile: Dict[str, Any],
                                       base_url: str = None):
    """
    向指定thread注入结构化的用户信息

    这个工具函数提供了专门用于注入结构化用户信息的便捷方法，
    支持前端表单数据的直接对接。

    Args:
        thread_id (str): 目标thread的ID
        user_profile (Dict[str, Any]): 结构化的用户信息，格式如下：
            {
                "name": "张雨晴",
                "sex": "女",
                "age": "25",
                "phone": "13800138000",
                "address": "北京市海淀区"
            }
        base_url (str, optional): LangGraph Cloud的基础URL

    Returns:
        dict: 操作结果，包含成功状态和详细信息

    Example:
        # 注入结构化用户信息
        result = inject_structured_context_to_thread(
            thread_id="thread_123",
            user_profile={
                "name": "张雨晴",
                "sex": "女",
                "age": "25",
                "phone": "13800138000",
                "address": "北京市海淀区"
            }
        )
    """
    try:
        import requests
        import os

        # 获取基础URL
        if not base_url:
            base_url = os.getenv("LANGGRAPH_BASE_URL", "http://127.0.0.1:2024")

        # 构建结构化的上下文消息数组
        context_messages = []

        # 将用户信息转换为标准格式
        field_mappings = {
            "name": "姓名",
            "sex": "性别",
            "age": "年龄",
            "phone": "电话",
            "address": "地址"
        }

        for field, value in user_profile.items():
            if value and str(value).strip():
                context_messages.append({
                    "type": field,
                    "content": str(value).strip()
                })

        # 添加上下文标记
        context_messages.append({
            "type": "additional_kwargs",
            "content": {
                "context_update": True,
                "update_type": "user_profile"
            }
        })

        # 构建请求payload
        payload = {
            "values": {
                "long_term_messages": context_messages
            }
        }

        # 发送请求
        state_url = f"{base_url}/threads/{thread_id}/state"
        response = requests.post(state_url, json=payload, timeout=30)

        if response.status_code == 200:
            return {
                "success": True,
                "thread_id": thread_id,
                "profile_fields": list(user_profile.keys()),
                "total_fields": len([v for v in user_profile.values() if v and str(v).strip()]),
                "timestamp": requests.utils.formatdate(timeval=None, localtime=True)
            }
        else:
            return {
                "success": False,
                "error": f"API请求失败: {response.status_code} - {response.text}"
            }

    except Exception as e:
        return {
            "success": False,
            "error": f"注入结构化上下文时出错: {str(e)}"
        }

@tool
def inject_context_to_thread(thread_id: str, context_messages: List[Dict[str, Any]],
                           update_type: str = "background_info", metadata: Dict[str, Any] = None):
    """
    向指定thread注入上下文信息的工具函数

    这个工具函数提供了向现有对话线程注入上下文信息的便捷方法，
    可以用来添加背景信息、用户画像、系统上下文等各种类型的上下文数据。

    Args:
        thread_id (str): 目标thread的ID
        context_messages (List[Dict[str, Any]]): 要注入的上下文消息列表
            每个消息格式: {"type": "human", "content": "消息内容"}
        update_type (str): 更新类型，可选值：
            - "background_info": 背景信息
            - "system_context": 系统上下文
            - "user_profile": 用户画像
        metadata (Dict[str, Any], optional): 额外的元数据

    Returns:
        dict: 操作结果，包含成功状态和详细信息

    Example:
        # 注入用户背景信息
        result = inject_context_to_thread(
            thread_id="thread_123",
            context_messages=[
                {"type": "human", "content": "{{}}是一位大学生"}
            ],
            update_type="background_info"
        )
    """
    try:
        import requests
        import os
        from datetime import datetime

        # 构建上下文更新请求
        context_request = {
            "thread_id": thread_id,
            "context_messages": context_messages,
            "update_type": update_type,
            "metadata": metadata or {}
        }

        # 获取LangGraph Cloud的基础URL
        base_url = os.getenv("LANGGRAPH_BASE_URL", "https://your-langgraph-cloud-url")
        if base_url == "https://your-langgraph-cloud-url":
            return {
                "success": False,
                "error": "LANGGRAPH_BASE_URL环境变量未设置"
            }

        # 构建API URL
        api_url = f"{base_url}/threads/{thread_id}/runs"

        # 构建请求payload
        payload = {
            "configurable": {
                "context_request": context_request
            }
        }

        # 发送请求
        response = requests.post(api_url, json=payload, timeout=30)

        if response.status_code == 200:
            return {
                "success": True,
                "thread_id": thread_id,
                "update_type": update_type,
                "messages_count": len(context_messages),
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "error": f"API请求失败: {response.status_code} - {response.text}"
            }

    except Exception as e:
        return {
            "success": False,
            "error": f"注入上下文时出错: {str(e)}"
        }

@tool
def verify_context_injection(thread_id: str, base_url: str = None):
    """
    验证指定thread中的上下文信息是否正确注入

    这个工具函数用于检查thread状态，验证之前注入的上下文消息
    是否被正确保存和处理。

    Args:
        thread_id (str): 目标thread的ID
        base_url (str, optional): LangGraph Cloud的基础URL

    Returns:
        dict: 验证结果，包含状态信息和上下文消息详情
    """
    try:
        import requests
        import os

        # 获取基础URL
        if not base_url:
            base_url = os.getenv("LANGGRAPH_BASE_URL", "http://127.0.0.1:2024")

        # 构建API URL
        state_url = f"{base_url}/threads/{thread_id}/state"

        # 发送请求获取当前状态
        response = requests.get(state_url, timeout=30)

        if response.status_code != 200:
            return {
                "success": False,
                "error": f"获取thread状态失败: {response.status_code} - {response.text}"
            }

        state_data = response.json()
        values = state_data.get("values", {})

        # 分析long_term_messages
        long_term_messages = values.get("long_term_messages", [])
        context_messages = []
        regular_messages = []

        for i, msg in enumerate(long_term_messages):
            if isinstance(msg, dict):
                additional_kwargs = msg.get("additional_kwargs", {})
                content = msg.get("content", "")
                msg_type = msg.get("type", "unknown")
            else:
                additional_kwargs = getattr(msg, 'additional_kwargs', {})
                content = getattr(msg, 'content', str(msg))
                msg_type = "Human" if isinstance(msg, HumanMessage) else "AI"

            if additional_kwargs.get("context_update"):
                context_messages.append({
                    "index": i,
                    "type": msg_type,
                    "content": content,
                    "update_type": additional_kwargs.get("update_type"),
                    "context_update": True
                })
            else:
                regular_messages.append({
                    "index": i,
                    "type": msg_type,
                    "content": content
                })

        return {
            "success": True,
            "thread_id": thread_id,
            "total_messages": len(long_term_messages),
            "context_messages": context_messages,
            "regular_messages": regular_messages,
            "context_message_count": len(context_messages),
            "regular_message_count": len(regular_messages)
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"验证上下文注入时出错: {str(e)}"
        }

@tool
def event_triggered_node(state_dict: dict):
    """
    事件已触发节点 - 事件驱动聊天系统的核心工具函数
    
    该函数在事件已触发但用户没有回复的情况下，重新生成下一个事件实例。
    主要用于维护主动聊天的连续性，确保销售流程不会因为用户沉默而中断。
    
    核心功能：
    1. 分析当前事件状态和用户行为模式
    2. 根据硬性判断规则或LLM决策生成下一个事件
    3. 计算合适的事件触发时间
    4. 更新相关的时间字段（最后活跃时间、用户最后回复时间等）
    5. 生成新的事件实例用于后续调度
    
    业务逻辑：
    - 当事件触发后用户没有回复时，需要生成下一个跟进事件
    - 保持原有的用户最后回复时间不变（因为用户确实没有回复）
    - 更新最后活跃发送时间（记录AI主动发送的时间）
    - 根据业务规则决定下一个事件的类型和时机
    
    Args:
        state_dict (dict): 包含对话状态和事件信息的字典
        
    Returns:
        dict: 包含新事件实例和相关时间信息的字典
        
    Note:
        - 支持硬性判断规则和LLM决策两种模式
        - 自动处理时区转换，使用北京时间
        - 包含完整的错误处理和默认值设置
        - 事件时间会自动调整到合适的业务时间
    """
    try:
        # 获取状态信息，提供默认值
        long_term_messages = state_dict.get("long_term_messages", [])
        appointment_time = state_dict.get("appointment_time", "")
        event_instance = state_dict.get("event_instance")
        if event_instance:
            if isinstance(event_instance, dict):
                event_instance = EventInstance(**event_instance)
            event_type = event_instance.event_type
            event_time = event_instance.event_time
        else:
            event_type = EventType.OPENING_GREETING  # 默认开场问候
            event_time = datetime.now(BEIJING_TZ)
        user_last_reply_time = state_dict.get("user_last_reply_time")
        if user_last_reply_time is None:
            user_last_reply_time = datetime.now(BEIJING_TZ).isoformat()
        elif isinstance(user_last_reply_time, datetime):
            user_last_reply_time = user_last_reply_time.isoformat()

        last_active_send_time = state_dict.get("last_active_send_time")
        if last_active_send_time is None:
            last_active_send_time = datetime.now(BEIJING_TZ).isoformat()
        elif isinstance(last_active_send_time, datetime):
            last_active_send_time = last_active_send_time.isoformat()

        user_treatment_completion_info = state_dict.get("user_treatment_completion_info", "")

        # 获取配置（优先运行时 persona_config，其次上下文默认配置）
        try:
            from agents.persona_config.config_manager import config_manager
            runtime_config = config_manager.get_config() or {}
            if isinstance(state_dict, dict):
                assistant_cfg = state_dict.get("assistant_config") or {}
                if assistant_cfg:
                    runtime_config = {**runtime_config, **assistant_cfg}
            if runtime_config:
                config = Configuration(**runtime_config)
            else:
                config = Configuration.from_context()
        except Exception:
            config = Configuration.from_context()

        # 生成决策提示词（事件已触发）
        prompt = get_event_decision_prompt_triggered(
            last_event_type=event_type,
            last_event_time=event_time,
            user_last_reply_time=user_last_reply_time,
            last_active_send_time=last_active_send_time,
            visit_info=appointment_time,
            conversation_history=long_term_messages,
            user_treatment_completion_info=user_treatment_completion_info,
            config=config
        )

        # 检查是否是硬性判断返回的JSON（直接返回事件）
        if prompt.strip().startswith('{') and '"event_type"' in prompt:
            # 硬性判断已返回事件，直接解析
            event_decision = parse_event_decision(prompt)
        else:
            # 需要调用LLM进行决策（仅使用运行时配置）
            from agents.persona_config.config_manager import config_manager
            runtime_config = config_manager.get_config() or {}
            model_provider = runtime_config.get("model_provider", "openrouter")
            model_name = runtime_config.get("decision_model", runtime_config.get("model_name", "x-ai/grok-code-fast-1"))
            
            llm = create_llm(
                model_provider=model_provider,
                model_name=model_name,
                temperature=0.5
            )
            
            system_msg = SystemMessage(
                content="你是一个专业的事件决策AI助手，负责根据对话内容和用户状态决定应该生成什么类型的事件。")
            user_msg = HumanMessage(content=prompt)

            response = llm.invoke([system_msg, user_msg])
            decision_response = response.content

            # 解析决策结果
            event_decision = parse_event_decision(decision_response)

        print(f"[DEBUG] 事件已触发 - 决策结果: {event_decision}")

        # 创建事件实例
        event_type_str = event_decision.get("event_type", "pending_activation")
        try:
            event_type = EventType(event_type_str)
        except ValueError:
            event_type = EventType.PENDING_ACTIVATION

        # 解析时间
        event_time_str = event_decision.get("event_time")
        if event_time_str:
            try:
                event_time = datetime.fromisoformat(event_time_str.replace('Z', '+00:00')).astimezone(BEIJING_TZ)
            except:
                event_time = datetime.now(BEIJING_TZ)
        else:
            event_time = datetime.now(BEIJING_TZ)

        appointment_time_str = event_decision.get("appointment_time")
        appointment_time = None
        if appointment_time_str:
            try:
                appointment_time = datetime.fromisoformat(appointment_time_str.replace('Z', '+00:00')).astimezone(
                    BEIJING_TZ)
            except:
                pass

        # 创建事件实例
        event_instance = EventInstance(
            event_type=event_type,
            event_time=event_time.isoformat()
        )

        # 设置时间字段
        now = datetime.now(BEIJING_TZ).replace(second=0, microsecond=0)
        last_active_send_time = now.isoformat()  # 当前发送消息时间

        # event_triggered_node: 事件已触发，用户没有回复，保持原来的 user_last_reply_time
        user_last_reply_time = state_dict.get("user_last_reply_time", "")
        if user_last_reply_time and isinstance(user_last_reply_time, datetime):
            user_last_reply_time = user_last_reply_time.replace(second=0, microsecond=0).isoformat()

        print(f"[DEBUG] 生成事件实例: {event_instance}")

        return {
            "event_instance": event_instance,
            "appointment_time": appointment_time.isoformat() if appointment_time else None,
            "user_last_reply_time": user_last_reply_time,
            "last_active_send_time": last_active_send_time,
            "error_message": None
        }

    except Exception as e:
        print(f"[ERROR] 事件已触发节点出错: {e}")
        return {
            "event_instance": None,
            "appointment_time": None,
            "user_last_reply_time": None,
            "last_active_send_time": None,
            "error_message": str(e)
        }

@tool
def event_untriggered_node(state_dict: Dict):
    """
    事件未触发节点 - 用户主动回复事件处理工具函数
    
    该函数在用户主动回复但事件尚未触发的情况下，重新生成事件实例。
    主要用于响应用户的主动行为，调整事件调度策略，确保事件与用户行为保持同步。
    
    核心功能：
    1. 检测用户主动回复行为
    2. 根据用户回复内容调整事件策略
    3. 重新计算事件触发时间
    4. 更新用户最后回复时间和最后活跃时间
    5. 生成新的事件实例用于后续调度
    
    业务逻辑：
    - 当用户主动回复时，说明用户有参与意愿
    - 需要重新评估事件时机，避免与用户行为冲突
    - 用户最后回复时间更新为当前时间（记录用户活跃状态）
    - 根据用户回复内容调整下一个事件的类型和时机
    - 确保事件调度与用户行为模式保持一致
    
    Args:
        state_dict (Dict): 包含对话状态和事件信息的字典
        
    Returns:
        dict: 包含新事件实例和相关时间信息的字典
        
    Note:
        - 专门处理用户主动回复的场景
        - 自动更新用户活跃时间戳
        - 支持LLM智能决策事件类型和时机
        - 包含完整的错误处理和默认值设置
    """
    try:
        # 获取状态信息，提供默认值
        long_term_messages = state_dict.get("long_term_messages", [])
        appointment_time = state_dict.get("appointment_time", "")
        user_last_reply_time = state_dict.get("user_last_reply_time")
        if user_last_reply_time is None:
            user_last_reply_time = datetime.now(BEIJING_TZ).isoformat()
        elif isinstance(user_last_reply_time, datetime):
            user_last_reply_time = user_last_reply_time.isoformat()
        last_active_send_time = state_dict.get("last_active_send_time")
        if last_active_send_time is None:
            last_active_send_time = datetime.now(BEIJING_TZ).isoformat()
        elif isinstance(last_active_send_time, datetime):
            last_active_send_time = last_active_send_time.isoformat()
        user_treatment_completion_info = state_dict.get("user_treatment_completion_info", "")
        event_instance = state_dict.get("event_instance")
        if event_instance:
            if isinstance(event_instance, dict):
                event_instance = EventInstance(**event_instance)
            event_type = event_instance.event_type
            event_time = event_instance.event_time
        else:
            event_type = "pending_activation"
            event_time = datetime.now(BEIJING_TZ)
        # 获取配置（优先运行时 persona_config，其次上下文默认配置）
        try:
            from agents.persona_config.config_manager import config_manager
            runtime_config = config_manager.get_config() or {}
            if isinstance(state_dict, dict):
                assistant_cfg = state_dict.get("assistant_config") or {}
                if assistant_cfg:
                    runtime_config = {**runtime_config, **assistant_cfg}
            if runtime_config:
                config = Configuration(**runtime_config)
            else:
                config = Configuration.from_context()
        except Exception:
            config = Configuration.from_context()

        # 生成决策提示词（事件未触发）
        prompt = get_event_decision_prompt_untriggered(
            last_event_type=event_type,
            last_event_time=event_time,
            user_last_reply_time=user_last_reply_time,
            last_active_send_time=last_active_send_time,
            visit_info=appointment_time,
            conversation_history=long_term_messages,
            user_treatment_completion_info=user_treatment_completion_info,
            config=config
        )
        # 调用LLM进行决策（仅使用运行时配置）
        from agents.persona_config.config_manager import config_manager
        runtime_config = config_manager.get_config() or {}
        model_provider = runtime_config.get("model_provider", "openrouter")
        model_name = runtime_config.get("decision_model", runtime_config.get("model_name", "x-ai/grok-code-fast-1"))
        
        llm = create_llm(
            model_provider=model_provider,
            model_name=model_name,
            temperature=0.5
        )
        
        system_msg = SystemMessage(
            content="你是一个专业的事件决策AI助手，负责根据对话内容和用户状态决定应该生成什么类型的事件。")
        user_msg = HumanMessage(content=prompt)

        response = llm.invoke([system_msg, user_msg])
        decision_response = response.content
        # 解析决策结果
        event_decision = parse_event_decision(decision_response)
        print(f"[DEBUG] 事件未触发 - 决策结果: {event_decision}")

        # 创建事件实例
        event_type_str = event_decision.get("event_type", "pending_activation")
        try:
            event_type = EventType(event_type_str)
        except ValueError:
            event_type = EventType.PENDING_ACTIVATION

        # 解析时间
        event_time_str = event_decision.get("event_time")
        if event_time_str:
            try:
                event_time = datetime.fromisoformat(event_time_str.replace('Z', '+00:00')).astimezone(BEIJING_TZ)
            except:
                event_time = datetime.now(BEIJING_TZ)
        else:
            event_time = datetime.now(BEIJING_TZ)

        appointment_time_str = event_decision.get("appointment_time")
        appointment_time = None
        if appointment_time_str:
            try:
                appointment_time = datetime.fromisoformat(appointment_time_str.replace('Z', '+00:00')).astimezone(
                    BEIJING_TZ)
            except:
                pass
        # 创建事件实例
        event_instance = EventInstance(
            event_type=event_type,
            event_time=event_time.isoformat()
        )

        # 设置时间字段
        now = datetime.now(BEIJING_TZ).replace(second=0, microsecond=0)
        last_active_send_time = now.isoformat()  # 当前发送消息时间

        # event_untriggered_node: 事件未触发，用户主动回复，user_last_reply_time 设为当前时间
        user_last_reply_time = now.isoformat()

        print(f"[DEBUG] 生成事件实例: {event_instance}")

        return {
            "event_instance": event_instance,
            "appointment_time": appointment_time.isoformat() if appointment_time else None,
            "user_last_reply_time": user_last_reply_time,
            "last_active_send_time": last_active_send_time,
            "error_message": None
        }

    except Exception as e:
        print(f"[ERROR] 事件未触发节点出错: {e}")
        return {
            "event_instance": None,
            "appointment_time": None,
            "user_last_reply_time": None,
            "last_active_send_time": None,
            "error_message": str(e)
        }

@tool
def event_driven_chat_node(state_dict: dict):
    """
    事件驱动聊天节点 - AI主动回复生成工具函数
    
    该函数根据事件类型和对话历史，生成AI的主动回复内容。
    主要用于实现销售主动营销、跟进提醒、关怀问候等主动聊天场景，
    提升用户参与度和转化率。
    
    核心功能：
    1. 根据事件类型选择对应的提示词模板
    2. 分析对话历史，了解用户状态和偏好
    3. 动态插入业务信息（如治疗完成情况）
    4. 调用LLM生成自然、个性化的回复内容
    5. 异步发送通知到后端系统
    
    业务场景：
    - 开场问候（opening_greeting）：新用户的第一声问候
    - 客户回访（customer_followup）：定期关心用户状态和需求
    - 邀约提醒（appointment_reminder）：提醒用户即将到来的预约
    - 待唤醒（pending_activation）：激活沉默用户的参与意愿
    - 连接尝试（connection_attempt）：建立与用户的初步联系
    
    Args:
        state_dict (dict): 包含对话状态、事件信息和历史消息的字典
        
    Returns:
        dict: 包含生成的AI回复内容的字典
        
    Note:
        - 支持多种事件类型的个性化回复
        - 自动格式化对话历史，限制长度避免token超限
        - 包含完整的错误处理和超时机制
        - 异步发送后端通知，不阻塞主流程
        - 回复内容控制在20-80字之间，符合业务要求
    """
    try:
        long_term_messages = state_dict.get("long_term_messages", [])
        event_instance = state_dict.get("event_instance")
        if not event_instance:
            logger.warning("No event_instance found")
            return {"last_message": ""}

        # 获取事件类型
        if isinstance(event_instance, dict):
            event_type = event_instance.get("event_type")
        else:
            event_type = getattr(event_instance, "event_type", None)
        if not event_type:
            logger.warning(f"Invalid event_type: {event_type}")
            return {"last_message": ""}

        # 获取配置（优先运行时 persona_config，其次上下文默认配置）
        try:
            from agents.persona_config.config_manager import config_manager
            runtime_config = config_manager.get_config() or {}
            if isinstance(state_dict, dict):
                assistant_cfg = state_dict.get("assistant_config") or {}
                if assistant_cfg:
                    runtime_config = {**runtime_config, **assistant_cfg}
            if runtime_config:
                config = Configuration(**runtime_config)
            else:
                config = Configuration.from_context()
        except Exception:
            config = Configuration.from_context()

        # 获取配置化的事件提示词
        event_action_mapping = get_event_action_mapping(config)
        event_config = event_action_mapping.get(event_type)
        if not event_config:
            logger.warning(f"No event config found for type: {event_type}")
            return {"last_message": f""}

        # 动态插入业务信息
        try:
            event_prompt = event_config["prompt"].format(
                user_treatment_completion_info=state_dict.get("user_treatment_completion_info", "")
            )
        except Exception as e:
            logger.error(f"Error formatting event prompt: {e}")
            event_prompt = event_config["prompt"]

        # 格式化历史消息
        def _format_messages(long_term_messages):
            if not long_term_messages:
                return ""
            lines = []
            for msg in long_term_messages[-50:]:
                role = "用户" if getattr(msg, "type", "") == "human" else "AI"
                lines.append(f"{role}: {msg.content}")
            return "\n".join(lines)

        formatted_history = _format_messages(long_term_messages)
        
        # 获取配置化的whoareyou_prompt
        whoareyou_prompt = get_whoareyou_prompt(config)
        
        if event_type == "opening_greeting":
            ai_input = f"{event_prompt}"
        else:
            ai_input = f"{whoareyou_prompt}\n\n{formatted_history}\n\n{event_prompt}"

        # ===== 新增：主动事件的多媒体内容感知 =====
        multimedia_context = ""

        # 检查是否有选中的图片即将发送
        selected_image = state_dict.get("selected_image")
        if selected_image and isinstance(selected_image, dict):
            image_name = selected_image.get("name", "图片")
            multimedia_context += f"\n\n【系统提示】你将同时发送一张图片给用户，图片名称为：{image_name}。请在回复中自然地提及或配合这张图片，让回复内容与图片协调一致。"

        # 检查是否有语音回复的意图（主动事件也可能发送语音）
        audio_reply = state_dict.get("audio_reply")
        if audio_reply:
            multimedia_context += f"\n\n【系统提示】你将以语音形式回复用户。请确保回复内容适合语音播放，语气自然、口语化，避免过长的句子。"

        # 将多媒体上下文添加到AI输入
        if multimedia_context:
            ai_input += multimedia_context
            logger.info(f"添加主动事件多媒体上下文: {multimedia_context.strip()}")

        # 添加输出要求
        ai_input += """

输出要求：
1. 直接输出回复内容，不要加引号
2. 字数控制在20-80字之间
3. 使用自然的口语化表达
4. 不要使用markdown格式
5. 句号用换行符替代
"""

        # 调用AI模型（添加超时和错误处理）
        try:
            # 使用运行时配置
            from agents.persona_config.config_manager import config_manager
            runtime_config = config_manager.get_config() or {}
            model_provider = runtime_config.get("model_provider", "openrouter")
            model_name = runtime_config.get("generation_model", runtime_config.get("model_name", "x-ai/grok-code-fast-1"))
            
            llm = create_llm(
                model_provider=model_provider,
                model_name=model_name,
                temperature=1
            )
            
            user_msg = SystemMessage(content=ai_input)
            logger.info(f"Calling LLM for event type: {event_type}")
            response = llm.invoke([user_msg])
            logger.info("LLM call successful")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return {"last_message": ""}

        # 向后端发送通知（异步处理，不阻塞主流程）
        try:
            print(f"[DEBUG] 已生成主动回复：{response.content}")
            print(f"[DEBUG] 即将向后端发送通知")
            # 使用异步方式发送通知，避免阻塞
            import threading
            def send_notification_async():
                try:
                    send_notification_to_backend(state_dict, response.content)
                except Exception as e:
                    logger.error(f"Failed to send notification: {e}")

            thread = threading.Thread(target=send_notification_async)
            thread.daemon = True
            thread.start()
        except Exception as e:
            logger.error(f"Error starting notification thread: {e}")

        # 返回last_message
        return {
            "last_message": response.content
        }
    except Exception as e:
        logger.error(f"Error in event_driven_chat_node: {e}")
        return {"last_message": ""}


def send_notification_to_backend(state_dict, response_content):
    """
    向后端发送通知 - 事件通知系统
    
    该函数负责将AI生成的主动回复内容异步发送到后端系统，
    实现事件驱动的通知机制，确保后端能够及时响应和处理AI的主动行为。
    
    核心功能：
    1. 构建标准化的通知数据格式
    2. 生成唯一的请求ID用于追踪
    3. 转换时间格式为毫秒级时间戳
    4. 发送HTTP POST请求到配置的后端URL
    5. 处理网络异常和超时情况
    
    通知数据结构：
    - reqId: 唯一请求标识符
    - graphId: 图标识符（固定为"agent"）
    - assistantId: 助手ID
    - threadId: 线程ID
    - eventId: 事件类型标识
    - eventTime: 事件时间（毫秒级时间戳）
    - eventContent: 事件内容（JSON格式的AI回复）
    
    Args:
        state_dict (dict): 包含助手ID、线程ID和事件信息的字典
        response_content (str): AI生成的回复内容
        
    Returns:
        None: 该函数不返回值，仅负责发送通知
        
    Note:
        - 使用环境变量BACKEND_URL配置后端地址
        - 设置3秒超时，避免长时间阻塞
        - 包含完整的错误处理和日志记录
        - 支持多种网络异常情况的处理
        - 通知失败不会影响主业务流程
    """
    try:
        # 获取必要参数
        assistant_id = state_dict.get("assistant_id", None)
        thread_id = state_dict.get("thread_id", None)
        event_instance = state_dict.get("event_instance", {})

        # 检查后端URL是否配置
        backend_url = os.getenv("BACKEND_URL")
        if not backend_url:
            logger.warning("BACKEND_URL not configured, skipping notification")
            return

        # 生成UUID
        req_id = str(uuid.uuid4())

        # 获取事件时间并转换为13位时间戳
        event_time_str = event_instance.get("event_time")
        if event_time_str:
            try:
                # 解析时间字符串并转换为毫秒级时间戳
                dt = datetime.fromisoformat(event_time_str.replace('Z', '+00:00'))
                event_time_ms = int(dt.timestamp() * 1000)
            except Exception as e:
                logger.error(f"Error parsing event_time: {e}")
                event_time_ms = int(datetime.now().timestamp() * 1000)
        else:
            # 如果没有时间，使用当前时间
            event_time_ms = int(datetime.now().timestamp() * 1000)

        # 获取事件类型
        event_id = event_instance.get("event_type", "unknown")

        # 构建通知数据
        notification_data = {
            "reqId": req_id,
            "graphId": "agent",
            "assistantId": assistant_id,
            "threadId": thread_id,
            "eventId": event_id,
            "eventTime": event_time_ms,
            "eventContent": json.dumps({"active_chat_response": response_content}, ensure_ascii=False)
        }

        logger.info(f"发送通知到后端: {backend_url}")
        logger.debug(f"通知数据: {notification_data}")

        # 发送到后端，设置更短的超时时间
        response = requests.post(backend_url, json=notification_data, timeout=3)
        if response.status_code == 200:
            logger.info(f"✅ 通知发送成功: {req_id}")
        else:
            logger.warning(f"❌ 通知发送失败: {response.status_code} - {response.text}")

    except requests.exceptions.Timeout:
        logger.warning("通知发送超时")
    except requests.exceptions.ConnectionError:
        logger.warning("无法连接到后端服务")
    except Exception as e:
        logger.error(f"发送通知时出错: {e}")


def schedule_event_node(state: Any, config=None):
    """
    调度事件节点 - 事件调度和通知管理
    
    该函数负责根据当前状态调度下一个事件，并将事件信息发送到本地事件服务器。
    主要用于实现事件的本地调度和通知，确保事件能够在指定时间准确触发。
    
    核心功能：
    1. 验证事件实例的完整性和有效性
    2. 检查事件时间是否在未来（避免过期事件）
    3. 构建标准化的事件通知数据
    4. 发送事件通知到本地事件服务器
    5. 管理事件调度的生命周期
    
    调度逻辑：
    - 只有当事件时间在未来时才进行调度
    - 自动过滤过期的事件实例
    - 支持多种事件实例格式（对象、字典等）
    - 确保事件类型字段的正确序列化
    
    通知机制：
    - 通过环境变量ALIYUN_URL发送到阿里云服务器
    - 包含完整的事件信息和上下文数据
    - 设置10秒超时，确保通知的及时性
    - 支持同步通知，确保事件调度的可靠性
    
    Args:
        state (Any): 包含事件实例和调度相关信息的对象
        config (dict, optional): 配置信息，用于事件调度
        
    Returns:
        dict: 包含事件调度状态和相关信息的字典
        
    Note:
        - 只调度未来时间的事件，自动过滤过期事件
        - 支持多种事件实例格式的自动转换
        - 包含完整的错误处理和日志记录
        - 事件调度失败不会影响主业务流程
        - 通过环境变量配置服务器地址，支持灵活部署
    """
    try:
        # 从state中获取必要字段
        assistant_id = state.get("assistant_id", None)
        thread_id = state.get("thread_id", None)
        event_instance = state.get("event_instance")
        appointment_time = state.get("appointment_time", "")
        user_last_reply_time = state.get("user_last_reply_time", "")
        last_active_send_time = state.get("last_active_send_time", "")
        # 如果有事件实例且必要参数都存在，发送通知
        if event_instance and assistant_id and thread_id:
            # 检查事件时间是否在未来
            from datetime import datetime, timezone, timedelta
            try:
                if isinstance(event_instance, dict):
                    event_time_str = event_instance.get("event_time")
                else:
                    event_time_str = getattr(event_instance, "event_time", None)

                if event_time_str:
                    event_time = datetime.fromisoformat(event_time_str.replace('Z', '+00:00')).astimezone(BEIJING_TZ)
                    current_time = datetime.now(BEIJING_TZ)

                    # 只有当事件时间在未来时才调度
                    if event_time > current_time:
                        # 将 EventInstance 对象转换为字典格式
                        if hasattr(event_instance, 'dict'):
                            event_instance_dict = event_instance.dict()
                            # 确保 event_type 是字符串而不是枚举对象
                            if 'event_type' in event_instance_dict and hasattr(event_instance_dict['event_type'],
                                                                               'value'):
                                event_instance_dict['event_type'] = event_instance_dict['event_type'].value
                        elif hasattr(event_instance, '__dict__'):
                            event_instance_dict = event_instance.__dict__.copy()
                            # 确保 event_type 是字符串而不是枚举对象
                            if 'event_type' in event_instance_dict and hasattr(event_instance_dict['event_type'],
                                                                               'value'):
                                event_instance_dict['event_type'] = event_instance_dict['event_type'].value
                        else:
                            event_type = getattr(event_instance, "event_type", "")
                            if hasattr(event_type, 'value'):
                                event_type = event_type.value
                            event_instance_dict = {
                                "event_type": event_type,
                                "event_time": getattr(event_instance, "event_time", "")
                            }

                        # 发送事件通知到本地端口（同步版本）
                        import requests
                        try:
                            # 构建通知数据
                            notification_payload = {
                                "assistant_id": assistant_id,
                                "thread_id": thread_id,
                                "event_instance": event_instance_dict,
                                "appointment_time": appointment_time,
                                "user_last_reply_time": user_last_reply_time,
                                "last_active_send_time": last_active_send_time,
                            }

                            logger.info(f"[DEBUG] 发送本地通知: {notification_payload}")
                            # 发送事件通知到阿里云URL
                            aliyun_url = os.getenv("ALIYUN_URL")
                            local_response = requests.post(f"{aliyun_url}/event_notification",
                                                           json=notification_payload, timeout=10)
                            if local_response.status_code == 200:
                                logger.info(f"[DEBUG] 本地事件通知发送成功: {local_response.json()}")
                            else:
                                logger.warning(
                                    f"[DEBUG] 本地事件通知发送失败: status_code={local_response.status_code}")
                        except Exception as local_e:
                            logger.warning(f"[DEBUG] 本地事件通知发送异常: {local_e}")

                        # 发送了通知，返回 event_info=True 和事件参数
                        return {
                            "event_info": True,
                            "assistant_id": assistant_id,
                            "thread_id": thread_id,
                            "event_instance": event_instance,  # 保持原始的 event_instance
                            "appointment_time": appointment_time,
                            "user_last_reply_time": user_last_reply_time,
                            "last_active_send_time": last_active_send_time,
                        }
                    else:
                        logger.info(f"[DEBUG] 事件时间已过期，跳过调度: {event_time}")
                        return {"event_info": False}
                else:
                    logger.warning("[DEBUG] 事件实例缺少event_time字段")
                    return {"event_info": False}
            except Exception as e:
                logger.error(f"[DEBUG] 解析事件时间失败: {e}")
                return {"event_info": False}
        else:
            # 记录跳过通知的原因
            if not event_instance:
                logger.info("[DEBUG] 跳过通知：没有事件实例")
            elif not assistant_id:
                logger.info("[DEBUG] 跳过通知：assistant_id 为 None")
            elif not thread_id:
                logger.info("[DEBUG] 跳过通知：thread_id 为 None")

            return {"event_info": False}

    except Exception as e:
        logger.error(f"[ERROR] 调度事件节点出错: {e}")
        return {
            "event_info": False,
            "error_message": str(e)
        }

async def event_generation_and_scheduling_active_send_node(state: Any, config=None):
    """
    事件生成和调度主动发送节点 - 事件处理的核心协调函数
    
    该函数根据事件是否已触发，智能选择相应的事件处理策略：
    - 事件已触发：并行执行主动聊天和事件重新生成
    - 事件未触发：执行事件未触发的处理逻辑
    
    核心功能：
    1. 判断当前事件状态（已触发 vs 未触发）
    2. 根据状态选择合适的事件处理策略
    3. 异步并行执行多个事件处理工具
    4. 合并处理结果，返回完整的事件信息
    
    处理策略：
    - 事件已触发模式：
      * 并行调用event_driven_chat_node生成主动回复
      * 并行调用event_triggered_node重新生成下一个事件
      * 使用asyncio.gather实现真正的异步并行执行
    
    - 事件未触发模式：
      * 调用event_untriggered_node处理用户主动回复
      * 重新评估事件时机和类型
    
    Args:
        state (Any): 包含事件状态和上下文信息的对象
        config (dict, optional): 配置信息，用于事件处理
        
    Returns:
        dict: 包含事件处理结果的字典，包括新事件实例和AI回复
        
    Note:
        - 使用异步并行执行，提升事件处理效率
        - 智能路由，根据事件状态选择最优处理策略
        - 支持结果合并，确保返回数据的完整性
        - 包含完整的错误处理和异常恢复机制
    """
    state_data = dict(state)
    event_happens = state_data["event_happens"]
    
    if event_happens:
        # 异步调用事件触发时的事件生成工具函数和主动事件聊天工具函数
        result1, result2 = await asyncio.gather(
            asyncio.to_thread(event_driven_chat_node.invoke, {
                "state_dict": state_data
            }),
            asyncio.to_thread(event_triggered_node.invoke, {
                "state_dict": state_data
            })
        )
        # 合并结果
        return {**result1, **result2}
    else:
        # 方法2: 调用事件未触发时的事件生成工具函数
        result = await asyncio.to_thread(event_untriggered_node.invoke, {
            "state_dict": state_data
        })
        return result

def create_event_generation_and_scheduling_workflow():
    """
    创建事件生成和调度工作流 - 事件管理系统的图构建函数
    
    该函数负责构建一个完整的LangGraph工作流，用于处理事件的生成、调度和管理。
    工作流包含两个核心节点：事件生成和调度节点，实现事件的全生命周期管理。
    
    工作流结构：
    1. START → event_generation_and_scheduling_active_send
    2. event_generation_and_scheduling_active_send → schedule_event
    3. schedule_event → END
    
    节点功能：
    - event_generation_and_scheduling_active_send: 
      * 根据事件状态选择处理策略
      * 并行执行事件生成和主动聊天
      * 智能路由不同的事件处理逻辑
    
    - schedule_event:
      * 验证事件实例的有效性
      * 检查事件时间是否在未来
      * 发送事件通知到本地事件服务器
      * 管理事件调度的生命周期
    
    配置支持：
    - 使用Configuration作为配置模式
    - 支持运行时配置的动态注入
    - 输出格式为Output TypedDict
    
    Args:
        None: 该函数不需要参数
        
    Returns:
        CompiledStateGraph: 编译完成的事件生成和调度工作流图
        
    Note:
        - 工作流使用StateGraph构建，支持复杂的状态管理
        - 支持配置模式的动态注入和验证
        - 输出格式标准化，便于与其他系统集成
        - 工作流编译后可直接执行，支持异步调用
    """
    # 创建主图
    config_schema = Configuration
    event_generation_and_scheduling_graph = StateGraph(AgentState,config_schema=config_schema,output=Output)
    # 添加节点
    event_generation_and_scheduling_graph.add_node(
        "event_generation_and_scheduling_active_send", event_generation_and_scheduling_active_send_node
    )
    event_generation_and_scheduling_graph.add_node("schedule_event", schedule_event_node)
    # 添加边
    event_generation_and_scheduling_graph.add_edge(START, "event_generation_and_scheduling_active_send")
    event_generation_and_scheduling_graph.add_edge("event_generation_and_scheduling_active_send", "schedule_event")
    event_generation_and_scheduling_graph.add_edge("schedule_event", END)  # 直接结束

    # 编译并返回
    return event_generation_and_scheduling_graph.compile()

def create_context_update_workflow():
    """
    创建上下文更新工作流 - 专门处理向现有thread注入上下文信息

    该工作流提供了一个专门的接口，用于向正在进行的对话线程注入新的上下文信息，
    这些信息会被无缝集成到对话历史中，影响后续的AI回复。

    工作流结构：
    1. START → context_update_processor
    2. context_update_processor → END

    节点功能：
    - context_update_processor: 处理上下文更新请求

    配置支持：
    - 使用Configuration作为配置模式
    - 支持运行时配置的动态注入
    - 输入格式为ContextUpdateRequest
    - 输出格式为更新后的AgentState

    Args:
        None: 该函数不需要参数

    Returns:
        CompiledStateGraph: 编译完成的事件生成和调度工作流图

    Note:
        - 工作流专门用于上下文更新，不涉及对话生成
        - 支持多种类型的上下文更新：背景信息、系统上下文、用户画像
        - 工作流编译后可直接执行，支持异步调用
        - 通过标准LangGraph API调用，不需要额外的HTTP端点
    """
    # 创建主图
    config_schema = Configuration
    context_update_graph = StateGraph(AgentState, config_schema=config_schema, output=AgentState)

    # 添加节点
    context_update_graph.add_node(
        "context_update_processor",
        lambda state, config: context_update_node(state, config.get("context_request") if config else None)
    )

    # 添加边
    context_update_graph.add_edge(START, "context_update_processor")
    context_update_graph.add_edge("context_update_processor", END)

    # 编译并返回
    return context_update_graph.compile()

async def detect_and_select_image(state: AgentState):
    """
    检测多媒体请求并选择合适的素材（图片、视频、卡片链接）

    Args:
        state: 当前代理状态

    Returns:
        None: 直接修改state中的素材相关字段
    """
    try:
        print("[MATERIAL] ======================================")
        print("[MATERIAL] 开始执行素材检测和选择流程")
        print("[MATERIAL] ======================================")

        # 获取用户最新消息
        processed_messages = state.get("processed_messages", [])
        print(f"[MATERIAL] 处理过的消息数量: {len(processed_messages)}")

        if not processed_messages:
            print("[MATERIAL] ❌ 没有处理过的消息，跳过素材检测")
            return

        # 找到最新的用户消息
        user_message = ""
        for msg in reversed(processed_messages):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                print(f"[MATERIAL] 找到用户消息: {user_message}")
                break

        if not user_message:
            print("[MATERIAL] ❌ 未找到用户消息，跳过素材检测")
            return

        print(f"[MATERIAL] 🔍 准备检测用户消息: '{user_message}'")

        # 检测是否包含素材请求
        from utils import detect_image_request
        has_image_request = await detect_image_request(user_message)

        if not has_image_request:
            print("[MATERIAL] ❌ 未检测到素材请求")
            state["image_request_detected"] = False
            return

        print("[MATERIAL] ✅ 检测到素材请求，开始查询素材")
        state["image_request_detected"] = True

        # 获取thread_id和assistant_id
        thread_id = state.get("thread_id")
        assistant_id = state.get("assistant_id")

        print(f"[MATERIAL] thread_id: {thread_id}")
        print(f"[MATERIAL] assistant_id: {assistant_id}")

        if not thread_id:
            print("[MATERIAL] ❌ 缺少thread_id，跳过素材查询")
            return

        # 查询可用的素材（所有类型）
        print(f"[MATERIAL] 📡 开始查询所有类型素材...")
        from utils import query_material_images
        materials = await query_material_images(thread_id, assistant_id)

        if not materials:
            print("[MATERIAL] ❌ 未找到可用的素材")
            return

        print(f"[MATERIAL] ✅ 找到 {len(materials)} 个素材")
        material_type_names = {2: "图片", 3: "视频", 4: "卡片链接", 5: "卡片", 6: "语音", 7: "文件"}
        for i, material in enumerate(materials):
            type_name = material_type_names.get(material.get('materialType', 2), '未知类型')
            print(f"[MATERIAL]   {i+1}. [{type_name}] {material['name']} (ID: {material['id']}, 类型: {material.get('materialType', 2)})")

        # 使用AI智能选择合适的素材和类型
        print("[MATERIAL] 🤖 开始AI智能选择合适的素材和类型...")
        from utils import select_relevant_meterials
        long_term_messages = state.get("long_term_messages", [])
        selected_material = await select_relevant_meterials(materials, user_message, long_term_messages)

        if selected_material:
            print(f"[MATERIAL] ✅ 选择素材: {selected_material['name']}")
            print(f"[MATERIAL]   素材ID: {selected_material['id']}")
            print(f"[MATERIAL]   素材类型: {selected_material.get('materialType', 2)}")
            state["selected_image"] = selected_material
            state["material_selection_success"] = True
        else:
            print("[MATERIAL] ❌ 未找到合适的素材")
            # 生成未找到合适素材的回复提示
            state["material_selection_success"] = False
            state["material_selection_failure_reason"] = "no_suitable_material"
            # 设置一个标志，让AI知道需要生成相应的回复
            state["need_material_failure_response"] = True

        print("[MATERIAL] ======================================")
        print("[MATERIAL] 素材检测和选择流程完成")
        print("[MATERIAL] ======================================")

    except Exception as e:
        print(f"[MATERIAL] ❌ 素材检测异常: {e}")
        import traceback
        print(f"[MATERIAL] 异常详情: {traceback.format_exc()}")
