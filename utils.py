import asyncio
import json
import os
from openai import AsyncOpenAI
from Configurations import Configuration
import aiohttp
import io
import re
from typing import List, Any, Optional
from datetime import datetime, timezone, timedelta
import random
import uuid
import hashlib
BEIJING_TZ = timezone(timedelta(hours=8))
from agents.persona_config.config_manager import config_manager
_cfg = config_manager.get_config() or {}

# 延迟初始化client，避免在模块导入时阻塞
_client = None

# 延迟加载 dashscope，避免在未安装或未配置时影响其他功能
_dashscope_loaded = False
def _ensure_dashscope_loaded() -> bool:
    global _dashscope_loaded
    if _dashscope_loaded:
        return True
    try:
        import dashscope  # noqa: F401
        _dashscope_loaded = True
        return True
    except Exception:
        return False

def _use_openrouter() -> bool:
    # 全面转向 OpenRouter：始终返回 True
    return True

def _normalize_model_name_for_openrouter(model_name: str) -> str:
    if not model_name:
        return model_name
    # 在 OpenRouter 上使用 OpenAI 家族模型时加前缀 openai/
    if model_name.startswith("gpt-") and "/" not in model_name:
        return f"openai/{model_name}"
    return model_name

# =====================
# StepFun TTS 集成
# 文档参考：`https://platform.stepfun.com/docs/guide/tts`、`https://platform.stepfun.com/docs/api-reference/audio/create_audio`
# =====================

async def synthesize_tts_stepfun(text: str, voice: str = None, audio_format: str = "mp3", speed: float = 1.0, pitch: float = 0.0) -> Optional[str]:
    """调用智能阶跃 StepFun TTS 生成语音，返回公网可访问的音频URL。

    参数按仓库运行时配置与函数入参合并；失败返回 None。
    """
    if not isinstance(text, str) or not text.strip():
        return None
    import os
    api_key = os.getenv("STEPFUN_API_KEY")
    if not api_key:
        print("[TTS] 未检测到 STEPFUN_API_KEY，跳过合成")
        return None
    # 运行时配置
    try:
        cfg = Configuration.from_context()
    except Exception:
        cfg = None
    voice = voice or (cfg.tts_voice if cfg else _cfg.get("tts_voice", "huolinvsheng"))
    audio_format = audio_format or (cfg.tts_format if cfg else _cfg.get("tts_format", "mp3"))
    speed = float(speed or (cfg.tts_speed if cfg else _cfg.get("tts_speed", 1.0)))
    pitch = float(pitch or (cfg.tts_pitch if cfg else _cfg.get("tts_pitch", 0.0)))

    # 端点/模型可由环境或运行时配置覆盖
    endpoint = os.getenv("STEPFUN_TTS_ENDPOINT") or "https://api.stepfun.com/v1/audio/speech"
    try:
        cfg_model = (cfg.tts_model if cfg else _cfg.get("tts_model", "step-tts-vivid"))
    except Exception:
        cfg_model = _cfg.get("tts_model", "step-tts-vivid")
    model = os.getenv("STEPFUN_TTS_MODEL") or cfg_model
    url = endpoint
    print(f"[TTS] 调用 StepFun: endpoint={url}, model={model}, voice={voice}, format={audio_format}, speed={speed}, pitch={pitch}")
    payload = {
        "input": text,
        "model": model,
        "voice": voice,
        "format": audio_format,
        "speed": speed,
        "pitch": pitch,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json, audio/mpeg, audio/mp3"
    }
    try:
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                ctype = resp.headers.get("Content-Type", "")
                print(f"[TTS] HTTP {resp.status}, content-type={ctype}")
                if resp.status != 200:
                    # 某些实现会直接返回音频流（content-type: audio/*），此时按二进制处理
                    try:
                        if ctype.startswith("audio/"):
                            audio_bytes = await resp.read()
                            fname = f"speech_{uuid.uuid4().hex[:8]}.{audio_format or 'mp3'}"
                            link = await _upload_bytes_public(audio_bytes, fname)
                            print(f"[TTS] 二进制音频→transfer.sh 上传结果: {link}")
                            return link
                    except Exception:
                        print("[TTS] 处理音频流失败")
                    return None

                # 优先尝试JSON返回
                text_ct = resp.headers.get("Content-Type", "")
                if "application/json" in text_ct or not text_ct or "json" in text_ct:
                    try:
                        j = await resp.json(content_type=None)
                        print(f"[TTS] JSON 返回: keys={list(j.keys()) if isinstance(j, dict) else type(j)}")
                    except Exception:
                        j = None
                    # 1) 直接URL
                    if isinstance(j, dict):
                        audio_url = None
                        # 常见字段兼容
                        audio_url = (
                            (j.get("data") or {}).get("url") if isinstance(j.get("data"), dict) else None
                        ) or j.get("url") or j.get("audio_url")
                        if audio_url and isinstance(audio_url, str) and audio_url.startswith("http"):
                            print(f"[TTS] 直接获得URL: {audio_url}")
                            return audio_url
                        # 2) base64 内容
                        base64_data = (
                            (j.get("data") or {}).get("audio") if isinstance(j.get("data"), dict) else None
                        ) or j.get("audio") or j.get("content")
                        if isinstance(base64_data, str) and base64_data:
                            try:
                                import base64
                                # 处理 data:audio/mpeg;base64, 前缀
                                if "," in base64_data and base64_data.strip().startswith("data:"):
                                    base64_data = base64_data.split(",", 1)[1]
                                audio_bytes = base64.b64decode(base64_data)
                                fname = f"speech_{uuid.uuid4().hex[:8]}.{audio_format or 'mp3'}"
                                link = await _upload_bytes_public(audio_bytes, fname)
                                print(f"[TTS] base64→transfer.sh 上传结果: {link}")
                                return link
                            except Exception:
                                print("[TTS] 解析base64失败")
                                return None
                    return None
                else:
                    # 非JSON：尝试按音频二进制处理
                    audio_bytes = await resp.read()
                    if audio_bytes:
                        fname = f"speech_{uuid.uuid4().hex[:8]}.{audio_format or 'mp3'}"
                        link = await _upload_bytes_public(audio_bytes, fname)
                        print(f"[TTS] 二进制→transfer.sh 上传结果: {link}")
                        return link
                    return None
    except Exception:
        print("[TTS] StepFun 请求异常")
        return None

async def _upload_bytes_public(data: bytes, filename: str) -> Optional[str]:
    """上传二进制到公共临时文件托管，返回公网可访问链接。

    顺序：transfer.sh → 0x0.st → file.io → tmpfiles.org
    """
    if not data:
        return None
    timeout = aiohttp.ClientTimeout(total=30)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # 1) transfer.sh (PUT)
            try:
                url = f"https://transfer.sh/{filename}"
                async with session.put(url, data=data, headers={"Content-Type": "application/octet-stream"}) as r:
                    body = await r.text()
                    print(f"[TTS-UP] transfer.sh status={r.status}, body={body[:80]}")
                    if r.status in (200, 201):
                        link = body.strip()
                        if link.startswith("http"):
                            return link
            except Exception as e:
                print(f"[TTS-UP] transfer.sh 失败: {e}")

            # 2) 0x0.st (multipart/form-data POST file)
            try:
                form = aiohttp.FormData()
                form.add_field("file", data, filename=filename, content_type="application/octet-stream")
                # 0x0.st 对默认 User-Agent 可能返回 403，模拟 curl UA
                async with session.post("https://0x0.st", data=form, headers={"User-Agent": "curl/8.0", "Accept": "*/*"}) as r:
                    text = (await r.text()).strip()
                    print(f"[TTS-UP] 0x0.st status={r.status}, body={text[:80]}")
                    if r.status in (200, 201) and text.startswith("http"):
                        return text
            except Exception as e:
                print(f"[TTS-UP] 0x0.st 失败: {e}")

            # 3) file.io (multipart/form-data)
            try:
                form = aiohttp.FormData()
                form.add_field("file", data, filename=filename, content_type="application/octet-stream")
                async with session.post("https://file.io", data=form) as r:
                    j = await r.json(content_type=None)
                    print(f"[TTS-UP] file.io status={r.status}, json_keys={list(j.keys()) if isinstance(j, dict) else type(j)}")
                    link = (j or {}).get("link")
                    if r.status in (200, 201) and isinstance(link, str) and link.startswith("http"):
                        return link
            except Exception as e:
                print(f"[TTS-UP] file.io 失败: {e}")

            # 4) tmpfiles.org (multipart/form-data)
            try:
                form = aiohttp.FormData()
                form.add_field("file", data, filename=filename, content_type="application/octet-stream")
                async with session.post("https://tmpfiles.org/api/v1/upload", data=form) as r:
                    j = await r.json(content_type=None)
                    print(f"[TTS-UP] tmpfiles status={r.status}, json_keys={list(j.keys()) if isinstance(j, dict) else type(j)}")
                    data_obj = (j or {}).get("data") if isinstance(j, dict) else None
                    page_url = (data_obj or {}).get("url") if isinstance(data_obj, dict) else None
                    file_name = (data_obj or {}).get("file_name") if isinstance(data_obj, dict) else None
                    if isinstance(page_url, str) and page_url.startswith("http"):
                        # 情况A：已经是直接下载链接，直接返回
                        if "/dl/" in page_url:
                            return page_url
                        # 情况B：分享页 /s/<id>[/<name>] 或 根路径 /<id>[/<name>] → 统一转换为 /dl/<id>/<name>
                        try:
                            parts = page_url.rstrip("/").split("/")
                            # 期望 parts 形如 [scheme, '', host, ...path]
                            path_parts = parts[3:] if len(parts) > 3 else []
                            if not path_parts:
                                return page_url
                            if path_parts[0] == "s":
                                tail = path_parts[1:]
                            else:
                                tail = path_parts
                            if len(tail) >= 2:
                                file_id, inferred_name = tail[0], tail[1]
                                return f"https://tmpfiles.org/dl/{file_id}/{inferred_name}"
                            elif len(tail) == 1:
                                file_id = tail[0]
                                name = file_name or filename
                                return f"https://tmpfiles.org/dl/{file_id}/{name}" if name else f"https://tmpfiles.org/dl/{file_id}"
                        except Exception:
                            pass
                        # 无法解析则返回页面链接作为兜底
                        return page_url
            except Exception as e:
                print(f"[TTS-UP] tmpfiles 失败: {e}")
    except Exception as e:
        print(f"[TTS-UP] 会话创建失败: {e}")
        return None
    return None

async def get_openai_client():
    """异步获取 OpenRouter 兼容客户端，统一走 OpenRouter。"""
    global _client
    if _client is None:
        print("[DEBUG] 初始化OpenRouter兼容客户端...")
        api_key = await asyncio.to_thread(os.getenv, "OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("未检测到 OPENROUTER_API_KEY。请设置后重试，已全面切换为 OpenRouter。")
        referer = os.getenv("HTTP_REFERER", "")
        title = os.getenv("X_TITLE", "")
        # 设置请求超时，避免云端长时间挂起
        http_client_timeout = float(os.getenv("OPENROUTER_HTTP_TIMEOUT", "30"))
        _client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": referer,
                "X-Title": title,
            },
            timeout=http_client_timeout,
        )
        print("[DEBUG] OpenRouter兼容客户端初始化完成")
    return _client

def generate_video_id(video_url: str) -> str:
    """为视频生成唯一ID"""
    # 使用URL的hash作为基础
    url_hash = hashlib.md5(video_url.encode()).hexdigest()[:8]
    # 添加时间戳和随机数确保唯一性
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = str(uuid.uuid4())[:4]
    return f"video_{url_hash}_{timestamp}_{random_suffix}"

def generate_frame_id(video_id: str, frame_index: int) -> str:
    """为视频帧生成唯一ID"""
    return f"{video_id}_frame_{frame_index:03d}"

def generate_audio_id(video_id: str) -> str:
    """为音频生成唯一ID"""
    return f"{video_id}_audio"

async def describe_image_urls(urls: List[str]) -> List[str]:
    """
    使用 GPT-4o 对图片链接进行描述（逐张处理）
    """
    print("=" * 80)
    print("🖼️ [DEBUG-视觉识别] 开始执行describe_image_urls")
    print("=" * 80)
    print(f"🖼️ [DEBUG-视觉识别] 需要处理的图片URL数量: {len(urls)}")
    for i, url in enumerate(urls, 1):
        print(f"🖼️ [DEBUG-视觉识别] 图片 {i}: {url}")

    if not urls:
        print("🖼️ [DEBUG-视觉识别] 没有图片URL需要处理，返回空列表")
        return []

    try:
        print("🖼️ [DEBUG-视觉识别] 正在获取OpenAI客户端...")
        client = await get_openai_client()
        print("🖼️ [DEBUG-视觉识别] OpenAI客户端获取成功")
    except Exception as e:
        print(f"🖼️ [DEBUG-视觉识别] 获取OpenAI客户端失败: {e}")
        import traceback
        print(f"🖼️ [DEBUG-视觉识别] 详细错误信息:\n{traceback.format_exc()}")
        return [f"[获取视觉模型客户端失败: {e}]" for _ in urls]

    descriptions = []
    # 优先使用运行时的 vision_model；未显式配置则强制使用 z-ai/glm-4.5v（不再回退到 model_name，避免选到不支持图像的聊天模型）
    vision_model = _normalize_model_name_for_openrouter(_cfg.get("vision_model") or "z-ai/glm-4.5v")
    print(f"🖼️ [DEBUG-视觉识别] 将使用的视觉模型: {vision_model}")

    for i, url in enumerate(urls, 1):
        print(f"🖼️ [DEBUG-视觉识别] 开始处理第 {i} 张图片...")
        try:
            print(f"🖼️ [DEBUG-视觉识别] 正在调用视觉模型分析图片: {url[:100]}...")
            response = await client.chat.completions.create(
                model=vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "请描述这张图片的内容："},
                            {"type": "image_url", "image_url": {"url": url}}
                        ]
                    }
                ],
                max_tokens=300
            )
            print(f"🖼️ [DEBUG-视觉识别] 视觉模型调用完成，响应类型: {type(response)}")
            description = response.choices[0].message.content.strip()
            print(f"🖼️ [DEBUG-视觉识别] 图片 {i} 描述成功，长度: {len(description)} 字符")
            print(f"🖼️ [DEBUG-视觉识别] 图片 {i} 描述内容: {description[:200]}...")
        except Exception as e:
            print(f"🖼️ [DEBUG-视觉识别] 图片 {i} 描述失败: {e}")
            import traceback
            print(f"🖼️ [DEBUG-视觉识别] 详细错误信息:\n{traceback.format_exc()}")
            description = f"[图片描述失败: {e}]"

        descriptions.append(description)

    print(f"🖼️ [DEBUG-视觉识别] 所有图片处理完成，共 {len(descriptions)} 个描述")
    return descriptions

async def transcribe_audio_urls(urls: List[str]) -> List[str]:
    """
    语音转写优先使用阿里云 SenseVoice（dashscope），失败时回退到 OpenAI Whisper。

    注意：SenseVoice 要求输入为公网可访问的 URL，不支持直接上传文件字节。
    """
    print("=" * 80)
    print("🎵 [DEBUG-音频转录] 开始执行transcribe_audio_urls")
    print("=" * 80)
    print(f"🎵 [DEBUG-音频转录] 需要处理的音频URL数量: {len(urls)}")
    for i, url in enumerate(urls, 1):
        print(f"🎵 [DEBUG-音频转录] 音频 {i}: {url}")

    if not urls:
        print("🎵 [DEBUG-音频转录] 没有音频URL需要处理，返回空列表")
        return []

    print("🎵 [DEBUG-音频转录] 检查阿里云SenseVoice配置...")
    # 如果配置了阿里云 API Key 且安装了 dashscope，则优先使用 SenseVoice
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    print(f"🎵 [DEBUG-音频转录] DASHSCOPE_API_KEY配置状态: {'已配置' if dashscope_api_key else '未配置'}")

    if dashscope_api_key and _ensure_dashscope_loaded():
        print("🎵 [DEBUG-音频转录] 尝试使用阿里云SenseVoice进行音频转录...")
        try:
            # 尝试从运行时配置读取 language hint，例如 "zh"/"en"/"yue"/"ja"/"ko"/"auto"
            language_hint = None
            try:
                language_hint = _cfg.get("asr_language") or _cfg.get("sensevoice_language")
                print(f"🎵 [DEBUG-音频转录] 语言设置: {language_hint}")
            except Exception:
                language_hint = None

            results = await _sensevoice_transcribe_urls(urls, language_code=language_hint)
            print(f"🎵 [DEBUG-音频转录] SenseVoice处理完成，结果数量: {len(results)}")

            # 若结果基本可用，则直接返回
            if results and any(isinstance(x, str) and x.strip() for x in results):
                print("🎵 [DEBUG-音频转录] SenseVoice结果有效，直接返回")
                return results
            else:
                print("🎵 [DEBUG-音频转录] SenseVoice结果无效，回退到Whisper")
        except Exception as e:
            print(f"🎵 [DEBUG-音频转录] SenseVoice调用失败，回退到Whisper: {e}")
            import traceback
            print(f"🎵 [DEBUG-音频转录] SenseVoice详细错误:\n{traceback.format_exc()}")

    print("🎵 [DEBUG-音频转录] 使用OpenAI Whisper进行音频转录...")
    # 回退到 Whisper（现有实现）
    try:
        client = await get_openai_client()
        print("🎵 [DEBUG-音频转录] OpenAI客户端获取成功")
    except Exception as e:
        print(f"🎵 [DEBUG-音频转录] 获取OpenAI客户端失败: {e}")
        return [f"[获取音频转录客户端失败: {e}]" for _ in urls]

    transcriptions: List[str] = []
    whisper_model = _cfg.get("whisper_model", "whisper-1")
    print(f"🎵 [DEBUG-音频转录] 将使用的Whisper模型: {whisper_model}")

    async with aiohttp.ClientSession() as session:
        for i, url in enumerate(urls, 1):
            print(f"🎵 [DEBUG-音频转录] 正在处理第 {i} 个音频: {url[:100]}...")
            try:
                print(f"🎵 [DEBUG-音频转录] 下载音频文件...")
                async with session.get(url) as resp:
                    status = resp.status
                    print(f"🎵 [DEBUG-音频转录] HTTP响应状态码: {status}")

                    if resp.status == 200:
                        audio_data = await resp.read()
                        print(f"🎵 [DEBUG-音频转录] 音频数据下载完成，大小: {len(audio_data)} bytes")

                        audio_file = io.BytesIO(audio_data)
                        audio_file.name = "audio.mp3"

                        prompt = "请直接提取这段语音的核心内容，控制在200字以内，保留关键信息。"
                        print(f"🎵 [DEBUG-音频转录] 转录提示词: {prompt}")

                        # 若未配置官方 OpenAI Key，跳过 Whisper 兜底
                        if not os.getenv("OPENAI_API_KEY"):
                            print("🎵 [DEBUG-音频转录] 未配置OPENAI_API_KEY，跳过音频转写")
                            transcriptions.append("[未配置OPENAI_API_KEY，跳过音频转写]")
                            continue

                        print("🎵 [DEBUG-音频转录] 正在调用Whisper API...")
                        response = await client.audio.transcriptions.create(
                            model=whisper_model,
                            file=audio_file,
                            prompt=prompt,
                            response_format="text"
                        )

                        transcribed_text = response.strip() if isinstance(response, str) else response.text.strip()
                        print(f"🎵 [DEBUG-音频转录] Whisper转录完成，原始长度: {len(transcribed_text)} 字符")

                        if len(transcribed_text) > 150:
                            print(f"🎵 [DEBUG-音频转录] 内容过长({len(transcribed_text)}字)，使用GPT提炼重要内容...")
                            try:
                                important_content = await extract_important_content(transcribed_text, max_length=100)
                                transcriptions.append(important_content)
                                print(f"🎵 [DEBUG-音频转录] 提炼完成，最终长度: {len(important_content)} 字")
                            except Exception as e:
                                print(f"🎵 [DEBUG-音频转录] 内容提炼失败: {e}")
                                transcriptions.append(transcribed_text[:150] + "...")
                        else:
                            transcriptions.append(transcribed_text)
                            print(f"🎵 [DEBUG-音频转录] 转录完成，长度: {len(transcribed_text)} 字")
                    else:
                        error_msg = f"[语音获取失败: {resp.status}]"
                        transcriptions.append(error_msg)
                        print(f"🎵 [DEBUG-音频转录] {error_msg}")
            except Exception as e:
                error_msg = f"[语音转录失败: {e}]"
                transcriptions.append(error_msg)
                print(f"🎵 [DEBUG-音频转录] {error_msg}")
                import traceback
                print(f"🎵 [DEBUG-音频转录] 详细错误信息:\n{traceback.format_exc()}")

    print(f"🎵 [DEBUG-音频转录] 所有音频处理完成，共 {len(transcriptions)} 个转录结果")
    return transcriptions

async def _sensevoice_transcribe_urls(urls: List[str], language_code: Optional[str] = None) -> List[str]:
    """
    使用阿里云 SenseVoice 录音语音识别（dashscope）对一组音频 URL 进行转写。
    返回与输入等长的结果列表。
    """
    # 获取结果元数据（内置 WAIT→FETCH 兜底、重试）
    results_meta = await _sensevoice_get_results_meta(urls, language_code)

    parsed_texts: List[str] = []
    for item in results_meta:
        if item.get("subtask_status") != "SUCCEEDED":
            code = item.get("code", "")
            msg = item.get("message", "")
            parsed_texts.append(f"[SenseVoice子任务失败: {code} {msg}]")
            continue
        t_url = item.get("transcription_url")
        if not t_url:
            parsed_texts.append("[SenseVoice缺少结果URL]")
            continue
        j = await _fetch_json_resilient(t_url)
        parsed_texts.append(_parse_sensevoice_json(j))

    # 长文本做摘要
    final_texts: List[str] = []
    for text in parsed_texts:
        if not isinstance(text, str):
            final_texts.append("[SenseVoice结果解析失败]")
            continue
        t = text.strip()
        if len(t) > 150:
            try:
                summarized = await extract_important_content(t, max_length=100)
                final_texts.append(summarized)
            except Exception:
                final_texts.append(t[:120] + "...")
        else:
            final_texts.append(t)
    return final_texts

def _parse_sensevoice_json(j: Any) -> str:
    """从 SenseVoice 的转写 JSON 中提取文本，移除情绪/事件标签。"""
    try:
        # 优先解析 transcripts -> sentences
        transcripts = (j or {}).get("transcripts") or (j or {}).get("Transcript")
        if transcripts and isinstance(transcripts, list):
            # 取第一个 channel 的内容
            first = transcripts[0]
            # 一些示例使用 key "text" 存储富文本（含标签）
            text_field = first.get("text") or ""
            sentences = first.get("sentences") or []
            if sentences and isinstance(sentences, list):
                joined = " ".join(s.get("text", "") for s in sentences if isinstance(s, dict))
                return _strip_sv_tags(joined)
            if text_field:
                return _strip_sv_tags(text_field)
        # 其他结构兜底：直接字符串化
        return _strip_sv_tags(str(j))
    except Exception:
        return "[SenseVoice结果解析异常]"

def _strip_sv_tags(text: str) -> str:
    """去除 SenseVoice 富文本中的标签，如 <|Speech|>、<|HAPPY|>、<|Applause|> 等。"""
    try:
        return re.sub(r"<\|/?[^|]+\|>", "", text).strip()
    except Exception:
        return text

async def _fetch_json_resilient(url: str, retries: int = 3, backoff_base: float = 0.5) -> Any:
    """带重试与指数退避的 JSON 拉取，缓解偶发 SSL EOF/网络闪断。
    - retries: 最大重试次数
    - backoff_base: 初始退避秒数
    """
    last_err = None
    for attempt in range(retries + 1):
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, ssl=False) as r:
                    if r.status == 200:
                        return await r.json(content_type=None)
                    # 特殊处理405错误，提供更友好的错误信息
                    elif r.status == 405:
                        print(f"[DEBUG] HTTP 405 Method Not Allowed for URL: {url}")
                        print("[DEBUG] 可能原因：LangSmith环境限制或服务器不支持GET方法")
                        last_err = RuntimeError(f"HTTP 405 - Method Not Allowed (LangSmith环境可能存在访问限制)")
                    else:
                        last_err = RuntimeError(f"HTTP {r.status}")
        except Exception as e:
            last_err = e
        if attempt < retries:
            await asyncio.sleep(backoff_base * (2 ** attempt))
    return {"error": str(last_err) if last_err else "unknown"}

async def _sensevoice_get_results_meta(urls: List[str], language_code: Optional[str]) -> List[dict]:
    """
    稳健地提交 SenseVoice 任务并获取 results 元数据：
    - 先 async_call → wait；若 wait 触发网络异常（如 SSL EOF），退避重试；
    - 若 wait 失败或状态非 OK，则尝试 fetch 轮询（避免单次长连接问题）。
    """
    from dashscope.audio.asr import Transcription  # type: ignore
    from http import HTTPStatus

    params = dict(
        model="sensevoice-v1",
        file_urls=urls,
        language_hints=[language_code] if language_code else ["auto"],
    )

    # 提交任务
    task_response = await asyncio.to_thread(Transcription.async_call, **params)
    if not task_response or not getattr(task_response, "output", None) or not getattr(task_response.output, "task_id", None):
        maybe_msg = getattr(task_response, "message", None) if task_response else None
        raise RuntimeError(f"SenseVoice 提交失败：未获得任务ID{f'（{maybe_msg}）' if maybe_msg else ''}")
    task_id = task_response.output.task_id

    # 等待结果（带重试，处理 SSL EOF 等瞬断问题）
    last_wait_err = None
    for attempt in range(3):
        try:
            transcribe_response = await asyncio.to_thread(Transcription.wait, task_id)
            if transcribe_response and transcribe_response.status_code == HTTPStatus.OK:
                output = transcribe_response.output or {}
                return output.get("results", []) or []
            last_wait_err = RuntimeError(f"非OK状态: {getattr(transcribe_response, 'status_code', None)}")
        except Exception as e:
            last_wait_err = e
        await asyncio.sleep(0.5 * (2 ** attempt))

    # 兜底：使用 fetch 轮询，避免长连接问题
    try:
        for _ in range(10):
            fetch_resp = await asyncio.to_thread(Transcription.fetch, task_id)
            if not fetch_resp:
                await asyncio.sleep(0.5)
                continue
            status = getattr(fetch_resp, "output", {}) or {}
            task_status = (status.get("task_status") or getattr(fetch_resp, "task_status", None))
            if task_status in ("SUCCEEDED", "FAILED"):
                out = getattr(fetch_resp, "output", {}) or {}
                return out.get("results", []) or []
            await asyncio.sleep(0.8)
    except Exception:
        pass

    raise RuntimeError(f"SenseVoice 等待失败：{last_wait_err}")

async def transcribe_audio_urls_with_emotion(urls: List[str]) -> List[dict]:
    """
    返回包含文本与情感的结果列表：[{"text": str, "emotion": str}]。
    - 优先 SenseVoice：从结果中解析情感标签（HAPPY/SAD/ANGRY/NEUTRAL），并清洗标签。
    - 回退 Whisper：仅返回文本，emotion 置为 "未知"。

    支持配置选项：
    - 设置环境变量 DISABLE_SENSEVOICE=1 可强制使用Whisper，避免LangSmith环境的网络限制
    """
    if not urls:
        return []

    # 检查是否禁用SenseVoice（适用于LangSmith等受限环境）
    disable_sensevoice = os.getenv("DISABLE_SENSEVOICE", "0") == "1"
    if disable_sensevoice:
        print("[DEBUG] SenseVoice已被禁用，直接使用Whisper")
        texts = await transcribe_audio_urls(urls)
        return [{"text": t, "emotion": "未知"} for t in texts]

    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    if dashscope_api_key and _ensure_dashscope_loaded():
        try:
            # 获取结果元数据（内置 WAIT→FETCH 兜底、重试）
            results_meta = await _sensevoice_get_results_meta(urls, None)

            parsed: List[dict] = []
            for item in results_meta:
                if item.get("subtask_status") != "SUCCEEDED":
                    code = item.get("code", "")
                    msg = item.get("message", "")
                    parsed.append({"text": f"[SenseVoice子任务失败: {code} {msg}]", "emotion": "未知"})
                    continue
                t_url = item.get("transcription_url")
                if not t_url:
                    parsed.append({"text": "[SenseVoice缺少结果URL]", "emotion": "未知"})
                    continue
                j = await _fetch_json_resilient(t_url)
                text, emotion = _parse_sensevoice_json_with_emotion(j)
                parsed.append({"text": text, "emotion": emotion})

            # 对过长文本做摘要
            final: List[dict] = []
            for r in parsed:
                txt = r.get("text", "")
                if isinstance(txt, str) and len(txt) > 150:
                    try:
                        summarized = await extract_important_content(txt, max_length=100)
                        final.append({"text": summarized, "emotion": r.get("emotion", "未知")})
                    except Exception:
                        final.append({"text": txt[:120] + "...", "emotion": r.get("emotion", "未知")})
                else:
                    final.append(r)
            return final
        except Exception as e:
            print(f"⚠️ SenseVoice（含情感）调用失败，回退 Whisper: {e}")

    # Whisper 回退：仅文本
    texts = await transcribe_audio_urls(urls)
    return [{"text": t, "emotion": "未知"} for t in texts]

def _parse_sensevoice_json_with_emotion(j: Any) -> tuple[str, str]:
    """
    从 SenseVoice 的转写 JSON 中提取 (text, emotion)。
    emotion 取值映射为中文：HAPPY→高兴, SAD→伤心, ANGRY→生气, NEUTRAL→中性；无则返回 "未知"。
    """
    try:
        # 检查是否是错误响应
        if isinstance(j, dict) and "error" in j:
            error_msg = j.get("error", "")
            if "405" in error_msg or "Method Not Allowed" in error_msg:
                return "[语音转录失败: Error code: 405]", "未知"
            else:
                return f"[语音转录失败: {error_msg}]", "未知"

        transcripts = (j or {}).get("transcripts") or (j or {}).get("Transcript")
        raw_text = ""
        if transcripts and isinstance(transcripts, list):
            first = transcripts[0]
            text_field = first.get("text") or ""
            sentences = first.get("sentences") or []
            if sentences and isinstance(sentences, list):
                raw_text = " ".join(s.get("text", "") for s in sentences if isinstance(s, dict))
            elif text_field:
                raw_text = text_field
        else:
            raw_text = str(j)

        emotion_en = _extract_sv_emotion_tag(raw_text)
        emotion_cn = _map_emotion_to_zh(emotion_en) if emotion_en else "未知"
        clean_text = _strip_sv_tags(raw_text)
        return clean_text, emotion_cn
    except Exception:
        return "[SenseVoice结果解析异常]", "未知"

def _extract_sv_emotion_tag(text: str) -> Optional[str]:
    """提取最后出现的情感标签，如 <|HAPPY|>。返回英文代号或 None。"""
    try:
        tags = re.findall(r"<\|([A-Z]+)\|>", text)
        # 仅关注情感标签
        valid = [t for t in tags if t in {"HAPPY", "SAD", "ANGRY", "NEUTRAL"}]
        return valid[-1] if valid else None
    except Exception:
        return None

def _map_emotion_to_zh(tag: Optional[str]) -> str:
    mapping = {
        "HAPPY": "高兴",
        "SAD": "伤心",
        "ANGRY": "生气",
        "NEUTRAL": "中性",
    }
    return mapping.get(tag or "", "未知")


async def extract_important_content(text: str, max_length: int = 100) -> str:
    """
    提取文本中的重要内容，控制在指定字数以内
    """
    client = await get_openai_client()
    if len(text) <= max_length:
        return text
    
    try:
        response = await client.chat.completions.create(
            model=_normalize_model_name_for_openrouter(_cfg.get("generation_model", _cfg.get("model_name", "gpt-4o-mini"))),  # 使用更快的模型
            messages=[
                {
                    "role": "system",
                    "content": f"你是一个文本摘要专家。请从以下语音转录文本中提取最重要的内容，控制在{max_length}字以内。保留关键信息，去除冗余内容。"
                },
                {
                    "role": "user", 
                    "content": f"请提取以下语音内容的重要信息，控制在{max_length}字以内：\n\n{text}"
                }
            ],
            max_tokens=300,
            temperature=0.1
        )
        
        result = response.choices[0].message.content.strip()
        return result
        
    except Exception as e:
        print(f"⚠️ 重要内容提取失败: {e}")
        # 降级处理：简单截取
        return text[:max_length] + "..." if len(text) > max_length else text


def parse_datetime_to_beijing(dt_str):
    """将ISO字符串转为东八区datetime对象"""
    if not dt_str:
        return None
    if isinstance(dt_str, datetime):
        if dt_str.tzinfo is None:
            return dt_str.replace(tzinfo=BEIJING_TZ)
        return dt_str.astimezone(BEIJING_TZ)
    if isinstance(dt_str, str):
        try:
            return datetime.fromisoformat(dt_str.replace('Z', '+08:00')).astimezone(BEIJING_TZ)
        except Exception:
            return None
    return None

def ensure_beijing_aware(dt):
    # 只做类型透传，不做任何转换，输入输出都为str或None
    return dt
def extract_xml(text: str, tag: str) -> str:
    """
    从给定的文本中提取指定XML标签的内容。
    这个函数是解析大语言模型返回的结构化响应的关键工具。

    工作原理:
    - 使用正则表达式 `re.search` 来查找模式 `<tag>(.*?)</tag>`。
    - `re.DOTALL` 标志允许 `.` 匹配包括换行符在内的任意字符，
      这对于提取可能包含多行内容的XML标签至关重要。
    - 如果找到匹配项，`match.group(1)`会返回第一个捕获组的内容，
      也就是开始和结束标签之间的所有文本。

    Args:
        text (str): 包含XML的文本。
        tag (str): 要提取内容的XML标签名。

    Returns:
        str: 指定XML标签的内容，如果未找到标签则返回空字符串。
    """
    # 使用正则表达式搜索指定标签对之间的内容
    match = re.search(f'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
    # 如果找到匹配项，返回捕获的内容，否则返回空字符串
    return match.group(1) if match else ""

def format_messages(messages: List[Any]) -> str:
    """格式化对话消息."""
    if not messages:
        return ""
    lines = []
    for msg in messages[-10:]:  # 保留最近10条对话
        # 兼容 dict 和对象
        role = None
        content = None
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
        elif hasattr(msg, "role") and hasattr(msg, "content"):
            role = getattr(msg, "role", "")
            content = getattr(msg, "content", "")
        else:
            # 兜底：转字符串
            role = str(type(msg))
            content = str(msg)

        if role == "user" or role == "human":
            lines.append(f"用户：{content}")
        elif role == "assistant" or role == "ai":
            lines.append(f"助手：{content}")
        else:
            lines.append(f"{role}：{content}")
    return "\n".join(lines)

def calculate_smart_interval(user_last_reply_time: str, last_active_send_time: str) -> int:
    """计算智能触发间隔。所有参数均为字符串，内部需要  时转为datetime"""
    now = datetime.now(BEIJING_TZ)

    user_last_reply_dt = parse_datetime_to_beijing(user_last_reply_time)
    last_active_send_dt = parse_datetime_to_beijing(last_active_send_time)

    if not user_last_reply_dt:
        return 86400  # 默认1天
    user_reply_diff = now - user_last_reply_dt
    user_reply_days = user_reply_diff.days

    # 智能间隔规则（可根据业务调整）
    if user_reply_days >= 60:
        return 31536000  # 1年，转人工
    if user_reply_days >= 30:
        return random.randint(10 * 86400, 20 * 86400)
    if user_reply_days >= 7:
        return random.randint(3 * 86400, 5 * 86400)
    if user_reply_days >= 1:
        return random.randint(86400, 2 * 86400)
    # 2小时内
    hours_since_reply = user_reply_diff.total_seconds() / 3600
    if hours_since_reply <= 2:
        return random.randint(3 * 3600, 18 * 3600)
    return 86400  # 默认1天

def parse_event_decision(response: str) -> dict:
    """解析事件决策响应."""
    try:
        # 尝试提取JSON
        if "{" in response and "}" in response:
            start = response.find("{")
            end = response.rfind("}") + 1
            json_str = response[start:end]
            return json.loads(json_str)
        else:
            raise ValueError("响应中没有找到有效的JSON")
    except Exception as e:
        print(f"[ERROR] 解析事件决策响应失败: {e}")
        print(f"[ERROR] 原始响应: {response}")
        return {
            "event_type": "pending_activation",
            "event_time": datetime.now(BEIJING_TZ).isoformat(),
            "appointment_time": None
        }

async def describe_video_urls(urls: List[str]) -> List[str]:
    """
    专业视频分析处理 - 基于阿里云百炼视频理解API思路（纯内存多帧处理）
    
    处理流程：
    1. 流式下载视频数据到内存
    2. 使用Python包提取多个关键帧（每秒1-2帧）
    3. 提取音频数据
    4. 使用aihubmix o4-mini分析多个关键帧
    5. 使用OpenAI Whisper转录音频
    6. 综合多模态信息生成视频描述
    
    参考：阿里云百炼视频理解API思路，但使用现有模型
    """
    # 支持的视频格式
    VIDEO_FORMATS = {
        "wmv", "asf", "asx", "rm", "rmvb", "mp4", "mpeg", "mpg", "3gp", 
        "mov", "m4v", "avi", "dat", "mkv", "flv", "vob", "ogv", "webm", 
        "ts", "mts", "m2ts", "divx", "xvid", "swf", "f4v", "f4p", "f4a", "f4b"
    }
    
    descriptions = []
    for url in urls:
        try:
            # 检查URL是否为支持的视频格式
            url_lower = url.lower()
            is_video = any(url_lower.endswith(f".{fmt}") for fmt in VIDEO_FORMATS) or any(f".{fmt}?" in url_lower for fmt in VIDEO_FORMATS)
            
            if not is_video:
                print(f"[DEBUG] URL格式检查失败: {url}")
                descriptions.append(f"[非视频格式或格式不支持: {url}]")
                continue
            
            print(f"🎬 开始专业视频分析: {url}")
            
            # 方案1：多帧视频分析（云平台友好）
            try:
                description = await _analyze_video_multiframe(url)
                descriptions.append(description)
                
            except Exception as multiframe_error:
                print(f"⚠️ 多帧视频分析失败: {multiframe_error}")
                
                # 方案2：直接URL分析（降级）
                try:
                    description = await _analyze_video_url_direct(url)
                    descriptions.append(description)
                    
                except Exception as direct_error:
                    print(f"⚠️ 直接URL分析失败: {direct_error}")
                    
                    # 方案3：智能URL分析
                    try:
                        description = await _analyze_video_url_intelligent(url)
                        descriptions.append(description)
                        
                    except Exception as intelligent_error:
                        print(f"⚠️ 智能URL分析失败: {intelligent_error}")
                        
                        # 方案4：降级处理 - 基本信息
                        filename = url.split('/')[-1].split('?')[0] if '/' in url else url
                        file_extension = filename.split('.')[-1].lower() if '.' in filename else "未知格式"
                        description = f"视频文件：{filename}（{file_extension}格式）。当前环境限制，无法进行详细的视频内容分析。"
                        descriptions.append(description)
            
        except Exception as e:
            description = f"[视频处理失败: {e}]"
            descriptions.append(description)
    
    return descriptions

async def _analyze_video_multiframe(video_url: str) -> str:
    """多帧视频分析 - 流式下载视频，提取多个关键帧和音频（内存优化版本）"""
    print(f"🎬 开始多帧视频分析: {video_url}")
    
    # 生成唯一视频ID
    video_id = generate_video_id(video_url)
    print(f"📋 视频ID: {video_id}")
    
    try:
        # 1. 流式下载视频数据到内存
        video_data = await _download_video_to_memory(video_url)
        video_size_mb = len(video_data) / (1024 * 1024)
        print(f"✅ 视频数据下载完成，大小: {video_size_mb:.2f} MB")
        
        # 检查视频大小，避免处理过大的文件
        if video_size_mb > 100:  # 超过100MB的视频
            print(f"⚠️ 视频文件过大({video_size_mb:.2f}MB)，使用降级处理")
            return await _analyze_video_url_direct(video_url)
        
        # 2. 提取多个关键帧（每秒1-2帧）
        frame_images = await _extract_frames_from_memory(video_data, video_id)
        print(f"✅ 提取了 {len(frame_images)} 个关键帧")
        
        # 3. 提取音频数据
        audio_data = await _extract_audio_from_memory(video_data, video_id)
        audio_size_mb = len(audio_data) / (1024 * 1024) if audio_data else 0
        print(f"✅ 音频数据提取完成，大小: {audio_size_mb:.2f} MB")
        
        # 4. 使用 aihubmix o4-mini 分析多个关键帧
        frame_descriptions = []
        if frame_images:
            try:
                frame_descriptions = await _analyze_frames_with_aihubmix(frame_images, video_id)
                print(f"✅ 关键帧分析完成，共 {len(frame_descriptions)} 个描述")
            except Exception as frame_error:
                print(f"⚠️ 关键帧分析失败: {frame_error}")
                frame_descriptions = [f"第{i+1}帧：分析失败" for i in range(min(len(frame_images), 5))]
        
        # 5. 使用 OpenAI Whisper 转录音频
        audio_transcription = ""
        if audio_data:
            try:
                audio_transcription = await _transcribe_audio_from_memory(audio_data, video_id)
                print(f"✅ 音频转录完成")
            except Exception as audio_error:
                print(f"⚠️ 音频转录失败: {audio_error}")
                audio_transcription = "无法提取音频内容"
        
        # 6. 综合多模态信息生成视频描述
        try:
            result = await _synthesize_multiframe_video_description(frame_descriptions, audio_transcription, video_url, video_id)
        except Exception as synthesis_error:
            print(f"⚠️ 综合描述生成失败: {synthesis_error}")
            # 降级处理
            frame_summary = "；".join(frame_descriptions[:3]) if frame_descriptions else "无法提取视频帧"
            audio_summary = audio_transcription if audio_transcription != "无法提取音频内容" else "无音频"
            result = f"🎬 视频内容：{frame_summary}。音频内容：{audio_summary}"
        
        # 7. 主动清理大内存对象（虽然Python会自动清理，但显式清理更安全）
        del video_data
        del frame_images
        del audio_data
        
        # 强制垃圾回收
        import gc
        gc.collect()
        
        print(f"✅ 内存清理完成，视频ID: {video_id}")
        return result
        
    except Exception as e:
        print(f"❌ 多帧视频分析失败: {e}")
        # 确保异常时也清理内存
        try:
            import gc
            gc.collect()
        except:
            pass
        raise

async def _analyze_video_url_direct(video_url: str) -> str:
    """直接分析视频URL，使用aihubmix o4-mini和OpenAI Whisper（纯内存处理）"""
    import asyncio
    print(f"🔧 直接分析视频URL: {video_url}")
    
    try:
        # 方案1：尝试使用 aihubmix o4-mini 分析视频画面
        try:
            # 异步获取API key，避免阻塞
            aihubmix_api_key = await asyncio.to_thread(os.getenv, "AIHUBMIX_API_KEY")
            print(f"[DEBUG] 获取AIHUBMIX_API_KEY: {aihubmix_api_key[:10] if aihubmix_api_key else 'None'}...")
            if aihubmix_api_key:
                from openai import AsyncOpenAI
                aihubmix_client = AsyncOpenAI(
                    api_key=aihubmix_api_key,
                    base_url="https://aihubmix.com/v1"
                )
                
                # 使用 image_url 处理视频（可能只看到第一帧或缩略图）
                response = await aihubmix_client.chat.completions.create(
                    model="o4-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "这是一个视频文件，请详细描述你看到的画面内容，包括人物、物体、动作、文字、字幕、镜头语言等。如果只能看到第一帧，请说明这是视频的静态画面："},
                                {"type": "image_url", "image_url": {"url": video_url}}
                            ]
                        }
                    ],
                    max_tokens=400
                )
                
                frame_description = response.choices[0].message.content.strip()
                print(f"✅ aihubmix视频画面分析完成")
                
                # 方案2：尝试使用 OpenAI Whisper 转录音频
                audio_transcription = ""
                try:
                    # 使用现有的 transcribe_audio_urls 函数
                    audio_transcriptions = await transcribe_audio_urls([video_url])
                    if audio_transcriptions and not audio_transcriptions[0].startswith("[语音转录失败"):
                        audio_transcription = audio_transcriptions[0]
                        print(f"✅ 音频转录完成")
                    else:
                        audio_transcription = "无法提取音频内容"
                        print(f"⚠️ 音频转录失败")
                except Exception as audio_error:
                    print(f"⚠️ 音频转录失败: {audio_error}")
                    audio_transcription = "无法提取音频内容"
                
                # 方案3：综合生成视频描述
                return await _synthesize_video_description_simple(frame_description, audio_transcription, video_url)
                
        except Exception as aihubmix_error:
            print(f"⚠️ aihubmix分析失败: {aihubmix_error}")
        
        # 方案4：尝试使用 OpenAI GPT-4o
        try:
            client = await get_openai_client()
            response = await client.chat.completions.create(
                model=_normalize_model_name_for_openrouter(_cfg.get("vision_model") or "z-ai/glm-4.5v"),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "这是一个视频文件，请详细描述你看到的画面内容，包括人物、物体、动作、文字、字幕、镜头语言等。如果只能看到第一帧，请说明这是视频的静态画面："},
                            {"type": "image_url", "image_url": {"url": video_url}}
                        ]
                    }
                ],
                max_tokens=400
            )
            
            frame_description = response.choices[0].message.content.strip()
            print(f"✅ OpenAI视频画面分析完成")
            
            # 尝试音频转录
            audio_transcription = ""
            try:
                audio_transcriptions = await transcribe_audio_urls([video_url])
                if audio_transcriptions and not audio_transcriptions[0].startswith("[语音转录失败"):
                    audio_transcription = audio_transcriptions[0]
                    print(f"✅ 音频转录完成")
                else:
                    audio_transcription = "无法提取音频内容"
                    print(f"⚠️ 音频转录失败")
            except Exception as audio_error:
                print(f"⚠️ 音频转录失败: {audio_error}")
                audio_transcription = "无法提取音频内容"
            
            return await _synthesize_video_description_simple(frame_description, audio_transcription, video_url)
            
        except Exception as openai_error:
            print(f"⚠️ OpenAI分析失败: {openai_error}")
        
        # 方案5：智能URL分析
        return await _analyze_video_url_intelligent(video_url)
        
    except Exception as e:
        print(f"❌ URL直接分析失败: {e}")
        raise

async def _analyze_video_url_intelligent(video_url: str) -> str:
    """智能分析视频URL，基于URL特征推测内容"""
    print(f"🧠 智能分析视频URL: {video_url}")
    
    # 从URL中提取信息
    url_lower = video_url.lower()
    filename = video_url.split('/')[-1].split('?')[0] if '/' in video_url else video_url
    file_extension = filename.split('.')[-1].lower() if '.' in filename else "未知格式"
    
    # 分析URL特征
    analysis_parts = []
    
    # 1. 文件格式分析
    if file_extension in ["mp4", "mov", "avi", "mkv"]:
        analysis_parts.append(f"标准视频格式（{file_extension}）")
    elif file_extension in ["3gp", "m4v"]:
        analysis_parts.append(f"移动设备视频格式（{file_extension}）")
    elif file_extension in ["webm", "ogv"]:
        analysis_parts.append(f"网页视频格式（{file_extension}）")
    else:
        analysis_parts.append(f"视频格式（{file_extension}）")
    
    # 2. 文件名分析
    if any(keyword in filename.lower() for keyword in ["video", "vid", "movie", "film"]):
        analysis_parts.append("文件名包含视频相关关键词")
    
    # 3. URL路径分析
    if "/wechat/" in url_lower:
        analysis_parts.append("来自微信的视频文件")
    elif "/video/" in url_lower:
        analysis_parts.append("来自视频目录")
    elif "/media/" in url_lower:
        analysis_parts.append("来自媒体目录")
    
    # 4. 时间戳分析
    import re
    timestamp_match = re.search(r'(\d{10,13})', filename)
    if timestamp_match:
        timestamp = timestamp_match.group(1)
        if len(timestamp) == 13:  # 毫秒时间戳
            from datetime import datetime
            try:
                dt = datetime.fromtimestamp(int(timestamp) / 1000)
                analysis_parts.append(f"创建时间：{dt.strftime('%Y-%m-%d %H:%M:%S')}")
            except:
                pass
    
    # 5. 生成智能描述
    if analysis_parts:
        description = f"视频文件：{filename}。特征分析：{'；'.join(analysis_parts)}。"
    else:
        description = f"视频文件：{filename}（{file_extension}格式）。"
    
    description += " 当前环境限制，无法进行详细的视频内容分析。如需完整分析，建议使用支持视频处理的专业API。"
    
    return f"🎬 {description}"

async def _synthesize_video_description_simple(frame_description: str, audio_transcription: str, video_url: str) -> str:
    """综合视频画面和音频信息生成简单描述"""
    
    # 构建综合提示词
    prompt = f"""
# 视频分析任务
请基于以下信息，生成一个完整的视频描述：

## 视频画面分析结果：
{frame_description}

## 音频转录内容：
{audio_transcription if audio_transcription != "无法提取音频内容" else "无音频内容"}

## 任务要求：
1. 结合画面和音频信息，生成视频的详细概述
2. 如果只能看到第一帧，请说明这是视频的静态画面
3. 突出视频的核心信息和关键情节
4. 保持客观准确，不添加推测内容

## 输出要求：
- 控制在300字以内
- 格式：先描述画面内容，再结合音频信息总结
"""

    try:
        # 使用现有模型生成综合描述
        response = await get_openai_client().chat.completions.create(
            model=_cfg.get("generation_model", _cfg.get("model_name", "gpt-4o-mini")),
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的视频分析师，擅长结合视觉和音频信息进行视频内容分析。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        description = response.choices[0].message.content.strip()
        print(f"✅ 视频分析完成: {video_url}")
        return f"🎬 视频分析结果：{description}"
        
    except Exception as e:
        print(f"⚠️ 视频描述生成失败: {e}")
        # 降级处理：简单拼接
        return f"🎬 视频内容：{frame_description}。音频内容：{audio_transcription}"

async def _download_video_to_memory(video_url: str) -> bytes:
    """流式下载视频数据到内存"""
    async with aiohttp.ClientSession() as session:
        async with session.get(video_url) as response:
            if response.status == 200:
                video_data = await response.read()
                return video_data
            else:
                raise Exception(f"视频下载失败: {response.status}")

async def _extract_frames_from_memory(video_data: bytes, video_id: str) -> List[bytes]:
    """从内存中的视频数据提取关键帧"""
    try:
        # 尝试使用 moviepy（推荐）
        try:
            return await _extract_frames_with_moviepy(video_data, video_id)
        except ImportError:
            print("⚠️ moviepy未安装，尝试使用opencv-python")
        except Exception as e:
            print(f"⚠️ moviepy处理失败: {e}，尝试其他方案")
        
        # 尝试使用 opencv-python
        try:
            return await _extract_frames_with_opencv(video_data, video_id)
        except ImportError:
            print("⚠️ opencv-python未安装，尝试使用imageio")
        except Exception as e:
            print(f"⚠️ opencv-python处理失败: {e}，尝试其他方案")
        
        # 尝试使用 imageio
        try:
            return await _extract_frames_with_imageio(video_data, video_id)
        except ImportError:
            print("⚠️ imageio未安装，使用降级方案")
        except Exception as e:
            print(f"⚠️ imageio处理失败: {e}，使用降级方案")
        
        # 降级方案：只提供基本信息
        raise Exception("所有Python视频处理包都不可用，请安装moviepy或opencv-python")
        
    except Exception as e:
        print(f"❌ 视频帧提取失败: {e}")
        raise

async def _extract_frames_with_moviepy(video_data: bytes, video_id: str) -> List[bytes]:
    """使用moviepy从内存数据提取关键帧（异步版本）"""
    import io
    import asyncio
    import tempfile
    import os
    
    print(f"🎬 使用moviepy处理视频: {video_id}")
    
    # 将同步操作移到线程池中执行
    def _extract_frames_sync(video_data: bytes, video_id: str) -> List[bytes]:
        from moviepy.editor import VideoFileClip
        import cv2
        import numpy as np
        
        # 创建临时文件，使用唯一ID命名
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4', prefix=f"{video_id}_", delete=False) as temp_file:
                temp_file.write(video_data)
                temp_file_path = temp_file.name
            
            print(f"📁 创建临时视频文件: {temp_file_path}")
            
            # 加载视频
            video = VideoFileClip(temp_file_path)
            duration = video.duration
            fps = video.fps
            
            # 提取关键帧（每秒1帧）
            frame_interval = int(fps)  # 每秒1帧
            frame_images = []
            
            # 使用正确的方式获取帧
            for i in range(0, int(duration * fps), frame_interval):
                try:
                    # 获取指定时间的帧
                    frame = video.get_frame(i / fps)
                    # 转换为JPEG格式的bytes
                    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    frame_images.append(buffer.tobytes())
                except Exception as e:
                    print(f"⚠️ 提取第{i}帧失败: {e}")
                    continue
            
            video.close()
            return frame_images
        finally:
            # 确保临时文件被删除
            _safe_delete_temp_file(temp_file_path)
    
    # 在线程池中执行同步操作
    frame_images = await asyncio.to_thread(_extract_frames_sync, video_data, video_id)
    print(f"✅ 使用moviepy提取了 {len(frame_images)} 个关键帧")
    return frame_images

async def _extract_frames_with_opencv(video_data: bytes, video_id: str) -> List[bytes]:
    """使用opencv-python从内存数据提取关键帧（异步版本）"""
    import io
    import asyncio
    import tempfile
    import os
    
    print(f"🎬 使用opencv-python处理视频: {video_id}")
    
    # 将同步操作移到线程池中执行
    def _extract_frames_sync(video_data: bytes, video_id: str) -> List[bytes]:
        import cv2
        import numpy as np
        
        # 创建临时文件，使用唯一ID命名
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4', prefix=f"{video_id}_", delete=False) as temp_file:
                temp_file.write(video_data)
                temp_file_path = temp_file.name
            
            print(f"📁 创建临时视频文件: {temp_file_path}")
            
            # 使用opencv读取视频
            cap = cv2.VideoCapture(temp_file_path)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_interval = int(fps)  # 每秒1帧
            frame_images = []
            
            for i in range(0, total_frames, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    # 转换为JPEG格式的bytes
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_images.append(buffer.tobytes())
            
            cap.release()
            return frame_images
        finally:
            # 确保临时文件被删除
            _safe_delete_temp_file(temp_file_path)
    
    # 在线程池中执行同步操作
    frame_images = await asyncio.to_thread(_extract_frames_sync, video_data, video_id)
    print(f"✅ 使用opencv-python提取了 {len(frame_images)} 个关键帧")
    return frame_images

async def _extract_frames_with_imageio(video_data: bytes, video_id: str) -> List[bytes]:
    """使用imageio从内存数据提取关键帧（异步版本）"""
    import io
    import asyncio
    import tempfile
    import os
    
    print(f"🎬 使用imageio处理视频: {video_id}")
    
    # 将同步操作移到线程池中执行
    def _extract_frames_sync(video_data: bytes, video_id: str) -> List[bytes]:
        import imageio
        import cv2
        import numpy as np
        
        # 创建临时文件，使用唯一ID命名
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4', prefix=f"{video_id}_", delete=False) as temp_file:
                temp_file.write(video_data)
                temp_file_path = temp_file.name
            
            print(f"📁 创建临时视频文件: {temp_file_path}")
            
            reader = imageio.get_reader(temp_file_path)
            fps = reader.get_meta_data()['fps']
            total_frames = reader.get_length()
            
            frame_interval = int(fps)  # 每秒1帧
            frame_images = []
            
            for i in range(0, total_frames, frame_interval):
                try:
                    frame = reader.get_data(i)
                    # 转换为JPEG格式的bytes
                    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    frame_images.append(buffer.tobytes())
                except IndexError:
                    break
            
            reader.close()
            return frame_images
        finally:
            # 确保临时文件被删除
            _safe_delete_temp_file(temp_file_path)
    
    # 在线程池中执行同步操作
    frame_images = await asyncio.to_thread(_extract_frames_sync, video_data, video_id)
    print(f"✅ 使用imageio提取了 {len(frame_images)} 个关键帧")
    return frame_images

async def _extract_audio_from_memory(video_data: bytes, video_id: str) -> bytes:
    """从内存中的视频数据提取音频"""
    try:
        # 尝试使用 moviepy
        try:
            return await _extract_audio_with_moviepy(video_data, video_id)
        except ImportError:
            print("⚠️ moviepy未安装，尝试使用pydub")
        except Exception as e:
            print(f"⚠️ moviepy音频提取失败: {e}，尝试其他方案")
        
        # 尝试使用 pydub
        try:
            return await _extract_audio_with_pydub(video_data, video_id)
        except ImportError:
            print("⚠️ pydub未安装，无法提取音频")
        except Exception as e:
            print(f"⚠️ pydub音频提取失败: {e}")
        
        return None
        
    except Exception as e:
        print(f"❌ 音频提取失败: {e}")
        return None

async def _extract_audio_with_moviepy(video_data: bytes, video_id: str) -> bytes:
    """使用moviepy从内存数据提取音频（异步版本）"""
    import io
    import asyncio
    import tempfile
    import os
    
    print(f"🎵 使用moviepy提取音频: {video_id}")
    
    # 将同步操作移到线程池中执行
    def _extract_audio_sync(video_data: bytes, video_id: str) -> bytes:
        from moviepy.editor import VideoFileClip
        
        # 创建临时文件，使用唯一ID命名
        temp_file_path = None
        audio_temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4', prefix=f"{video_id}_", delete=False) as temp_file:
                temp_file.write(video_data)
                temp_file_path = temp_file.name
            
            print(f"📁 创建临时视频文件: {temp_file_path}")
            
            # 加载视频
            video = VideoFileClip(temp_file_path)
            audio = video.audio
            
            if audio is not None:
                # 创建临时音频文件，使用唯一ID命名
                with tempfile.NamedTemporaryFile(suffix='.mp3', prefix=f"{video_id}_audio_", delete=False) as audio_temp_file:
                    audio_temp_path = audio_temp_file.name
                
                print(f"📁 创建临时音频文件: {audio_temp_path}")
                
                # 提取音频到临时文件
                audio.write_audiofile(audio_temp_path, verbose=False, logger=None)
                
                # 读取音频数据
                with open(audio_temp_path, 'rb') as f:
                    audio_data = f.read()
                
                return audio_data
            else:
                return None
        finally:
            # 确保所有临时文件被删除
            _safe_delete_temp_file(temp_file_path)
            
            _safe_delete_temp_file(audio_temp_path)
    
    # 在线程池中执行同步操作
    audio_data = await asyncio.to_thread(_extract_audio_sync, video_data, video_id)
    if audio_data:
        print(f"✅ 音频提取完成，大小: {len(audio_data)} bytes")
    else:
        print("⚠️ 视频中没有音频轨道")
    return audio_data

async def _extract_audio_with_pydub(video_data: bytes, video_id: str) -> bytes:
    """使用pydub从内存数据提取音频（异步版本）"""
    import io
    import asyncio
    import tempfile
    import os
    
    print(f"🎵 使用pydub提取音频: {video_id}")
    
    # 将同步操作移到线程池中执行
    def _extract_audio_sync(video_data: bytes, video_id: str) -> bytes:
        from pydub import AudioSegment
        
        # 创建临时文件，使用唯一ID命名
        temp_file_path = None
        audio_temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4', prefix=f"{video_id}_", delete=False) as temp_file:
                temp_file.write(video_data)
                temp_file_path = temp_file.name
            
            print(f"📁 创建临时视频文件: {temp_file_path}")
            
            # 加载音频
            audio = AudioSegment.from_file(temp_file_path)
            
            # 创建临时音频文件，使用唯一ID命名
            with tempfile.NamedTemporaryFile(suffix='.mp3', prefix=f"{video_id}_audio_", delete=False) as audio_temp_file:
                audio_temp_path = audio_temp_file.name
            
            print(f"📁 创建临时音频文件: {audio_temp_path}")
            
            # 导出为MP3格式
            audio.export(audio_temp_path, format="mp3")
            
            # 读取音频数据
            with open(audio_temp_path, 'rb') as f:
                audio_data = f.read()
            
            return audio_data
        finally:
            # 确保所有临时文件被删除
            _safe_delete_temp_file(temp_file_path)
            
            _safe_delete_temp_file(audio_temp_path)
    
    # 在线程池中执行同步操作
    audio_data = await asyncio.to_thread(_extract_audio_sync, video_data, video_id)
    print(f"✅ 音频提取完成，大小: {len(audio_data)} bytes")
    return audio_data

async def _analyze_frames_with_aihubmix(frame_images: List[bytes], video_id: str) -> List[str]:
    """使用aihubmix o4-mini分析多个关键帧"""
    import asyncio
    client = await get_openai_client()
    # 异步获取API key，避免阻塞
    aihubmix_api_key = await asyncio.to_thread(os.getenv, "AIHUBMIX_API_KEY")
    print(f"[DEBUG] 获取AIHUBMIX_API_KEY: {aihubmix_api_key[:10] if aihubmix_api_key else 'None'}...")
    if not aihubmix_api_key:
        print("⚠️ 未配置AIHUBMIX_API_KEY，使用OpenAI GPT-4o")
        return await _analyze_frames_with_openai(frame_images, video_id)
    
    # from openai import AsyncOpenAI # This line is now redundant as client is global
    # aihubmix_client = AsyncOpenAI( # This line is now redundant as client is global
    #     api_key=aihubmix_api_key,
    #     base_url="https://aihubmix.com/v1"
    # )
    
    frame_descriptions = []
    
    # 限制处理帧数，避免API调用过多
    max_frames = min(len(frame_images), 5)  # 最多处理5帧
    
    for i, frame_data in enumerate(frame_images[:max_frames]):
        try:
            # 生成帧ID
            frame_id = generate_frame_id(video_id, i)
            print(f"🔍 分析帧 {frame_id}")
            
            # 将图片数据转换为base64
            import base64
            frame_base64 = base64.b64encode(frame_data).decode('utf-8')
            frame_url = f"data:image/jpeg;base64,{frame_base64}"
            
            response = await client.chat.completions.create(
                model="o4-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"这是视频的第{i+1}个关键帧，请详细描述画面内容，包括人物、物体、动作、文字、字幕、镜头语言等："},
                            {"type": "image_url", "image_url": {"url": frame_url}}
                        ]
                    }
                ],
                max_completion_tokens=200
            )
            
            description = response.choices[0].message.content.strip()
            frame_descriptions.append(f"第{i+1}帧：{description}")
            print(f"✅ 帧 {frame_id} 分析完成")
            
        except Exception as e:
            print(f"⚠️ 第{i+1}帧分析失败: {e}")
            frame_descriptions.append(f"第{i+1}帧：分析失败")
    
    return frame_descriptions

async def _analyze_frames_with_openai(frame_images: List[bytes], video_id: str) -> List[str]:
    """使用OpenAI GPT-4o分析多个关键帧"""
    client = await get_openai_client()
    frame_descriptions = []
    
    # 限制处理帧数，避免API调用过多
    max_frames = min(len(frame_images), 5)  # 最多处理5帧
    
    for i, frame_data in enumerate(frame_images[:max_frames]):
        try:
            # 生成帧ID
            frame_id = generate_frame_id(video_id, i)
            print(f"🔍 分析帧 {frame_id}")
            
            # 将图片数据转换为base64
            import base64
            frame_base64 = base64.b64encode(frame_data).decode('utf-8')
            frame_url = f"data:image/jpeg;base64,{frame_base64}"
            
            response = await client.chat.completions.create(
                model=_normalize_model_name_for_openrouter(_cfg.get("vision_model") or "z-ai/glm-4.5v"),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"这是视频的第{i+1}个关键帧，请详细描述画面内容，包括人物、物体、动作、文字、字幕、镜头语言等："},
                            {"type": "image_url", "image_url": {"url": frame_url}}
                        ]
                    }
                ],
                max_tokens=200
            )
            
            description = response.choices[0].message.content.strip()
            frame_descriptions.append(f"第{i+1}帧：{description}")
            print(f"✅ 帧 {frame_id} 分析完成")
            
        except Exception as e:
            print(f"⚠️ 第{i+1}帧分析失败: {e}")
            frame_descriptions.append(f"第{i+1}帧：分析失败")
    
    return frame_descriptions

async def _transcribe_audio_from_memory(audio_data: bytes, video_id: str) -> str:
    """从内存中的音频数据转录音频"""
    client = await get_openai_client()
    try:
        # 生成音频ID
        audio_id = generate_audio_id(video_id)
        print(f"🎵 开始转录音频: {audio_id}")
        
        # 创建内存文件对象
        audio_file = io.BytesIO(audio_data)
        audio_file.name = "audio.mp3"  # OpenAI SDK要求有name
        
        # 使用prompt指导Whisper直接输出压缩内容
        prompt = "请直接提取这段语音的核心内容，控制在200字以内，保留关键信息。"
        
        if not os.getenv("OPENAI_API_KEY"):
            return "无法提取音频内容"
        response = await client.audio.transcriptions.create(
            model=_cfg.get("whisper_model", "whisper-1"),
            file=audio_file,
            prompt=prompt,
            response_format="text"
        )
        
        # 当使用 response_format="text" 时，API 直接返回字符串
        transcribed_text = response.strip() if isinstance(response, str) else response.text.strip()
        
        # 检查字数，如果超过150字就用GPT提炼重要内容
        if len(transcribed_text) > 150:
            print(f"🎵 语音内容过长({len(transcribed_text)}字)，使用GPT提炼重要内容...")
            important_content = await extract_important_content(transcribed_text, max_length=100)
            print(f"✅ 语音内容已提炼，长度: {len(important_content)}字")
            return important_content
        else:
            print(f"✅ 语音内容已处理，长度: {len(transcribed_text)}字")
            return transcribed_text
            
    except Exception as e:
        print(f"⚠️ 音频转录失败: {e}")
        return "无法提取音频内容"

async def _synthesize_multiframe_video_description(frame_descriptions: List[str], audio_transcription: str, video_url: str, video_id: str) -> str:
    """综合多帧视频画面和音频信息生成详细描述"""
    
    print(f"📝 开始合成视频描述: {video_id}")
    
    # 构建综合提示词
    prompt = f"""
# 视频分析任务
请基于以下多帧信息，生成一个完整的视频描述：

## 关键帧分析结果（按时间顺序）：
{chr(10).join(frame_descriptions) if frame_descriptions else "无法提取视频帧"}

## 音频转录内容：
{audio_transcription if audio_transcription != "无法提取音频内容" else "无音频内容"}

## 任务要求：
1. 分析每个关键帧的画面信息，包括人物、物体、动作、文字、字幕、镜头语言等
2. 将关键帧信息按时间顺序串联起来，生成视频的详细概述
3. 结合音频转录内容，还原该片段的完整剧情
4. 输出格式：先描述各关键帧，再总结整个视频内容

## 输出要求：
- 控制在400字以内
- 保持客观准确，不添加推测内容
- 突出视频的核心信息和关键情节
- 体现视频的时序变化和剧情发展
"""

    try:
        # 使用现有模型生成综合描述
        client = await get_openai_client()
        response = await client.chat.completions.create(
            model=_normalize_model_name_for_openrouter(_cfg.get("generation_model", _cfg.get("model_name", "gpt-4o-mini"))),
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的视频分析师，擅长结合多帧视觉和音频信息进行视频内容分析。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=600,
            temperature=0.3,
            timeout=30  # 添加超时设置
        )
        
        description = response.choices[0].message.content.strip()
        print(f"✅ 多帧视频分析完成: {video_id}")
        return f"🎬 多帧视频分析结果：{description}"
            
    except Exception as e:
        print(f"⚠️ 多帧视频描述生成失败: {e}")
        # 降级处理：简单拼接
        frame_summary = "；".join(frame_descriptions[:3]) if frame_descriptions else "无法提取视频帧"
        audio_summary = audio_transcription if audio_transcription != "无法提取音频内容" else "无音频"
        
        # 如果帧描述为空，提供基本信息
        if not frame_descriptions or all("分析失败" in desc for desc in frame_descriptions):
            frame_summary = f"成功提取了{len(frame_descriptions)}个关键帧，但分析失败"
        
        return f"🎬 视频内容：{frame_summary}。音频内容：{audio_summary}"

def _get_memory_usage():
    """获取当前内存使用情况"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)  # 转换为MB
        return f"{memory_mb:.2f} MB"
    except ImportError:
        return "未知（需要安装psutil）"

def _log_memory_usage(stage: str):
    """记录内存使用情况"""
    memory_usage = _get_memory_usage()
    print(f"📊 内存使用情况 [{stage}]: {memory_usage}")

async def describe_webpage_urls(urls: List[str]) -> List[str]:
    """
    抓取并提炼通用网页链接的主要内容（支持公众号文章等），用于在对话中理解链接内容。

    - 对每个URL并发下载HTML
    - 优先提取 <article> / <main> / 公众号 #js_content 的正文
    - 结合<title>生成简短摘要（约180字）
    - 网络/解析失败时返回可读的错误提示
    """
    print("=" * 80)
    print("🌐 [DEBUG-外部链接识别] 开始执行describe_webpage_urls")
    print("=" * 80)
    print(f"🌐 [DEBUG-外部链接识别] 需要处理的网页URL数量: {len(urls)}")
    for i, url in enumerate(urls, 1):
        print(f"🌐 [DEBUG-外部链接识别] 网页 {i}: {url}")

    if not urls:
        print("🌐 [DEBUG-外部链接识别] 没有网页URL需要处理，返回空列表")
        return []

    import aiohttp
    from bs4 import BeautifulSoup

    headers_base = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    async def fetch_and_summarize(url: str) -> str:
        print(f"🌐 [DEBUG-外部链接识别] 开始处理网页: {url}")

        try:
            print(f"🌐 [DEBUG-外部链接识别] 正在设置请求头...")
            # 针对特定域名添加额外头（例如微信公众号）
            headers = dict(headers_base)
            if "mp.weixin.qq.com" in url:
                headers.update({
                    "Referer": "https://weixin.qq.com/",
                    "Upgrade-Insecure-Requests": "1",
                })
                print(f"🌐 [DEBUG-外部链接识别] 检测到微信公众号，添加特殊请求头")

            print(f"🌐 [DEBUG-外部链接识别] 正在发起HTTP请求...")
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(url, timeout=15) as resp:
                    status = resp.status
                    print(f"🌐 [DEBUG-外部链接识别] HTTP响应状态码: {status}")

                    text_body = await resp.text(errors="ignore")
                    print(f"🌐 [DEBUG-外部链接识别] 获取响应内容，长度: {len(text_body)} 字符")

                    if status != 200 or ("环境异常" in text_body and "去验证" in text_body):
                        print(f"🌐 [DEBUG-外部链接识别] 检测到异常响应，使用Jina AI代理...")
                        # 兜底：使用 Jina AI Reader 代理拉取纯文本
                        proxy_url = f"https://r.jina.ai/{url}"
                        try:
                            print(f"🌐 [DEBUG-外部链接识别] 正在调用代理: {proxy_url}")
                            async with session.get(proxy_url, timeout=20) as proxy_resp:
                                proxy_status = proxy_resp.status
                                print(f"🌐 [DEBUG-外部链接识别] 代理响应状态码: {proxy_status}")

                                if proxy_resp.status == 200:
                                    proxy_text = await proxy_resp.text(errors="ignore")
                                    print(f"🌐 [DEBUG-外部链接识别] 代理获取内容成功，长度: {len(proxy_text)} 字符")
                                    # 代理返回已是文本，直接进入后续提炼
                                    html = f"<html><body><article>{proxy_text}</article></body></html>"
                                    print(f"🌐 [DEBUG-外部链接识别] 使用代理内容进行解析")
                                else:
                                    print(f"🌐 [DEBUG-外部链接识别] 代理调用失败: HTTP {proxy_status}")
                                    return f"[网页获取失败: HTTP {status}，代理 {proxy_resp.status}]"
                        except Exception as proxy_err:
                            print(f"🌐 [DEBUG-外部链接识别] 代理调用异常: {proxy_err}")
                            return f"[网页获取失败: HTTP {status}，代理异常: {proxy_err}]"
                    else:
                        # 正常HTML
                        html = text_body
                        print(f"🌐 [DEBUG-外部链接识别] 使用原始HTML内容进行解析")
        except Exception as e:
            print(f"🌐 [DEBUG-外部链接识别] 网页获取异常: {e}")
            import traceback
            print(f"🌐 [DEBUG-外部链接识别] 详细错误信息:\n{traceback.format_exc()}")
            return f"[网页获取失败: {e}]"

        try:
            print(f"🌐 [DEBUG-外部链接识别] 开始HTML解析...")
            soup = BeautifulSoup(html, "html.parser")
            print(f"🌐 [DEBUG-外部链接识别] BeautifulSoup解析完成")

            # 标题
            title = ""
            try:
                if soup.title and soup.title.get_text():
                    title = soup.title.get_text(strip=True)
                    print(f"🌐 [DEBUG-外部链接识别] 提取到标题: {title}")
            except Exception as e:
                print(f"🌐 [DEBUG-外部链接识别] 提取标题失败: {e}")
                title = ""

            # 针对公众号文章的特化选择器
            content_node = None
            if "mp.weixin.qq.com" in url:
                content_node = soup.find(id="js_content") or soup.select_one("#js_content, .rich_media_content")
                if content_node:
                    print(f"🌐 [DEBUG-外部链接识别] 使用微信公众号专用选择器提取内容")
                else:
                    print(f"🌐 [DEBUG-外部链接识别] 微信公众号专用选择器未找到内容")

            # 通用节点
            if content_node is None:
                content_node = soup.find("article") or soup.find("main")
                if content_node:
                    print(f"🌐 [DEBUG-外部链接识别] 使用通用选择器(article/main)提取内容")
                else:
                    print(f"🌐 [DEBUG-外部链接识别] 通用选择器未找到内容，使用兜底方法")

            # 兜底：聚合常见文本标签
            if content_node is None:
                parts = []
                for tag in soup.find_all(["h1", "h2", "h3", "p", "li"]):
                    text_piece = tag.get_text(" ", strip=True)
                    if text_piece:
                        parts.append(text_piece)
                content_text = "\n".join(parts)
                print(f"🌐 [DEBUG-外部链接识别] 兜底方法提取到 {len(parts)} 个文本片段")
            else:
                content_text = content_node.get_text(" ", strip=True)

            # 清理空白
            content_text = re.sub(r"\s+", " ", content_text).strip()
            print(f"🌐 [DEBUG-外部链接识别] 清理后内容长度: {len(content_text)} 字符")

            if not content_text:
                print(f"🌐 [DEBUG-外部链接识别] 未提取到有效内容")
                return title or "[未能解析网页正文]"

            # 内容过长时提炼要点（~180字）
            summary = content_text
            max_len = 180
            if len(summary) > max_len:
                print(f"🌐 [DEBUG-外部链接识别] 内容过长({len(summary)}字符)，正在提炼...")
                try:
                    summary = await extract_important_content(summary, max_length=max_len)
                    print(f"🌐 [DEBUG-外部链接识别] AI提炼完成，长度: {len(summary)} 字符")
                except Exception as e:
                    print(f"🌐 [DEBUG-外部链接识别] AI提炼失败，使用截断: {e}")
                    summary = summary[:max_len] + "..."
                    print(f"🌐 [DEBUG-外部链接识别] 截断完成，长度: {len(summary)} 字符")

            if title:
                result = f"{title}：{summary}"
            else:
                result = summary

            print(f"🌐 [DEBUG-外部链接识别] 网页处理完成，最终结果长度: {len(result)} 字符")
            return result
        except Exception as e:
            print(f"🌐 [DEBUG-外部链接识别] 网页解析异常: {e}")
            import traceback
            print(f"🌐 [DEBUG-外部链接识别] 详细错误信息:\n{traceback.format_exc()}")
            return f"[网页解析失败: {e}]"

    # 顺序处理以控制并发，避免外部站点风控；如需更快可切换为gather
    print(f"🌐 [DEBUG-外部链接识别] 开始顺序处理 {len(urls)} 个URL...")
    results: List[str] = []
    for i, u in enumerate(urls, 1):
        print(f"🌐 [DEBUG-外部链接识别] 正在处理第 {i}/{len(urls)} 个URL: {u[:100]}...")
        desc = await fetch_and_summarize(u)
        results.append(desc)
        print(f"🌐 [DEBUG-外部链接识别] 第 {i} 个URL处理完成，结果长度: {len(desc)} 字符")

    print(f"🌐 [DEBUG-外部链接识别] 所有网页处理完成，共 {len(results)} 个结果")
    return results

def _safe_delete_temp_file(file_path: str, max_retries: int = 3, delay: float = 0.1):
    """安全删除临时文件，适用于LangSmith部署环境"""
    if not file_path or not os.path.exists(file_path):
        return
    
    for attempt in range(max_retries):
        try:
            # 在LangSmith环境中，文件可能被短暂锁定
            import time
            time.sleep(delay * (attempt + 1))  # 递增延迟
            
            os.unlink(file_path)
            print(f"🗑️ 删除临时文件: {file_path}")
            return
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"⚠️ 无法删除临时文件 {file_path}: {e}")
                # 在LangSmith中，如果无法删除，记录但继续执行
                print(f"📝 临时文件将在系统清理时自动删除: {file_path}")
            else:
                print(f"⚠️ 删除临时文件失败，重试 {attempt + 1}/{max_retries}: {e}")

def _cleanup_temp_files(temp_files: list):
    """批量清理临时文件"""
    for file_path in temp_files:
        _safe_delete_temp_file(file_path)

async def get_audio_duration_ms(audio_url: str) -> Optional[int]:
    """获取音频文件的时长（毫秒）

    Args:
        audio_url: 音频文件的URL

    Returns:
        int: 音频时长（毫秒），获取失败时返回None
    """
    if not audio_url or not isinstance(audio_url, str):
        return None

    try:
        # 下载音频文件
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(audio_url) as response:
                if response.status != 200:
                    print(f"[AUDIO] 下载失败: HTTP {response.status}")
                    return None

                # 读取音频数据
                audio_data = await response.read()
                if not audio_data:
                    print("[AUDIO] 下载到的音频数据为空")
                    return None

        # 方案1：尝试使用mutagen（轻量级，无阻塞调用问题）
        try:
            from mutagen.mp3 import MP3
            from mutagen import File
            from io import BytesIO

            def _get_duration_mutagen(data):
                try:
                    audio_file = BytesIO(data)
                    audio = File(audio_file)
                    if audio and hasattr(audio, 'info') and hasattr(audio.info, 'length'):
                        return int(audio.info.length * 1000)  # 转换为毫秒
                    return None
                except Exception as e:
                    print(f"[AUDIO] Mutagen解析失败: {e}")
                    return None

            import asyncio
            duration_ms = await asyncio.to_thread(_get_duration_mutagen, audio_data)

            if duration_ms is not None:
                print(f"[AUDIO] 音频时长: {duration_ms}毫秒 (使用mutagen)")
                return duration_ms

        except ImportError:
            print("[AUDIO] 未安装mutagen库，尝试使用pydub")

        # 方案2：使用pydub（如果mutagen不可用）
        try:
            from pydub import AudioSegment
            from io import BytesIO

            def _get_audio_duration_pydub(audio_data):
                try:
                    audio_file = BytesIO(audio_data)
                    audio = AudioSegment.from_file(audio_file)
                    if audio is None:
                        return None
                    return len(audio)
                except Exception as e:
                    print(f"[AUDIO] Pydub解析失败: {e}")
                    return None

            duration_ms = await asyncio.to_thread(_get_audio_duration_pydub, audio_data)

            if duration_ms is not None:
                print(f"[AUDIO] 音频时长: {duration_ms}毫秒 (使用pydub)")
                return duration_ms

        except ImportError:
            print("[AUDIO] 未安装pydub库")
        except Exception as e:
            print(f"[AUDIO] Pydub异步执行失败: {e}")

        # 方案3：简单估算（基于文件大小粗略估算）
        try:
            # MP3平均比特率估算（粗略）
            file_size_kb = len(audio_data) / 1024
            # 假设平均128kbps，计算时长（秒）
            estimated_seconds = (file_size_kb * 8) / 128
            estimated_ms = int(estimated_seconds * 1000)

            print(f"[AUDIO] 音频时长估算: {estimated_ms}毫秒 (基于文件大小)")
            return estimated_ms

        except Exception as e:
            print(f"[AUDIO] 估算时长失败: {e}")

        print("[AUDIO] 所有方法都无法获取音频时长")
        return None

    except Exception as e:
        print(f"[AUDIO] 获取音频时长异常: {e}")
        return None

# =====================
# 图片材料查询功能
# =====================

async def query_material_images(thread_id: str, assistant_id: str = None) -> List[dict]:
    """
    查询可用的多媒体材料 - 获取所有类型

    Args:
        thread_id: 对话线程ID
        assistant_id: 助手ID（可选）

    Returns:
        List[dict]: 所有材料的列表，格式为[{"id": str, "name": str, "materialType": int, "content": str}]
    """
    try:
        # 构建查询URL
        base_url = os.getenv("BACKEND_URL", "")
        url = f"{base_url}"

        # 构建请求数据 - 使用type=0获取所有类型
        payload = {
            "threadId": thread_id,
            "page": 1,
            "limit": 50,
            "type": 0,  # 获取所有类型的素材
            "flag": 0
        }

        print(f"[MATERIAL_QUERY] ===== 发送素材查询请求 =====")
        print(f"[MATERIAL_QUERY] 请求URL: {url}")
        print(f"[MATERIAL_QUERY] 请求方法: POST")
        print(f"[MATERIAL_QUERY] 请求体 (JSON):")
        import json
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"[MATERIAL_QUERY] 请求头: Content-Type: application/json")
        print(f"[MATERIAL_QUERY] 超时设置: 30秒")
        print(f"[MATERIAL_QUERY] ===== 请求发送完成 =====")

        # 发送请求
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                print(f"[MATERIAL_QUERY] ===== 接收响应 =====")
                print(f"[MATERIAL_QUERY] 响应状态码: {response.status}")
                print(f"[MATERIAL_QUERY] 响应头: {dict(response.headers)}")

                if response.status != 200:
                    print(f"[MATERIAL_QUERY] ❌ 请求失败: HTTP {response.status}")
                    response_text = await response.text()
                    print(f"[MATERIAL_QUERY] 错误响应内容: {response_text}")
                    print(f"[MATERIAL_QUERY] ===== 响应处理完成 =====")
                    return []

                data = await response.json()
                print(f"[MATERIAL_QUERY] 响应体 (JSON):")
                print(json.dumps(data, indent=2, ensure_ascii=False))
                print(f"[MATERIAL_QUERY] API响应状态码: {data.get('code', 'unknown')}")

                if data.get('code') != 200:
                    print(f"[MATERIAL_QUERY] ❌ API返回业务错误: {data.get('msg', 'unknown error')}")
                    print(f"[MATERIAL_QUERY] ===== 响应处理完成 =====")
                    return []

                materials = data.get('data', [])
                print(f"[MATERIAL_QUERY] 获取到 {len(materials)} 个材料")

                # 保留完整的材料信息，包括materialType
                filtered_materials = []
                print(f"[MATERIAL_QUERY] ===== 处理材料数据 =====")
                for i, material in enumerate(materials):
                    material_id = material.get('id', '').strip()
                    name = material.get('name', '').strip()
                    material_type = material.get('materialType', 2)  # 默认图片类型
                    content = material.get('content', '')

                    print(f"[MATERIAL_QUERY] 材料 {i+1}:")
                    print(f"  - ID: {material_id}")
                    print(f"  - 名称: {name}")
                    print(f"  - 类型: {material_type}")
                    print(f"  - 内容: {content[:50]}{'...' if len(content) > 50 else ''}")

                    if material_id and name:
                        filtered_materials.append({
                            "id": material_id,
                            "name": name,
                            "materialType": material_type,
                            "content": content
                        })
                    else:
                        print(f"  ❌ 跳过无效材料 (缺少ID或名称)")

                print(f"[MATERIAL_QUERY] ===== 数据处理完成 =====")
                print(f"[MATERIAL_QUERY] 过滤后剩余 {len(filtered_materials)} 个有效材料")
                print(f"[MATERIAL_QUERY] ===== 素材查询流程结束 =====")
                return filtered_materials

    except Exception as e:
        print(f"[MATERIAL_QUERY] ===== 发生异常 =====")
        print(f"[MATERIAL_QUERY] ❌ 查询材料异常: {e}")
        import traceback
        print(f"[MATERIAL_QUERY] 异常堆栈:")
        print(traceback.format_exc())
        print(f"[MATERIAL_QUERY] ===== 异常处理完成 =====")
        return []

async def select_relevant_meterials(materials: List[dict], user_message: str, context_messages: List = None) -> Optional[dict]:
    """
    使用AI判断当前语境需要发送哪个材料（支持所有类型）

    Args:
        materials: 可用的材料列表（包含所有类型：图片、视频、卡片链接等）
        user_message: 用户消息
        context_messages: 对话上下文（可选）

    Returns:
        Optional[dict]: 选中的材料信息，格式为{"id": str, "name": str, "materialType": int, "content": str}，无合适材料时返回None
    """
    if not materials:
        print("[MATERIAL_SELECT] 没有可用的材料")
        return None

    try:
        # 构建材料列表描述
        material_type_names = {
            2: "图片", 3: "视频", 4: "卡片链接", 5: "卡片", 6: "语音", 7: "文件"
        }

        materials_text = "\n".join([
            f"{i+1}. [{material_type_names.get(m.get('materialType', 2), '未知类型')}] {m['name']}"
            for i, m in enumerate(materials)
        ])

        # 添加对话上下文
        context_text = ""
        if context_messages:
            recent_messages = context_messages[-5:]  # 最近5条消息
            context_list = []
            for msg in recent_messages:
                if hasattr(msg, 'content'):
                    content = msg.content
                elif isinstance(msg, dict):
                    content = msg.get('content', '')
                else:
                    content = str(msg)

                # 区分用户和AI消息
                if hasattr(msg, 'type') and msg.type == 'human':
                    context_list.append(f"用户: {content}")
                elif hasattr(msg, 'type') and msg.type in ['ai', 'assistant']:
                    context_list.append(f"AI: {content}")
                else:
                    context_list.append(f"消息: {content}")

            context_text = "\n".join(context_list)

        # 构建智能选择提示词
        prompt = f"""
你是一个专业的智能材料选择助手，负责根据用户的需求从材料库中选择最合适的材料。

**用户最新消息**: "{user_message}"

**对话上下文**:
{context_text}

**可用的材料列表**:
{materials_text}

**材料类型说明**:
- 图片（materialType=2）：适合展示静态内容，如环境、案例、效果图、位置等
- 视频（materialType=3）：适合展示动态内容，如介绍、演示、操作流程等
- 卡片链接（materialType=4）：适合提供详细信息，如商品详情、价格、服务介绍等
- 卡片（materialType=5）：适合展示结构化信息，如产品卡片、服务卡片等
- 语音（materialType=6）：适合语音回复，如音频消息等
- 文件（materialType=7）：适合文档资料，如PDF、Word等

**智能选择指南**:
1. **理解用户意图**: 深入分析用户消息的真实需求
2. **类型匹配**: 根据用户需求选择最合适的材料类型
3. **内容相关性**: 选择内容最相关、能够直接回答用户问题的材料
4. **上下文关联**: 考虑对话历史，选择能够延续对话逻辑的材料
5. **实用性优先**: 选择最能帮助用户的材料

**选择策略**:
- 如果用户询问"在哪里"或"怎么走"或"位置"，选择包含位置信息的图片
- 如果用户想要"看看样子"或"长什么样"或"外观"，选择图片类型
- 如果用户询问"环境怎么样"或"店面"，选择展示环境的图片
- 如果用户想要"客户案例"或"效果图"，选择展示案例的图片或视频
- 如果用户想要"看介绍"或"了解详情"或"演示"，选择视频或卡片链接
- 如果用户询问"价格"或"服务详情"，选择卡片链接类型
- 如果用户想要"文档"或"资料"，选择文件类型
- 如果用户想要语音回复，选择语音类型

**输出要求**:
请返回JSON格式：
{{"selected_name": "材料名称", "material_type": 材料类型数字, "reason": "选择理由"}}
如果没有合适的材料，返回：
{{"selected_name": null, "material_type": null, "reason": "没有找到合适的材料"}}
"""

        # 调用AI进行智能选择
        try:
            client = await get_openai_client()
            response = await client.chat.completions.create(
                model=_normalize_model_name_for_openrouter(_cfg.get("generation_model", _cfg.get("model_name", "gpt-4o-mini"))),
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个专业的材料选择助手，擅长理解用户需求并从材料库中选择最合适的材料类型和内容。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=300,
                temperature=0.2
            )

            result_text = response.choices[0].message.content.strip()
            print(f"[MATERIAL_SELECT] AI选择结果: {result_text}")

            # 解析JSON结果
            try:
                result = json.loads(result_text)
                selected_name = result.get("selected_name")
                material_type = result.get("material_type")
                reason = result.get("reason", "")

                print(f"[MATERIAL_SELECT] 解析结果 - 名称: {selected_name}, 类型: {material_type}, 理由: {reason}")

                # 检查是否有合适的选择
                if not selected_name or selected_name is None:
                    print("[MATERIAL_SELECT] AI判断无合适材料")
                    return None

                # 查找对应的材料
                for material in materials:
                    if material['name'] == selected_name and material.get('materialType') == material_type:
                        print(f"[MATERIAL_SELECT] 找到精确匹配材料: {selected_name} (类型: {material_type})")
                        return material

                # 如果精确匹配失败，尝试仅名称匹配
                for material in materials:
                    if material['name'] == selected_name:
                        print(f"[MATERIAL_SELECT] 找到名称匹配材料: {selected_name} (实际类型: {material.get('materialType')})")
                        return material

                # 尝试模糊匹配
                for material in materials:
                    if selected_name in material['name'] or material['name'] in selected_name:
                        print(f"[MATERIAL_SELECT] 找到模糊匹配材料: {material['name']}")
                        return material

                print(f"[MATERIAL_SELECT] 未找到匹配材料: {selected_name}")
                return None

            except json.JSONDecodeError as e:
                print(f"[MATERIAL_SELECT] JSON解析失败: {e}")
                print(f"[MATERIAL_SELECT] 原始响应: {result_text}")
                return None

        except Exception as e:
            print(f"[MATERIAL_SELECT] AI选择失败: {e}")
            return None

    except Exception as e:
        print(f"[MATERIAL_SELECT] 选择材料异常: {e}")
        return None



async def detect_image_request(user_message: str) -> bool:
    """
    使用AI模型检测用户消息是否包含发送图片的请求

    Args:
        user_message: 用户消息

    Returns:
        bool: 是否需要发送图片
    """
    print("=" * 80)
    print("🔍 [DEBUG-图片请求检测] 开始执行detect_image_request")
    print("=" * 80)
    print(f"🔍 [DEBUG-图片请求检测] 输入消息: '{user_message}'")

    try:
        print("🔍 [DEBUG-图片请求检测] 正在构建AI检测提示词...")

        # 构建AI检测提示词
        prompt = f"""
你是一个智能助手，专门判断用户是否需要发送多媒体内容（图片、视频、卡片链接）。

请分析以下用户消息，判断用户是否明确或隐含表达了需要你发送多媒体内容的需求。

**用户消息**: "{user_message}"

**判断标准**:
1. 明确请求图片：如"发张照片"、"来张图片"、"看一下效果"等
2. 询问视觉信息：如"长什么样"、"在哪里"、"怎么走"等
3. 要求查看案例：如"客户案例"、"效果图"、"环境照片"等
4. 其他需要视觉展示的情况
5. 请求视频内容：如"发个视频"、"看看视频"、"视频介绍"等
6. 请求卡片链接：如"发个链接"、"看看详情"、"卡片链接"等
7. 要求查看演示：如"演示视频"、"操作视频"等

**输出要求**:
- 如果需要发送多媒体内容，返回：YES
- 如果不需要发送多媒体内容，返回：NO
- 只返回YES或NO，不要其他内容

请判断：
"""
        print(f"🔍 [DEBUG-图片请求检测] 提示词构建完成，长度: {len(prompt)} 字符")

        print("🔍 [DEBUG-图片请求检测] 正在获取OpenAI客户端...")
        client = await get_openai_client()
        print("🔍 [DEBUG-图片请求检测] OpenAI客户端获取成功")

        model_name = _normalize_model_name_for_openrouter(_cfg.get("generation_model", _cfg.get("model_name", "gpt-4o-mini")))
        print(f"🔍 [DEBUG-图片请求检测] 将使用的模型: {model_name}")

        print("🔍 [DEBUG-图片请求检测] 正在调用AI模型进行判断...")
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个多媒体内容请求检测助手，判断用户是否需要发送多媒体内容（图片、视频、卡片链接），返回YES或NO。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=10,  # 只返回YES或NO，10个token足够
            temperature=0.1  # 降低随机性，提高一致性
        )
        print(f"🔍 [DEBUG-图片请求检测] AI模型调用完成，响应类型: {type(response)}")

        result = response.choices[0].message.content.strip().upper()
        print(f"🔍 [DEBUG-图片请求检测] AI判断结果: '{result}'")

        # 判断结果
        if result == "YES":
            print(f"🔍 [DEBUG-图片请求检测] ✅ AI检测到图片请求: '{user_message}'")
            return True
        else:
            print(f"🔍 [DEBUG-图片请求检测] ❌ AI判断不需要发送图片: '{user_message}'")
            return False

    except Exception as e:
        print(f"🔍 [DEBUG-图片请求检测] ❌ AI检测图片请求异常: {e}")
        import traceback
        print(f"🔍 [DEBUG-图片请求检测] 异常详情:\n{traceback.format_exc()}")

        # AI调用失败时，使用简单的关键词兜底
        print("🔍 [DEBUG-图片请求检测] 使用关键词兜底检测...")
        simple_keywords = ["图片", "照片", "案例", "效果", "地址", "位置", "环境"]
        has_keyword = any(keyword in user_message.lower() for keyword in simple_keywords)
        print(f"🔍 [DEBUG-图片请求检测] 关键词检测结果: {has_keyword}")
        print(f"🔍 [DEBUG-图片请求检测] 关键词兜底最终结果: {has_keyword}")
        return has_keyword

