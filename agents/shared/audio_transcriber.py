import asyncio
import io
import os
import requests
from typing import List
import openai

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

async def transcribe_audio_from_url(url: str) -> str:
    """
    Downloads an MP3 from a URL, transcribes it using OpenAI's Whisper model,
    and returns the transcription text.

    Args:
        url: The URL of the MP3 file.

    Returns:
        The transcribed text, or an error message if transcription fails.
    """
    # 优先 SenseVoice（只接受公网 URL）
    das_api_key = os.getenv("DASHSCOPE_API_KEY")
    if das_api_key and _ensure_dashscope_loaded():
        try:
            from dashscope.audio.asr import Transcription  # type: ignore
            from http import HTTPStatus
            task_resp = await asyncio.to_thread(
                Transcription.async_call,
                model="sensevoice-v1",
                file_urls=[url],
                language_hints=["auto"],
            )
            final_resp = await asyncio.to_thread(Transcription.wait, task_resp.output.task_id)
            if final_resp.status_code == HTTPStatus.OK:
                out = final_resp.output or {}
                results = (out.get("results") or [])
                if results and results[0].get("subtask_status") == "SUCCEEDED":
                    t_url = results[0].get("transcription_url")
                    if t_url:
                        r = requests.get(t_url, timeout=30)
                        r.raise_for_status()
                        j = r.json()
                        # 简单解析（与 utils.py 保持一致的标签清理）
                        import re
                        transcripts = j.get("transcripts") or []
                        if transcripts:
                            t = transcripts[0]
                            text_field = t.get("text") or ""
                            sentences = t.get("sentences") or []
                            if sentences:
                                joined = " ".join(s.get("text", "") for s in sentences if isinstance(s, dict))
                                return re.sub(r"<\|/?[^|]+\|>", "", joined).strip()
                            if text_field:
                                return re.sub(r"<\|/?[^|]+\|>", "", text_field).strip()
                        return ""
        except Exception as e:
            print(f"[WARN] SenseVoice failed, fallback to Whisper for {url}: {e}")

    # Whisper 回退（下载音频并上载）
    try:
        def download_audio(u: str) -> bytes:
            response = requests.get(u, timeout=30)
            response.raise_for_status()
            return response.content

        audio_content = await asyncio.to_thread(download_audio, url)
        audio_file = io.BytesIO(audio_content)
        audio_file.name = "audio.mp3"
        client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        transcription = await client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )
        # 兼容 text 或 raw string
        return transcription.text if hasattr(transcription, "text") else str(transcription).strip()
    except Exception as e:
        print(f"[ERROR] Audio transcription failed for URL {url}: {e}")
        return f"[语音处理失败: {url}]"

async def process_audio_urls_to_text(urls: List[str]) -> List[str]:
    """
    Processes a list of audio URLs in parallel and returns their transcriptions.

    Args:
        urls: A list of MP3 file URLs.

    Returns:
        A list of transcription texts.
    """
    if not urls:
        return []
    
    tasks = [transcribe_audio_from_url(url) for url in urls]
    transcriptions = await asyncio.gather(*tasks)
    return transcriptions 