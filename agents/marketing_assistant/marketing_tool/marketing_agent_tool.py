"""营销文案生成工具模块

将营销文案生成agent包装成工具函数，供react_agent调用
"""

import json
import re
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from typing import Dict, Any, Optional
from llm import create_llm
from ..persona_prompt_template import MarketingCopyPromptTemplate


@tool
def marketing_copy_generator(requirement: str, conversation_memory: str = "", min_word_count: int = 50, use_previous_copy: bool = False) -> str:
    """生成营销文案的工具函数。
    
    Args:
        requirement: 文案需求描述
        conversation_memory: 用户近几轮的对话上下文
        min_word_count: 最小字数要求，默认100字
        use_previous_copy: 是否使用之前的文案作为参考，默认False
        
    Returns:
        str: 生成的营销文案
    """
    print(f"📝 工具调用参数 - requirement: {requirement[:50]}...")  # 打印1
    print(f"💬 对话上下文参数 - conversation_memory 完整结构: {conversation_memory}")  # 打印2 - 查看完整结构
    # print(f"use_previous_copy: {use_previous_copy}")
    """生成营销文案的工具，根据需求生成不同风格的营销文案
    
    Args:
        requirement: 营销文案需求描述
        min_word_count: 每条文案的最少字数要求，默认150字
        use_previous_copy: 是否使用之前生成的文案进行修改，默认False
    """
    
    # print(f"📝 开始生成营销文案: {requirement}")
    
    try:
        # 使用传入的对话上下文
        print(f"🔄 处理后的上下文长度: {len(conversation_memory)} 字符")  # 打印3
        
        
        # 构建完整提示词 - 优化组织结构和逻辑顺序
        prompt_parts = []
        
        # 1. 角色定义 - 建立AI身份和专业背景
        prompt_parts.append(MarketingCopyPromptTemplate.COMMON_PROMPTS["role_definition"])
        
        # 2. 使用场景和修改规则 - 明确任务类型和执行规则
        if use_previous_copy:
            prompt_parts.append(f"\n\n{MarketingCopyPromptTemplate.OLD_COPY_PROMPTS['usage_scenario']}")
            prompt_parts.append(f"\n{MarketingCopyPromptTemplate.OLD_COPY_PROMPTS['modification_types']}")
            prompt_parts.append(f"\n{MarketingCopyPromptTemplate.OLD_COPY_PROMPTS['reference_mapping']}")
        
        # 3. 对话上下文 - 提供历史文案信息（在需求之前，便于理解用户指代）
        if conversation_memory:
            prompt_parts.append(f"\n\n## 对话上下文\n以下是之前的对话记录，包含已生成的营销文案：\n{conversation_memory}")
        
        # 4. 核心需求 - 用户的具体要求
        prompt_parts.append(f"\n\n## 用户需求\n{requirement}")
        
        # 5. 内容创作要求 - 文案质量和风格标准
        prompt_parts.append(f"\n\n{MarketingCopyPromptTemplate.COMMON_PROMPTS['content_requirements']}")
        
        # 6. 字数要求 - 动态处理字数限制
        if min_word_count != 50:  # 只在非默认值时添加字数要求
            prompt_parts.append(f"\n\n## 字数要求\n每条文案字数不少于{min_word_count}字")
        
        # 7. 输出格式和任务指令
        if use_previous_copy:
            prompt_parts.append(f"\n\n{MarketingCopyPromptTemplate.OLD_COPY_PROMPTS['task_instruction']}")
        else:
            prompt_parts.append(f"\n\n{MarketingCopyPromptTemplate.NEW_COPY_PROMPTS['task_instruction']}")
        
        # 8. 格式要求 - 确保输出规范
        prompt_parts.append(f"\n\n{MarketingCopyPromptTemplate.COMMON_PROMPTS['format_requirements']}")
        
        # 9. 示例参考 - 仅在新文案生成时提供
        if not use_previous_copy:
            prompt_parts.append(f"\n\n{MarketingCopyPromptTemplate.NEW_COPY_PROMPTS['examples']}")
        
        prompt = "".join(prompt_parts)
        
        # 调用LLM生成营销文案
        llm = create_llm("openrouter", "openai/chatgpt-4o-latest", temperature=0.7)
        response = llm.invoke(prompt)
        
        # 解析生成的文案并返回结构化数据
        content = response.content.strip()
        # print(f"✅ 营销文案生成成功")
        # print(f"🔍 LLM原始输出: {content[:200]}...")
        
        # 解析三个文案
        copies = {}
        lines = content.split('\n')
        current_copy = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('1. '):
                current_copy = 'copy_1'
                current_content = [line[3:]]  # 去掉"1. "
            elif line.startswith('2. '):
                if current_copy == 'copy_1' and current_content:
                    copies['copy_1'] = '\n'.join(current_content).strip()
                current_copy = 'copy_2'
                current_content = [line[3:]]  # 去掉"2. "
            elif line.startswith('3. '):
                if current_copy == 'copy_2' and current_content:
                    copies['copy_2'] = '\n'.join(current_content).strip()
                current_copy = 'copy_3'
                current_content = [line[3:]]  # 去掉"3. "
            elif current_copy and line:
                current_content.append(line)
        
        # 处理最后一个文案
        if current_copy and current_content:
            copies[current_copy] = '\n'.join(current_content).strip()
        
        # 将文案转换为数组格式
        marketing_copies_array = [
            {"id": "copy_1", "content": copies.get("copy_1", "")},
            {"id": "copy_2", "content": copies.get("copy_2", "")},
            {"id": "copy_3", "content": copies.get("copy_3", "")}
        ]
        
        # 返回JSON格式的结果，包含原始内容和解析后的文案
        result = {
            "raw_content": content,
            "marketing_copies": marketing_copies_array
        }
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        error_msg = f"营销文案生成出现错误：{str(e)}"
        # print(f"❌ {error_msg}")
        # 返回JSON格式的错误结果
        error_result = {
            "raw_content": error_msg,
            "marketing_copies": []
        }
        return json.dumps(error_result, ensure_ascii=False)