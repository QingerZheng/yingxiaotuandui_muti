import os
from typing import Optional

def load_prompt(name: str, include_base_context: bool = True, custom_base_context: Optional[str] = None) -> str:
    """
    从 prompts/ 目录加载一个prompt文本文件。硬文末拼接，而不是自然融合，与自定义agent仍有差距
    对应需求中的人设调整部分。

    Args:
        name (str): prompt的名称，对应文件名（不含扩展名）。
                    例如, name='state_evaluator' 将加载 'state_evaluator.txt'。
        include_base_context (bool): 是否包含base_context.txt的内容，默认为True。
                                    RAG等场景可以设置为False避免拼接。

    Returns:
        str: prompt文件的内容。
        
    Raises:
        FileNotFoundError: 如果对应的prompt文件不存在。
    """
    # 获取当前文件所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 智能处理文件名，如果用户传入的文件名不带.txt，则自动补全
    if not name.endswith('.txt'):
        name_with_ext = f"{name}.txt"
    else:
        name_with_ext = name
    
    # 构造完整的提示词文件路径
    prompt_path = os.path.join(current_dir, name_with_ext)
    
    if not os.path.exists(prompt_path):
        # 增加更详细的错误提示
        raise FileNotFoundError(f"Error: Prompt file not found at {prompt_path}. Please ensure '{name_with_ext}' exists in the 'prompts' directory.")

    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_body = f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {prompt_path}")
        raise

    # 如果存在全局上下文且需要拼接，自动拼接
    if custom_base_context is not None:
        base_context = custom_base_context
    elif include_base_context:
        base_path = os.path.join(current_dir, "base_context.txt")
        if os.path.exists(base_path):
            try:
                with open(base_path, 'r', encoding='utf-8') as f:
                    base_context = f.read().strip()
            except Exception:
                base_context = ""
    else:
        base_context = ""
    return f"{base_context}\n\n{prompt_body}" if base_context else prompt_body 