from typing import List, Dict, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from Configurations import Configuration
from tools import web_search
from states import AgentState
import asyncio
# 定义输出状态
class Output(TypedDict):
    """图的输出状态 - 只包含最终回复"""
    used_tools: Optional[List[Dict[str, str]]]  # 工具调用列表，如 [{"tool": "rag", "reason": "查询原因..."}, {"tool": "search", "reason": "搜索原因..."}]
    tool_results: Optional[List[Dict[str, str]]]  # 工具执行结果列表，如 [{"tool": "rag", "result": "查询结果..."}, {"tool": "search", "result": "搜索结果..."}]


class ToolRegistry:
    """工具注册表"""
    def __init__(self):
        self.tools = {
            "search": web_search,  # 联网搜索外部知识
        }
    def get_tool(self, tool_name: str):
        return self.tools.get(tool_name)
# 全局工具注册表
tool_registry = ToolRegistry()
def intelligent_tool_planning(state: AgentState):
    """智能工具规划节点 - 使用LLM分析用户输入并生成工具执行列表"""
    print(f"\n🧠 === 智能工具规划开始 ===")

    # 获取从最新消息开始连续的人类消息，直到遇到AI消息
    user_messages = []
    for msg in reversed(state["long_term_messages"]):
        if isinstance(msg, HumanMessage):
            user_messages.insert(0, msg.content)  # 保持原有顺序
        elif isinstance(msg, AIMessage):
            break  # 遇到AI消息就停止
    user_message = "".join(user_messages) if user_messages else ""
    if user_message=="":
        print("最近没有人类消息，可能是主动聊天触发的。")
        return {"used_tools": [], "tool_results": []}
    else:
        print(f"💬 最近用户消息: {user_message}")

        # 初始化LLM，仅使用运行时配置
        from agents.persona_config.config_manager import config_manager
        cfg = config_manager.get_config() or {}
        from llm import create_llm
        llm = create_llm(
            model_provider=cfg.get("model_provider", "openrouter"),
            model_name=cfg.get("planning_model", cfg.get("model_name", "x-ai/grok-code-fast-1")),
            temperature=0.0
        )

        # 设计工具规划prompt
        system_prompt = """
你是一个专业的医美客服AI的工具调用决策中枢。你的核心任务是判断当前用户的提问是否超出了你预设的知识库和角色定位，并决定是否需要调用网络搜索工具来补充信息。

# 你的已知信息与角色定位
- **身份**: 你是 ""，一名。
- **核心知识库**: 
- **对话目标**: 你的首要目标是邀约客户到店，而不是解答世界上的所有问题。
- **人设**: 专业、亲切，但知识有边界。对于专业外的问题，你会像普通人一样表示不了解，而不是立刻去查。

# 工具清单
- `search`: 网络搜索工具。用于查询你知识库之外的、但与医美领域高度相关的最新、或非常具体的信息。

# 调用规则 (请严格遵守)
1.  **优先使用已知信息**：如果问题能通过你的核心知识库回答（如“你们的光子嫩肤多少钱一次？”或“你们地址在哪？”），绝对不要调用搜索。返回 `[]`。
2.  **坚决拒绝无关搜索**：对于和医美、护肤、公司业务完全无关的常识性问题（如做菜、天气、新闻、娱乐八卦等），坚决不能调用搜索。这不符合你的客服人设。你需要自然地回复表示不了解。返回 `[]`。
3.  **谨慎处理相关领域问题**：对于医美护肤相关，但超出了你的核心知识库的具体问题，才考虑调用搜索。目的是为了更好地服务客户，对比信息，并最终引导回我们自己的项目。

# 示例分析 (关键)

### 场景一：无需调用 (返回 [])
- **用户**: 你好呀，今天过得怎么样？
  - **判断**: 纯粹的闲聊，无需联网。
  - **返回**: `[]`
- **用户**: 你们的Fotona4D效果怎么样？
  - **判断**: 核心业务问题，你的知识库里有详细资料。
  - **返回**: `[]`
- **用户**: 做鱼的正确步骤是什么？
  - **判断**: 完全无关的领域。一个医美客服不应该知道这个。她应该回复“这个我不太清楚耶，我平时不做饭”，而不是去搜索。
  - **返回**: `[]`
- **用户**: 今天{{}}天气怎么样？
  - **判断**: 虽然和客户到店有关，但客服人设更倾向于提醒客户“您出门前可以看看天气预报哦”，而不是自己去查。保持人设的局限性。
  - **返回**: `[]`
- **用户**: 最近哪种医美项目最火？
  - **判断**: 这是一个行业趋势问题。作为一个专业顾问，你应该基于自己的认知和公司主推的项目来回答，而不是去实时搜索。搜索会显得你很不专业。
  - **返回**: `[]`

### 场景二：需要调用 (返回 search)
- **用户**: 我听说最近有个叫{{}}的东西，你们有吗？它和你们的玻尿酸有什么区别？
  - **判断**: {{}}是一个具体的、可能较新的产品名，知识库里没有。为了和我们自己的玻尿酸做专业对比，需要查询其成分和原理。
  - **返回**: `[{"tool": "search", "reason": "查询'{{}}'的成分、作用原理，以便和玻尿酸进行对比"}]`
- **用户**: AestheFill这个牌子和你们用的产品比怎么样？
  - **判断**: 这是一个客户提到的、我们知识库里没有的具体竞品品牌。需要查询以进行专业解答。
  - **返回**: `[{"tool": "search", "reason": "查询医美品牌AestheFill的产品特点和技术"}]`

# 输出格式
请严格只输出 JSON 数组（可以为空），不要添加任何解释性文字。"""
    # 构建messages
    planning_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"用户消息：{user_message}", additional_kwargs={"send_style": "text"})
    ]
    try:
        print(f"🚀 调用LLM进行工具规划...")
        response = llm.invoke(planning_messages)
        tool_plan_text = response.content.strip()
        print(f"📝 LLM原始响应: {tool_plan_text}")
        # 解析JSON响应
        import json
        from json_parser_utils import robust_json_parse, create_fallback_dict
        
        print(f"[DEBUG-工具规划] 原始模型响应: {tool_plan_text}")
        
        # 使用鲁棒的JSON解析工具
        planned_tools_raw = robust_json_parse(
            tool_plan_text, 
            context="工具规划", 
            fallback_dict=[],
            debug=True
        )
        # 只保留合法的 search 工具项
        planned_tools = [
            {"tool": "search", "reason": str(item.get("reason", "")).strip()}
            for item in (planned_tools_raw if isinstance(planned_tools_raw, list) else [])
            if isinstance(item, dict) and item.get("tool") == "search" and str(item.get("reason", "")).strip()
        ]
        print(f"🔧 解析并过滤后的工具列表: {planned_tools}")
    except Exception as e:
        print(f"❌ LLM工具规划失败: {e}")
        return {"used_tools": [], "tool_results": []}
    print(f"🎯 最终工具规划: {planned_tools}")
    print(f"🧠 === 智能工具规划结束 ===\n")
    return {"used_tools": planned_tools, "tool_results": []}

async def parallel_tools_execution_node(state: AgentState):
    """并行执行工具节点 - 异步调用工具列表中的工具，获得tool_results"""
    print(f"\n🔧 === 并行工具执行开始 ===")
    
    used_tools = state.get("used_tools", [])
    
    # 情况1: 空工具列表
    if not used_tools or used_tools is None or len(used_tools) == 0:
        print("📝 没有需要执行的工具")
        return {"tool_results": []}
    
    # 情况2: 有工具需要执行
    print(f"📋 需要执行的工具: {len(used_tools)}个")
    print("🔍 工具分析: 仅支持 Search 工具")
    
    # 打印每个工具的详细信息
    for i, tool_info in enumerate(used_tools):
        print(f"  {i+1}. {tool_info['tool']}: {tool_info['reason']}")
    
    # 并行执行所有工具
    async def execute_tool(tool_info):
        tool_name = tool_info["tool"]
        reason = tool_info["reason"]
        
        print(f"🚀 开始执行工具: {tool_name} ({reason})")
        
        try:
            tool_func = tool_registry.get_tool(tool_name)
            if not tool_func:
                print(f"❌ 工具 {tool_name} 未找到")
                return {
                    "tool": tool_name,
                    "result": f"工具 {tool_name} 未找到"
                }
            
            # 根据工具类型调用，使用reason作为查询内容
            if tool_name == "search":
                print(f"  🌐 执行网络搜索: {reason}")
                result = await asyncio.to_thread(tool_func.invoke, {"query": reason, "max_results": 3})
            else:
                print(f"  ⚙️  执行未知工具: {tool_name}")
                result = await asyncio.to_thread(tool_func.invoke, reason)
            
            print(f"✅ 工具 {tool_name} 执行成功")
            return {
                "tool": tool_name,
                "result": str(result)
            }
            
        except Exception as e:
            print(f"❌ 工具 {tool_name} 执行失败: {e}")
            return {
                "tool": tool_name,
                "result": f"执行失败: {str(e)}"
            }
    
    # 并行执行所有工具
    print(f"🔄 开始并行执行 {len(used_tools)} 个工具...")
    tool_results = await asyncio.gather(*[execute_tool(tool_info) for tool_info in used_tools])
    
    # 统计执行结果
    successful_tools = [r for r in tool_results if r['result'] and not r['result'].startswith('执行失败')]
    failed_tools = [r for r in tool_results if not r['result'] or r['result'].startswith('执行失败')]
    
    print(f"✅ 工具执行完成统计:")
    print(f"  - 成功: {len(successful_tools)}个")
    print(f"  - 失败: {len(failed_tools)}个")
    
    for result in tool_results:
        status = "✅" if result['result'] and not result['result'].startswith('执行失败') else "❌"
        print(f"  {status} {result['tool']}: {len(result['result'])}字符")
    
    print(f"🔧 === 并行工具执行结束 ===\n")
    return {"tool_results": tool_results}

# ===== 主要节点函数 =====

def create_outside_info_workflow():
    """创建外部信息查询工作流"""
    # 创建主图
    outside_info_graph = StateGraph(AgentState, output=Output)
    
    # 添加节点
    outside_info_graph.add_node("planning", intelligent_tool_planning)
    outside_info_graph.add_node("parallel_tools_execution", parallel_tools_execution_node)  #并行执行工具，获得tool_results
    
    # 添加边
    outside_info_graph.add_edge(START, "planning")
    outside_info_graph.add_edge("planning", "parallel_tools_execution")
    outside_info_graph.add_edge("parallel_tools_execution", END)  # 直接结束
    
    # 编译并返回
    return outside_info_graph.compile()