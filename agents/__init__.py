"""
Agent模块
包含从移植的各种智能体
"""

__version__ = "0.1.0"

# 使用延迟导入避免在模块导入时初始化模型
def get_wechat_moment_graph():
    """获取微信朋友圈Agent"""
    from .wechat import wechat_moment_graph
    return wechat_moment_graph

def get_user_analysis_graph():
    """获取用户分析报告Agent"""
    from .analysis_report import user_analysis_graph
    return user_analysis_graph

def get_profile_label_graph():
    """获取用户画像标签Agent"""
    from .user_profile import profile_label_graph
    return profile_label_graph

def get_profile_graph():
    """获取用户画像Agent"""
    from .user_profile import profile_graph
    return profile_graph

def get_comment_analysis_graph():
    """获取朋友圈评论Agent"""
    from .post_comments import comment_analysis_graph
    return comment_analysis_graph

# 为了向后兼容，提供直接访问的属性
class _LazyAgentLoader:
    @property
    def wechat_moment_graph(self):
        return get_wechat_moment_graph()
    
    @property
    def user_analysis_graph(self):
        return get_user_analysis_graph()
    
    @property
    def profile_label_graph(self):
        return get_profile_label_graph()
    
    @property
    def profile_graph(self):
        return get_profile_graph()
    
    @property
    def comment_analysis_graph(self):
        return get_comment_analysis_graph()

# 创建延迟加载器实例
_agents = _LazyAgentLoader()

# 为了模块级别的访问，重新定义这些属性
def __getattr__(name):
    if name in ["wechat_moment_graph", "user_analysis_graph", "profile_label_graph", 
                "profile_graph", "comment_analysis_graph"]:
        return getattr(_agents, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "get_wechat_moment_graph",
    "get_user_analysis_graph",
    "get_profile_label_graph", 
    "get_profile_graph",
    "get_comment_analysis_graph",
    "wechat_moment_graph",
    "user_analysis_graph",
    "profile_label_graph",
    "profile_graph",
    "comment_analysis_graph"
] 