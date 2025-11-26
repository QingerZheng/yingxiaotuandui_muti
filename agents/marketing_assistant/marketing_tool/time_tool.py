"""时间工具模块

提供获取当前时间、日期等信息的工具函数
"""

import pytz
from datetime import datetime
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List


class TimeInfo(BaseModel):
    """时间信息"""
    current_time: str = Field(description="当前时间")
    current_date: str = Field(description="当前日期")
    weekday: str = Field(description="星期几")


@tool
def get_current_time() -> str:
    """获取当前时间和日期信息"""
    # 获取北京时间
    beijing_tz = pytz.timezone('Asia/Shanghai')
    now = datetime.now(beijing_tz)
    
    # 基本时间信息
    current_time = now.strftime("%H:%M:%S")
    current_date = now.strftime("%Y年%m月%d日")
    weekday = now.strftime("%A")
    
    # 中文星期
    weekday_map = {
        'Monday': '星期一', 'Tuesday': '星期二', 'Wednesday': '星期三',
        'Thursday': '星期四', 'Friday': '星期五', 'Saturday': '星期六', 'Sunday': '星期日'
    }
    weekday_cn = weekday_map.get(weekday, weekday)
    
    time_info = TimeInfo(
        current_time=current_time,
        current_date=current_date,
        weekday=weekday_cn
    )
    
    return time_info.model_dump_json()