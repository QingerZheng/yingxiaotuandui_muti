#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
线程状态查看工具

功能:
1. 获取线程基本信息
2. 获取线程运行状态
3. 显示线程中的消息列表
4. 支持详细和简洁两种显示模式
5. 显示原始API请求响应数据
6. 支持交互式选择显示模式
"""

import requests
import json
from typing import Dict, List, Optional

class ThreadStateViewer:
    """线程状态查看器"""
    
    def __init__(self):
        self.base_url = "https://test-1bed77dd8e49563a9a13ba443f0cce46.us.langgraph.app"
        self.api_key = "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"
        self.headers = {
            'x-api-key': self.api_key,
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json',
            'Accept': '*/*',
            'Host': 'test-1bed77dd8e49563a9a13ba443f0cce46.us.langgraph.app',
            'Connection': 'keep-alive'
        }
    
    def display_raw_response(self, response: requests.Response, request_type: str) -> None:
        """
        显示原始API响应数据
        
        Args:
            response: requests响应对象
            request_type: 请求类型描述
        """
        print(f"\n🔍 原始{request_type}响应数据:")
        print("=" * 80)
        
        # 显示请求信息
        print(f"📤 请求方法: {response.request.method}")
        print(f"📤 请求URL: {response.request.url}")
        print(f"📤 请求头:")
        for key, value in response.request.headers.items():
            # 隐藏敏感信息
            if 'api-key' in key.lower():
                value = value[:10] + "..." + value[-5:] if len(value) > 15 else "***"
            print(f"     {key}: {value}")
        
        if response.request.body:
            print(f"📤 请求体:")
            try:
                body_json = json.loads(response.request.body)
                print(json.dumps(body_json, indent=2, ensure_ascii=False))
            except:
                print(f"     {response.request.body}")
        
        # 显示响应信息
        print(f"\n📥 响应状态码: {response.status_code}")
        print(f"📥 响应头:")
        for key, value in response.headers.items():
            print(f"     {key}: {value}")
        
        print(f"📥 响应体:")
        try:
            response_json = response.json()
            print(json.dumps(response_json, indent=2, ensure_ascii=False))
        except:
            print(f"     {response.text}")
        
        print("=" * 80)
    
    def get_thread_info_with_raw(self, thread_id: str, show_raw: bool = False) -> Optional[Dict]:
        """
        获取线程基本信息（支持显示原始响应）
        
        Args:
            thread_id: 线程ID
            show_raw: 是否显示原始响应数据
            
        Returns:
            线程信息字典，失败时返回None
        """
        url = f"{self.base_url}/threads/{thread_id}"
        
        try:
            print(f"🔍 正在获取线程基本信息...")
            print(f"请求URL: {url}")
            
            response = requests.get(url, headers=self.headers)
            
            if show_raw:
                self.display_raw_response(response, "线程信息")
            
            print(f"响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                if not show_raw:
                    print("✅ 线程信息获取成功!")
                    print(f"📋 线程信息: {json.dumps(result, indent=2, ensure_ascii=False)}")
                return result
            else:
                print(f"❌ 获取线程信息失败，状态码: {response.status_code}")
                print(f"错误信息: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ 获取线程信息时出错: {str(e)}")
            return None
    
    def get_thread_state_with_raw(self, thread_id: str, show_raw: bool = False) -> Optional[Dict]:
        """
        获取线程运行状态（支持显示原始响应）
        
        Args:
            thread_id: 线程ID
            show_raw: 是否显示原始响应数据
            
        Returns:
            线程状态字典，失败时返回None
        """
        url = f"{self.base_url}/threads/{thread_id}/state"
        
        try:
            print(f"🔍 正在获取线程运行状态...")
            print(f"请求URL: {url}")
            
            response = requests.get(url, headers=self.headers)
            
            if show_raw:
                self.display_raw_response(response, "线程状态")
            
            print(f"响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                if not show_raw:
                    print("✅ 线程状态获取成功!")
                    print(f"📋 线程状态: {json.dumps(result, indent=2, ensure_ascii=False)}")
                return result
            else:
                print(f"❌ 获取线程状态失败，状态码: {response.status_code}")
                print(f"错误信息: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ 获取线程状态时出错: {str(e)}")
            return None
    
    def get_thread_info(self, thread_id: str, verbose: bool = False) -> Optional[Dict]:
        """
        获取线程基本信息
        
        Args:
            thread_id: 线程ID
            verbose: 是否显示详细信息
            
        Returns:
            线程信息字典，失败时返回None
        """
        url = f"{self.base_url}/threads/{thread_id}"
        
        try:
            if verbose:
                print(f"🔍 正在获取线程基本信息...")
                print(f"请求URL: {url}")
            
            response = requests.get(url, headers=self.headers)
            
            if verbose:
                print(f"响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                if verbose:
                    print("✅ 线程信息获取成功!")
                    print(f"📋 线程信息: {json.dumps(result, indent=2, ensure_ascii=False)}")
                return result
            else:
                print(f"❌ 获取线程信息失败，状态码: {response.status_code}")
                print(f"错误信息: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ 获取线程信息时出错: {str(e)}")
            return None
    
    def get_thread_state(self, thread_id: str, verbose: bool = False) -> Optional[Dict]:
        """
        获取线程运行状态
        
        Args:
            thread_id: 线程ID
            verbose: 是否显示详细信息
            
        Returns:
            线程状态字典，失败时返回None
        """
        url = f"{self.base_url}/threads/{thread_id}/state"
        
        try:
            if verbose:
                print(f"🔍 正在获取线程运行状态...")
                print(f"请求URL: {url}")
            
            response = requests.get(url, headers=self.headers)
            
            if verbose:
                print(f"响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                if verbose:
                    print("✅ 线程状态获取成功!")
                    print(f"📋 线程状态: {json.dumps(result, indent=2, ensure_ascii=False)}")
                return result
            else:
                print(f"❌ 获取线程状态失败，状态码: {response.status_code}")
                print(f"错误信息: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ 获取线程状态时出错: {str(e)}")
            return None
    
    def get_thread_messages(self, thread_id: str, verbose: bool = False) -> List[Dict]:
        """
        获取线程中的消息列表
        
        Args:
            thread_id: 线程ID
            verbose: 是否显示详细信息
            
        Returns:
            消息列表
        """
        state = self.get_thread_state(thread_id, verbose=False)
        if not state:
            return []
        
        messages = state.get("values", {}).get("messages", [])
        
        if verbose:
            print(f"📝 线程中共有 {len(messages)} 条消息:")
            for i, msg in enumerate(messages, 1):
                msg_type = msg.get("type", "unknown")
                msg_id = msg.get("id", "N/A")
                content = msg.get("content", "")
                content_preview = content[:50] + "..." if len(content) > 50 else content
                print(f"  {i}. [{msg_type}] ID: {msg_id}")
                print(f"     内容: {content_preview}")
        
        return messages
    
    def display_thread_summary(self, thread_id: str):
        """
        显示线程摘要信息
        
        Args:
            thread_id: 线程ID
        """
        print(f"📊 线程摘要: {thread_id}")
        print("=" * 60)
        
        # 获取基本信息
        info = self.get_thread_info(thread_id)
        if info:
            print(f"✅ 线程状态: {info.get('status', 'unknown')}")
            print(f"📅 创建时间: {info.get('created_at', 'N/A')}")
            print(f"🔄 更新时间: {info.get('updated_at', 'N/A')}")
            
            metadata = info.get('metadata', {})
            if metadata:
                print(f"📋 元数据: {json.dumps(metadata, ensure_ascii=False)}")
        
        # 获取消息信息
        messages = self.get_thread_messages(thread_id)
        print(f"💬 消息数量: {len(messages)}")
        
        if messages:
            print(f"\n📝 最近的消息:")
            for msg in messages[-3:]:  # 显示最后3条消息
                msg_type = msg.get("type", "unknown")
                content = msg.get("content", "")
                content_preview = content[:100] + "..." if len(content) > 100 else content
                print(f"  [{msg_type}] {content_preview}")
    
    def load_config(self) -> List[str]:
        """
        从config.json加载线程ID列表
        
        Returns:
            线程ID列表
        """
        try:
            with open("config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
                return config.get("thread_ids", [])
        except Exception as e:
            print(f"❌ 加载配置文件失败: {str(e)}")
            return []
    
    def load_full_config(self) -> Dict:
        """
        从config.json加载完整配置信息
        
        Returns:
            完整配置字典
        """
        try:
            with open("config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
                return config
        except Exception as e:
            print(f"❌ 加载配置文件失败: {str(e)}")
            return {}
    
    def get_config_value(self, key: str, default=None):
        """
        获取配置文件中的特定值
        
        Args:
            key: 配置键名
            default: 默认值
            
        Returns:
            配置值
        """
        config = self.load_full_config()
        return config.get(key, default)

def main():
    """主函数"""
    print("🔍 LangGraph 线程状态查看工具")
    print("=" * 50)
    
    viewer = ThreadStateViewer()
    
    # 加载配置
    thread_ids = viewer.load_config()
    if not thread_ids:
        print("❌ 未找到有效的线程配置信息！")
        print("请确保 config.json 文件存在且包含有效的线程ID")
        return
    
    # 选择显示模式
    print("\n📋 请选择显示模式:")
    print("1. 摘要模式 (默认)")
    print("2. 详细模式")
    print("3. 原始响应模式")
    print("4. 线程信息原始响应")
    print("5. 线程状态原始响应")
    
    try:
        choice = input("\n请输入选择 (1-5, 默认为1): ").strip()
        if not choice:
            choice = "1"
    except KeyboardInterrupt:
        print("\n\n👋 用户取消操作")
        return
    
    # 处理每个线程
    for i, thread_id in enumerate(thread_ids, 1):
        if len(thread_ids) > 1:
            print(f"\n🎯 处理线程 {i}/{len(thread_ids)}: {thread_id}")
        
        if choice == "1":
            # 摘要模式
            viewer.display_thread_summary(thread_id)
        elif choice == "2":
            # 详细模式
            print(f"📊 线程详细信息: {thread_id}")
            print("=" * 60)
            
            # 获取并显示详细的线程信息
            info = viewer.get_thread_info(thread_id, verbose=True)
            print("\n" + "-" * 40)
            
            # 获取并显示详细的线程状态
            state = viewer.get_thread_state(thread_id, verbose=True)
            print("\n" + "-" * 40)
            
            # 获取并显示详细的消息列表
            messages = viewer.get_thread_messages(thread_id, verbose=True)
            
        elif choice == "3":
            # 原始响应模式 - 显示所有原始响应
            print(f"📊 线程原始响应数据: {thread_id}")
            print("=" * 60)
            
            # 获取线程信息的原始响应
            info = viewer.get_thread_info_with_raw(thread_id, show_raw=True)
            
            # 获取线程状态的原始响应
            state = viewer.get_thread_state_with_raw(thread_id, show_raw=True)
            
        elif choice == "4":
            # 仅显示线程信息的原始响应
            print(f"📊 线程信息原始响应: {thread_id}")
            print("=" * 60)
            
            info = viewer.get_thread_info_with_raw(thread_id, show_raw=True)
            
        elif choice == "5":
            # 仅显示线程状态的原始响应
            print(f"📊 线程状态原始响应: {thread_id}")
            print("=" * 60)
            
            state = viewer.get_thread_state_with_raw(thread_id, show_raw=True)
            
        else:
            print(f"❌ 无效的选择: {choice}，使用默认摘要模式")
            viewer.display_thread_summary(thread_id)
        
        # 如果有多个线程，添加分隔符
        if i < len(thread_ids):
            print("\n" + "="*80 + "\n")
    
    print(f"\n🎉 完成！共处理了 {len(thread_ids)} 个线程")

if __name__ == "__main__":
    main()