#!/usr/bin/env python3
"""
LangGraph 线程重建工具

功能流程:
1. 查看当前线程状态
2. 删除线程
3. 创建同名线程
4. 根据初始状态重新patch（保留消息ID结构但清空内容）

主要特性:
- 支持批量重建多个线程
- 保留原始消息ID结构
- 自动验证重建结果
- 详细的进度显示和错误处理
"""

import requests
import json
import os
import time
from typing import Dict, Optional, Any, List
from thread_state import ThreadStateViewer

class ThreadRebuilder(ThreadStateViewer):
    """线程重建工具类
    
    继承自ThreadStateViewer，提供完整的线程重建功能
    """
    
    # 配置常量
    WAIT_AFTER_DELETE = 2  # 删除后等待时间（秒）
    WAIT_AFTER_CREATE = 3  # 创建后等待时间（秒）
    WAIT_BETWEEN_THREADS = 3  # 线程间等待时间（秒）
    
    def __init__(self):
        super().__init__()  # 继承 ThreadStateViewer 的初始化
        # 从配置文件读取 assistant_id
        self.DEFAULT_ASSISTANT_ID = self.get_config_value("assistant_id", "{{}}")
    
    def _log_step(self, step_num: int, description: str, url: str = None) -> None:
        """统一的步骤日志输出"""
        print(f"\n🔧 步骤{step_num}: {description}...")
        if url:
            print(f"请求URL: {url}")
    
    def _handle_response(self, response: requests.Response, success_message: str, 
                        error_message: str) -> bool:
        """统一的响应处理"""
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code in [200, 201, 204]:
            print(f"✅ {success_message}")
            return True
        else:
            print(f"❌ {error_message}，状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
            return False
    
    def delete_thread(self, thread_id: str) -> bool:
        """
        步骤2: 删除线程
        
        Args:
            thread_id: 要删除的线程ID
            
        Returns:
            bool: 删除是否成功
        """
        try:
            url = f"{self.base_url}/threads/{thread_id}"
            self._log_step(2, "正在删除线程", url)
            
            response = requests.delete(url, headers=self.headers)
            
            if response.status_code in [200, 204]:
                print(f"✅ 成功删除线程 {thread_id}")
                return True
            elif response.status_code == 404:
                print(f"⚠️ 线程 {thread_id} 不存在，可能已被删除")
                return True
            else:
                return self._handle_response(response, "", "删除线程失败")
                
        except Exception as e:
            print(f"❌ 删除线程时出错: {str(e)}")
            return False
    
    def create_thread(self, thread_id: str, original_state: Dict) -> bool:
        """
        步骤3: 创建同名线程（只保留指定的状态字段）
        
        Args:
            thread_id: 要创建的线程ID
            original_state: 原始线程状态
            
        Returns:
            bool: 创建是否成功
        """
        try:
            url = f"{self.base_url}/threads"
            self._log_step(3, "正在创建新线程", url)
            
            # 只保留指定的状态字段
            # 构造创建线程的payload（只包含指定的基本信息）
            payload = {
                "thread_id": thread_id,
                "created_at": original_state.get("created_at"),
                "updated_at": original_state.get("updated_at"),
                "metadata": {
                    "assistant_id": self.DEFAULT_ASSISTANT_ID
                },
                "status": "idle",
                "config": {},
                "values": None,
                "interrupts": {},
                "error": None
            }
            
            # 如果原始状态中有metadata，使用原始的assistant_id
            if original_state.get("metadata") and original_state.get("metadata").get("assistant_id"):
                payload["metadata"]["assistant_id"] = original_state["metadata"]["assistant_id"]
            
            print(f"📤 请求体:")
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            
            response = requests.post(url, headers=self.headers, json=payload)
            
            if response.status_code in [200, 201]:
                new_thread = response.json()
                print(f"✅ 成功创建新线程")
                print(f"📋 新线程信息:")
                print(json.dumps(new_thread, indent=2, ensure_ascii=False))
                return True
            elif response.status_code == 409:
                print(f"⚠️ 线程 {thread_id} 已存在，继续执行后续步骤...")
                return True
            else:
                return self._handle_response(response, "", "创建线程失败")
                
        except Exception as e:
            print(f"❌ 创建线程时出错: {str(e)}")
            return False
    
    def _extract_state_without_messages(self, original_state: Dict) -> Dict:
        """
        提取指定的状态字段
        
        Args:
            original_state: 原始线程状态
            
        Returns:
            Dict: 只包含指定状态字段的字典
        """
        print(f"📝 正在提取指定的状态字段...")
        
        # 只保留指定的状态字段
        preserved_state = {
            "thread_id": original_state.get("thread_id"),
            "created_at": original_state.get("created_at"),
            "updated_at": original_state.get("updated_at"),
            "metadata": {
                "assistant_id": original_state.get("metadata", {}).get("assistant_id", self.DEFAULT_ASSISTANT_ID)
            },
            "status": "idle",
            "config": {},
            "values": None,
            "interrupts": {},
            "error": None
        }
        
        # 统计保留的字段
        preserved_fields = [
            "thread_id", "created_at", "updated_at", "metadata.assistant_id",
            "status", "config", "values", "interrupts", "error"
        ]
        
        print(f"   - 保留的字段: {', '.join(preserved_fields)}")
        print(f"   - 总计保留 {len(preserved_fields)} 个字段")
        
        return preserved_state
    
    def _process_messages(self, original_messages: List[Dict]) -> List[Dict]:
        """
        处理消息列表：不再保留消息结构
        
        Args:
            original_messages: 原始消息列表
            
        Returns:
            List[Dict]: 空消息列表
        """
        if not original_messages:
            print(f"📝 原线程无消息，保持空消息列表")
            return []
        
        print(f"📝 原线程有 {len(original_messages)} 条消息，但不再保留消息结构")
        print(f"   - 根据需求，不保留任何消息结构")
        
        # 返回空列表，不保留任何消息
        return []
    
    def patch_thread_state(self, thread_id: str, original_state: Dict) -> bool:
        """
        步骤4: 通过POST操作设置指定的状态字段
        
        Args:
            thread_id: 线程ID
            original_state: 原始线程状态
            
        Returns:
            bool: POST是否成功
        """
        try:
            url = f"{self.base_url}/threads/{thread_id}/state"
            self._log_step(4, "正在通过POST设置线程状态", url)
            
            # 由于在创建线程时已经设置了所有需要的状态字段，这里不需要再设置
            print(f"ℹ️ 所有必要的状态字段已在创建线程时设置，跳过POST操作")
            
            # 记录状态字段
            print(f"📝 已设置的状态字段:")
            print(f"   - thread_id: {thread_id}")
            print(f"   - created_at: {original_state.get('created_at')}")
            print(f"   - updated_at: {original_state.get('updated_at')}")
            print(f"   - metadata.assistant_id: {original_state.get('metadata', {}).get('assistant_id', self.DEFAULT_ASSISTANT_ID)}")
            print(f"   - status: idle")
            print("   - config: {}")
            print(f"   - values: None")
            print("   - interrupts: {}")
            print(f"   - error: None")
            
            return True
                
        except Exception as e:
            print(f"❌ 处理线程状态时出错: {str(e)}")
            return False
    
    def _verify_rebuild_result(self, thread_id: str) -> bool:
        """
        验证重建结果
        
        Args:
            thread_id: 线程ID
            
        Returns:
            bool: 验证是否成功
        """
        print(f"\n🔍 最终验证: 检查重建后的线程...")
        
        # 验证1: 检查线程基本信息
        thread_info = self.get_thread_info(thread_id, verbose=False)
        if not thread_info:
            print(f"❌ 无法获取线程基本信息")
            return False
        
        print(f"✅ 线程基本信息验证成功:")
        print(f"   - 线程ID: {thread_info.get('thread_id')}")
        print(f"   - 创建时间: {thread_info.get('created_at')}")
        print(f"   - 更新时间: {thread_info.get('updated_at')}")
        print(f"   - 状态: {thread_info.get('status')}")
        print(f"   - 元数据: {json.dumps(thread_info.get('metadata', {}), ensure_ascii=False)}")
        
        # 验证2: 检查线程运行状态
        final_state = self.get_thread_state(thread_id, verbose=False)
        if not final_state:
            print(f"❌ 验证失败: 无法获取重建后的线程运行状态")
            return False
        
        # 验证指定的状态字段
        print(f"✅ 线程状态验证:")
        print(f"   - thread_id: {final_state.get('thread_id') == thread_id}")
        print(f"   - status: {final_state.get('status') == 'idle'}")
        print(f"   - config: {final_state.get('config') == {}}")
        print(f"   - values: {final_state.get('values') is None}")
        print(f"   - interrupts: {final_state.get('interrupts') == {}}")
        print(f"   - error: {final_state.get('error') is None}")
        
        # 安全地检查metadata
        metadata = final_state.get('metadata')
        if metadata is not None:
            assistant_id_exists = metadata.get('assistant_id') is not None
        else:
            assistant_id_exists = False
        print(f"   - metadata.assistant_id: {assistant_id_exists}")
        
        return True
    
    def rebuild_thread(self, thread_id: str) -> bool:
        """
        执行完整的线程重建流程
        
        Args:
            thread_id: 要重建的线程ID
            
        Returns:
            bool: 重建是否成功
        """
        print(f"🚀 开始重建线程: {thread_id}")
        print("=" * 60)
        
        # 步骤1: 获取当前线程状态
        print(f"🔍 步骤1: 正在获取线程状态...")
        original_state = self.get_thread_state(thread_id)
        if not original_state:
            print(f"❌ 无法获取线程状态，重建失败")
            return False
        
        # 打印获取到的线程状态信息
        print(f"✅ 成功获取线程状态")
        print(f"📋 当前线程状态:")
        print(json.dumps(original_state, indent=2, ensure_ascii=False))
        
        # 显示消息数量统计
        messages = original_state.get("values", {}).get("messages", [])
        print(f"📝 当前线程包含 {len(messages)} 条消息")
        if messages:
            print(f"   - 最早消息: {messages[0].get('id', 'unknown')} (类型: {messages[0].get('type', 'unknown')})")
            print(f"   - 最新消息: {messages[-1].get('id', 'unknown')} (类型: {messages[-1].get('type', 'unknown')})")
        
        time.sleep(1)  # 短暂等待
        
        # 步骤2: 删除线程
        if not self.delete_thread(thread_id):
            print(f"❌ 删除线程失败，重建终止")
            return False
        
        time.sleep(self.WAIT_AFTER_DELETE)  # 等待删除完成
        
        # 步骤3: 创建同名线程
        if not self.create_thread(thread_id, original_state):
            print(f"❌ 创建新线程失败，重建失败")
            return False
        
        time.sleep(self.WAIT_AFTER_CREATE)  # 等待创建完成
        
        # 步骤4: 重新patch状态
        if not self.patch_thread_state(thread_id, original_state):
            print(f"❌ Patch线程状态失败，但线程已创建")
            return False
        
        print(f"\n🎉 线程 {thread_id} 重建完成!")
        print("=" * 60)
        
        # 最终验证
        return self._verify_rebuild_result(thread_id)
    
    def _display_summary(self, success_count: int, fail_count: int, total_count: int) -> None:
        """
        显示批量重建结果摘要
        
        Args:
            success_count: 成功数量
            fail_count: 失败数量
            total_count: 总数量
        """
        print(f"\n📊 批量重建完成:")
        print(f"  - 成功重建: {success_count}")
        print(f"  - 重建失败: {fail_count}")
        print(f"  - 总计处理: {total_count}")
        
        if success_count == total_count:
            print(f"\n🎉 所有线程重建成功！")
        elif success_count > 0:
            print(f"\n⚠️ 部分线程重建成功，请检查失败的线程")
        else:
            print(f"\n❌ 所有线程重建失败，请检查配置和网络连接")
    
    def batch_rebuild(self, thread_ids: List[str]) -> Dict[str, int]:
        """
        批量重建线程
        
        Args:
            thread_ids: 要重建的线程ID列表
            
        Returns:
            Dict[str, int]: 包含成功和失败数量的统计
        """
        print(f"\n📋 将要重建以下 {len(thread_ids)} 个线程:")
        for i, thread_id in enumerate(thread_ids, 1):
            print(f"  {i}. {thread_id}")
        
        print(f"\n🚀 开始自动重建...")
        
        success_count = 0
        fail_count = 0
        
        for i, thread_id in enumerate(thread_ids, 1):
            print(f"\n--- 重建线程 {i}/{len(thread_ids)} ---")
            
            if self.rebuild_thread(thread_id):
                success_count += 1
                print(f"✅ 线程 {thread_id} 重建成功")
            else:
                fail_count += 1
                print(f"❌ 线程 {thread_id} 重建失败")
            
            # 避免请求过于频繁
            if i < len(thread_ids):
                print(f"⏳ 等待{self.WAIT_BETWEEN_THREADS}秒后处理下一个线程...")
                time.sleep(self.WAIT_BETWEEN_THREADS)
        
        return {"success": success_count, "fail": fail_count}

def main():
    """主函数 - 自动执行模式"""
    print("🔄 LangGraph 线程重建工具 - 自动执行模式")
    print("=" * 50)
    
    rebuilder = ThreadRebuilder()
    
    # 加载配置
    thread_ids = rebuilder.load_config()
    if not thread_ids:
        print("❌ 未找到有效的线程配置信息！")
        return
    
    # 执行批量重建
    result = rebuilder.batch_rebuild(thread_ids)
    
    # 显示结果摘要
    rebuilder._display_summary(result["success"], result["fail"], len(thread_ids))

if __name__ == "__main__":
    main()