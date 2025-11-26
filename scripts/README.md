<<<<<<< HEAD
# APIFOX
=======
# LangGraph 线程重建工具 - 独立版本

这是一个独立的线程重建工具包，包含了运行 `rebuild_thread.py` 所需的所有文件。

## 文件说明

- `rebuild_thread.py` - 主要的线程重建工具
- `thread_state.py` - 线程状态查看器基类
- `config.json` - 配置文件，包含API密钥和线程ID列表
- `requirements.txt` - Python依赖包列表
- `README.md` - 本说明文件

## 功能特性

- 支持批量重建多个线程
- 只保留指定的状态字段（不保留消息结构）
- 自动验证重建结果
- 详细的进度显示和错误处理

## 保留的状态字段

重建线程时，只保留以下状态字段：

```json
{
   "thread_id": "...", 
   "created_at": "...", 
   "updated_at": "...", 
   "metadata": { 
     "assistant_id": "..."
   }, 
   "status": "idle", 
   "config": {}, 
   "values": null, 
   "interrupts": {}, 
   "error": null 
}
```

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置设置

编辑 `config.json` 文件，确保包含正确的配置信息：

```json
{
  "assistant_id": "your-assistant-id",
  "graph_id": "agent",
  "thread_ids": [
    "thread-id-1",
    "thread-id-2",
    "..."
  ]
}
```

### 3. 运行工具

```bash
python rebuild_thread.py
```

## 工作流程

1. **获取线程状态** - 读取当前线程的完整状态信息
2. **删除线程** - 删除现有线程
3. **创建新线程** - 使用相同ID创建新线程
4. **重置状态** - 恢复除消息外的所有状态，消息内容被清空
5. **验证结果** - 确认重建操作成功完成

## 注意事项

- 确保API密钥和端点URL正确配置
- 重建操作会清空线程中的所有消息内容
- 建议在重建前备份重要数据
- 工具会自动处理批量操作的间隔时间，避免请求过于频繁

## 错误处理

工具包含完善的错误处理机制：
- 网络连接错误
- API响应错误
- 配置文件错误
- 线程不存在等情况

如遇到问题，请检查：
1. 网络连接是否正常
2. API密钥是否有效
3. 线程ID是否存在
4. 配置文件格式是否正确
>>>>>>> master
