# 从 C# 到 Python：企业知识库 Agent 学习指南

这份指南不要求你先背完 Python 语法。学习顺序跟随一次真实请求：配置 → 文档 → 向量 → Agent → API → 数据库 → 测试。

## 1. 先认识入口

Python 的 `app/main.py` 类似 ASP.NET Core 的 `Program.cs`。`create_app()` 创建 FastAPI 应用，`app.include_router(router)` 类似注册 Controller 路由。

```powershell
python -m uvicorn app.main:app --reload
```

这里的第一个 `app` 是 Python 包，`main` 是模块，第二个 `app` 是模块中的 FastAPI 对象。

**练习 1：** 把应用标题临时改成自己的名称，打开 `/docs` 观察标题变化，再恢复。

## 2. 函数、类和类型标注

C#：

```csharp
public string Parse(string fileName, byte[] content)
```

Python：

```python
def parse(self, filename: str, content: bytes) -> str:
```

Python 类型标注主要帮助阅读、编辑器和静态检查，运行时仍是动态语言。项目用 `dataclass` 表示简单数据，用 Pydantic 表示 API 输入输出。

**练习 2：** 打开 `app/documents/models.py`，解释为什么 `TextChunk` 使用 `frozen=True`。

## 3. Protocol 对应接口

`app/retrieval/embeddings.py` 中的 `EmbeddingProvider` 是 Python 的 `Protocol`。它类似 C# 接口：调用者关心方法形状，不关心具体类。

```python
class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
```

`DeterministicEmbeddingProvider` 用于测试，`FastEmbedEmbeddingProvider` 用于真实语义向量。业务代码不需要知道使用哪一个实现。

**练习 3：** 给确定性向量的维度传入 `0`，运行对应 pytest，观察异常来自哪里。

## 4. 文档切片

`TextChunker` 使用固定窗口和重叠区域。假设块大小为 10、重叠为 3，下一块会从前一块第 8 个字符开始。重叠可以减少答案刚好落在边界时的信息损失。

```powershell
python -m pytest tests/documents/test_chunker.py -v
```

**练习 4：** 将测试中的重叠从 3 改为 0，先写出你认为的切片结果，再运行测试验证。

## 5. RAG 与租户隔离

RAG 流程是：问题向量化 → Qdrant 相似度检索 → 提取相关片段 → 交给模型回答。每条向量都保存 `tenant_id`，查询时必须使用完全相同的过滤条件。

`tenant_id` 隔离是业务安全要求，不是界面筛选。接口即使收到其他企业的文档 ID，也不能跨企业召回内容。

```powershell
python -m pytest tests/retrieval/test_vector_store.py -v
```

**练习 5：** 在测试中新增第三个企业，确认查询企业 A 时结果集合仍然只有 A。

## 6. LangGraph 工作流

`app/agent/workflow.py` 把每一步写成节点。节点接收 `AgentState`，只返回要更新的字段。条件边根据 `in_scope` 和 `evidence_ok` 决定下一步。

这和 C# 中把业务写成多个 Service 方法类似，但图结构能明确展示分支和状态变化。

**练习 6：** 把测试阈值从 `0.3` 改为 `0.99`，预测原本能回答的问题会走哪条分支，然后运行测试。

## 7. FastAPI 与依赖注入

FastAPI 的路由函数类似 Controller Action。`AppContainer` 保存共享依赖，测试可以把真实 Qdrant、PostgreSQL 和 OpenAI 替换成内存实现。

```powershell
python -m pytest tests/api/test_api.py -v
```

**练习 7：** 请求一个不存在的工单 ID，确认 API 返回 404 和 `ticket not found`。

## 8. 数据库事务

问答接口创建工单和执行记录后统一 `commit()`。出现异常时执行 `rollback()`。这与 C# 中 `DbContext.SaveChanges()` 和事务回滚的目的相同。

**练习 8：** 阅读 `tests/persistence/test_repositories.py`，指出 `flush()` 和 `commit()` 分别在哪里发生、为什么需要区分。

## 9. pytest 的红绿循环

每个功能先写 pytest：

1. 测试因为缺少功能而失败。
2. 写最小实现。
3. 测试通过。
4. 重构并再次测试。

不要只看最终的绿色结果。面试时应能说明某个测试防止了什么错误。

**练习 9：** 为上传文件大小限制新增一个测试，先确认失败，再实现或调整代码使其通过。

## 10. 你的讲解顺序

1. 从 `app/main.py` 找入口。
2. 从 `app/api/routes.py` 跟踪请求。
3. 从 `app/documents/service.py` 跟踪导入。
4. 从 `app/agent/workflow.py` 跟踪问答。
5. 从 `app/retrieval/vector_store.py` 解释隔离。
6. 从测试文件证明行为。

当你能不看稿完成这条调用链，并现场修改一个阈值或返回字段时，才可以在面试中说自己“理解并能维护”这个项目。

