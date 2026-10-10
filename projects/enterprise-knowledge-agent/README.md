# Enterprise Knowledge Agent

一个面向学习和面试展示的企业知识库 Agent。它可以导入企业文档，通过 RAG 检索回答问题并返回引用；当证据不足时，LangGraph 会转入人工工单分支。

这是可部署的个人项目，不是生产客户项目。默认配置使用确定性模型和确定性向量，方便无 API Key 学习与测试；真实模型模式使用 OpenAI Responses API，实际语义检索可切换 FastEmbed。

## 能力对应

- Python 独立项目结构与类型标注
- FastAPI 接口和自动 OpenAPI 文档
- LangGraph 多步骤状态工作流
- Qdrant 向量数据库和 `tenant_id` 数据隔离
- PostgreSQL 工单与运行记录
- OpenAI 模型调用
- pytest 测试和 Docker Compose 部署

## 本地安装

要求 Python 3.11 或 3.12。

```powershell
cd projects/enterprise-knowledge-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

## 运行测试

```powershell
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

测试不会调用真实 OpenAI API，也不会产生模型费用。

## 本地运行

默认使用 SQLite、内存 Qdrant 和确定性模型：

```powershell
python -m uvicorn app.main:app --reload
```

打开 `http://127.0.0.1:8000/docs` 查看交互式 API 文档。

## Docker Compose 运行

安装并启动 Docker Desktop 后执行：

```powershell
docker compose up --build -d
docker compose ps
docker compose logs -f api
```

停止服务但保留数据卷：

```powershell
docker compose down
```

Docker 模式启动 FastAPI、Qdrant 和 PostgreSQL 三个服务。

## 导入示例知识库

PowerShell：

```powershell
curl.exe -X POST http://localhost:8000/api/v1/documents `
  -F "tenant_id=demo-company" `
  -F "file=@sample_data/employee_handbook.md;type=text/markdown"
```

## 提问

有知识库证据的问题：

```powershell
curl.exe -X POST http://localhost:8000/api/v1/questions `
  -H "Content-Type: application/json" `
  -d '{"tenant_id":"demo-company","question":"员工每年有多少天年假？"}'
```

知识库未覆盖的问题会返回 `status=escalated` 和 `ticket_id`：

```powershell
curl.exe -X POST http://localhost:8000/api/v1/questions `
  -H "Content-Type: application/json" `
  -d '{"tenant_id":"demo-company","question":"公司班车几点发车？"}'
```

查询工单：

```powershell
curl.exe http://localhost:8000/api/v1/tickets/1
```

## 使用真实模型和本地语义向量

在 `.env` 中设置：

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=你的密钥
OPENAI_MODEL=gpt-5.5
EMBEDDING_PROVIDER=fastembed
```

密钥只放在本地 `.env`，不要提交到 Git。首次使用 FastEmbed 会下载模型文件。

## 项目结构

- `app/api`：HTTP 请求和响应
- `app/agent`：LangGraph 状态与节点
- `app/documents`：解析、切片和导入
- `app/retrieval`：Embedding 和 Qdrant
- `app/llm`：确定性与 OpenAI 模型提供者
- `app/tickets`、`app/runs`：SQL 持久化
- `tests`：业务行为测试
- `docs`：学习、架构和面试材料

## 故障排查

### `ModuleNotFoundError`

确认终端位于项目目录，并重新执行 `python -m pip install -e ".[dev]"`。

### OpenAI 模式提示缺少密钥

确认 `.env` 包含 `OPENAI_API_KEY`，并且启动服务的终端位于 `.env` 所在目录。

### Docker 命令不存在

安装 Docker Desktop，启动后重新打开终端，通过 `docker version` 验证。

### Qdrant 无法连接

运行 `docker compose ps` 和 `docker compose logs qdrant`，确认端口 6333 没有被其他程序占用。

### 问题总是创建工单

先确认文档已经导入同一个 `tenant_id`，然后查看检索分数。开发模式可适当降低 `EVIDENCE_THRESHOLD`，生产环境必须通过评测集确定阈值。

更多说明见 [学习指南](docs/learning-guide.md)、[架构说明](docs/architecture.md)和[面试指南](docs/interview-guide.md)。

