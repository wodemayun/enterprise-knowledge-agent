# 企业知识库与工单 Agent 设计规格

## 1. 项目目标

构建一个可运行、可测试、可部署、适合面试演示的企业知识库 Agent。系统接收企业文档，完成文本切片、向量化和检索；用户提问后，Agent 根据检索证据生成带来源的回答。证据不足时，系统拒绝编造答案，并自动创建人工工单。

项目同时承担教学目标：让具有 C# 基础但 Python 经验有限的学习者理解并能独立讲解 Python、FastAPI、RAG、向量数据库、Agent 工作流、API 对接、测试和 Docker 部署。

该项目在简历和面试中统一描述为“可部署的个人项目”，不描述为一年以上生产项目或真实商业交付。

## 2. 成功标准

项目完成时必须满足以下条件：

1. 可以通过 API 上传 `.txt`、`.md` 和文本型 `.pdf` 文档。
2. 文档按企业标识隔离，切片后写入 Qdrant。
3. 用户提问后可以返回答案、引用片段、来源文件和检索分数。
4. 当检索证据不足时不生成确定性答案，而是在 PostgreSQL 中创建人工工单。
5. LangGraph 明确展示分类、检索、证据判断、回答和工单分支。
6. 使用 OpenAI Responses API 时可以生成真实模型答案。
7. 没有 API Key 时，自动化测试可使用确定性替身运行，不调用外部模型。
8. `docker compose up --build` 可以启动 API、Qdrant 和 PostgreSQL。
9. `pytest` 覆盖核心业务行为并全部通过。
10. README 和中文学习手册能帮助学习者运行、修改并讲解项目。

## 3. 范围

### 3.1 第一版包含

- 企业文档上传与解析。
- 固定长度、带重叠的文本切片。
- 本地 FastEmbed 向量化。
- Qdrant 向量存储和按 `tenant_id` 过滤检索。
- 基于 LangGraph 的可观察工作流。
- OpenAI 模型提供者和确定性测试模型提供者。
- 带来源引用的回答。
- 低置信度工单创建。
- PostgreSQL 工单和 Agent 执行记录。
- FastAPI 接口、健康检查和 OpenAPI 文档。
- Docker Compose 本地部署。
- 单元测试与 API 集成测试。
- 中文教学文档和面试讲解材料。

### 3.2 第一版不包含

- 用户注册、登录和复杂权限系统。
- 管理后台或前端聊天页面。
- 图片 OCR、扫描版 PDF 识别和表格抽取。
- 多 Agent 协作、长期记忆和自动网页搜索。
- Kubernetes、云平台发布和高并发压测。
- 用训练数据微调模型。

这些功能不影响第一版展示岗位要求中的 Python、RAG、向量数据库、Agent 工作流、API 对接和部署能力。

## 4. 技术方案选择

### 4.1 Python 与 API

使用 Python 3.11 和 FastAPI。FastAPI 负责请求验证、依赖注入、异步接口和自动 OpenAPI 文档。Pydantic 模型用于定义请求、响应和 Agent 状态边界。

### 4.2 Agent 工作流

使用 LangGraph 的 `StateGraph` 表达确定性的业务工作流。模型只负责需要语义判断和语言生成的部分；关键业务分支由显式状态和条件边控制，便于测试、审计和面试讲解。

工作流节点：

1. `classify_question`：判断问题是否属于知识库问答范围。
2. `retrieve_documents`：按企业标识从 Qdrant 检索相关片段。
3. `grade_evidence`：根据最高分、有效片段数量和配置阈值判断证据是否充分。
4. `generate_answer`：使用检索上下文生成带引用的答案。
5. `create_ticket`：保存人工处理工单。
6. `record_run`：记录最终状态、耗时和分支结果。

### 4.3 RAG 与向量数据库

使用 Qdrant 保存向量和文档元数据。Embedding 默认通过 FastEmbed 在本地生成，避免测试和基础演示依赖付费接口。每条向量包含：

- `tenant_id`
- `document_id`
- `source_name`
- `chunk_index`
- `text`

检索时必须携带 `tenant_id` 过滤条件，防止不同企业之间的数据串读。第一版使用稠密向量相似度检索，不增加混合检索和重排，以控制学习范围。

### 4.4 大模型调用

定义小型 `LLMProvider` 接口：

- `OpenAIProvider` 使用 OpenAI Responses API，密钥从环境变量读取。
- `DeterministicProvider` 仅用于测试和离线演示，按照固定规则返回结果。

业务代码只依赖接口，不直接依赖某个模型 SDK，便于测试，也能用 C# 中“依赖倒置”的概念解释。

### 4.5 关系数据库

PostgreSQL 保存人工工单和 Agent 执行记录。使用 SQLAlchemy 2.x 定义模型和会话边界。第一版包含两张表：

`tickets`：

- `id`
- `tenant_id`
- `question`
- `reason`
- `status`
- `created_at`

`agent_runs`：

- `id`
- `tenant_id`
- `question`
- `route`
- `top_score`
- `ticket_id`
- `duration_ms`
- `created_at`

## 5. API 设计

### `GET /health`

返回服务状态，并检查 PostgreSQL 和 Qdrant 是否可访问。

### `POST /api/v1/documents`

使用 multipart 上传文档。请求必须包含 `tenant_id`。响应返回文档 ID、切片数量和来源文件名。

### `POST /api/v1/questions`

请求：

```json
{
  "tenant_id": "demo-company",
  "question": "员工每年有多少天年假？"
}
```

回答分支响应：

```json
{
  "status": "answered",
  "answer": "根据员工手册，入职满一年后享有 5 天年假。",
  "citations": [
    {
      "source_name": "员工手册.md",
      "chunk_index": 3,
      "text": "员工入职满一年后，每年享有5天带薪年假。",
      "score": 0.91
    }
  ],
  "ticket_id": null
}
```

工单分支响应：

```json
{
  "status": "escalated",
  "answer": "知识库中没有足够证据，已创建人工工单。",
  "citations": [],
  "ticket_id": 12
}
```

### `GET /api/v1/tickets/{ticket_id}`

按 ID 查询人工工单状态。

## 6. 数据流

### 6.1 文档导入

1. API 校验企业标识、文件扩展名和文件大小。
2. 解析文本并规范化空白字符。
3. 按字符长度切片并保留重叠区域。
4. 为每个片段生成向量。
5. 将向量、文本和来源元数据写入 Qdrant。
6. 返回文档 ID 和写入数量。

如果文档没有可提取文本，接口返回明确的 422 错误，不生成空索引。

### 6.2 问答

1. API 构造初始 Agent 状态。
2. LangGraph 判断问题是否适合知识库回答。
3. 检索节点按企业标识召回片段。
4. 证据节点检查召回数量和相似度阈值。
5. 证据充分时调用模型生成答案并附加引用。
6. 证据不足或问题不在知识库范围时创建工单。
7. 无论走哪条分支，都记录 Agent 执行结果。

## 7. 错误处理与安全边界

- 不把 API Key 写入代码、测试、日志或 Git。
- `.env` 只存本地配置，仓库只提供 `.env.example`。
- 上传文件限制类型和大小，文件名只作为元数据，不直接拼接成磁盘路径。
- 所有向量检索必须包含 `tenant_id` 过滤。
- 外部模型调用设置超时；认证错误不重试，限流和临时服务错误只进行有限次数重试。
- 数据库写操作在事务中完成，失败时回滚。
- 错误响应不包含堆栈、数据库地址或密钥。
- 日志包含请求 ID、节点、耗时和错误类型，不记录完整密钥或不必要的文档内容。

## 8. 测试策略

开发采用测试先行。每个业务行为先写失败测试，再写最小实现。

核心测试包括：

1. 文本切片覆盖边界和重叠行为。
2. 空文档被拒绝。
3. 检索只能返回相同 `tenant_id` 的数据。
4. 高分证据进入回答分支。
5. 无结果或低分证据进入工单分支。
6. 回答必须包含引用。
7. 工单分支写入数据库并返回 ID。
8. OpenAI Provider 缺少密钥时给出明确配置错误。
9. 健康检查反映依赖服务状态。
10. API 请求和响应符合定义的 Pydantic 模型。

自动化测试不访问真实 OpenAI API，避免费用和不确定输出。

## 9. 项目结构

```text
projects/enterprise-knowledge-agent/
├── app/
│   ├── api/              # FastAPI 路由与依赖
│   ├── agent/            # LangGraph 状态、节点和图
│   ├── core/             # 配置、日志和通用异常
│   ├── documents/        # 文档解析与切片
│   ├── llm/              # 模型提供者接口和实现
│   ├── retrieval/        # Embedding 与 Qdrant 检索
│   ├── tickets/          # 工单模型、仓储和服务
│   ├── runs/             # Agent 执行记录
│   └── main.py           # 应用入口
├── sample_data/          # 可公开演示的企业制度样例
├── tests/                # 单元测试与 API 集成测试
├── docs/
│   ├── learning-guide.md # 从 C# 迁移到 Python 的教学
│   ├── interview-guide.md
│   └── architecture.md
├── .env.example
├── .gitignore
├── Dockerfile
├── compose.yaml
├── pyproject.toml
└── README.md
```

每个模块只有一个主要职责，API 层不直接实现检索、模型调用或数据库业务。

## 10. 教学交付方式

学习手册按运行链路组织，而不是按 Python 语法大全组织：

1. 认识项目目录和 Python 入口。
2. 对照 C# 理解函数、类、协议、类型标注和依赖注入。
3. 从测试理解文本切片。
4. 从测试理解向量检索和企业隔离。
5. 手动画出 LangGraph 节点与分支。
6. 跟踪一次完整问答请求。
7. 修改阈值并观察回答分支变化。
8. 人为制造依赖故障并根据日志排查。
9. 使用 Docker Compose 启动和停止服务。
10. 根据面试题独立讲解技术选择和限制。

每章都包含目标、核心代码位置、运行命令、观察结果和一个小练习。

## 11. 部署设计

Docker Compose 启动三个服务：

- `api`：FastAPI 应用。
- `qdrant`：向量数据库。
- `postgres`：工单和执行记录数据库。

API 通过环境变量取得依赖地址。服务提供健康检查并使用命名卷保存数据库数据。第一版不加入 Nginx；README 提供 Nginx 反向代理的扩展说明，避免为了技能标签增加无法充分讲解的运行组件。

## 12. 面试展示方式

演示顺序固定为：

1. 上传两份企业制度文档。
2. 提问一个可以从知识库回答的问题，展示引用。
3. 提问一个知识库没有覆盖的问题，展示自动工单。
4. 打开 LangGraph 工作流图解释节点和分支。
5. 展示 Qdrant 中的企业过滤字段。
6. 运行自动化测试。
7. 展示 Docker Compose 服务状态和日志。

项目陈述必须说明这是个人项目，并主动讲出第一版没有实现的 OCR、混合检索、重排、鉴权和生产监控。

## 13. 设计假设

- 开发环境为 Windows，部署通过 Docker Desktop。
- 用户可以安装或已经安装 Docker；纯 Python 测试不依赖 Docker。
- 真实模型调用需要用户自行配置 `OPENAI_API_KEY`。
- 示例文档不包含真实公司机密或个人隐私。
- 第一版面向学习和面试展示，不承诺生产级 SLA。
