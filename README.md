# AgentGraph-RAG

基于双层认知导航图的多跳问答系统。

## 系统架构

```
rag/
├── src/                        # 核心代码
│   ├── agent/                  # Agent 实现
│   │   ├── base.py             # BaseAgent 基类
│   │   └── prompt.py           # System prompts
│   ├── core/                   # 核心组件
│   │   ├── llm.py              # LLM 客户端
│   │   └── context.py          # Agent 上下文管理
│   ├── graph/
│   │   └── dual_graph.py       # 双层认知图构建与检索
│   ├── tools/                  # 检索工具
│   │   ├── graph_hop.py        # 图遍历工具
│   │   ├── keyword_search.py   # 关键词检索
│   │   ├── semantic_search.py  # 语义检索
│   │   └── read_chunk.py       # 读取文档块
│   └── rag/
│       ├── engine.py           # RAG 引擎
│       └── run.py              # CLI 入口
├── config/
│   └── setting.py              # 全局配置
├── scripts/                    # 运行脚本
│   ├── batch_runner.py         # 批量运行
│   ├── build_index.py          # 构建语义索引
│   ├── build_dual_graph.py     # 构建双层认知图
│   ├── chunk_corpus.py         # 文档分块
│   └── eval.py                 # 评估脚本
├── data/                       # 数据集
└── results/                    # 输出结果
```

## 模型说明

基于双层认知导航图实现的多跳问答：

- **Dual-Graph**: 实体层 + 篇章层
  - **实体节点**: 使用 spaCy NER 提取的名词实体
  - **篇章节点**: 文档块
  - **包含边**: 篇章 → 实体
  - **共现边**: 实体 ↔ 实体（权重为共现次数）

- **检索工具**（自动注册，按需启用）:
  - `keyword_search`: 关键词检索 + 图遍历
  - `semantic_search`: 语义检索（需构建索引）
  - `graph_hop`: 图遍历检索（需构建双层图）

---

## 快速开始

### 1. 配置

编辑 `config/setting.py`：

```python
DATASET = "hotpotqa"
LLM_MODEL = "qwen3.5-flash"
LLM_API_KEY = "your-api-key"
EMBEDDING_MODEL = "/path/to/embedding/model"
```

### 2. 准备数据

将数据集放置在 `data/{dataset}/` 目录下：
- `corpus.json`：语料库
- `qa_hard.json`：问题集

### 3. 构建索引

```bash
# 文档分块
python scripts/chunk_corpus.py

# 语义索引
python scripts/build_index.py

# 双层认知图
python scripts/build_dual_graph.py
```

### 4. 运行

```bash
# 单条查询
python src/rag/run.py "your question"

# 批量运行
python scripts/batch_runner.py
```

### 5. 评估

```bash
python scripts/eval.py --predictions results/.../predictions.jsonl
```

---

## 输出格式

```json
{
  "qid": "问题ID",
  "question": "问题内容",
  "pred_answer": "预测答案",
  "gold_answer": "标准答案",
  "supporting_facts": ["支持事实"]
}
```

## 评估指标

| 指标 | 说明 |
|------|------|
| **LLMAcc** | LLM 判断答案正确性 |
| **ContAcc** | 预测答案包含支持事实 |
