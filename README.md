# AgentGraph-RAG

基于双层认知导航图的多跳问答系统。

## 系统架构

```
rag/
├── src/                        # 主模型核心代码
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
│   └── index/                  # 语义索引存储
├── config/
│   └── setting.py              # 配置文件
├── scripts/                    # 运行脚本
│   ├── batch_runner.py         # 批量运行主模型
│   ├── build_index.py          # 构建语义索引
│   ├── build_dual_graph.py     # 构建双层认知图
│   └── eval.py                 # 评估脚本
├── data/                       # 数据集
└── results/                    # 输出结果
```

## 模型说明

### AgentGraph-RAG（主模型）

基于双层认知导航图实现的多跳问答：

- **Dual-Graph**: 实体层 + 篇章层
  - **实体节点**: 使用 spaCy NER 提取的名词实体
  - **篇章节点**: 文档块
  - **包含边**: 篇章 → 实体 (实体出现在该篇章中)
  - **共现边**: 实体 ↔ 实体 (在同一篇章中共现，权重为共现次数)

- **graph_hop**: 图遍历检索 - 查找共现实体

- **检索模式**:
  - `keyword`: 关键词检索 + 图遍历
  - `semantic`: 语义检索 + 图遍历
  - `hybrid`: 关键词 + 语义混合检索

---

## 快速开始

### 1. 配置

编辑 `config/setting.py`：

```python
DATASET = "2wikimultihopqa"
RETRIEVAL_TOOL = "keyword"
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
# 语义索引（所有模式都需要）
python scripts/build_index.py

# 双层认知图
python scripts/build_dual_graph.py
```

### 4. 运行

```bash
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

### 评估指标

| 指标 | 说明 |
|------|------|
| **LLMAcc** | LLM 判断答案正确性 |
| **ContAcc** | 预测答案包含支持事实 |

---

## Baseline

用于对比的 baseline 模型放在根目录下，以其模型名字命名。每个 baseline 模型都有其自己的 config、results 目录。baseline 模型的数据使用 data，模型调用方式、评估方式沿用 AgentGraph-RAG 的代码，具体实现在各自的文件夹里，不得修改 baseline 文件夹之外的内容。

## 注意事项

1. LLM 通过 API 调用，需要配置 API Key
2. Embedding 模型需要下载到本地，建议使用 sentence-transformers 模型
3. 默认 LLM: qwen3.5-flash
4. 默认 Embedding: qwen3-4B
