"""Settings"""

import os
import logging
from pathlib import Path

os.environ.pop("OMP_NUM_THREADS", None)

project_root = Path(__file__).parent.parent

LOG_LEVEL = logging.WARNING

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def _get_basename(path):
    """Extract basename with parent dir from path (without extension)."""
    parts = path.replace(".json", "").split("/")
    if len(parts) >= 2:
        return f"{parts[-2]}_{parts[-1]}"
    return os.path.splitext(os.path.basename(path))[0]


def get_output_dir():
    """Generate output directory path based on configuration."""
    chunks_basename = _get_basename(DATA_CHUNKS_FILE)
    questions_basename = _get_basename(DATA_QUESTIONS_FILE)
    llm_name = LLM_MODEL.replace(".", "_")
    
    # 模型名映射：允许不同模型名输出到相同目录
    if "Qwen3.5-122B" in LLM_MODEL or "Qwen3.5" in LLM_MODEL:
        llm_name = "qwen3_5-flash"
    
    emb_name = os.path.basename(EMBEDDING_MODEL).replace(".", "_")
    
    return f"{project_root}/results/{chunks_basename}_{questions_basename}/{llm_name}_{emb_name}/src_all/"


def get_baseline_output_dir(model_name):
    """Generate output directory for baseline models."""
    chunks_basename = _get_basename(DATA_CHUNKS_FILE)
    questions_basename = _get_basename(DATA_QUESTIONS_FILE)
    llm_name = LLM_MODEL.replace(".", "_")
    emb_name = os.path.basename(BASELINE_EMBEDDING_MODEL).replace(".", "_")
    return f"{project_root}/results/baseline/{model_name}/{chunks_basename}_{questions_basename}/{llm_name}_{emb_name}/"


# ============================================================================
# Model Configuration
# ============================================================================

# Dataset selection
DATASET = "hotpotqa"
QA_FILE = "qa_hard.json"

# Dataset paths
DATA_CHUNKS_FILE = f"{project_root}/data/{DATASET}/corpus.json"
DATA_QUESTIONS_FILE = f"{project_root}/data/{DATASET}/{QA_FILE}"
DATA_INDEX_DIR = f"{project_root}/data/indices/{DATASET}"
DATA_GRAPH_DIR = f"{project_root}/data/graphs"

# LLM Configuration
LLM_MODEL = "qwen3.5-flash"
LLM_API_KEY = "sk-84efbff494e54a0aa620dcae02e7e43d"
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 16384
LLM_REASONING_EFFORT = None

# Embedding Configuration
EMBEDDING_PROVIDER = "local"  # "local" or "api"

if EMBEDDING_PROVIDER == "local":
    EMBEDDING_MODEL = f"{project_root}/qwen3-4B"
    EMBEDDING_DIMENSIONS = 1024
else:  # api
    EMBEDDING_MODEL = "text-embedding-v4"
    EMBEDDING_DIMENSIONS = 1024

EMBEDDING_DEVICE = "cuda:0"
EMBEDDING_BATCH_SIZE = 32

# Vision Model Configuration (for image/PDF parsing)
VL_MODEL = "qwen3.6-flash"
VL_API_KEY = LLM_API_KEY
VL_BASE_URL = LLM_BASE_URL
VL_PROMPT = "Extract ALL text from this image. Output ONLY plain text - no JSON, no markdown code blocks, no quotes. Just the raw text exactly as it appears. If there is no text, output nothing."

# spaCy model (sm=fast, trf=accurate+GPU)
SPACY_MODEL = "en_core_web_trf"

# Chunking Configuration
CHUNK_MAX_TOKENS = 1000
CHUNK_TOKENIZER_MODEL = "gpt-4o"

# Agent Configuration
AGENT_MAX_LOOPS = 15
AGENT_MAX_TOKEN_BUDGET = 128000
AGENT_VERBOSE = False

# Batch Processing
OUTPUT_RESULTS_DIR = f"{project_root}/results/"
BATCH_WORKERS = 10
BATCH_LIMIT = None

# Evaluation
EVAL_PREDICTIONS_FILE = f"{project_root}/results/predictions.jsonl"
EVAL_WORKERS = 10
EVAL_OUTPUT_DIR = None