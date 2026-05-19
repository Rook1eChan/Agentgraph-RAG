#!/usr/bin/env python3
"""
Evaluation script for ARAG predictions.

Usage:
    python scripts/eval.py

Configuration is read from config/setting.py
"""

import os
import sys
import json
import re
import string
import argparse
import logging
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import setting as settings
from src.core.llm import LLMClient

logger = logging.getLogger(__name__)


def normalize_answer(s):
    """Normalize answer for comparison."""
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(pred: str, gold: str) -> float:
    """Exact Match: 1.0 if normalized pred == normalized gold, else 0.0."""
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def compute_f1_score(pred: str, gold: str) -> float:
    """Token-level F1 (HotpotQA official protocol)."""
    gold_tokens = normalize_answer(gold).split()
    pred_tokens = normalize_answer(pred).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


class Evaluator:
    """Evaluator for ARAG predictions."""

    def __init__(self, llm_client, predictions_path):
        self.llm_client = llm_client
        self.predictions_path = predictions_path
        self.prediction_results = self.load_predictions()
        logger.info(f"Loaded {len(self.prediction_results)} predictions")

    def extract_retrieved_facts(self, trajectory):
        """Extract all retrieved facts (titles) from trajectory."""
        retrieved = set()

        for step in trajectory:
            tool_name = step.get('tool_name', '')
            arguments = step.get('arguments', {})
            tool_result = str(step.get('tool_result', ''))

            if tool_name == 'keyword_search':
                import re
                matches = re.findall(r'Chunk ID: ([^\n,]+)', tool_result)
                for m in matches:
                    retrieved.add(m.strip())

            elif tool_name == 'read_chunk':
                chunk_ids = arguments.get('chunk_ids', [])
                for cid in chunk_ids:
                    retrieved.add(str(cid))

            elif tool_name == 'semantic_search':
                import re
                matches = re.findall(r'Chunk ID: ([^\n(]+)', tool_result)
                for m in matches:
                    retrieved.add(m.strip())

        return list(dict.fromkeys(retrieved))  # 去重保持顺序

    def load_predictions(self):
        """Load predictions from file."""
        if self.predictions_path.endswith('.jsonl'):
            prediction_results = []
            with open(self.predictions_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        prediction_results.append(json.loads(line))
        else:
            with open(self.predictions_path, 'r', encoding='utf-8') as f:
                prediction_results = json.load(f)
        return prediction_results

    def calculate_llm_accuracy(self, pred_answer, gold_answer):
        """Use LLM to judge if prediction is correct."""
        system_prompt = "You are an expert evaluator."
        user_prompt = f"""Please evaluate if the generated answer is correct by comparing it with the gold answer.
Generated answer: {pred_answer}
Gold answer: {gold_answer}

The generated answer should be considered correct if it:
1. Contains the key information from the gold answer
2. Is factually accurate and consistent with the gold answer
3. Does not contain any contradicting information

Respond with ONLY 'correct' or 'incorrect'.
Response:"""

        response, _ = self.llm_client.generate(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )

        if response.strip().lower() == "correct":
            return 1.0
        else:
            return 0.0

    def calculate_contain(self, pred_answer, supporting_facts):
        """Check if any supporting fact is contained in prediction."""
        if not pred_answer:
            return 0

        s1 = normalize_answer(pred_answer)

        # 如果 supporting_facts 是列表，逐个检查
        if isinstance(supporting_facts, list):
            for fact in supporting_facts:
                fact_norm = normalize_answer(str(fact))
                if fact_norm in s1:
                    return 1
            return 0
        else:
            # 兼容字符串格式
            fact_norm = normalize_answer(str(supporting_facts))
            return 1 if fact_norm in s1 else 0

    @staticmethod
    def _get_gold_list(prediction):
        """Extract list of gold answer strings from prediction dict."""
        raw = prediction.get('gold_answer') or prediction.get('answer', '')
        if isinstance(raw, list):
            return [str(g) for g in raw if g]
        return [str(raw)] if raw else []

    @staticmethod
    def _compute_em_f1(pred_answer, gold_list):
        """Compute EM (max) and F1 (max) across multiple gold answers."""
        em = max(compute_exact_match(pred_answer, g) for g in gold_list) if gold_list else 0.0
        f1 = max(compute_f1_score(pred_answer, g) for g in gold_list) if gold_list else 0.0
        return em, f1

    def evaluate_single(self, idx, prediction):
        """Evaluate single prediction."""
        qid = prediction.get('qid') or prediction.get('id', f'qid_{idx}')
        pred_answer = prediction.get("pred_answer", "")
        gold_answer = prediction.get("gold_answer") or prediction.get("answer", "")
        supporting_facts = prediction.get("supporting_facts", [])
        trajectory = prediction.get("trajectory", [])

        retrieved_facts = self.extract_retrieved_facts(trajectory)

        if not isinstance(pred_answer, str):
            result = {
                'qid': qid,
                'answer_em': 0.0,
                'answer_f1': 0.0,
                'llm_accuracy': 0.0,
                'contain_accuracy': 0.0,
                'retrieved_facts': retrieved_facts,
                'supporting_facts': supporting_facts
            }
            return idx, result

        gold_list = self._get_gold_list(prediction)
        em, f1 = self._compute_em_f1(pred_answer, gold_list)

        has_answer = pred_answer and pred_answer.strip() != ""

        if has_answer:
            llm_acc = self.calculate_llm_accuracy(pred_answer, gold_answer)
            contain_acc = self.calculate_contain(pred_answer, supporting_facts)
        else:
            llm_acc = 0.0
            contain_acc = 0.0

        result = {
            'qid': qid,
            'answer_em': em,
            'answer_f1': f1,
            'llm_accuracy': llm_acc,
            'contain_accuracy': contain_acc,
            'retrieved_facts': retrieved_facts,
            'supporting_facts': supporting_facts
        }

        return idx, result

    def evaluate(self, max_workers, output_dir=None):
        """Run evaluation with incremental processing and batch write-back."""
        # 过滤掉已评估的条目 (eval=True)
        pending_items = []
        for idx, pred in enumerate(self.prediction_results):
            if not pred.get('eval'):
                pending_items.append((idx, pred))

        total_items = len(self.prediction_results)
        completed_count = total_items - len(pending_items)

        logger.info(f"Total: {total_items}, Completed: {completed_count}, Pending: {len(pending_items)}")

        # Always compute EM/F1 for all items (fast, no LLM)
        for pred in self.prediction_results:
            if not pred.get('answer_em'):
                gold_list = self._get_gold_list(pred)
                pred_ans = pred.get('pred_answer', '')
                if not isinstance(pred_ans, str):
                    pred_ans = ''
                em, f1 = self._compute_em_f1(pred_ans, gold_list)
                pred['answer_em'] = em
                pred['answer_f1'] = f1

        if len(pending_items) == 0:
            logger.info("All items already evaluated!")
            return self.compute_summary()

        # 初始化统计
        llm_scores = [0.0] * total_items
        contain_scores = [0.0] * total_items
        em_scores = [p.get('answer_em', 0.0) for p in self.prediction_results]
        f1_scores = [p.get('answer_f1', 0.0) for p in self.prediction_results]

        # 填充已完成的统计
        for idx, pred in enumerate(self.prediction_results):
            if pred.get('eval'):
                llm_scores[idx] = pred.get('llm_accuracy', 0.0)
                contain_scores[idx] = pred.get('contain_accuracy', 0.0)

        # 处理pending items
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.evaluate_single, idx, pred): idx
                for idx, pred in pending_items
            }

            answered_count = completed_count
            total_llm_score = sum(llm_scores)
            total_contain_score = sum(contain_scores)

            pbar = tqdm(total=len(pending_items), desc="Evaluating", unit="sample")
            processed_since_write = 0

            for future in as_completed(futures):
                idx, result = future.result()

                # 更新内存中的记录
                pred = self.prediction_results[idx]
                pred['eval'] = True
                pred['answer_em'] = result.get('answer_em', 0.0)
                pred['answer_f1'] = result.get('answer_f1', 0.0)
                pred['supporting_facts'] = result.get('supporting_facts', [])
                pred['llm_accuracy'] = result.get('llm_accuracy', 0.0)
                pred['contain_accuracy'] = result.get('contain_accuracy', 0.0)
                pred['retrieved_facts'] = result.get('retrieved_facts', [])

                em_scores[idx] = result.get('answer_em', 0.0)
                f1_scores[idx] = result.get('answer_f1', 0.0)
                llm_scores[idx] = result.get('llm_accuracy', 0.0)
                contain_scores[idx] = result.get('contain_accuracy', 0.0)

                llm_acc = result.get('llm_accuracy', 0.0)
                contain_acc = result.get('contain_accuracy', 0.0)

                if llm_acc > 0 or contain_acc > 0:
                    answered_count += 1

                total_llm_score += llm_acc
                total_contain_score += contain_acc

                processed_since_write += 1

                # 每max_workers条批量写回
                if processed_since_write >= max_workers:
                    self.batch_write_predictions()
                    processed_since_write = 0

                if answered_count > 0:
                    current_llm_acc = total_llm_score / answered_count
                    current_contain_acc = total_contain_score / answered_count
                else:
                    current_llm_acc = 0.0
                    current_contain_acc = 0.0

                pbar.set_postfix({
                    'LLM_Acc': f'{current_llm_acc:.3f}',
                })
                pbar.update(1)
            pbar.close()

            # 最后写回剩余的
            self.batch_write_predictions()

        return self.compute_summary(llm_scores, contain_scores, em_scores, f1_scores)

    def batch_write_predictions(self):
        """Write all predictions to file."""
        with open(self.predictions_path, 'w', encoding='utf-8') as f:
            for pred in self.prediction_results:
                f.write(json.dumps(pred, ensure_ascii=False) + '\n')

    def compute_summary(self, llm_scores=None, contain_scores=None, em_scores=None, f1_scores=None):
        """Compute and save summary statistics."""
        if llm_scores is None:
            llm_scores = [p.get('llm_accuracy', 0.0) for p in self.prediction_results]
            contain_scores = [p.get('contain_accuracy', 0.0) for p in self.prediction_results]
            em_scores = [p.get('answer_em', 0.0) for p in self.prediction_results]
            f1_scores = [p.get('answer_f1', 0.0) for p in self.prediction_results]

        total_samples = len(self.prediction_results)

        llm_accuracy = sum(llm_scores) / total_samples
        contain_accuracy = sum(contain_scores) / total_samples
        answer_em = sum(em_scores) / total_samples
        answer_f1 = sum(f1_scores) / total_samples

        logger.info(f"Evaluation Results:")
        logger.info(f"  Total Samples: {total_samples}")
        logger.info(f"  Answer EM:     {answer_em:.4f}")
        logger.info(f"  Answer F1:     {answer_f1:.4f}")
        logger.info(f"  LLM Accuracy:  {llm_accuracy:.4f}")
        logger.info(f"  Contain Acc:   {contain_accuracy:.4f}")

        # 保存summary到文件
        summary_path = self.predictions_path.replace('.jsonl', '_eval_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                "total_samples": total_samples,
                "answer_em": answer_em,
                "answer_f1": answer_f1,
                "llm_accuracy": llm_accuracy,
                "contain_accuracy": contain_accuracy,
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"Summary saved to: {summary_path}")

        return llm_accuracy, contain_accuracy, answer_em, answer_f1


    def evaluate_answer_metrics(self, output_path=None):
        """Compute Exact Match and F1 from gold_answer vs pred_answer (no LLM needed)."""
        if output_path is None:
            output_path = self.predictions_path.replace('.jsonl', '_answer_metrics.json')

        em_scores = []
        f1_scores = []
        per_sample = []

        for pred in self.prediction_results:
            raw_gold = pred.get('gold_answer', '')
            pred_ans = pred.get('pred_answer', '')

            if not isinstance(pred_ans, str):
                pred_ans = str(pred_ans) if pred_ans else ''

            # Support both single string and list of multiple gold answers
            if isinstance(raw_gold, list):
                gold_list = [str(g) if not isinstance(g, str) else g for g in raw_gold if g]
            else:
                gold_list = [str(raw_gold) if not isinstance(raw_gold, str) else raw_gold]

            em = max(compute_exact_match(pred_ans, g) for g in gold_list) if gold_list else 0.0
            f1 = max(compute_f1_score(pred_ans, g) for g in gold_list) if gold_list else 0.0
            em_scores.append(em)
            f1_scores.append(f1)
            per_sample.append({'qid': pred.get('qid'), 'exact_match': em, 'f1': f1})

        total = len(self.prediction_results)
        avg_em = sum(em_scores) / total if total else 0.0
        avg_f1 = sum(f1_scores) / total if total else 0.0

        summary = {
            'total_samples': total,
            'exact_match': avg_em,
            'f1': avg_f1,
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({'summary': summary, 'per_sample': per_sample}, f, ensure_ascii=False, indent=2)

        logger.info(f"Answer metrics saved to: {output_path}")
        return avg_em, avg_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--answer-metrics', action='store_true', help='Run only answer EM/F1 evaluation (no LLM)')
    parser.add_argument('--predictions', type=str, default=None, help='Path to predictions.jsonl (overrides config)')
    args = parser.parse_args()

    predictions_file = args.predictions or os.path.join(settings.get_output_dir(), "predictions.jsonl")
    workers = settings.EVAL_WORKERS
    output_dir = settings.EVAL_OUTPUT_DIR

    logger.info(f"Predictions: {predictions_file}")
    logger.info(f"Workers: {workers}")
    logger.info(f"{'=' * 10}\n")

    # Create LLM client for evaluation
    llm_client = LLMClient(
        model=settings.LLM_MODEL,
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
    )

    evaluator = Evaluator(llm_client, predictions_file)

    if args.answer_metrics:
        logger.info(f"\n===== Answer EM / F1 Evaluation =====")
        em, f1 = evaluator.evaluate_answer_metrics()
        logger.info(f"\n{'=' * 10}")
        logger.info(f"Results")
        logger.info(f"{'=' * 10}")
        logger.info(f"Exact Match (EM): {em:.4f}")
        logger.info(f"F1:              {f1:.4f}")
        return

    llm_acc, contain_acc, answer_em, answer_f1 = evaluator.evaluate(max_workers=workers,
                                                                     output_dir=output_dir)

    logger.info(f"\n{'=' * 10}")
    logger.info(f"Results")
    logger.info(f"{'=' * 10}")
    logger.info(f"Answer EM:      {answer_em:.4f}")
    logger.info(f"Answer F1:      {answer_f1:.4f}")
    logger.info(f"LLM Accuracy:   {llm_acc:.4f}")
    logger.info(f"Contain Acc:    {contain_acc:.4f}")


if __name__ == "__main__":
    main()
