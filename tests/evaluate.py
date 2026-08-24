# tests/evaluate.py
"""评估：检索质量（Hit Rate + MRR + 拒答）
只测检索层，不调 LLM —— 快、便宜、纯粹看检索效果
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
# 把项目根目录加进 sys.path，保证直接运行本脚本时也能 import app 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retriever import search
from tests.qa_pairs import QA_PAIRS


def _normalize(text: str) -> str:
    """匹配前去掉所有空格：入库文本可能带异常空格（如"每月 十号"），
    关键词"每月十号"必须去掉空格才能匹配上"""
    return text.replace(" ", "").replace("\u3000", "")


def eval_retrieval():
    """跑全部 QA 对，算检索指标"""
    total = hit = 0          # Hit Rate 统计
    mrr_sum = 0.0            # MRR 累加
    correct_reject = 0       # 拒答统计
    reject_total = 0

    print("=" * 70)
    print(f"{'问题':<28} {'排名':<6} {'命中':<4} 状态")
    print("=" * 70)

    for q, keywords, answerable, *_ in QA_PAIRS:
        # 检索 top 5（评估时放宽到 5，看答案在不在前 5）
        docs = search(q, top_k=5)

        # 找到第一个命中的位置（答案关键词出现在检索片段里；两边都去空格再比）
        rank = None
        for i, d in enumerate(docs, start=1):
            if any(_normalize(kw) in _normalize(d.page_content) for kw in keywords):
                rank = i
                break

        if not answerable:
            # 不可答问题：期望检索不到（或检索到的都是无关内容）
            reject_total += 1
            if not docs:
                correct_reject += 1
                status = "✅ 拒答正确"
            else:
                status = f"⚠️ 检索到 {len(docs)} 个片段（可能误召回）"
            print(f"{q:<28} {'-':<6} {'-':<4} {status}")
        else:
            total += 1
            if rank is not None:
                hit += 1
                mrr_sum += 1.0 / rank
                status = f"✅ 命中 @{rank}"
            else:
                status = "❌ 未命中"
            print(f"{q:<28} {str(rank or '-'):<6} {'✅' if rank else '❌':<4} {status}")

    # 汇总
    print("\n" + "=" * 70)
    print(f"可答问题: {total} 个, 命中 {hit} 个")
    print(f"✅ Hit Rate: {hit / total:.1%}" if total else "无可答问题")
    print(f"✅ MRR: {mrr_sum / total:.3f}" if total else "")
    if reject_total:
        print(f"✅ 拒答准确率: {correct_reject}/{reject_total} ({correct_reject / reject_total:.1%})")
    print("=" * 70)


def show_score_distribution():
    """辅助工具：打印若干问题的 rerank 分数分布，用于校准 threshold"""
    from app.reranker import rerank
    from app.retriever import get_vectorstore

    print("\n" + "=" * 70)
    print("rerank 分数分布（用于校准 threshold）")
    print("=" * 70)
    vs = get_vectorstore()
    for q, keywords, answerable, *_ in QA_PAIRS[:6]:  # 看前 6 个问题足够
        texts = [d.page_content for d in vs.similarity_search(q, k=10)]
        if not texts:
            print(f"\n{q}: 召回为空")
            continue
        ranked = rerank(q, texts, top_n=10)
        print(f"\n问: {q}  ({'可答' if answerable else '不可答'})")
        for score, idx, text in ranked[:5]:  # 只看前 5 个
            hit = any(_normalize(kw) in _normalize(text) for kw in keywords)
            mark = "✔含答案" if hit else "  无关"
            print(f"  {score:.4f}  {mark}  {text[:30]}...")


if __name__ == "__main__":
    eval_retrieval()
    # 需要看分数分布校准阈值时，取消下一行注释：
    # show_score_distribution()