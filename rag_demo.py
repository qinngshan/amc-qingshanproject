# -*- coding: utf-8 -*-
"""
最小 RAG（检索增强生成）Demo —— 投福富「大模型方向」岗的动手项目
流程：文档切分 → TF-IDF 向量化 → 相似度检索 → 调 DeepSeek 生成带出处的回答
说明：检索用 TF-IDF（简单、能跑），生产环境会换语义 embedding（如 bge 模型）。
"""
import os
import re
import json
import sys
import urllib.request
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "项目知识点.md")


def load_api_key():
    env = open(os.path.expanduser("~/.hermes/.env"), encoding="utf-8").read()
    for line in env.splitlines():
        if line.startswith("DEEPSEEK_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("未找到 DEEPSEEK_API_KEY")


def load_chunks(path):
    """按 ## 标题把文档切成独立的「知识块」"""
    text = open(path, encoding="utf-8").read()
    chunks = re.split(r"\n(?=#{1,3} )", text)
    return [c.strip() for c in chunks if len(c.strip()) > 30]


def build_index(chunks):
    """TF-IDF 向量化，把每个 chunk 变成向量（中文用字符级 ngram，无需分词）"""
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), max_features=5000)
    chunk_vecs = vec.fit_transform(chunks)
    return vec, chunk_vecs


def retrieve(vec, chunk_vecs, chunks, query, top_k=3):
    """问题也向量化，用余弦相似度找最相关的 top_k 个 chunk"""
    qvec = vec.transform([query])
    sims = cosine_similarity(qvec, chunk_vecs)[0]
    top_idx = np.argsort(sims)[-top_k:][::-1]
    return [(chunks[i], float(sims[i])) for i in top_idx]


def generate(query, retrieved):
    """把检索到的资料拼进 prompt，调 DeepSeek 生成回答"""
    key = load_api_key()
    context = "\n\n".join(f"【资料片段】\n{c[:1200]}" for c, _ in retrieved)
    prompt = (
        "你是一个知识库问答助手。请只根据下面的资料回答用户问题，"
        "资料里没有的信息就明确说「资料中没有提到」。\n\n"
        f"{context}\n\n【用户问题】{query}"
    )
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个严谨的知识库问答助手，只依据给定资料回答。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 600,
    }
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["choices"][0]["message"]["content"]


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "调制识别用了哪些特征？累积量是什么？"
    print(f"知识库：{os.path.basename(DOC_PATH)}")
    print(f"问题：{query}\n")

    chunks = load_chunks(DOC_PATH)
    print(f"文档切分成 {len(chunks)} 个知识块")

    vec, chunk_vecs = build_index(chunks)
    retrieved = retrieve(vec, chunk_vecs, chunks, query)

    print("检索到的相关片段（相似度）：")
    for c, s in retrieved:
        title = c.split("\n")[0][:40]
        print(f"  [{s:.3f}] {title}")
    print()

    answer = generate(query, retrieved)
    print("=== 生成回答 ===")
    print(answer)


if __name__ == "__main__":
    main()
