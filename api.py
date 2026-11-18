# -*- coding: utf-8 -*-
"""给 RAG demo 加 FastAPI 接口：把检索+生成封装成 REST API。

启动：uvicorn api:app --host 127.0.0.1 --port 8000
或：  python api.py
调用：POST /ask  {"question": "..."}
"""
from fastapi import FastAPI
from pydantic import BaseModel

import rag_demo

app = FastAPI(title="RAG 问答 API", description="最小 RAG 检索增强生成 demo（TF-IDF + DeepSeek）")

# 启动时构建索引（知识库：项目知识点.md）
chunks = rag_demo.load_chunks(rag_demo.DOC_PATH)
vec, chunk_vecs = rag_demo.build_index(chunks)


class AskRequest(BaseModel):
    question: str
    top_k: int = 3


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(chunks)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    retrieved = rag_demo.retrieve(vec, chunk_vecs, chunks, req.question, req.top_k)
    answer = rag_demo.generate(req.question, retrieved)
    sources = [c.split("\n")[0][:60] for c, _ in retrieved]
    return AskResponse(question=req.question, answer=answer, sources=sources)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
