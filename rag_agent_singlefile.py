# LANGGRAPH-RAG-AGENT (Jupyter / .py single-file)
# Copy into a notebook cell or save as langgraph_rag_agent_notebook.py

import os
import glob
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
print("Starting agent file...")

# --- Dependencies (import guarded so we can show helpful errors) ---
try:
    import chromadb
    from chromadb.config import Settings
except Exception as e:
    chromadb = None
    print("WARN: chromadb not available:", e)

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None
    print("WARN: sentence-transformers not available:", e)

# LangChain/OpenAI (optional)
try:
    from langchain.llms import OpenAI
    from langchain.embeddings import OpenAIEmbeddings
except Exception as e:
    OpenAI = None
    OpenAIEmbeddings = None
    print("WARN: langchain/OpenAI LLM or embeddings not available:", e)

# Utility: cosine similarity (digit-by-digit safe arithmetic)
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # guard against zero-length vectors
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

# -----------------------
# Node: Plan
# -----------------------
@dataclass
class PlanNode:
    def run(self, query: str) -> Dict[str, Any]:
        print("[Plan] Query received:", query)
        tokens = query.strip().split()
        # heuristic: if > 3 words, retrieval recommended
        need_retrieval = len(tokens) > 3
        plan = {
            "query": query,
            "need_retrieval": need_retrieval,
            "note": "Use retrieval" if need_retrieval else "Short query — might not need retrieval"
        }
        print("[Plan] Plan:", plan)
        return plan

# -----------------------
# Node: Retrieve (Chroma + embeddings)
# -----------------------
@dataclass
class RetrieveNode:
    docs_path: str = "docs"
    collection_name: str = "rag_collection"
    chunk_size: int = 800
    chunk_overlap: int = 100
    embedding_model_name: str = "all-MiniLM-L6-v2"  # sentence-transformers model

    chroma_client: Any = None
    local_encoder: Any = None

    def _load_txts(self) -> List[Dict[str, Any]]:
        files = glob.glob(os.path.join(self.docs_path, "**"), recursive=True)
        docs = []
        for f in files:
            if f.lower().endswith(".txt"):
                with open(f, "r", encoding="utf-8") as fh:
                    text = fh.read()
                docs.append({"id": os.path.basename(f), "text": text})
        print(f"[Retrieve] Found {len(docs)} .txt docs under {self.docs_path}")
        return docs

    def _chunk_text(self, text: str) -> List[str]:
        # naive chunker by characters, can be improved by sentence splitting
        chunks = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + self.chunk_size, length)
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
            if start < 0:
                start = 0
        return chunks

    def _get_embed_fn(self):
        # Prefer sentence-transformers locally
        if SentenceTransformer is not None:
            if self.local_encoder is None:
                print("[Retrieve] Loading local sentence-transformers model:", self.embedding_model_name)
                self.local_encoder = SentenceTransformer(self.embedding_model_name)
            def embed_texts(texts: List[str]) -> List[List[float]]:
                emb = self.local_encoder.encode(texts, convert_to_numpy=True)
                return [e.tolist() for e in emb]
            return embed_texts
        # Fallback to OpenAIEmbeddings if key present
        if OpenAIEmbeddings is not None and os.getenv("OPENAI_API_KEY"):
            emb = OpenAIEmbeddings()
            def embed_texts(texts: List[str]) -> List[List[float]]:
                return emb.embed_documents(texts)
            return embed_texts
        raise RuntimeError("No embedding backend available. Install sentence-transformers or set OPENAI_API_KEY for OpenAIEmbeddings.")

    def build_or_query(self, query: str, top_k:int=3) -> Dict[str, Any]:
        docs = self._load_txts()
        if not docs:
            print("[Retrieve] No docs found; returning empty.")
            return {"docs": [], "query": query}

        # Prepare chunks and ids
        chunk_texts = []
        chunk_ids = []
        for d in docs:
            chunks = self._chunk_text(d["text"])
            for i, c in enumerate(chunks):
                chunk_ids.append(f"{d['id']}_chunk_{i}")
                chunk_texts.append(c)

        print(f"[Retrieve] Created {len(chunk_texts)} chunks from {len(docs)} docs.")

        embed_fn = self._get_embed_fn()

        # Initialize chroma client (in-memory default)
        if chromadb is None:
            raise RuntimeError("chromadb not installed. Install chromadb in requirements.")
        # Use default local client (no persistence) for simplicity
        client = chromadb.Client()
        # create or get collection
        try:
            coll = client.get_collection(self.collection_name)
            # delete then re-add for idempotent demo
            try:
                client.delete_collection(self.collection_name)
                coll = client.create_collection(self.collection_name)
            except Exception:
                coll = client.get_collection(self.collection_name)
        except Exception:
            coll = client.create_collection(self.collection_name)

        embeddings = embed_fn(chunk_texts)
        coll.add(ids=chunk_ids, documents=chunk_texts, embeddings=embeddings)
        print(f"[Retrieve] Indexed {len(chunk_ids)} chunks into Chroma collection '{self.collection_name}'")

        # embed the query
        q_emb = embed_fn([query])[0]
        results = coll.query(query_embeddings=[q_emb], n_results=top_k, include=['documents','distances','ids'])
        retrieved = []
        if results and 'documents' in results and len(results['documents'])>0:
            for doc_text, dist, doc_id in zip(results['documents'][0], results['distances'][0], results['ids'][0]):
                retrieved.append({"id": doc_id, "text": doc_text, "distance": float(dist)})
        print(f"[Retrieve] Retrieved {len(retrieved)} results for query.")
        return {"docs": retrieved, "query": query, "query_embedding": q_emb}

# -----------------------
# Node: Answer (LLM generation using LangChain/OpenAI OR fallback)
# -----------------------
@dataclass
class AnswerNode:
    llm_choice: str = "openai"  # 'openai' or 'local-fallback'
    temperature: float = 0.0

    def run(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Create a prompt that concisely includes retrieved context
        context = "\n\n---\n\n".join([f"Document ({d['id']}):\n{d['text'][:1500]}" for d in retrieved_docs])
        prompt = (
            "You are a helpful assistant that must answer using the provided context. "
            "If the context does not contain the answer, say exactly: 'I don't know based on provided documents.'\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )
        print("[Answer] Prompt preview (first 600 chars):")
        print(prompt[:600])

        # Use OpenAI LLM if available
        if OpenAI is not None and os.getenv("OPENAI_API_KEY"):
            llm = OpenAI(temperature=self.temperature)
            response = llm(prompt)  # returns text
            answer_text = response
        else:
            # fallback "cheap" method — summarise top doc or fail
            if retrieved_docs and len(retrieved_docs)>0:
                top = retrieved_docs[0]['text']
                # naive: return first 400 chars as 'answer'
                answer_text = "[Fallback LLM]\n" + (top[:800] + ("..." if len(top)>800 else ""))
            else:
                answer_text = "[Fallback LLM]\nI don't know based on provided documents."
        print("[Answer] Generated answer (first 400 chars):")
        print(answer_text[:400])
        return {"answer": answer_text}

# -----------------------
# Node: Reflect (validate relevance using embeddings similarity)
# -----------------------
@dataclass
class ReflectNode:
    embedding_model_name: str = "all-MiniLM-L6-v2"
    local_encoder: Any = None

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if SentenceTransformer is not None:
            if self.local_encoder is None:
                print("[Reflect] Loading embedding model for reflection:", self.embedding_model_name)
                self.local_encoder = SentenceTransformer(self.embedding_model_name)
            emb = self.local_encoder.encode(texts, convert_to_numpy=True)
            return [e.tolist() for e in emb]
        if OpenAIEmbeddings is not None and os.getenv("OPENAI_API_KEY"):
            emb = OpenAIEmbeddings()
            return emb.embed_documents(texts)
        raise RuntimeError("No embedding backend available for reflection. Install sentence-transformers or set OPENAI_API_KEY.")

    def run(self, query: str, answer: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        # compute similarity between query and answer
        vecs = self._embed([query, answer])
        qv, av = np.array(vecs[0]), np.array(vecs[1])
        sim_q_a = cosine_sim(qv, av)

        # compare query to top doc (if exists)
        top_doc_text = retrieved_docs[0]['text'] if retrieved_docs else ""
        doc_sim = 0.0
        if top_doc_text:
            doc_vec = self._embed([top_doc_text])[0]
            doc_sim = cosine_sim(qv, np.array(doc_vec))
        verdict = {
            "query_answer_similarity": float(sim_q_a),
            "query_topdoc_similarity": float(doc_sim),
            "is_relevant": (sim_q_a > 0.6) or (doc_sim > 0.55)
        }
        print("[Reflect] Verdict:", verdict)
        return verdict

# -----------------------
# Orchestrator: LangGraph-style pipeline
# -----------------------
class LangGraphStyleAgent:
    def __init__(self, docs_path: str = "docs"):
        self.plan = PlanNode()
        self.retrieve = RetrieveNode(docs_path=docs_path)
        self.answer = AnswerNode()
        self.reflect = ReflectNode()

    def run(self, query: str) -> Dict[str, Any]:
        print("\n=== Agent run started ===")
        p = self.plan.run(query)
        retrieved = {"docs": []}
        if p["need_retrieval"]:
            retrieved = self.retrieve.build_or_query(query)
        ans = self.answer.run(query, retrieved.get("docs", []))
        verdict = self.reflect.run(query, ans["answer"], retrieved.get("docs", []))
        print("=== Agent run finished ===\n")
        return {"query": query, "answer": ans["answer"], "retrieved": retrieved.get("docs", []), "verdict": verdict}

# -----------------------
# Example usage when running as script / notebook cell
# -----------------------
if __name__ == "__main__":
    # Quick sample doc creation for demo (if docs empty)
    os.makedirs("docs", exist_ok=True)
    if len(glob.glob("docs/*.txt")) == 0:
        sample_text = (
            "Renewable energy sources (solar, wind, hydro) reduce carbon emissions by "
            "displacing fossil fuel electricity generation. They also reduce air pollution, "
            "provide long-term cost reductions, and support energy independence."
        )
        with open("docs/renewables.txt", "w", encoding="utf-8") as f:
            f.write(sample_text)
        print("[Main] Created sample doc at docs/renewables.txt")

    agent = LangGraphStyleAgent(docs_path="docs")
    question = "What are the benefits of renewable energy for reducing carbon emissions?"
    out = agent.run(question)
    print("Answer preview:\n", out["answer"][:800])
    print("Reflection:", out["verdict"])
