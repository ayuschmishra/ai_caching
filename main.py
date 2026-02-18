import os
import time
import hashlib
from collections import OrderedDict
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import openai
import numpy as np

load_dotenv()

# ── aipipe.org client (OpenAI-compatible) ─────────────────────────────────────
client = openai.OpenAI(
    api_key=os.getenv("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openrouter/v1"
)

app = FastAPI()

# ── Constants ──────────────────────────────────────────────────────────────────
CACHE_MAX_SIZE   = 500
TTL_SECONDS      = 86400      # 24 hours
SIM_THRESHOLD    = 0.95
AVG_TOKENS       = 3000
MODEL_COST_PER_M = 1.00       # $ per 1M tokens
BASELINE_DAILY   = 8.11

# ── In-Memory Stores ──────────────────────────────────────────────────────────
exact_cache: OrderedDict = OrderedDict()
semantic_cache: list     = []

# ── Analytics Counters ────────────────────────────────────────────────────────
stats = {"hits": 0, "misses": 0}

# ── Helper: MD5 hash of normalized query ──────────────────────────────────────
def make_key(query: str) -> str:
    normalized = query.strip().lower()
    return hashlib.md5(normalized.encode()).hexdigest()

# ── Helper: Get embedding via aipipe ──────────────────────────────────────────
def get_embedding(text: str) -> list:
    resp = client.embeddings.create(
        model="openai/text-embedding-3-small",  # openrouter model name format
        input=text.strip().lower()
    )
    return resp.data[0].embedding

# ── Helper: Cosine similarity ─────────────────────────────────────────────────
def cosine_similarity(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ── Helper: LRU + TTL eviction ────────────────────────────────────────────────
def evict():
    now = time.time()

    # TTL — exact cache
    expired = [k for k, v in exact_cache.items() if now - v["ts"] > TTL_SECONDS]
    for k in expired:
        del exact_cache[k]

    # TTL — semantic cache
    global semantic_cache
    semantic_cache = [e for e in semantic_cache if now - e["ts"] <= TTL_SECONDS]

    # LRU — remove oldest when over size limit
    while len(exact_cache) > CACHE_MAX_SIZE:
        exact_cache.popitem(last=False)

# ── Helper: Call LLM via aipipe ───────────────────────────────────────────────
def call_llm(query: str) -> str:
    resp = client.chat.completions.create(
        model="openai/gpt-4.1-nano",            # cheap, fast model on openrouter
        messages=[
            {"role": "system", "content": "You are a helpful document summarizer."},
            {"role": "user",   "content": query}
        ]
    )
    return resp.choices[0].message.content

# ── Request Model ─────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    application: str = "document summarizer"

# ── POST / ────────────────────────────────────────────────────────────────────
@app.post("/")
def handle_query(req: QueryRequest):
    evict()
    start = time.time()
    query = req.query
    key   = make_key(query)

    # 1. Exact match
    if key in exact_cache:
        exact_cache.move_to_end(key)
        stats["hits"] += 1
        latency = int((time.time() - start) * 1000)
        return {
            "answer":   exact_cache[key]["answer"],
            "cached":   True,
            "latency":  latency,
            "cacheKey": key
        }

    # 2. Semantic match
    embedding = get_embedding(query)
    for entry in semantic_cache:
        sim = cosine_similarity(embedding, entry["embedding"])
        if sim >= SIM_THRESHOLD:
            stats["hits"] += 1
            # Promote to exact cache too
            exact_cache[key] = {"answer": entry["answer"], "ts": time.time()}
            exact_cache.move_to_end(key)
            latency = int((time.time() - start) * 1000)
            return {
                "answer":   entry["answer"],
                "cached":   True,
                "latency":  latency,
                "cacheKey": f"semantic:{key}"
            }

    # 3. Cache miss → call LLM
    stats["misses"] += 1
    answer = call_llm(query)

    exact_cache[key] = {"answer": answer, "ts": time.time()}
    exact_cache.move_to_end(key)
    semantic_cache.append({"embedding": embedding, "answer": answer, "ts": time.time()})

    latency = int((time.time() - start) * 1000)
    return {
        "answer":   answer,
        "cached":   False,
        "latency":  latency,
        "cacheKey": key
    }

# ── GET /analytics ────────────────────────────────────────────────────────────
@app.get("/analytics")
def analytics():
    total    = stats["hits"] + stats["misses"]
    hit_rate = round(stats["hits"] / total, 4) if total > 0 else 0

    saved_tokens  = stats["hits"] * AVG_TOKENS
    cost_savings  = round(saved_tokens * MODEL_COST_PER_M / 1_000_000, 4)
    savings_pct   = round((cost_savings / BASELINE_DAILY) * 100, 2) if BASELINE_DAILY else 0

    return {
        "hitRate":        hit_rate,
        "totalRequests":  total,
        "cacheHits":      stats["hits"],
        "cacheMisses":    stats["misses"],
        "cacheSize":      len(exact_cache),
        "costSavings":    cost_savings,
        "savingsPercent": savings_pct,
        "strategies":     ["exact match", "semantic similarity", "LRU eviction", "TTL expiration"]
    }