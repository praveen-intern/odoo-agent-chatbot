# -*- coding: utf-8 -*-
"""
PostgreSQL (pgvector) backend for the KB/RAG helpers used by adv_ai_bot.

Requires:
  - psycopg2
  - openai (>=1.x)

Environment:
  ADV_AI_PG_DSN        = postgresql://user:pass@host:5432/dbname
  ADV_AI_EMBED_MODEL   = openai/text-embedding-3-small        (default)
  ADV_AI_EMBED_DIM     = 1536                                  (default)
  OPENAI_API_KEY       = real OpenAI key OR your OpenRouter key
  OPENAI_BASE_URL      = set to https://openrouter.ai/api/v1 for OpenRouter
  OPENROUTER_API_KEY   = optional; used if OPENAI_API_KEY not set
  OPENROUTER_REFERRER  = optional; e.g. http://localhost
  OPENROUTER_TITLE     = optional; e.g. Odoo Agent Bot
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import psycopg2
import psycopg2.extras
from openai import OpenAI

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
PG_DSN = os.environ.get("ADV_AI_PG_DSN")
EMBED_MODEL = os.environ.get("ADV_AI_EMBED_MODEL", "openai/text-embedding-3-small")
EMBED_DIM = int(os.environ.get("ADV_AI_EMBED_DIM", "1536"))  # kept for reference

# Lazy client (so env can be set before Odoo imports this file)
_CLIENT: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """
    OpenAI-compatible client for OpenRouter (or plain OpenAI).
    Uses OPENAI_API_KEY + OPENAI_BASE_URL; falls back to OPENROUTER_API_KEY.
    """
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or (
        "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None
    )
    if not api_key:
        raise RuntimeError(
            "Missing API key: set OPENAI_API_KEY (or OPENROUTER_API_KEY + OPENAI_BASE_URL)."
        )

    default_headers = {
        # OpenRouter recommends sending these; harmless for plain OpenAI
        "HTTP-Referer": os.getenv("OPENROUTER_REFERRER", "http://localhost"),
        "X-Title": os.getenv("OPENROUTER_TITLE", "Odoo Agent Bot"),
    }

    _CLIENT = OpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers)
    return _CLIENT


def _conn():
    if not PG_DSN:
        raise RuntimeError("ADV_AI_PG_DSN is not set")
    return psycopg2.connect(PG_DSN)


def _to_vector_literal(vec: Sequence[float]) -> str:
    """pgvector accepts a text literal like: [0.1,0.2,...]"""
    return "[" + ",".join(str(float(x)) for x in vec) + "]"


def _embed(texts: List[str]) -> List[List[float]]:
    """Batch-embed texts."""
    if not texts:
        return []
    client = _get_client()
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def _get_or_create_collection_id(cur, name: str, meta: Optional[Dict[str, Any]] = None) -> int:
    cur.execute("SELECT id FROM ai_collections WHERE name=%s", (name,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute(
        "INSERT INTO ai_collections (name, metadata) VALUES (%s, %s) RETURNING id",
        (name, json.dumps(meta or {})),
    )
    return cur.fetchone()[0]


# ---------------------------------------------------------------------
# Public API expected by adv_ai_bot
# ---------------------------------------------------------------------
def get_vector_db() -> str:
    """Compatibility shim; returns the DSN string."""
    return PG_DSN


def add_to_collection(
    collection: str,
    items: Iterable[Dict[str, Any]],
    *,
    kind: Optional[str] = None,
) -> List[int]:
    """
    items: iterable of dicts with keys:
      - text (required)
      - external_id (optional)
      - metadata (optional dict)
      - kind (optional; overrides function arg)
    Returns the inserted/updated row ids.
    Upsert on (collection_id, external_id) when external_id is provided.
    """
    items = list(items)
    if not items:
        return []

    texts = [i["text"] for i in items]
    embs = _embed(texts)

    with _conn() as conn, conn.cursor() as cur:
        cid = _get_or_create_collection_id(cur, collection)

        ids: List[int] = []
        for i, it in enumerate(items):
            ext_id = it.get("external_id")
            meta = it.get("metadata") or {}
            _kind = it.get("kind") or kind or "doc"
            content = it["text"]
            vec = _to_vector_literal(embs[i])

            if ext_id:
                # upsert by (collection_id, external_id)
                cur.execute(
                    """
                    INSERT INTO ai_items (collection_id, external_id, kind, content, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (collection_id, external_id)
                    DO UPDATE SET kind=EXCLUDED.kind,
                                  content=EXCLUDED.content,
                                  metadata=EXCLUDED.metadata,
                                  embedding=EXCLUDED.embedding
                    RETURNING id
                    """,
                    (cid, ext_id, _kind, content, json.dumps(meta), vec),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO ai_items (collection_id, kind, content, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s::vector)
                    RETURNING id
                    """,
                    (cid, _kind, content, json.dumps(meta), vec),
                )
            ids.append(cur.fetchone()[0])
        return ids


def remove_from_collection(
    collection: str,
    external_ids: Optional[Iterable[str]] = None,
) -> int:
    """
    Delete by external_ids; if None -> delete entire collection content
    (not the collection row). Returns number of rows deleted.
    """
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM ai_collections WHERE name=%s", (collection,))
        row = cur.fetchone()
        if not row:
            return 0
        cid = row[0]

        if external_ids:
            ext = list(external_ids)
            if not ext:
                return 0
            cur.execute(
                "DELETE FROM ai_items WHERE collection_id=%s AND external_id = ANY(%s)",
                (cid, ext),
            )
            return cur.rowcount or 0
        else:
            cur.execute("DELETE FROM ai_items WHERE collection_id=%s", (cid,))
            return cur.rowcount or 0


def _search(
    collection: str,
    query_text: str,
    k: int = 5,
    *,
    kind: Optional[str] = None,
    min_score: Optional[float] = None,  # cosine similarity (0..1)
) -> List[Dict[str, Any]]:
    emb = _embed([query_text])[0]
    vec = _to_vector_literal(emb)

    with _conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute("SELECT id FROM ai_collections WHERE name=%s", (collection,))
        r = cur.fetchone()
        if not r:
            return []
        cid = r[0]

        where = ["collection_id=%s"]
        params: List[Any] = [cid]
        if kind:
            where.append("kind=%s")
            params.append(kind)

        sql = f"""
            SELECT id, external_id, kind, content, metadata,
                   (1 - (embedding <=> %s::vector)) AS score
            FROM ai_items
            WHERE {' AND '.join(where)}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        cur.execute(sql, [vec] + params + [vec, k])
        rows = cur.fetchall()

    results: List[Dict[str, Any]] = []
    for row in rows:
        d = dict(row)
        if min_score is not None and d["score"] < float(min_score):
            continue
        if isinstance(d.get("metadata"), str):
            try:
                d["metadata"] = json.loads(d["metadata"])
            except Exception:
                pass
        results.append(d)
    return results


def get_related_documents(collection: str, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
    return _search(collection, query_text, k=k, kind=None)


def get_related_articles(collection: str, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
    return _search(collection, query_text, k=k, kind="article")


def get_related_qas(collection: str, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
    return _search(collection, query_text, k=k, kind="qa")


# ---------------------------------------------------------------------
# Convenience wrappers expected by adv_ai_knowledge_base.py
# (flexible signatures; accept single dict, list of dicts, or field args)
# ---------------------------------------------------------------------
def _norm_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def add_article(
    collection: str,
    article: Optional[Union[Dict[str, Any], str]] = None,
    *,
    external_id: Optional[str] = None,
    title: Optional[str] = None,
    content: Optional[str] = None,
    text: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    articles: Optional[Iterable[Dict[str, Any]]] = None,
) -> List[int]:
    """
    Accepts:
      - a dict or list of dicts with keys like: external_id/id, title/name, content/text, metadata
      - or field args (external_id, title/content/text, metadata)
    """
    payload: List[Dict[str, Any]] = []

    for it in _norm_list(articles) + _norm_list(article):
        if isinstance(it, dict):
            ext = it.get("external_id") or it.get("id")
            ttl = it.get("title") or it.get("name")
            txt = it.get("content") or it.get("text") or ttl or ""
            meta = it.get("metadata") or {}
            payload.append({"external_id": ext, "text": txt, "metadata": meta, "kind": "article"})
        elif isinstance(it, str):
            payload.append({"external_id": None, "text": it, "metadata": {}, "kind": "article"})

    if not payload:
        # use explicit args
        txt = content or text or (title or "")
        payload = [{
            "external_id": external_id,
            "text": txt,
            "metadata": metadata or {},
            "kind": "article",
        }]

    return add_to_collection(collection, payload, kind="article")


def add_qa(
    collection: str,
    qa: Optional[Union[Dict[str, Any], str]] = None,
    *,
    external_id: Optional[str] = None,
    question: Optional[str] = None,
    answer: Optional[str] = None,
    text: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    qas: Optional[Iterable[Dict[str, Any]]] = None,
) -> List[int]:
    """
    Accepts:
      - a dict or list of dicts with keys: external_id/id, question, answer, text/content, metadata
      - or field args (external_id, question/answer/text, metadata)
    """
    payload: List[Dict[str, Any]] = []

    for it in _norm_list(qas) + _norm_list(qa):
        if isinstance(it, dict):
            ext = it.get("external_id") or it.get("id")
            q = it.get("question")
            a = it.get("answer")
            txt = it.get("text") or it.get("content") or (f"Q: {q}\nA: {a}" if (q or a) else "")
            meta = it.get("metadata") or {}
            payload.append({"external_id": ext, "text": txt, "metadata": meta, "kind": "qa"})
        elif isinstance(it, str):
            payload.append({"external_id": None, "text": it, "metadata": {}, "kind": "qa"})

    if not payload:
        txt = text or (f"Q: {question}\nA: {answer}" if (question or answer) else "")
        payload = [{
            "external_id": external_id,
            "text": txt,
            "metadata": metadata or {},
            "kind": "qa",
        }]

    return add_to_collection(collection, payload, kind="qa")


def remove_article(collection: str, external_ids: Union[str, Iterable[str]]) -> int:
    ids_list = _norm_list(external_ids)
    return remove_from_collection(collection, ids_list)


def remove_qa(collection: str, external_ids: Union[str, Iterable[str]]) -> int:
    ids_list = _norm_list(external_ids)
    return remove_from_collection(collection, ids_list)
