"""
╔══════════════════════════════════════════════════════════════════╗
║         NEMI — Búsqueda Semántica de Eventos                    ║
║         NEM-110                                                  ║
║                                                                  ║
║  Convierte una query de texto en embedding y busca eventos       ║
║  relevantes usando similitud coseno en pgvector (Supabase).      ║
║                                                                  ║
║  Modo de uso:                                                    ║
║    from semantic_search import search_events                     ║
║    results = search_events("algo para bailar este viernes")      ║
║                                                                  ║
║  Como Lambda (AWS):                                              ║
║    Ver handler() al final del archivo                            ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import time
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
SUPABASE_URL    = os.getenv("SUPABASE_URL",    "https://imzjqgnlphbddlrfocei.supabase.co")
SUPABASE_KEY    = os.getenv("SUPABASE_SECRET_KEY", "")   # Secret key — no usar publishable

EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CITY    = "Monterrey"
DEFAULT_LIMIT   = 20
DEFAULT_THRESHOLD = 0.30   # Score semántico mínimo (0-1)
RESPONSE_TIMEOUT_MS = 300  # Advertir si se supera

# ─────────────────────────────────────────────────────────────────
# EMBEDDING DEL QUERY
# ─────────────────────────────────────────────────────────────────

def embed_query(query_text: str) -> list[float]:
    """
    Convierte el texto del query en un vector de 1536 dims.
    Usa el mismo modelo que el pipeline de eventos (text-embedding-3-small)
    para garantizar que el espacio vectorial sea compatible.
    """
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query_text.strip(),
    )
    return response.data[0].embedding


# ─────────────────────────────────────────────────────────────────
# BÚSQUEDA EN SUPABASE
# ─────────────────────────────────────────────────────────────────

def search_events(
    query_text:      str,
    city:            str   = DEFAULT_CITY,
    limit:           int   = DEFAULT_LIMIT,
    threshold:       float = DEFAULT_THRESHOLD,
    date_from:       Optional[str] = None,   # "YYYY-MM-DD" — filtro adicional
    price_max:       Optional[float] = None, # precio máximo (price_min <= price_max)
    category:        Optional[str] = None,   # filtro de categoría post-query
) -> dict:
    """
    Busca eventos relevantes para el query dado.

    Args:
        query_text:  Texto libre del usuario ("algo para bailar", "rock en vivo")
        city:        Ciudad para filtrar (default: Monterrey)
        limit:       Máximo de resultados a devolver
        threshold:   Score semántico mínimo — bajar si hay pocos resultados
        date_from:   Solo eventos desde esta fecha (default: hoy, manejado en SQL)
        price_max:   Filtro de precio máximo post-query
        category:    Filtro de categoría post-query ("music", "sports", etc.)

    Returns:
        {
          "results": [...],       # lista de eventos ordenados por relevancia
          "query":   "...",       # query original
          "count":   N,
          "latency_ms": {
            "embedding": N,
            "search": N,
            "total": N
          }
        }
    """
    t_start = time.time()

    # 1. Embeber el query
    t0 = time.time()
    try:
        query_embedding = embed_query(query_text)
    except Exception as e:
        logger.error(f"Error al embeber query: {e}")
        # Fallback a búsqueda por texto exacto
        return _fallback_text_search(query_text, city, limit)
    t_embed = int((time.time() - t0) * 1000)

    # 2. Buscar en Supabase via RPC
    t0 = time.time()
    try:
        from supabase import create_client
        client = create_client(SUPABASE_URL, SUPABASE_KEY)

        response = client.rpc(
            "match_events",
            {
                "query_embedding": query_embedding,
                "query_text":      query_text,
                "filter_city":     city,
                "match_count":     limit * 2,   # Pedir más para filtrar después
                "match_threshold": threshold,
            }
        ).execute()

        raw_results = response.data or []

    except Exception as e:
        logger.error(f"Error en búsqueda vectorial: {e}")
        return _fallback_text_search(query_text, city, limit)

    t_search = int((time.time() - t0) * 1000)

    # 3. Post-filtros opcionales (aplicados en Python para no complicar el SQL)
    results = raw_results
    if price_max is not None:
        results = [r for r in results if (r.get("price_min") or 0) <= price_max]
    if category:
        results = [r for r in results if r.get("category","").lower() == category.lower()]
    if date_from:
        results = [r for r in results if (r.get("start_date") or "") >= date_from]

    # Limitar al count pedido original
    results = results[:limit]

    t_total = int((time.time() - t_start) * 1000)

    if t_total > RESPONSE_TIMEOUT_MS:
        logger.warning(f"⚠️  Latencia {t_total}ms supera el objetivo de {RESPONSE_TIMEOUT_MS}ms")

    return {
        "results":    _format_results(results),
        "query":      query_text,
        "count":      len(results),
        "latency_ms": {
            "embedding": t_embed,
            "search":    t_search,
            "total":     t_total,
        },
    }


# ─────────────────────────────────────────────────────────────────
# FALLBACK — Búsqueda por texto exacto
# ─────────────────────────────────────────────────────────────────

def _fallback_text_search(query_text: str, city: str, limit: int) -> dict:
    """
    Fallback a búsqueda por ILIKE cuando el vector DB no está disponible.
    Cumple el acceptance criteria de NEM-110.
    """
    logger.warning("Usando fallback a búsqueda por texto exacto")
    try:
        from supabase import create_client
        client = create_client(SUPABASE_URL, SUPABASE_KEY)

        response = (
            client.table("events")
            .select("id, title, description, category, start_date, start_time, price_tier, price_min, image_url, ai_tags, source")
            .ilike("title", f"%{query_text}%")
            .eq("status", "published")
            .gte("start_date", time.strftime("%Y-%m-%d"))
            .limit(limit)
            .execute()
        )
        results = response.data or []
        # Añadir scores dummy para mantener el mismo formato
        for r in results:
            r["semantic_score"] = 0.0
            r["text_score"]     = 1.0
            r["final_score"]    = 1.0

        return {
            "results":    _format_results(results),
            "query":      query_text,
            "count":      len(results),
            "fallback":   True,
            "latency_ms": {"total": 0},
        }
    except Exception as e:
        logger.error(f"Error en fallback: {e}")
        return {"results": [], "query": query_text, "count": 0, "error": str(e)}


# ─────────────────────────────────────────────────────────────────
# FORMATO DE RESULTADOS
# ─────────────────────────────────────────────────────────────────

def _format_results(raw: list[dict]) -> list[dict]:
    """Limpia y normaliza los resultados para la respuesta al frontend."""
    out = []
    for r in raw:
        # Parsear ai_tags si viene como string
        ai_tags = r.get("ai_tags") or {}
        if isinstance(ai_tags, str):
            try:
                ai_tags = json.loads(ai_tags)
            except Exception:
                ai_tags = {}

        out.append({
            "id":            r.get("id"),
            "title":         r.get("title"),
            "description":   (r.get("description") or "")[:300],  # Truncar para la respuesta
            "category":      r.get("category"),
            "start_date":    r.get("start_date"),
            "start_time":    str(r.get("start_time") or ""),
            "price_tier":    r.get("price_tier"),
            "price_min":     r.get("price_min"),
            "image_url":     r.get("image_url"),
            "venue_name":    r.get("venue_name"),
            "venue_city":    r.get("venue_city"),
            "source_url":    r.get("source_url"),
            "ai_tags":       ai_tags,
            # Scores de relevancia (útiles para debugging / A/B testing)
            "_scores": {
                "semantic": round(r.get("semantic_score", 0), 4),
                "text":     round(r.get("text_score", 0), 4),
                "final":    round(r.get("final_score", 0), 4),
            },
        })
    return out


# ─────────────────────────────────────────────────────────────────
# AWS LAMBDA HANDLER
# ─────────────────────────────────────────────────────────────────

def handler(event: dict, context) -> dict:
    """
    Entry point para AWS Lambda.

    Evento esperado (desde API Gateway):
    {
        "queryStringParameters": {
            "q":         "algo para bailar",   # requerido
            "city":      "Monterrey",           # opcional
            "limit":     "20",                  # opcional
            "threshold": "0.3",                 # opcional
            "price_max": "500",                 # opcional
            "category":  "music"               # opcional
        }
    }
    """
    params = event.get("queryStringParameters") or {}

    query_text = params.get("q", "").strip()
    if not query_text:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Parámetro 'q' es requerido"}),
        }

    city      = params.get("city", DEFAULT_CITY)
    limit     = int(params.get("limit", DEFAULT_LIMIT))
    threshold = float(params.get("threshold", DEFAULT_THRESHOLD))
    price_max = float(params["price_max"]) if "price_max" in params else None
    category  = params.get("category")

    result = search_events(
        query_text=query_text,
        city=city,
        limit=min(limit, 50),   # Cap de seguridad
        threshold=threshold,
        price_max=price_max,
        category=category,
    )

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type":                "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(result, ensure_ascii=False, default=str),
    }


# ─────────────────────────────────────────────────────────────────
# MODO LOCAL — para pruebas directas
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "concierto de rock"
    print(f"\n🔍 Buscando: '{query}'\n")

    result = search_events(query, city="Monterrey", limit=10)

    print(f"Resultados: {result['count']}")
    print(f"Latencia:   embed={result['latency_ms'].get('embedding')}ms  "
          f"search={result['latency_ms'].get('search')}ms  "
          f"total={result['latency_ms'].get('total')}ms\n")

    for i, r in enumerate(result["results"], 1):
        scores = r["_scores"]
        print(f"  {i:2d}. [{scores['final']:.3f}] {r['title'][:55]}")
        print(f"       {r['venue_name']} | {r['start_date']} | {r['price_tier']}")
        if r["ai_tags"].get("actividad"):
            print(f"       tags: {r['ai_tags'].get('actividad')} / {r['ai_tags'].get('mood')}")
        print()
