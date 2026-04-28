"""
╔══════════════════════════════════════════════════════════════════╗
║         NEMI — Lambda: embed_event                              ║
║                                                                  ║
║  Triggered por: SQS queue "nemi-events-to-embed"                ║
║  Cuando:        lambda_enrich_event termina el tagging           ║
║                                                                  ║
║  Flujo:                                                          ║
║    1. Lee event_id del mensaje SQS                               ║
║    2. Fetcha el evento + ai_tags de Supabase                     ║
║    3. Construye enriched_text (título + descripción + venue      ║
║       + ai_tags)                                                 ║
║    4. Genera embedding con text-embedding-3-small (OpenAI)       ║
║    5. Guarda en event_embeddings (upsert idempotente)            ║
║                                                                  ║
║  Garantía: si el embedding falla, el evento sigue publicado      ║
║  y puede re-intentarse en el siguiente mensaje SQS.              ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import json
import time
import logging
import datetime

from sqs_utils import (
    parse_sqs_records, get_supabase_client,
    fetch_event, fetch_venue_name,
    lambda_response,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS  = 1536

COST_PER_1M_TOKENS = 0.020   # USD — text-embedding-3-small


# ─────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DEL ENRICHED TEXT
# ─────────────────────────────────────────────────────────────────

def build_enriched_text(event: dict, venue_name: str) -> str:
    """
    Construye el texto que se va a embeber combinando:
      - Contenido semántico: título + descripción + venue
      - Clasificación estructurada: ai_tags (mood, ambiente, actividad, ideal_para)

    Esto hace que el embedding capture ambas dimensiones del evento,
    mejorando la calidad de retrieval semántico.
    """
    title       = (event.get("title") or "").strip()
    description = (event.get("description") or "").strip()
    venue       = venue_name.strip()

    parts = []
    if title:
        parts.append(f"Título: {title}")
    if description:
        parts.append(f"Descripción: {description}")
    if venue:
        parts.append(f"Venue: {venue}")

    # ai_tags puede venir como dict o como string JSON
    tags = event.get("ai_tags") or {}
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except Exception:
            tags = {}

    for field in ["mood", "ambiente", "actividad", "ideal_para"]:
        vals = tags.get(field, [])
        if vals:
            label = field.capitalize().replace("_", " ")
            parts.append(f"{label}: {', '.join(vals)}")

    if tags.get("dress_code"):
        parts.append(f"Dress code: {tags['dress_code']}")

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────
# GENERACIÓN DE EMBEDDING
# ─────────────────────────────────────────────────────────────────

def generate_embedding(text: str) -> tuple[list[float], int]:
    """
    Genera el embedding de un texto con text-embedding-3-small.
    Retorna: (vector, tokens_used)
    """
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text.strip(),
    )
    vector = response.data[0].embedding
    tokens = response.usage.total_tokens if response.usage else 0
    return vector, tokens


# ─────────────────────────────────────────────────────────────────
# GUARDAR EN SUPABASE
# ─────────────────────────────────────────────────────────────────

def save_embedding(sb, event_id: str, vector: list[float]) -> bool:
    """
    Upsert del embedding en event_embeddings.
    Si ya existía un embedding para este event_id, lo sobreescribe
    (upsert idempotente por event_id PK).
    """
    try:
        sb.table("event_embeddings").upsert({
            "event_id":     event_id,
            "model_version": EMBEDDING_MODEL,
            "computed_at":  datetime.datetime.utcnow().isoformat() + "+00:00",
            "embedding":    vector,
        }, on_conflict="event_id").execute()
        return True
    except Exception as e:
        logger.error(f"Error guardando embedding para {event_id}: {e}")
        return False


# ─────────────────────────────────────────────────────────────────
# HANDLER PRINCIPAL
# ─────────────────────────────────────────────────────────────────

def handler(event: dict, context) -> dict:
    """
    Lambda handler. Procesa uno o varios mensajes SQS en batch.
    SQS puede enviar hasta 10 mensajes por invocación.
    """
    messages = parse_sqs_records(event)
    if not messages:
        return lambda_response(200, {"message": "Sin mensajes válidos", "processed": 0})

    sb      = get_supabase_client()
    results = []

    for msg in messages:
        event_id = msg["event_id"]
        t_start  = time.time()
        outcome  = {"event_id": event_id, "status": "error"}

        try:
            # 1. Fetch evento (incluye ai_tags)
            ev = fetch_event(sb, event_id)
            if not ev:
                logger.warning(f"Evento no encontrado: {event_id}")
                outcome["status"] = "not_found"
                results.append(outcome)
                continue

            # 2. Fetch venue
            venue_name = fetch_venue_name(sb, ev.get("venue_id", ""))

            # 3. Construir enriched text
            enriched = build_enriched_text(ev, venue_name)
            if not enriched.strip():
                logger.warning(f"Enriched text vacío para {event_id} — saltando embedding")
                outcome["status"]  = "empty_text"
                outcome["message"] = "Sin texto suficiente para embeber"
                results.append(outcome)
                continue

            # 4. Verificar si ya tiene embedding actualizado (idempotente)
            #    Solo re-embebe si no existe o si se forzó (mensaje con force=True)
            force = msg.get("force", False)
            if not force:
                existing = (
                    sb.table("event_embeddings")
                    .select("event_id")
                    .eq("event_id", event_id)
                    .execute()
                )
                if existing.data:
                    logger.info(f"Embedding ya existe, saltando: {event_id}")
                    outcome["status"] = "already_embedded"
                    results.append(outcome)
                    continue

            # 5. Generar embedding
            t0 = time.time()
            try:
                vector, tokens_used = generate_embedding(enriched)
            except Exception as e:
                logger.error(f"OpenAI falló para {event_id}: {e}")
                outcome["status"]  = "embedding_failed"
                outcome["error"]   = str(e)
                results.append(outcome)
                continue
            t_embed = int((time.time() - t0) * 1000)

            # 6. Guardar en Supabase
            saved = save_embedding(sb, event_id, vector)
            if saved:
                cost = (tokens_used / 1_000_000) * COST_PER_1M_TOKENS
                outcome["status"]      = "ok"
                outcome["tokens"]      = tokens_used
                outcome["cost_usd"]    = round(cost, 8)
                outcome["latency_embedding_ms"] = t_embed
                logger.info(
                    f"✅ Embedded [{tokens_used} tokens, ${cost:.6f}]: "
                    f"{ev.get('title','')[:50]}"
                )
            else:
                outcome["status"] = "save_failed"

        except Exception as e:
            logger.error(f"Error procesando {event_id}: {e}")
            outcome["error"] = str(e)

        outcome["latency_ms"] = int((time.time() - t_start) * 1000)
        results.append(outcome)

    ok_count = sum(1 for r in results if r["status"] == "ok")
    logger.info(f"Batch completado: {ok_count}/{len(results)} exitosos")

    return lambda_response(200, {"processed": len(results), "ok": ok_count, "results": results})
