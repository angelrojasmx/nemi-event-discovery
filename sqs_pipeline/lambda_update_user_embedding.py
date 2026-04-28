"""
╔══════════════════════════════════════════════════════════════════╗
║         NEMI — Lambda: update_user_embedding                    ║
║                                                                  ║
║  Triggered cuando un usuario interactúa con un evento:          ║
║    · save     → señal fuerte (peso 1.0)                         ║
║    · purchase → señal muy fuerte (peso 2.0)                     ║
║    · view     → señal débil (peso 0.2)                          ║
║                                                                  ║
║  Flujo:                                                          ║
║    1. Lee user_id del mensaje SQS                               ║
║    2. Carga todas las interacciones del usuario                  ║
║    3. Carga los embeddings de los eventos interactuados          ║
║    4. Calcula promedio ponderado → perfil vectorial del usuario  ║
║    5. Upsert en user_embeddings                                 ║
║                                                                  ║
║  Por qué importa:                                                ║
║    · Activa el modo "warm user" del recommendation engine        ║
║    · Permite que notification_targeting calcule match scores     ║
║      reales en lugar de usar fallback por categoría             ║
║                                                                  ║
║  Idempotente: recalcular el mismo usuario siempre da el mismo   ║
║  resultado si las interacciones no cambiaron.                   ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import json
import math
import time
import logging
import datetime

from sqs_utils import (
    parse_sqs_records, get_supabase_client, lambda_response,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "text-embedding-3-small"

# Pesos por tipo de interacción (mismos que recommendation_engine.py)
SIGNAL_WEIGHTS = {
    "purchase": 2.0,
    "save":     1.0,
    "view":     0.2,
}

# Límite de historial a considerar — las 200 interacciones más recientes
MAX_INTERACTIONS = 200


# ─────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DEL PERFIL VECTORIAL
# ─────────────────────────────────────────────────────────────────

def build_user_profile(interactions: list[dict], embeddings_map: dict) -> list[float] | None:
    """
    Promedio ponderado de los embeddings de los eventos con los que
    el usuario interactuó. Vectores de compra valen más que saves,
    y saves valen más que vistas.

    Retorna vector normalizado de 1536 dims, o None si no hay
    suficiente historial con embeddings disponibles.
    """
    weighted_sum = None
    total_weight = 0.0
    events_used  = 0

    for interaction in interactions:
        event_id    = interaction.get("event_id")
        action_type = interaction.get("action_type", "view")
        weight      = SIGNAL_WEIGHTS.get(action_type, 0.2)

        vec = embeddings_map.get(event_id)
        if not vec:
            continue   # Sin embedding para este evento, saltar

        if weighted_sum is None:
            weighted_sum = [0.0] * len(vec)

        for i, v in enumerate(vec):
            weighted_sum[i] += v * weight
        total_weight += weight
        events_used  += 1

    if weighted_sum is None or total_weight == 0:
        return None

    # Promedio ponderado
    avg = [x / total_weight for x in weighted_sum]

    # Normalizar a norma 1 (necesario para que dot product == cosine similarity)
    norm = math.sqrt(sum(x * x for x in avg))
    if norm == 0:
        return None

    profile = [x / norm for x in avg]
    logger.info(f"Perfil construido con {events_used} eventos (de {len(interactions)} interacciones)")
    return profile


# ─────────────────────────────────────────────────────────────────
# GUARDAR EN SUPABASE
# ─────────────────────────────────────────────────────────────────

def save_user_embedding(sb, user_id: str, vector: list[float]) -> bool:
    """
    Upsert del perfil vectorial en user_embeddings.
    Si ya existía, lo sobreescribe con el perfil actualizado.
    """
    try:
        sb.table("user_embeddings").upsert({
            "user_id":       user_id,
            "model_version": EMBEDDING_MODEL,
            "computed_at":   datetime.datetime.utcnow().isoformat() + "+00:00",
            "embedding":     vector,
        }, on_conflict="user_id").execute()
        return True
    except Exception as e:
        logger.error(f"Error guardando user_embedding para {user_id}: {e}")
        return False


# ─────────────────────────────────────────────────────────────────
# HANDLER PRINCIPAL
# ─────────────────────────────────────────────────────────────────

def handler(event: dict, context) -> dict:
    """
    Lambda handler. Procesa uno o varios mensajes SQS.

    Mensaje SQS esperado:
    {
        "user_id": "uuid-del-usuario"
    }

    El mensaje lo manda el backend cada vez que registra una
    interacción nueva en user_event_interactions.
    """
    messages = parse_sqs_records(event)
    if not messages:
        return lambda_response(200, {"message": "Sin mensajes válidos", "processed": 0})

    sb      = get_supabase_client()
    results = []

    for msg in messages:
        user_id = msg.get("user_id", "")
        if not user_id:
            logger.warning("Mensaje sin user_id, saltando")
            continue

        t_start = time.time()
        outcome = {"user_id": user_id, "status": "error"}

        try:
            # 1. Cargar historial de interacciones del usuario
            interactions_resp = (
                sb.table("user_event_interactions")
                .select("event_id, action_type, created_at")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .limit(MAX_INTERACTIONS)
                .execute()
            )
            interactions = interactions_resp.data or []

            if not interactions:
                logger.info(f"Usuario sin interacciones: {user_id}")
                outcome["status"]  = "no_interactions"
                outcome["message"] = "Sin historial suficiente para construir perfil"
                results.append(outcome)
                continue

            # 2. Cargar embeddings de los eventos interactuados
            event_ids = list({i["event_id"] for i in interactions})
            emb_resp  = (
                sb.table("event_embeddings")
                .select("event_id, embedding")
                .in_("event_id", event_ids)
                .execute()
            )
            embeddings_map = {}
            for row in (emb_resp.data or []):
                vec = row["embedding"]
                if isinstance(vec, str):
                    try: vec = json.loads(vec)
                    except: continue
                embeddings_map[row["event_id"]] = vec

            if not embeddings_map:
                logger.warning(f"Ningún evento con embedding para usuario {user_id}")
                outcome["status"]  = "no_embeddings"
                outcome["message"] = "Los eventos interactuados no tienen embeddings aún"
                results.append(outcome)
                continue

            # 3. Construir perfil vectorial
            profile = build_user_profile(interactions, embeddings_map)

            if profile is None:
                outcome["status"]  = "profile_failed"
                outcome["message"] = "No se pudo construir el perfil (embeddings insuficientes)"
                results.append(outcome)
                continue

            # 4. Guardar en user_embeddings
            saved = save_user_embedding(sb, user_id, profile)

            if saved:
                outcome["status"]       = "ok"
                outcome["interactions"] = len(interactions)
                outcome["events_used"]  = len(embeddings_map)
                logger.info(
                    f"✅ Perfil actualizado para {user_id} "
                    f"({len(embeddings_map)} eventos, {len(interactions)} interacciones)"
                )
            else:
                outcome["status"] = "save_failed"

        except Exception as e:
            logger.error(f"Error procesando usuario {user_id}: {e}")
            outcome["error"] = str(e)

        outcome["latency_ms"] = int((time.time() - t_start) * 1000)
        results.append(outcome)

    ok_count = sum(1 for r in results if r["status"] == "ok")
    logger.info(f"Batch completado: {ok_count}/{len(results)} exitosos")

    return lambda_response(200, {
        "processed": len(results),
        "ok":        ok_count,
        "results":   results,
    })
