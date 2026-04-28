"""
╔══════════════════════════════════════════════════════════════════╗
║         NEMI — Utilidades compartidas del pipeline SQS          ║
╚══════════════════════════════════════════════════════════════════╝

Funciones comunes usadas por lambda_enrich_event y lambda_embed_event.
"""

import os
import json
import logging
import datetime

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# CONFIGURACIÓN (via variables de entorno en Lambda)
# ─────────────────────────────────────────────────────────────────

SUPABASE_URL = os.getenv("SUPABASE_URL",        "https://imzjqgnlphbddlrfocei.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY", "")

SQS_EMBED_QUEUE_URL = os.getenv("SQS_EMBED_QUEUE_URL", "")   # ARN de la cola nemi-events-to-embed


# ─────────────────────────────────────────────────────────────────
# PARSEO DE MENSAJES SQS
# ─────────────────────────────────────────────────────────────────

def parse_sqs_records(event: dict) -> list[dict]:
    """
    Extrae los mensajes de un evento SQS de Lambda.
    Cada record tiene un body JSON con al menos { "event_id": "uuid" }.

    Formato esperado del mensaje:
    {
        "event_id":  "uuid-del-evento",
        "source":    "partner_publish",   # contexto (opcional)
        "timestamp": "2026-04-03T12:00:00Z"
    }
    """
    messages = []
    for record in event.get("Records", []):
        try:
            body = json.loads(record.get("body", "{}"))
            if "event_id" not in body:
                logger.warning(f"Mensaje sin event_id ignorado: {body}")
                continue
            messages.append(body)
        except json.JSONDecodeError as e:
            logger.error(f"Error parseando mensaje SQS: {e} | body: {record.get('body')}")
    return messages


# ─────────────────────────────────────────────────────────────────
# CLIENTE SUPABASE
# ─────────────────────────────────────────────────────────────────

def get_supabase_client():
    from supabase import create_client
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Faltan SUPABASE_URL o SUPABASE_SECRET_KEY en variables de entorno")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_event(sb, event_id: str) -> dict | None:
    """Carga un evento completo de Supabase por ID."""
    resp = (
        sb.table("events")
        .select(
            "id, title, description, category, status, start_date, "
            "venue_id, source, ai_tags"
        )
        .eq("id", event_id)
        .single()
        .execute()
    )
    return resp.data


def fetch_venue_name(sb, venue_id: str) -> str:
    """Carga el nombre del venue por ID."""
    if not venue_id:
        return ""
    try:
        resp = sb.table("venues").select("name").eq("id", venue_id).single().execute()
        return (resp.data or {}).get("name", "")
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────
# ENVÍO A SQS
# ─────────────────────────────────────────────────────────────────

def send_to_embed_queue(event_id: str) -> bool:
    """
    Envía un mensaje a la cola nemi-events-to-embed para disparar
    la generación de embedding una vez que el tagging terminó.
    """
    if not SQS_EMBED_QUEUE_URL:
        logger.warning("SQS_EMBED_QUEUE_URL no configurado — saltando envío a cola de embeddings")
        return False
    try:
        import boto3
        sqs = boto3.client("sqs")
        sqs.send_message(
            QueueUrl    = SQS_EMBED_QUEUE_URL,
            MessageBody = json.dumps({
                "event_id":  event_id,
                "source":    "post_tagging",
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            }),
        )
        logger.info(f"Enviado a embed queue: {event_id}")
        return True
    except Exception as e:
        logger.error(f"Error enviando a SQS embed queue: {e}")
        return False


# ─────────────────────────────────────────────────────────────────
# RESPUESTA LAMBDA ESTÁNDAR
# ─────────────────────────────────────────────────────────────────

def lambda_response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "body": json.dumps(body, ensure_ascii=False, default=str),
    }
