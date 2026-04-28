"""
╔══════════════════════════════════════════════════════════════════╗
║         NEMI — Lambda: enrich_event                             ║
║                                                                  ║
║  Triggered por: SQS queue "nemi-events-to-tag"                  ║
║  Cuando:        Un partner publica un evento nuevo               ║
║                                                                  ║
║  Flujo:                                                          ║
║    1. Lee event_id del mensaje SQS                               ║
║    2. Fetcha el evento de Supabase                               ║
║    3. Genera ai_tags con Gemini (fallback: Claude Haiku)         ║
║    4. Guarda ai_tags en events.ai_tags                           ║
║    5. Envía event_id a la cola nemi-events-to-embed              ║
║                                                                  ║
║  Garantía: si el tagging falla, el evento queda publicado        ║
║  igual — nunca bloquea el flujo principal.                       ║
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
    send_to_embed_queue, lambda_response,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────

GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY",    "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

PRIMARY_MODEL  = {"provider": "gemini",    "model_id": "gemini-2.5-flash",       "cost_input_per_1m": 0.15, "cost_output_per_1m": 0.60}
FALLBACK_MODEL = {"provider": "anthropic", "model_id": "claude-haiku-4-5-20251001", "cost_input_per_1m": 0.80, "cost_output_per_1m": 4.00}

# ─────────────────────────────────────────────────────────────────
# PROMPT (mismo que tagging_pipeline.py — versionado aquí)
# ─────────────────────────────────────────────────────────────────

PROMPT_VERSION = "v2"

SYSTEM_PROMPT = """Eres un sistema de clasificación de eventos para una app de discovery en Monterrey, México.
Dado el título, descripción y venue de un evento, extrae metadatos estructurados.

REGLAS ESTRICTAS:
- Responde SOLO con un JSON válido, sin texto adicional ni markdown
- Solo llena un campo si hay evidencia clara en el texto o el título
- Si no hay evidencia suficiente, usa null (campos simples) o [] (listas)
- NO inventes ni supongas más allá del texto disponible
- Es mejor un campo vacío que información incorrecta
- Todos los valores de texto en minúsculas y sin acentos

SCHEMA — usa exactamente estos campos y valores:
{
  "mood": [],
  "ambiente": [],
  "actividad": [],
  "ideal_para": [],
  "dress_code": null,
  "es_18_plus": null,
  "duracion_estimada_hrs": null
}

VALORES PERMITIDOS:
- mood       (máx 3): romantico, divertido, energico, relajado, cultural, familiar, nocturno, sofisticado, espiritual, deportivo, creativo, intenso
- ambiente   (máx 3): intimo, masivo, al_aire_libre, bar, teatro, estadio, galeria, salon, casual, adulto, familiar
- actividad  (máx 2): concierto, teatro, deporte, exposicion, cine, taller, festival, gastronomia, danza, comedia, conferencia, lucha_libre, musica_en_vivo
- ideal_para (máx 3): pareja, amigos, familia, solo, cita, grupos_grandes, ninos, profesionales, estudiantes, jovenes
- dress_code       : "casual" | "smart_casual" | "formal" | null
- es_18_plus       : true | false | null  (solo si hay mención explícita)
- duracion_estimada_hrs : número decimal | null"""

VALID_VALUES = {
    "mood":       {"romantico","divertido","energico","relajado","cultural","familiar","nocturno","sofisticado","espiritual","deportivo","creativo","intenso"},
    "ambiente":   {"intimo","masivo","al_aire_libre","bar","teatro","estadio","galeria","salon","casual","adulto","familiar"},
    "actividad":  {"concierto","teatro","deporte","exposicion","cine","taller","festival","gastronomia","danza","comedia","conferencia","lucha_libre","musica_en_vivo"},
    "ideal_para": {"pareja","amigos","familia","solo","cita","grupos_grandes","ninos","profesionales","estudiantes","jovenes"},
}

# ─────────────────────────────────────────────────────────────────
# TAGGING
# ─────────────────────────────────────────────────────────────────

def build_prompt(event: dict, venue_name: str) -> str:
    title       = (event.get("title") or "").strip()
    description = (event.get("description") or "").strip() or "(sin descripción)"
    venue       = venue_name.strip() or "(desconocido)"
    return f"Título: {title}\nDescripción: {description}\nVenue: {venue}"


def call_gemini(user_message: str) -> dict:
    from openai import OpenAI
    client = OpenAI(
        api_key  = GEMINI_API_KEY,
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    t0 = time.time()
    resp = client.chat.completions.create(
        model       = PRIMARY_MODEL["model_id"],
        messages    = [{"role": "system", "content": SYSTEM_PROMPT},
                       {"role": "user",   "content": user_message}],
        temperature = 0,
        max_tokens  = 300,
    )
    return {
        "text":          resp.choices[0].message.content or "",
        "tokens_input":  resp.usage.prompt_tokens if resp.usage else 0,
        "tokens_output": resp.usage.completion_tokens if resp.usage else 0,
        "latency_ms":    int((time.time() - t0) * 1000),
        "model":         PRIMARY_MODEL["model_id"],
    }


def call_claude(user_message: str) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    t0 = time.time()
    resp = client.messages.create(
        model       = FALLBACK_MODEL["model_id"],
        system      = SYSTEM_PROMPT,
        messages    = [{"role": "user", "content": user_message}],
        max_tokens  = 300,
        temperature = 0,
    )
    return {
        "text":          resp.content[0].text if resp.content else "",
        "tokens_input":  resp.usage.input_tokens if resp.usage else 0,
        "tokens_output": resp.usage.output_tokens if resp.usage else 0,
        "latency_ms":    int((time.time() - t0) * 1000),
        "model":         FALLBACK_MODEL["model_id"],
    }


def parse_and_validate(raw_text: str) -> dict | None:
    """Parsea el JSON del modelo y valida valores contra el vocabulario."""
    text = raw_text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    required = ["mood", "ambiente", "actividad", "ideal_para", "dress_code", "es_18_plus", "duracion_estimada_hrs"]
    for field in required:
        if field not in data:
            data[field] = [] if field in VALID_VALUES else None

    for field, valid_set in VALID_VALUES.items():
        vals = data.get(field, [])
        if isinstance(vals, list):
            data[field] = [v.lower().strip() for v in vals if isinstance(v, str) and v.lower().strip() in valid_set]

    return data


def tag_event(event: dict, venue_name: str) -> tuple[dict | None, str, bool]:
    """
    Intenta taggear un evento.
    Retorna: (tags_dict | None, model_used, fallback_used)
    """
    prompt = build_prompt(event, venue_name)

    # Intento 1: Gemini
    try:
        result = call_gemini(prompt)
        tags   = parse_and_validate(result["text"])
        if tags is not None:
            return tags, result["model"], False
    except Exception as e:
        logger.warning(f"Gemini falló: {e}")

    # Fallback: Claude Haiku
    try:
        result = call_claude(prompt)
        tags   = parse_and_validate(result["text"])
        if tags is not None:
            return tags, result["model"], True
    except Exception as e:
        logger.warning(f"Claude falló: {e}")

    return None, "none", False


# ─────────────────────────────────────────────────────────────────
# GUARDAR EN SUPABASE
# ─────────────────────────────────────────────────────────────────

def save_ai_tags(sb, event_id: str, tags: dict) -> bool:
    """Guarda los ai_tags en la tabla events."""
    try:
        sb.table("events").update({
            "ai_tags":    tags,
            "updated_at": datetime.datetime.utcnow().isoformat() + "+00:00",
        }).eq("id", event_id).execute()
        return True
    except Exception as e:
        logger.error(f"Error guardando ai_tags para {event_id}: {e}")
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

    sb = get_supabase_client()
    results = []

    for msg in messages:
        event_id = msg["event_id"]
        t_start  = time.time()
        outcome  = {"event_id": event_id, "status": "error"}

        try:
            # 1. Fetch evento
            ev = fetch_event(sb, event_id)
            if not ev:
                logger.warning(f"Evento no encontrado: {event_id}")
                outcome["status"] = "not_found"
                results.append(outcome)
                continue

            # 2. Si ya tiene ai_tags, saltar (idempotente)
            if ev.get("ai_tags"):
                logger.info(f"Evento ya taggeado, saltando: {event_id}")
                send_to_embed_queue(event_id)
                outcome["status"] = "already_tagged"
                results.append(outcome)
                continue

            # 3. Fetch venue
            venue_name = fetch_venue_name(sb, ev.get("venue_id", ""))

            # 4. Tagging
            tags, model_used, fallback = tag_event(ev, venue_name)

            if tags is None:
                # Tagging falló — evento sigue publicado, sin ai_tags
                logger.error(f"Tagging falló para {event_id} — publicando sin ai_tags")
                outcome["status"]  = "tagging_failed"
                outcome["message"] = "Evento publicado sin ai_tags"
            else:
                # 5. Guardar ai_tags
                saved = save_ai_tags(sb, event_id, tags)
                if saved:
                    # 6. Disparar Lambda de embeddings
                    send_to_embed_queue(event_id)
                    outcome["status"]   = "ok"
                    outcome["model"]    = model_used
                    outcome["fallback"] = fallback
                    logger.info(f"✅ Taggeado [{model_used}]: {ev.get('title','')[:50]}")
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
