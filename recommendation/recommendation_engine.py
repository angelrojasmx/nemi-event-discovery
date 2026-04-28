"""
╔══════════════════════════════════════════════════════════════════╗
║         NEMI — Recommendation Engine (Feed Personalizado)       ║
║         NEM-109                                                  ║
║                                                                  ║
║  Genera el feed personalizado de eventos para cada usuario.      ║
║                                                                  ║
║  Dos modos:                                                      ║
║    · Cold start  — usuario nuevo sin historial                   ║
║    · Warm user   — ranking por similitud con perfil del usuario  ║
║                                                                  ║
║  A/B testing via env var RANKING_STRATEGY:                       ║
║    "semantic"   — perfil de usuario + similitud coseno (default) ║
║    "popularity" — temporal + popularidad (fallback / control)    ║
║    "hybrid"     — mezcla de ambos                                ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import json
import time
import math
import logging
import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────

SUPABASE_URL = os.getenv("SUPABASE_URL",        "https://imzjqgnlphbddlrfocei.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY", "")

# A/B testing: cambiar sin redeploy via variable de entorno
RANKING_STRATEGY = os.getenv("RANKING_STRATEGY", "semantic")  # "semantic" | "popularity" | "hybrid"

DEFAULT_CITY  = "Monterrey"
DEFAULT_LIMIT = 20

# Pesos del score final (modo semantic/hybrid)
W_SEMANTIC   = 0.50   # Similitud con perfil del usuario
W_TEMPORAL   = 0.20   # Qué tan pronto es el evento
W_POPULARITY = 0.15   # Popularidad / calidad del evento
W_FRESHNESS  = 0.15   # Qué tan reciente fue agregado

# Pesos del perfil de usuario (qué acciones valen más)
USER_SIGNAL_WEIGHTS = {
    "purchase": 2.0,   # Compra = señal más fuerte
    "save":     1.0,   # Save = señal fuerte
    "view":     0.2,   # Vista sin acción = señal débil
}


# ─────────────────────────────────────────────────────────────────
# CÁLCULO DE SCORES INDIVIDUALES
# ─────────────────────────────────────────────────────────────────

def temporal_score(start_date_str: str) -> float:
    """
    Score de proximidad temporal (0-1).
    Prioriza eventos próximos sobre lejanos.

    Hoy:              1.00
    Mañana:           0.95
    Esta semana:      0.80
    Próxima semana:   0.60
    Este mes:         0.40
    Más de 30 días:   0.20
    """
    if not start_date_str:
        return 0.20
    try:
        event_date = datetime.date.fromisoformat(start_date_str[:10])
        today      = datetime.date.today()
        days_away  = (event_date - today).days
        if days_away < 0:
            return 0.0   # Evento pasado — no debería aparecer pero por si acaso
        elif days_away == 0:
            return 1.00
        elif days_away == 1:
            return 0.95
        elif days_away <= 7:
            return 0.80
        elif days_away <= 14:
            return 0.60
        elif days_away <= 30:
            return 0.40
        else:
            return 0.20
    except Exception:
        return 0.20


def popularity_score(event: dict) -> float:
    """
    Score de popularidad (0-1).
    Por ahora usa quality_score y source_trust_score de la tabla events
    como proxy. Cuando tengamos datos reales de tickets vendidos /
    registros, se reemplaza esta función.
    """
    quality     = float(event.get("quality_score")      or 0.5)
    trust       = float(event.get("source_trust_score") or 0.5)
    # Promedio ponderado: confiamos más en quality que en trust
    return min(1.0, quality * 0.7 + trust * 0.3)


def freshness_score(event: dict) -> float:
    """
    Score de frescura (0-1): qué tan reciente fue agregado el evento.
    Eventos agregados en los últimos 3 días tienen boost de novedad.
    """
    scraped_at = event.get("created_at") or event.get("scraped_at") or ""
    if not scraped_at:
        return 0.5
    try:
        # Parsear timestamp (puede venir con timezone)
        ts = datetime.datetime.fromisoformat(scraped_at[:19])
        days_old = (datetime.datetime.utcnow() - ts).days
        if days_old <= 1:   return 1.00
        elif days_old <= 3:  return 0.90
        elif days_old <= 7:  return 0.70
        elif days_old <= 14: return 0.50
        else:                return 0.30
    except Exception:
        return 0.5


def cosine_similarity(a: list, b: list) -> float:
    """Similitud coseno entre dos vectores (asume que están normalizados → dot product)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    # Si los vectores están normalizados (norma ~1), dot product = cosine similarity
    return max(0.0, min(1.0, dot))


# ─────────────────────────────────────────────────────────────────
# PERFIL DE USUARIO
# ─────────────────────────────────────────────────────────────────

def build_user_profile(user_interactions: list[dict], embeddings_map: dict) -> Optional[list[float]]:
    """
    Construye el embedding de perfil del usuario como promedio ponderado
    de los embeddings de los eventos con los que interactuó.

    Args:
        user_interactions: lista de {event_id, action_type, created_at}
                           action_type: "purchase" | "save" | "view"
        embeddings_map:    {event_id: [float, ...]} — vectores de 1536 dims

    Returns:
        Vector normalizado de 1536 dims, o None si no hay historial útil.
    """
    weighted_sum = None
    total_weight = 0.0

    for interaction in user_interactions:
        event_id    = interaction.get("event_id")
        action_type = interaction.get("action_type", "view")
        weight      = USER_SIGNAL_WEIGHTS.get(action_type, 0.2)

        vec = embeddings_map.get(event_id)
        if not vec:
            continue

        if weighted_sum is None:
            weighted_sum = [0.0] * len(vec)

        for i, v in enumerate(vec):
            weighted_sum[i] += v * weight
        total_weight += weight

    if weighted_sum is None or total_weight == 0:
        return None

    # Normalizar
    avg = [x / total_weight for x in weighted_sum]
    norm = math.sqrt(sum(x * x for x in avg))
    if norm == 0:
        return None
    return [x / norm for x in avg]


# ─────────────────────────────────────────────────────────────────
# COLD START
# ─────────────────────────────────────────────────────────────────

def score_cold_start(event: dict, onboarding_prefs: dict = None) -> float:
    """
    Scoring para usuarios nuevos sin historial.
    Usa temporal + popularidad + preferencias de onboarding si existen.

    onboarding_prefs: {
        "categories":       ["music", "sports"],
        "moods":            ["energico", "familiar"],
        "music_genres":     ["rock", "jazz", "indie"]   ← viene de Spotify/Apple Music (NEM-120/121)
    }

    Lógica de boost por géneros musicales:
    - Compara los géneros del usuario contra el campo 'actividad' en ai_tags del evento
      (ej. usuario escucha "jazz" → evento tiene actividad=["jazz","musica_en_vivo"] → match)
    - También compara contra la categoría del evento
    - El boost es menor que el de categoría porque es una señal menos directa
    """
    t_score = temporal_score(event.get("start_date", ""))
    p_score = popularity_score(event)
    f_score = freshness_score(event)

    base = t_score * 0.45 + p_score * 0.35 + f_score * 0.20

    if onboarding_prefs:
        boost = 0.0
        pref_categories  = [c.lower() for c in (onboarding_prefs.get("categories") or [])]
        pref_moods       = (onboarding_prefs.get("moods") or [])
        pref_music       = [g.lower() for g in (onboarding_prefs.get("music_genres") or [])]

        ai_tags = event.get("ai_tags") or {}
        if isinstance(ai_tags, str):
            try: ai_tags = json.loads(ai_tags)
            except: ai_tags = {}

        # Boost por categoría (señal directa)
        if event.get("category", "").lower() in pref_categories:
            boost += 0.15

        # Boost por mood
        event_moods  = ai_tags.get("mood", [])
        mood_matches = len(set(pref_moods) & set(event_moods))
        boost += min(0.10, mood_matches * 0.05)

        # Boost por géneros musicales (Spotify/Apple Music)
        # Compara géneros del usuario contra:
        #   1. genre/subgenre del evento (fuente más específica)
        #   2. actividad en ai_tags (ej. si el tagger puso "jazz" explícitamente)
        #   3. categoría del evento
        if pref_music:
            event_genre     = event.get("genre", "").lower()
            event_subgenre  = event.get("subgenre", "").lower()
            event_actividad = [a.lower() for a in (ai_tags.get("actividad") or [])]
            event_category  = event.get("category", "").lower()
            music_matches   = sum(
                1 for g in pref_music
                if g in event_genre                              # "jazz" en genre="Jazz"
                or g in event_subgenre                          # "blues" en subgenre="Blues/Rock"
                or g in event_actividad                         # "jazz" en actividad=["jazz"]
                or any(g in a for a in event_actividad)         # "electro" en "musica_electronica"
                or g in event_category                          # genre en categoría
            )
            boost += min(0.12, music_matches * 0.06)   # tope menor que categoría (señal indirecta)

        base = base + boost

    return round(min(1.0, base), 4)


# ─────────────────────────────────────────────────────────────────
# WARM USER
# ─────────────────────────────────────────────────────────────────

def score_warm_user(event: dict, event_embedding: list, user_profile: list,
                    strategy: str = "semantic") -> float:
    """
    Scoring para usuarios con historial.

    strategy:
      "semantic"   — 50% similitud + 20% temporal + 15% popularidad + 15% frescura
      "popularity" — 0% similitud + 45% temporal + 35% popularidad + 20% frescura
      "hybrid"     — 35% similitud + 25% temporal + 25% popularidad + 15% frescura
    """
    sem   = cosine_similarity(user_profile, event_embedding) if event_embedding else 0.0
    temp  = temporal_score(event.get("start_date", ""))
    pop   = popularity_score(event)
    fresh = freshness_score(event)

    if strategy == "popularity":
        return round(temp * 0.45 + pop * 0.35 + fresh * 0.20, 4)
    elif strategy == "hybrid":
        return round(sem * 0.35 + temp * 0.25 + pop * 0.25 + fresh * 0.15, 4)
    else:  # semantic (default)
        return round(sem * W_SEMANTIC + temp * W_TEMPORAL + pop * W_POPULARITY + fresh * W_FRESHNESS, 4)


# ─────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────

def get_personalized_feed(
    user_id:           str,
    city:              str   = DEFAULT_CITY,
    limit:             int   = DEFAULT_LIMIT,
    offset:            int   = 0,
    strategy:          str   = None,
    onboarding_prefs:  dict  = None,
    excluded_ids:      list  = None,   # IDs ya mostrados en sesión → excluir del feed
    exploration_factor: float = 0.25,  # Fracción del feed dedicada a exploración (0.0-1.0)
    date_range_days:   int   = 90,     # Solo mostrar eventos en los próximos N días
) -> dict:
    """
    Genera el feed personalizado de eventos para un usuario.

    Args:
        user_id:            UUID del usuario
        city:               Ciudad para filtrar eventos
        limit:              Número de eventos a devolver
        offset:             Para paginación
        strategy:           Override del RANKING_STRATEGY env var
        onboarding_prefs:   Preferencias del onboarding (para cold start)
        excluded_ids:       IDs ya mostrados en esta sesión (para variedad al recargar)
        exploration_factor: Fracción del feed para exploración fuera del perfil.
                            0.25 = 25% exploración, 75% explotación (default).
                            0.0 = solo recomendaciones personalizadas (sin exploración).

    Returns:
        {
          "feed":              [...],   # eventos ordenados, con exploración intercalada
          "mode":              "warm" | "cold_start",
          "strategy":          "semantic" | "popularity" | "hybrid",
          "count":             N,
          "exploration_count": N,      # cuántos eventos de exploración se incluyeron
          "latency_ms":        N
        }
    """
    t_start  = time.time()
    strategy = strategy or RANKING_STRATEGY

    try:
        from supabase import create_client
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)

        # 1. Cargar interacciones del usuario
        interactions = _get_user_interactions(sb, user_id)
        mode = "warm" if interactions else "cold_start"

        # 2. Cargar eventos disponibles (publicados, futuros, en la ciudad)
        events = _get_candidate_events(sb, city, date_range_days=date_range_days)

        # 3. IDs ya interactuados (para excluir del feed)
        seen_ids = {i["event_id"] for i in interactions}

        # IDs ya mostrados en esta sesión (para variedad al recargar)
        session_excluded = set(excluded_ids or [])

        # 4. Candidatos = eventos no vistos ni mostrados en esta sesión
        candidates = [
            e for e in events
            if e["id"] not in seen_ids
            and e["id"] not in session_excluded
        ]

        if mode == "warm":
            # Cargar embeddings de los eventos con los que el usuario interactuó
            interaction_ids   = list(seen_ids)
            user_emb_rows     = _get_embeddings(sb, interaction_ids)
            embeddings_map    = {r["event_id"]: json.loads(r["embedding"]) for r in user_emb_rows}
            user_profile      = build_user_profile(interactions, embeddings_map)

            if user_profile is None:
                mode = "cold_start"
            else:
                # Cargar embeddings de candidatos
                candidate_ids     = [e["id"] for e in candidates]
                candidate_emb     = _get_embeddings(sb, candidate_ids)
                candidate_emb_map = {r["event_id"]: json.loads(r["embedding"]) for r in candidate_emb}

                scored = [
                    {
                        **_format_event(e),
                        "_score": {
                            "final":    score_warm_user(e, candidate_emb_map.get(e["id"]), user_profile, strategy),
                            "semantic": cosine_similarity(user_profile, candidate_emb_map.get(e["id"]) or []),
                            "temporal": temporal_score(e.get("start_date", "")),
                            "mode":     mode,
                        }
                    }
                    for e in candidates
                ]

        if mode == "cold_start":
            scored = [
                {
                    **_format_event(e),
                    "_score": {
                        "final":    score_cold_start(e, onboarding_prefs),
                        "temporal": temporal_score(e.get("start_date", "")),
                        "mode":     mode,
                    }
                }
                for e in candidates
            ]

        # 5. Separar explotación y exploración (solo en modo warm)
        exploration_count = 0
        if mode == "warm" and exploration_factor > 0:
            scored.sort(key=lambda x: x["_score"]["final"], reverse=True)

            n_explore  = max(1, round(limit * exploration_factor))
            n_exploit  = limit - n_explore

            # Explotación: top eventos según el perfil del usuario
            exploit_pool = scored[:max(n_exploit * 3, 60)]   # pool amplio para paginar
            exploit_page = exploit_pool[offset : offset + n_exploit]

            # Exploración: eventos con menor similitud semántica al perfil del usuario,
            # pero con buen score temporal (cosas distintas a lo habitual, pero próximas).
            # Usamos umbral relativo (percentil 30 de similitud) en lugar de absoluto,
            # para adaptarnos a distribuciones distintas según el perfil del usuario.
            sem_scores = sorted([e["_score"].get("semantic", 0.0) for e in scored])
            explore_threshold = sem_scores[max(0, int(len(sem_scores) * 0.30))]  # percentil 30

            explore_pool = sorted(
                [e for e in scored if e["_score"].get("semantic", 1.0) <= explore_threshold],
                key=lambda x: x["_score"]["temporal"],
                reverse=True,
            )
            explore_page = explore_pool[:n_explore]
            exploration_count = len(explore_page)

            # Intercalar: 1 exploración cada ~4 eventos de explotación
            page = _interleave(exploit_page, explore_page)[:limit]

        else:
            # Cold start o exploración desactivada: ranking directo
            scored.sort(key=lambda x: x["_score"]["final"], reverse=True)
            page = scored[offset : offset + limit]

        latency = int((time.time() - t_start) * 1000)
        if latency > 500:
            logger.warning(f"⚠️ Feed latency {latency}ms supera objetivo de 500ms")

        return {
            "feed":              page,
            "mode":              mode,
            "strategy":          strategy,
            "count":             len(page),
            "total":             len(scored),
            "exploration_count": exploration_count,
            "date_range_days":   date_range_days,
            "latency_ms":        latency,
        }

    except Exception as e:
        logger.error(f"Error en recommendation engine: {e}")
        return {"feed": [], "mode": "error", "error": str(e), "latency_ms": 0}


# ─────────────────────────────────────────────────────────────────
# HELPERS DE SUPABASE
# ─────────────────────────────────────────────────────────────────

def _interleave(exploit: list, explore: list) -> list:
    """
    Intercala eventos de exploración entre los de explotación.
    Patrón: E E E X E E E X E E E X ...
    (una exploración cada ~4 eventos de explotación)

    Ejemplo con 6 exploit + 2 explore:
        [exploit0, exploit1, exploit2, EXPLORE0, exploit3, exploit4, exploit5, EXPLORE1]
    """
    result  = []
    ei      = 0   # índice en exploit
    xi      = 0   # índice en explore
    step    = max(1, round(len(exploit) / max(len(explore), 1)))

    for i in range(len(exploit) + len(explore)):
        if xi < len(explore) and ei > 0 and ei % step == 0:
            result.append(explore[xi])
            xi += 1
        if ei < len(exploit):
            result.append(exploit[ei])
            ei += 1

    # Si quedaron exploraciones sin insertar, las agregamos al final
    result.extend(explore[xi:])
    return result


def _get_user_interactions(sb, user_id: str) -> list[dict]:
    """
    Carga el historial de interacciones del usuario.
    Tabla esperada: user_event_interactions(user_id, event_id, action_type, created_at)
    action_type: "save" | "purchase" | "view"
    """
    try:
        resp = (
            sb.table("user_event_interactions")
            .select("event_id, action_type, created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(200)   # Máximo de historial a considerar
            .execute()
        )
        return resp.data or []
    except Exception:
        return []   # Si la tabla no existe todavía, retornar vacío (cold start)


def _get_candidate_events(sb, city: str, date_range_days: int = 90) -> list[dict]:
    """
    Carga eventos publicados y futuros filtrando por ciudad via JOIN con venues.
    Solo trae eventos entre hoy y los próximos date_range_days días.
    No trae description — campo pesado innecesario para el ranking.
    Si se necesita la descripción completa, el frontend la pide por event_id.
    """
    today    = datetime.date.today()
    max_date = (today + datetime.timedelta(days=date_range_days)).isoformat()
    today    = today.isoformat()

    resp = (
        sb.table("events")
        .select(
            "id, title, category, start_date, start_time, "
            "price_tier, price_min, image_url, ai_tags, source, source_url, "
            "venue_id, quality_score, source_trust_score, created_at, "
            "venues!inner(city)"          # JOIN con venues para filtrar por ciudad
        )
        .eq("status", "published")
        .gte("start_date", today)         # No mostrar eventos pasados
        .lte("start_date", max_date)      # No mostrar eventos demasiado lejanos
        .eq("venues.city", city)          # Filtro de ciudad en SQL, no en Python
        .execute()
    )
    return resp.data or []


def _get_embeddings(sb, event_ids: list[str]) -> list[dict]:
    """Carga embeddings para una lista de event_ids."""
    if not event_ids:
        return []
    resp = (
        sb.table("event_embeddings")
        .select("event_id, embedding")
        .in_("event_id", event_ids)
        .execute()
    )
    return resp.data or []


def _format_event(event: dict) -> dict:
    """Normaliza el evento para la respuesta del feed."""
    ai_tags = event.get("ai_tags") or {}
    if isinstance(ai_tags, str):
        try: ai_tags = json.loads(ai_tags)
        except: ai_tags = {}

    return {
        "id":         event.get("id"),
        "title":      event.get("title"),
        "category":   event.get("category"),
        "start_date": event.get("start_date"),
        "start_time": str(event.get("start_time") or ""),
        "price_tier": event.get("price_tier"),
        "price_min":  event.get("price_min"),
        "image_url":  event.get("image_url"),
        "source_url": event.get("source_url"),
        "ai_tags":    ai_tags,
        # description se omite aquí — el frontend la pide por event_id si la necesita
    }


# ─────────────────────────────────────────────────────────────────
# AWS LAMBDA HANDLER
# ─────────────────────────────────────────────────────────────────

def handler(event: dict, context) -> dict:
    """
    Entry point para AWS Lambda.

    Evento esperado (desde API Gateway / backend):
    {
        "user_id":  "uuid-del-usuario",
        "city":     "Monterrey",
        "limit":    20,
        "offset":   0,
        "strategy": "semantic"   (opcional — override del env var)
    }
    """
    body = {}
    if event.get("body"):
        try: body = json.loads(event["body"])
        except: pass
    # También acepta query params
    params = event.get("queryStringParameters") or {}
    body   = {**params, **body}

    user_id = body.get("user_id", "")
    if not user_id:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "user_id es requerido"}),
        }

    # excluded_ids puede venir como lista JSON o como string separado por comas
    excluded_raw = body.get("excluded_ids", [])
    if isinstance(excluded_raw, str):
        excluded_ids = [x.strip() for x in excluded_raw.split(",") if x.strip()]
    else:
        excluded_ids = excluded_raw or []

    result = get_personalized_feed(
        user_id            = user_id,
        city               = body.get("city", DEFAULT_CITY),
        limit              = int(body.get("limit", DEFAULT_LIMIT)),
        offset             = int(body.get("offset", 0)),
        strategy           = body.get("strategy"),
        excluded_ids       = excluded_ids,
        exploration_factor = float(body.get("exploration_factor", 0.25)),
        date_range_days    = int(body.get("date_range_days", 90)),
    )

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type":                "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(result, ensure_ascii=False, default=str),
    }
