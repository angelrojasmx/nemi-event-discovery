"""
╔══════════════════════════════════════════════════════════════════╗
║         NEMI — Tagging Pipeline (Producción)                    ║
║                                                                  ║
║  Procesa todos los eventos con:                                  ║
║    · Gemini 2.5 Flash como modelo primario                       ║
║    · Claude Haiku 3.5 como fallback automático                   ║
║    · Guardado incremental — reanudable si se interrumpe          ║
║    · Output CSV listo para importar a Supabase                   ║
╚══════════════════════════════════════════════════════════════════╝

USO:
    python tagging_pipeline.py                   # todos los eventos
    python tagging_pipeline.py --limit 50        # primeros N eventos (prueba)
    python tagging_pipeline.py --resume          # continuar desde donde quedó

OUTPUT:
    tagging_output_YYYYMMDD.csv   — resultados completos con métricas
    tagging_supabase_YYYYMMDD.csv — solo id + ai_tags, listo para upsert en Supabase
"""

import os
import csv
import json
import time
import argparse
import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# CONFIGURACIÓN — API Keys
# ─────────────────────────────────────────────────────────────────

API_KEYS = {
    "gemini":    os.getenv("GEMINI_API_KEY", ""),
    "anthropic": os.getenv("ANTHROPIC_API_KEY", ""),
}

# Hardcodea aquí si prefieres (solo para uso local):
# API_KEYS["gemini"]    = "AIza..."
# API_KEYS["anthropic"] = "sk-ant-..."

# ─────────────────────────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────────────────────────

OUTPUT_DIR  = Path(__file__).parent
EVENTS_DIR  = Path(__file__).parent.parent / "Events"

def _latest_events_csv() -> Path:
    """Busca el CSV de eventos más reciente en la carpeta Events/ (por fecha de modificación)."""
    import glob
    files = glob.glob(str(EVENTS_DIR / "events_rows*.csv"))
    # Excluir archivos de prueba (eventos muy pequeños < 5 KB)
    real_files = [f for f in files if Path(f).stat().st_size > 5_000]
    if not real_files:
        raise FileNotFoundError(
            f"No se encontró ningún events_rows*.csv en {EVENTS_DIR}.\n"
            "Corre primero con MODO = 'download' o coloca el CSV manualmente."
        )
    return Path(max(real_files, key=lambda f: Path(f).stat().st_mtime))

def _load_venue_map() -> dict:
    """Carga el CSV de venues y retorna {venue_id: venue_name}."""
    venues_path = EVENTS_DIR / "venues_rows.csv"
    if not venues_path.exists():
        return {}
    with open(venues_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["id"]: r["name"] for r in rows if r.get("id") and r.get("name","").strip()}

EVENTS_CSV = _latest_events_csv()

# Mapa global venue_id → nombre; se popula en run_pipeline() antes de procesar
_VENUE_MAP: dict = {}

# ─────────────────────────────────────────────────────────────────
# MODELOS
# ─────────────────────────────────────────────────────────────────

PRIMARY_MODEL = {
    "name":                 "gemini-2.5-flash",
    "provider":             "gemini",
    "model_id":             "gemini-2.5-flash",
    "cost_input_per_1m":    0.15,
    "cost_output_per_1m":   0.60,
}

FALLBACK_MODEL = {
    "name":                 "claude-haiku-3-5",
    "provider":             "anthropic",
    "model_id":             "claude-haiku-4-5-20251001",
    "cost_input_per_1m":    0.80,
    "cost_output_per_1m":   4.00,
}

# ─────────────────────────────────────────────────────────────────
# PROMPT — Vocabulario actualizado con hallazgos del benchmark
# ─────────────────────────────────────────────────────────────────

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

VALORES PERMITIDOS (usa solo estos, máximo indicado):
- mood       (máx 3): romantico, divertido, energico, relajado, cultural, familiar, nocturno, sofisticado, espiritual, deportivo, creativo, intenso
- ambiente   (máx 3): intimo, masivo, al_aire_libre, bar, teatro, estadio, galeria, salon, casual, adulto, familiar
- actividad  (máx 2): concierto, teatro, deporte, exposicion, cine, taller, festival, gastronomia, danza, comedia, conferencia, lucha_libre, musica_en_vivo
- ideal_para (máx 3): pareja, amigos, familia, solo, cita, grupos_grandes, ninos, profesionales, estudiantes, jovenes
- dress_code       : "casual" | "smart_casual" | "formal" | null
- es_18_plus       : true | false | null  (solo si hay mención explícita)
- duracion_estimada_hrs : número decimal | null  (solo si hay hora inicio Y fin claras, o se menciona duración)"""

USER_TEMPLATE = """Título: {title}
Descripción: {description}
Venue: {venue}"""

# Vocabulario para validación
VALID_VALUES = {
    "mood":       {"romantico","divertido","energico","relajado","cultural","familiar","nocturno","sofisticado","espiritual","deportivo","creativo","intenso"},
    "ambiente":   {"intimo","masivo","al_aire_libre","bar","teatro","estadio","galeria","salon","casual","adulto","familiar"},
    "actividad":  {"concierto","teatro","deporte","exposicion","cine","taller","festival","gastronomia","danza","comedia","conferencia","lucha_libre","musica_en_vivo"},
    "ideal_para": {"pareja","amigos","familia","solo","cita","grupos_grandes","ninos","profesionales","estudiantes","jovenes"},
}
REQUIRED_FIELDS = ["mood","ambiente","actividad","ideal_para","dress_code","es_18_plus","duracion_estimada_hrs"]

# ─────────────────────────────────────────────────────────────────
# LLAMADAS A MODELOS
# ─────────────────────────────────────────────────────────────────

def call_gemini(user_message: str, max_retries: int = 3) -> dict:
    from openai import OpenAI
    client = OpenAI(
        api_key  = API_KEYS["gemini"],
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            response = client.chat.completions.create(
                model      = PRIMARY_MODEL["model_id"],
                messages   = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                temperature = 0,
                max_tokens  = 2000,
            )
            return {
                "raw_text":      response.choices[0].message.content or "",
                "tokens_input":  response.usage.prompt_tokens if response.usage else 0,
                "tokens_output": response.usage.completion_tokens if response.usage else 0,
                "latency_ms":    int((time.time() - t0) * 1000),
                "error":         None,
            }
        except Exception as e:
            if attempt == max_retries - 1:
                return {"raw_text":"","tokens_input":0,"tokens_output":0,"latency_ms":0,"error":str(e)}
            time.sleep(2 ** attempt)
    return {"raw_text":"","tokens_input":0,"tokens_output":0,"latency_ms":0,"error":"max_retries"}


def call_claude(user_message: str, max_retries: int = 3) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=API_KEYS["anthropic"])
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            response = client.messages.create(
                model      = FALLBACK_MODEL["model_id"],
                system     = SYSTEM_PROMPT,
                messages   = [{"role": "user", "content": user_message}],
                max_tokens = 300,
                temperature = 0,
            )
            return {
                "raw_text":      response.content[0].text if response.content else "",
                "tokens_input":  response.usage.input_tokens if response.usage else 0,
                "tokens_output": response.usage.output_tokens if response.usage else 0,
                "latency_ms":    int((time.time() - t0) * 1000),
                "error":         None,
            }
        except Exception as e:
            if attempt == max_retries - 1:
                return {"raw_text":"","tokens_input":0,"tokens_output":0,"latency_ms":0,"error":str(e)}
            time.sleep(2 ** attempt)
    return {"raw_text":"","tokens_input":0,"tokens_output":0,"latency_ms":0,"error":"max_retries"}


# ─────────────────────────────────────────────────────────────────
# PARSEO Y VALIDACIÓN
# ─────────────────────────────────────────────────────────────────

def parse_tags(raw_text: str) -> dict:
    """Parsea y valida el JSON del modelo. Retorna métricas + tags limpios."""
    result = {
        "tags":           None,
        "valid":          False,
        "coverage_score": 0.0,
        "oov_count":      0,
        "parse_error":    None,
    }

    text = raw_text.strip()
    if text.startswith("```"):
        text = "\n".join(l for l in text.split("\n") if not l.startswith("```"))

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        result["parse_error"] = str(e)
        return result

    if not isinstance(data, dict):
        result["parse_error"] = "response_not_dict"
        return result

    # Asegurar que todos los campos requeridos existen
    for field in REQUIRED_FIELDS:
        if field not in data:
            data[field] = [] if field in VALID_VALUES else None

    # Limpiar y validar valores de lista
    oov = 0
    for field, valid_set in VALID_VALUES.items():
        vals = data.get(field, [])
        if not isinstance(vals, list):
            data[field] = []
            continue
        cleaned = []
        for v in vals:
            if isinstance(v, str):
                v_clean = v.lower().strip()
                if v_clean in valid_set:
                    cleaned.append(v_clean)
                else:
                    oov += 1
        data[field] = cleaned

    # Coverage: fracción de campos de lista con al menos 1 valor
    filled = sum(1 for f in VALID_VALUES if len(data.get(f, [])) > 0)
    coverage = round(filled / len(VALID_VALUES), 3)

    result["tags"]           = data
    result["valid"]          = True
    result["coverage_score"] = coverage
    result["oov_count"]      = oov
    return result


def estimate_cost(model_config: dict, tokens_in: int, tokens_out: int) -> float:
    return round(
        (tokens_in  / 1_000_000) * model_config["cost_input_per_1m"] +
        (tokens_out / 1_000_000) * model_config["cost_output_per_1m"],
        8
    )


# ─────────────────────────────────────────────────────────────────
# PRE-PROCESAMIENTO: deduplicación + reglas fijas
# ─────────────────────────────────────────────────────────────────

# Prioridad de fuente para deduplicación (mayor = mejor)
SOURCE_PRIORITY = {
    # Taquilleras nacionales — mayor info estructurada
    "ticketmaster": 10, "superboletos": 9, "boletia": 8,
    # Plataformas con descripción generalmente buena
    "eventbrite": 7, "feverup": 7,
    "foro13mty": 7, "civitatis": 6,
    # Venues propios — info directa pero variable
    "arena_monterrey": 6, "forocorona": 6, "estadiobbva": 6,
    "escenariognp": 6, "showcenter": 6, "autodromomty": 6,
    "arema": 6, "cintermex": 6,
    # Institucionales y municipales
    "conarte_agenda": 5, "conarte_exposiciones": 5, "conarte_convocatorias": 5,
    "3museos": 5, "sanpedro_vive": 5, "nlgobmx": 5,
    # Equipos deportivos
    "sultanes": 5, "tigres": 5, "rayados": 5,
    # Agregadores con menos metadata
    "mty360": 3, "nuevoleon_travel": 2, "uanl": 2,
}

def _src_priority(src: str) -> int:
    return SOURCE_PRIORITY.get(src, 4)

def _info_score(ev: dict) -> int:
    return len(ev.get("title") or "") + len(ev.get("description") or "") + len(ev.get("venue") or "")

def _cineteca_tags(event: dict) -> dict:
    """Tags fijos para eventos de conarte_cineteca (películas con listas de actores)."""
    cat  = (event.get("category") or "").lower()
    desc = (event.get("description") or "").lower()
    if "comedia" in cat or "comedy" in cat:
        mood = ["cultural", "divertido"]
    elif "animaci" in cat:
        mood = ["cultural", "familiar"]
    elif "terror" in cat or "horror" in cat:
        mood = ["cultural", "intenso"]
    elif "drama" in cat:
        mood = ["cultural", "intenso"]
    else:
        mood = ["cultural"]
    ideal = ["familia", "pareja", "amigos"] if "animaci" in cat else ["pareja", "amigos"]
    return {
        "mood": mood, "ambiente": ["teatro"], "actividad": ["cine"],
        "ideal_para": ideal, "dress_code": None, "es_18_plus": None,
        "duracion_estimada_hrs": None,
    }

def preprocess_events(events: list[dict]) -> tuple[list[dict], dict[str, dict], dict[str, str]]:
    """
    Pre-procesa la lista de eventos antes del pipeline LLM.

    Retorna:
      - to_tag:      eventos que sí van al LLM
      - fixed_tags:  {event_id: tags_dict} para eventos con regla fija (cineteca)
      - duplicates:  {event_id_removed: event_id_kept}
    """
    # 1. Detectar duplicados por título + fecha
    from collections import defaultdict
    title_date_groups: dict = defaultdict(list)
    for ev in events:
        # Usar start_date como clave canónica (date_text ya no viene poblado en el nuevo schema)
        date_key = (ev.get("start_date") or ev.get("date_text") or "").strip()
        key = ((ev.get("title") or "").strip().lower(), date_key)
        title_date_groups[key].append(ev)

    duplicates: dict[str, str] = {}
    for group in title_date_groups.values():
        if len(group) < 2:
            continue
        # Elegir el mejor evento del grupo
        best = max(group, key=lambda e: (_src_priority(e.get("source", "")), _info_score(e)))
        for ev in group:
            if ev["id"] != best["id"]:
                duplicates[ev["id"]] = best["id"]

    dup_ids = set(duplicates.keys())

    # 2. Separar: cineteca con regla fija vs resto que va al LLM
    fixed_tags: dict[str, dict] = {}
    to_tag: list[dict] = []

    for ev in events:
        if ev["id"] in dup_ids:
            continue  # Saltar duplicados

        # Nota: la regla fija de conarte_cineteca se eliminó porque esa fuente ya
        # no existe en el nuevo schema. conarte_agenda/exposiciones/convocatorias
        # tienen descripciones narrativas y van directo al LLM.

        to_tag.append(ev)

    return to_tag, fixed_tags, duplicates


# ─────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────

def build_user_message(event: dict) -> str:
    title       = (event.get("title") or "").strip()
    description = (event.get("description") or "").strip() or "(sin descripción)"
    # Resolver venue: primero texto legacy, luego join por venue_id, luego desconocido
    venue = (event.get("venue") or event.get("location") or "").strip()
    if not venue:
        venue_id = event.get("venue_id", "")
        venue = _VENUE_MAP.get(venue_id, "").strip() or "(desconocido)"
    return USER_TEMPLATE.format(title=title, description=description, venue=venue)


def tag_event(event: dict) -> dict:
    """
    Intenta taggear un evento. Usa Gemini primero; si falla o devuelve JSON inválido,
    reintenta automáticamente con Claude.
    """
    user_msg = build_user_message(event)
    desc_len = len((event.get("description") or "").strip())
    is_title_only = desc_len < 30

    # --- Intento 1: Gemini ---
    api_result = call_gemini(user_msg)
    parsed     = parse_tags(api_result["raw_text"])
    model_used = PRIMARY_MODEL["name"]
    fallback   = False

    # --- Fallback: Claude (si Gemini falla o JSON inválido) ---
    if not parsed["valid"] or api_result["error"]:
        api_result = call_claude(user_msg)
        parsed     = parse_tags(api_result["raw_text"])
        model_used = FALLBACK_MODEL["name"]
        fallback   = True

    # Estimar costo con el modelo que se usó
    model_cfg = PRIMARY_MODEL if not fallback else FALLBACK_MODEL
    cost = estimate_cost(model_cfg, api_result["tokens_input"], api_result["tokens_output"])

    return {
        "event_id":           event.get("id", ""),
        "source":             event.get("source", ""),
        "title":              (event.get("title") or "")[:100],
        "desc_length":        desc_len,
        "is_title_only":      is_title_only,
        "ai_tags_json":       json.dumps(parsed["tags"], ensure_ascii=False) if parsed["tags"] else "{}",
        "valid_json":         parsed["valid"],
        "coverage_score":     parsed["coverage_score"],
        "oov_count":          parsed["oov_count"],
        "model_used":         model_used,
        "fallback_used":      fallback,
        "tokens_input":       api_result["tokens_input"],
        "tokens_output":      api_result["tokens_output"],
        "estimated_cost_usd": cost,
        "latency_ms":         api_result["latency_ms"],
        "parse_error":        parsed.get("parse_error") or "",
        "processed_at":       datetime.datetime.utcnow().isoformat(),
    }


def run_pipeline(limit: int = None, resume: bool = False, events_csv: Path = None):
    # Cargar mapa de venues (venue_id → nombre)
    global _VENUE_MAP
    _VENUE_MAP = _load_venue_map()
    print(f"🗺️  Venues cargados: {len(_VENUE_MAP)}")

    # Cargar eventos
    csv_path = events_csv or EVENTS_CSV
    with open(csv_path, encoding="utf-8") as f:
        all_events = list(csv.DictReader(f))

    if limit:
        all_events = all_events[:limit]

    # ── Pre-procesamiento ────────────────────────────────────────
    to_tag, fixed_tags, duplicates = preprocess_events(all_events)

    print(f"\n📋 Pre-procesamiento completado:")
    print(f"   Total eventos:        {len(all_events)}")
    print(f"   Duplicados excluidos: {len(duplicates)}")
    print(f"   Regla fija (cine):    {len(fixed_tags)}")
    print(f"   A procesar con LLM:   {len(to_tag)}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    output_path    = OUTPUT_DIR / f"tagging_output_{timestamp}.csv"
    supabase_path  = OUTPUT_DIR / f"tagging_supabase_{timestamp}.csv"

    # Cargar IDs ya procesados (para reanudar)
    processed_ids = set()
    if resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                processed_ids.add(row["event_id"])
        print(f"♻️  Reanudando — {len(processed_ids)} eventos ya procesados")

    # Encabezados CSV
    fieldnames = [
        "event_id","source","title","desc_length","is_title_only",
        "ai_tags_json","valid_json","coverage_score","oov_count",
        "model_used","fallback_used","tokens_input","tokens_output",
        "estimated_cost_usd","latency_ms","parse_error","processed_at",
    ]

    mode = "a" if resume else "w"
    with open(output_path, mode, newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        if not resume:
            writer.writeheader()

            # Escribir eventos con regla fija (sin llamar al LLM)
            for eid, tags in fixed_tags.items():
                ev = next((e for e in all_events if e["id"] == eid), {})
                writer.writerow({
                    "event_id": eid, "source": ev.get("source",""),
                    "title": (ev.get("title") or "")[:100],
                    "desc_length": len(ev.get("description") or ""),
                    "is_title_only": False,
                    "ai_tags_json": json.dumps(tags, ensure_ascii=False),
                    "valid_json": True, "coverage_score": 1.0, "oov_count": 0,
                    "model_used": "fixed_rule", "fallback_used": False,
                    "tokens_input": 0, "tokens_output": 0,
                    "estimated_cost_usd": 0.0, "latency_ms": 0,
                    "parse_error": "", "processed_at": datetime.datetime.utcnow().isoformat(),
                })

        pending = [e for e in to_tag if str(e.get("id","")) not in processed_ids]
        print(f"\n🚀 Iniciando pipeline LLM — {len(pending)} eventos")
        print(f"   Modelo primario:  {PRIMARY_MODEL['name']}")
        print(f"   Modelo fallback:  {FALLBACK_MODEL['name']}")
        print(f"   Output:           {output_path.name}\n")

        total_cost   = 0.0
        fallback_cnt = 0
        empty_cnt    = 0
        cov_sum      = 0.0

        for i, event in enumerate(pending, 1):
            result = tag_event(event)

            writer.writerow(result)
            out_f.flush()

            total_cost   += result["estimated_cost_usd"]
            cov_sum      += result["coverage_score"]
            if result["fallback_used"]:  fallback_cnt += 1
            if result["coverage_score"] == 0.0: empty_cnt += 1

            fb_icon  = "↩️ " if result["fallback_used"] else "  "
            cov_icon = "⚠️ " if result["coverage_score"] == 0.0 else "✅"
            print(f"  [{i:4d}/{len(pending)}] {cov_icon}{fb_icon}"
                  f"cov={result['coverage_score']:.2f} "
                  f"${result['estimated_cost_usd']:.6f} "
                  f"{result['latency_ms']:4d}ms "
                  f"[{event.get('source','?')[:12]:<12}] "
                  f"{(event.get('title') or '')[:40]}")

            time.sleep(0.2)  # Pequeña pausa para no saturar rate limits

    # Generar CSV para Supabase (solo id + ai_tags)
    with open(output_path, encoding="utf-8") as f:
        results = list(csv.DictReader(f))

    with open(supabase_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id","ai_tags"])
        writer.writeheader()
        for r in results:
            writer.writerow({"id": r["event_id"], "ai_tags": r["ai_tags_json"]})

    # Resumen final
    n = len(results)
    avg_cov  = cov_sum / len(pending) if pending else 0
    print(f"\n{'═'*60}")
    print(f"  ✅ Pipeline completado — {n} eventos procesados")
    print(f"  Coverage promedio:  {avg_cov:.3f}")
    print(f"  Fallbacks usados:   {fallback_cnt} ({fallback_cnt/n*100:.1f}%)")
    print(f"  Eventos vacíos:     {empty_cnt} ({empty_cnt/n*100:.1f}%)")
    print(f"  Costo total:        ${total_cost:.4f} USD")
    print(f"  Costo promedio:     ${total_cost/n:.6f} USD/evento")
    print(f"{'═'*60}")
    print(f"\n  📁 Resultados:  {output_path.name}")
    print(f"  📁 Supabase:    {supabase_path.name}")
    print(f"\n  Para subir a Supabase:")
    print(f"  Table Editor → Import CSV → {supabase_path.name}")
    print(f"  (asegúrate de que la columna 'ai_tags' sea de tipo jsonb)")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEMI Tagging Pipeline")
    parser.add_argument("--limit",  type=int, default=None,  help="Procesar solo los primeros N eventos")
    parser.add_argument("--resume", action="store_true",      help="Continuar desde donde quedó")
   