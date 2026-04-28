"""
╔══════════════════════════════════════════════════════════════════╗
║         NEMI — LLM Tagging Benchmark                            ║
║         Compara múltiples proveedores para auto-tagging          ║
║         de eventos y encuentra la mejor relación calidad/costo   ║
╚══════════════════════════════════════════════════════════════════╝

USO:
    # Verificar muestra sin gastar tokens
    python benchmark_llm_tagging.py --dry-run

    # Correr solo con OpenAI (ya tienes la key)
    python benchmark_llm_tagging.py --models gpt-4o-mini

    # Correr varios modelos
    python benchmark_llm_tagging.py --models gpt-4o-mini gemini-2.0-flash llama-3.3-70b

    # Correr todos los modelos configurados
    python benchmark_llm_tagging.py --models all

    # Cambiar tamaño de muestra
    python benchmark_llm_tagging.py --models gpt-4o-mini --sample-size 30

OUTPUTS:
    benchmark_results_YYYYMMDD_HHMMSS.csv  — resultados detallados por evento
    benchmark_report_YYYYMMDD_HHMMSS.txt   — resumen comparativo por modelo
"""

import os
import csv
import json
import time
import random
import argparse
import datetime
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────
# CONFIGURACIÓN — Pon tus API keys aquí o usa variables de entorno
# ─────────────────────────────────────────────────────────────────

#API_KEYS = {
#    "openai":    os.getenv("OPENAI_API_KEY",    ""),
#    "gemini":    os.getenv("GEMINI_API_KEY",    ""),
#    "anthropic": os.getenv("ANTHROPIC_API_KEY", ""),
#    "groq":      os.getenv("GROQ_API_KEY",      ""),
#}

# Si prefieres hardcodear las keys para pruebas locales, descomenta y llena:
API_KEYS = {}
API_KEYS["openai"]    = os.getenv("OPENAI_API_KEY", "")
API_KEYS["gemini"]    = os.getenv("GEMINI_API_KEY", "")
API_KEYS["anthropic"] = os.getenv("ANTHROPIC_API_KEY", "")
API_KEYS["groq"]      = os.getenv("GROQ_API_KEY", "")

# Ruta al CSV exportado de Supabase
EVENTS_CSV = Path(__file__).parent.parent / "Events" / "events_rows.csv"

# Ruta de salida para resultados
OUTPUT_DIR = Path(__file__).parent

# Semilla aleatoria para reproducibilidad del muestreo
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────────
# DEFINICIÓN DE MODELOS
# Agrega o quita modelos según las keys que tengas disponibles
# ─────────────────────────────────────────────────────────────────

MODELS = {
    "gpt-4o-mini": {
        "provider":             "openai",
        "model_id":             "gpt-4o-mini",
        "cost_input_per_1m":    0.15,
        "cost_output_per_1m":   0.60,
        "description":          "Baseline del piloto — barato y confiable",
    },
    "gpt-4o": {
        "provider":             "openai",
        "model_id":             "gpt-4o",
        "cost_input_per_1m":    2.50,
        "cost_output_per_1m":   10.00,
        "description":          "Máxima calidad OpenAI — referencia de oro",
    },
    "gemini-2.5-flash": {
        "provider":             "gemini",
        "model_id":             "gemini-2.5-flash",
        "cost_input_per_1m":    0.15,
        "cost_output_per_1m":   0.60,
        "description":          "Reemplazo de gemini-2.0-flash, disponible para cuentas nuevas",
    },
    "gemini-2.5-flash-lite": {
        "provider":             "gemini",
        "model_id":             "gemini-2.5-flash-lite",
        "cost_input_per_1m":    0.075,
        "cost_output_per_1m":   0.30,
        "description":          "Opción más barata de Gemini, optimizada para baja latencia",
    },
    "claude-haiku-3-5": {
        "provider":             "anthropic",
        "model_id":             "claude-haiku-4-5-20251001",
        "cost_input_per_1m":    0.80,
        "cost_output_per_1m":   4.00,
        "description":          "Fuerte en seguir instrucciones precisas en español",
    },
    "llama-3.3-70b": {
        "provider":             "groq",
        "model_id":             "llama-3.3-70b-versatile",
        "cost_input_per_1m":    0.59,
        "cost_output_per_1m":   0.79,
        "description":          "Open source vía Groq — casi gratis, mide la calidad",
    },
}

# ─────────────────────────────────────────────────────────────────
# PROMPT — Mismo para todos los modelos (comparación justa)
# ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Eres un sistema de clasificación de eventos para una app de discovery en Monterrey, México.
Dado el título y descripción de un evento, extrae metadatos estructurados.

REGLAS ESTRICTAS:
- Responde SOLO con un JSON válido, sin texto adicional ni markdown
- Solo llena un campo si hay evidencia clara en el texto
- Si no hay evidencia suficiente, usa null (campos simples) o [] (listas)
- NO inventes ni supongas más allá del texto
- Es mejor un campo vacío que información inventada
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

VALORES PERMITIDOS (usa solo estos, máximo indicado por campo):
- mood       (máx 3): romantico, divertido, energico, relajado, cultural, familiar, nocturno, sofisticado, espiritual, deportivo
- ambiente   (máx 3): intimo, masivo, al_aire_libre, bar, teatro, estadio, galeria, salon, casual, adulto, familiar
- actividad  (máx 2): concierto, teatro, deporte, exposicion, cine, taller, festival, gastronomia, danza, comedia, conferencia, lucha_libre, musica_en_vivo
- ideal_para (máx 3): pareja, amigos, familia, solo, cita, grupos_grandes, ninos, profesionales
- dress_code       : "casual" | "smart_casual" | "formal" | null  (solo si se menciona explícitamente)
- es_18_plus       : true | false | null  (solo si se menciona explícitamente restricción de edad)
- duracion_estimada_hrs : número decimal | null  (solo si se menciona duración o hay hora inicio Y hora fin claras)"""

USER_TEMPLATE = """Título: {title}
Descripción: {description}
Venue: {venue}"""

# ─────────────────────────────────────────────────────────────────
# SCHEMA DE VALIDACIÓN
# ─────────────────────────────────────────────────────────────────

VALID_VALUES = {
    "mood":       {"romantico","divertido","energico","relajado","cultural","familiar","nocturno","sofisticado","espiritual","deportivo"},
    "ambiente":   {"intimo","masivo","al_aire_libre","bar","teatro","estadio","galeria","salon","casual","adulto","familiar"},
    "actividad":  {"concierto","teatro","deporte","exposicion","cine","taller","festival","gastronomia","danza","comedia","conferencia","lucha_libre","musica_en_vivo"},
    "ideal_para": {"pareja","amigos","familia","solo","cita","grupos_grandes","ninos","profesionales"},
}
REQUIRED_FIELDS = ["mood", "ambiente", "actividad", "ideal_para", "dress_code", "es_18_plus", "duracion_estimada_hrs"]


# ─────────────────────────────────────────────────────────────────
# MUESTREO ESTRATIFICADO
# ─────────────────────────────────────────────────────────────────

def load_and_sample_events(csv_path: Path, n_per_strata: tuple = (35, 35, 20), seed: int = RANDOM_SEED) -> list[dict]:
    """
    Carga eventos y genera muestra estratificada en 3 tipos:
      Tipo A (>=300 chars): descripción rica — prueba calidad máxima
      Tipo B (50-299 chars): descripción media — prueba el caso real más común
      Tipo C (<50 chars)  : descripción pobre — prueba comportamiento conservador

    Filtra fuentes que producen ruido estructural (conarte_cineteca usa
    el campo 'category' para metadata de películas, no categorías de evento).
    """
    # Fuentes a excluir del muestreo (categorías de ruido estructural)
    NOISY_SOURCES = {"conarte_cineteca"}

    with open(csv_path, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    type_a, type_b, type_c = [], [], []
    for row in all_rows:
        if row.get("source") in NOISY_SOURCES:
            continue
        desc = (row.get("description") or "").strip()
        if len(desc) >= 300:
            type_a.append(row)
        elif len(desc) >= 50:
            type_b.append(row)
        else:
            type_c.append(row)

    rng = random.Random(seed)
    n_a, n_b, n_c = n_per_strata

    sample_a = rng.sample(type_a, min(n_a, len(type_a)))
    sample_b = rng.sample(type_b, min(n_b, len(type_b)))
    sample_c = rng.sample(type_c, min(n_c, len(type_c)))

    for row in sample_a: row["_strata"] = "A_rich"
    for row in sample_b: row["_strata"] = "B_medium"
    for row in sample_c: row["_strata"] = "C_sparse"

    sample = sample_a + sample_b + sample_c
    rng.shuffle(sample)

    print(f"\n📋 Muestra generada: {len(sample)} eventos")
    print(f"   Tipo A (>=300 chars): {len(sample_a)}")
    print(f"   Tipo B (50-299 chars): {len(sample_b)}")
    print(f"   Tipo C (<50 chars):   {len(sample_c)}")
    print(f"   Sources representadas: {len(set(r['source'] for r in sample))}")
    return sample


def build_user_message(row: dict) -> str:
    title       = (row.get("title") or "").strip()
    description = (row.get("description") or "").strip() or "(sin descripción)"
    venue       = (row.get("venue") or row.get("location") or "").strip() or "(desconocido)"
    return USER_TEMPLATE.format(title=title, description=description, venue=venue)


# ─────────────────────────────────────────────────────────────────
# LLAMADAS A PROVEEDORES
# ─────────────────────────────────────────────────────────────────

def call_openai_compatible(
    model_id: str,
    api_key: str,
    base_url: str,
    user_message: str,
    max_retries: int = 3,
    is_gemini: bool = False,
) -> dict:
    """Llama a cualquier API compatible con OpenAI (OpenAI, Gemini vía compat, Groq).

    Gemini 2.5 Flash es un modelo con "thinking" (razonamiento interno).
    Requiere max_tokens más alto y no soporta response_format=json_object
    de la misma forma que OpenAI — se depende del prompt para forzar JSON.
    """
    try:
        from openai import OpenAI, RateLimitError, APIError
    except ImportError:
        raise ImportError("Instala el SDK: pip install openai")

    client = OpenAI(api_key=api_key, base_url=base_url)

    # Gemini thinking models necesitan más tokens de salida
    max_output_tokens = 2000 if is_gemini else 300

    # Gemini no soporta response_format igual que OpenAI — usamos solo prompt
    extra_kwargs = {} if is_gemini else {"response_format": {"type": "json_object"}}

    for attempt in range(max_retries):
        try:
            t0 = time.time()
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                temperature=0,
                max_tokens=max_output_tokens,
                **extra_kwargs,
            )
            latency_ms = int((time.time() - t0) * 1000)

            raw_text      = response.choices[0].message.content or ""
            tokens_input  = response.usage.prompt_tokens if response.usage else 0
            tokens_output = response.usage.completion_tokens if response.usage else 0

            return {
                "raw_text":      raw_text,
                "tokens_input":  tokens_input,
                "tokens_output": tokens_output,
                "latency_ms":    latency_ms,
                "error":         None,
            }

        except RateLimitError:
            wait = 2 ** attempt
            print(f"     ⚠️  Rate limit, reintentando en {wait}s...")
            time.sleep(wait)
        except APIError as e:
            if attempt == max_retries - 1:
                return {"raw_text": "", "tokens_input": 0, "tokens_output": 0, "latency_ms": 0, "error": str(e)}
            time.sleep(1)

    return {"raw_text": "", "tokens_input": 0, "tokens_output": 0, "latency_ms": 0, "error": "max_retries_exceeded"}


def call_anthropic(
    model_id: str,
    api_key: str,
    user_message: str,
    max_retries: int = 3,
) -> dict:
    """Llama a la API de Anthropic (Claude)."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("Instala el SDK: pip install anthropic")

    client = anthropic.Anthropic(api_key=api_key)

    for attempt in range(max_retries):
        try:
            t0 = time.time()
            response = client.messages.create(
                model=model_id,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
                max_tokens=300,
                temperature=0,
            )
            latency_ms = int((time.time() - t0) * 1000)

            raw_text      = response.content[0].text if response.content else ""
            tokens_input  = response.usage.input_tokens if response.usage else 0
            tokens_output = response.usage.output_tokens if response.usage else 0

            return {
                "raw_text":      raw_text,
                "tokens_input":  tokens_input,
                "tokens_output": tokens_output,
                "latency_ms":    latency_ms,
                "error":         None,
            }

        except anthropic.RateLimitError:
            wait = 2 ** attempt
            print(f"     ⚠️  Rate limit, reintentando en {wait}s...")
            time.sleep(wait)
        except Exception as e:
            if attempt == max_retries - 1:
                return {"raw_text": "", "tokens_input": 0, "tokens_output": 0, "latency_ms": 0, "error": str(e)}
            time.sleep(1)

    return {"raw_text": "", "tokens_input": 0, "tokens_output": 0, "latency_ms": 0, "error": "max_retries_exceeded"}


def call_model(model_name: str, user_message: str) -> dict:
    """Dispatcher: elige el proveedor correcto según la config del modelo."""
    config   = MODELS[model_name]
    provider = config["provider"]
    model_id = config["model_id"]

    if provider == "openai":
        return call_openai_compatible(
            model_id    = model_id,
            api_key     = API_KEYS["openai"],
            base_url    = "https://api.openai.com/v1",
            user_message = user_message,
        )
    elif provider == "gemini":
        return call_openai_compatible(
            model_id     = model_id,
            api_key      = API_KEYS["gemini"],
            base_url     = "https://generativelanguage.googleapis.com/v1beta/openai/",
            user_message = user_message,
            is_gemini    = True,
        )
    elif provider == "groq":
        return call_openai_compatible(
            model_id    = model_id,
            api_key     = API_KEYS["groq"],
            base_url    = "https://api.groq.com/openai/v1",
            user_message = user_message,
        )
    elif provider == "anthropic":
        return call_anthropic(
            model_id    = model_id,
            api_key     = API_KEYS["anthropic"],
            user_message = user_message,
        )
    else:
        raise ValueError(f"Proveedor desconocido: {provider}")


# ─────────────────────────────────────────────────────────────────
# VALIDACIÓN Y MÉTRICAS
# ─────────────────────────────────────────────────────────────────

def parse_and_validate(raw_text: str) -> dict:
    """
    Parsea el JSON del modelo y valida que cumple el schema.
    Retorna métricas de cobertura y calidad.
    """
    result = {
        "parsed_json":        None,
        "valid_json":         False,
        "schema_ok":          False,
        "fallback_used":      False,
        "out_of_vocab_count": 0,
        "mood_count":         0,
        "ambiente_count":     0,
        "actividad_count":    0,
        "ideal_para_count":   0,
        "has_dress_code":     False,
        "has_18plus":         False,
        "has_duracion":       False,
        "coverage_score":     0.0,
        "parse_error":        None,
    }

    # 1. Intentar parsear JSON
    text = raw_text.strip()
    # Quitar posibles bloques markdown ```json ... ```
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(l for l in lines if not l.startswith("```"))

    try:
        data = json.loads(text)
        result["valid_json"]  = True
        result["parsed_json"] = data
    except json.JSONDecodeError as e:
        result["parse_error"]   = str(e)
        result["fallback_used"] = True
        return result

    # 2. Validar schema
    if not isinstance(data, dict):
        result["parse_error"] = "response_not_dict"
        return result

    missing_fields = [f for f in REQUIRED_FIELDS if f not in data]
    if missing_fields:
        result["parse_error"] = f"missing_fields: {missing_fields}"
    else:
        result["schema_ok"] = True

    # 3. Contar cobertura por campo
    oov_count = 0
    for field, valid_set in VALID_VALUES.items():
        val = data.get(field, [])
        if isinstance(val, list) and len(val) > 0:
            count_key = f"{field}_count"
            result[count_key] = len(val)
            # Contar valores fuera del vocabulario controlado
            for v in val:
                if isinstance(v, str) and v.lower() not in valid_set:
                    oov_count += 1

    result["has_dress_code"] = bool(data.get("dress_code"))
    result["has_18plus"]     = data.get("es_18_plus") is not None
    result["has_duracion"]   = data.get("duracion_estimada_hrs") is not None
    result["out_of_vocab_count"] = oov_count

    # 4. Coverage score: fracción de campos de lista que tienen al menos 1 valor
    list_fields_filled = sum(1 for f in VALID_VALUES if result.get(f"{f}_count", 0) > 0)
    result["coverage_score"] = round(list_fields_filled / len(VALID_VALUES), 3)

    return result


def estimate_cost(model_name: str, tokens_input: int, tokens_output: int) -> float:
    config = MODELS[model_name]
    cost = (tokens_input / 1_000_000) * config["cost_input_per_1m"]
    cost += (tokens_output / 1_000_000) * config["cost_output_per_1m"]
    return round(cost, 8)


# ─────────────────────────────────────────────────────────────────
# LOOP PRINCIPAL DEL BENCHMARK
# ─────────────────────────────────────────────────────────────────

def run_benchmark(model_names: list[str], sample: list[dict], output_dir: Path) -> Path:
    """
    Corre el benchmark para los modelos indicados sobre la muestra.
    Guarda resultados incrementalmente (si se interrumpe, no se pierde todo).
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"benchmark_results_{timestamp}.csv"

    fieldnames = [
        "model_name", "event_id", "event_strata", "event_source",
        "event_title", "desc_length",
        "valid_json", "schema_ok", "fallback_used",
        "mood_count", "ambiente_count", "actividad_count", "ideal_para_count",
        "has_dress_code", "has_18plus", "has_duracion",
        "out_of_vocab_count", "coverage_score",
        "tokens_input", "tokens_output", "estimated_cost_usd", "latency_ms",
        "ai_tags_json", "parse_error",
    ]

    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model_name in model_names:
            config = MODELS[model_name]
            print(f"\n{'═'*60}")
            print(f"  Modelo: {model_name}  ({config['description']})")
            print(f"  Proveedor: {config['provider']}  |  ID: {config['model_id']}")
            print(f"{'═'*60}")

            # Verificar que hay API key para este proveedor
            provider_key = API_KEYS.get(config["provider"], "")
            if not provider_key:
                print(f"  ⛔ Sin API key para '{config['provider']}'. Saltando modelo.")
                print(f"     Configura la variable de entorno {config['provider'].upper()}_API_KEY")
                continue

            ok_count   = 0
            fail_count = 0
            total_cost = 0.0

            for i, event in enumerate(sample, 1):
                user_msg = build_user_message(event)
                desc_len = len((event.get("description") or "").strip())

                # Llamada al modelo
                api_result = call_model(model_name, user_msg)

                # Manejar error de API
                if api_result["error"] and not api_result["raw_text"]:
                    metrics = {
                        "valid_json": False, "schema_ok": False, "fallback_used": True,
                        "mood_count": 0, "ambiente_count": 0, "actividad_count": 0,
                        "ideal_para_count": 0, "has_dress_code": False,
                        "has_18plus": False, "has_duracion": False,
                        "out_of_vocab_count": 0, "coverage_score": 0.0,
                        "parsed_json": None, "parse_error": api_result["error"],
                    }
                    fail_count += 1
                else:
                    metrics = parse_and_validate(api_result["raw_text"])
                    if metrics["valid_json"]:
                        ok_count += 1
                    else:
                        fail_count += 1

                cost = estimate_cost(model_name, api_result["tokens_input"], api_result["tokens_output"])
                total_cost += cost

                # Guardar fila de resultado
                row = {
                    "model_name":          model_name,
                    "event_id":            event.get("id", ""),
                    "event_strata":        event.get("_strata", ""),
                    "event_source":        event.get("source", ""),
                    "event_title":         (event.get("title") or "")[:80],
                    "desc_length":         desc_len,
                    "valid_json":          metrics["valid_json"],
                    "schema_ok":           metrics["schema_ok"],
                    "fallback_used":       metrics["fallback_used"],
                    "mood_count":          metrics["mood_count"],
                    "ambiente_count":      metrics["ambiente_count"],
                    "actividad_count":     metrics["actividad_count"],
                    "ideal_para_count":    metrics["ideal_para_count"],
                    "has_dress_code":      metrics["has_dress_code"],
                    "has_18plus":          metrics["has_18plus"],
                    "has_duracion":        metrics["has_duracion"],
                    "out_of_vocab_count":  metrics["out_of_vocab_count"],
                    "coverage_score":      metrics["coverage_score"],
                    "tokens_input":        api_result["tokens_input"],
                    "tokens_output":       api_result["tokens_output"],
                    "estimated_cost_usd":  cost,
                    "latency_ms":          api_result["latency_ms"],
                    "ai_tags_json":        json.dumps(metrics["parsed_json"], ensure_ascii=False) if metrics["parsed_json"] else "",
                    "parse_error":         metrics.get("parse_error") or "",
                }
                writer.writerow(row)
                f.flush()  # Guardar inmediatamente

                # Progreso en consola
                status_icon = "✅" if metrics["valid_json"] else "❌"
                cov = metrics["coverage_score"]
                print(f"  [{i:3d}/{len(sample)}] {status_icon} "
                      f"cov={cov:.2f} "
                      f"cost=${cost:.6f} "
                      f"{api_result['latency_ms']:4d}ms "
                      f"[{event.get('_strata','?')}] "
                      f"{(event.get('title') or '')[:40]}")

                # Pequeña pausa para no saturar rate limits
                time.sleep(0.3)

            print(f"\n  📊 {model_name}: {ok_count} ok / {fail_count} fallos | costo total: ${total_cost:.4f}")

    print(f"\n✅ Resultados guardados en: {results_path.name}")
    return results_path


# ─────────────────────────────────────────────────────────────────
# REPORTE COMPARATIVO
# ─────────────────────────────────────────────────────────────────

def generate_report(results_path: Path) -> Path:
    """Lee el CSV de resultados y genera un reporte de texto comparativo."""
    from collections import defaultdict

    with open(results_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("⚠️  Sin resultados para reportar.")
        return results_path

    # Agrupar por modelo
    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model_name"]].append(row)

    report_path = results_path.with_name(results_path.name.replace("results", "report").replace(".csv", ".txt"))

    lines = []
    lines.append("═" * 70)
    lines.append("  NEMI — LLM Tagging Benchmark — Reporte Comparativo")
    lines.append(f"  Generado: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("═" * 70)
    lines.append("")

    # Tabla resumen
    header = f"{'Modelo':<22} {'JSON%':>6} {'Cov':>5} {'mood':>5} {'amb':>5} {'act':>5} {'para':>5} {'OoV':>4} {'Avg$':>10} {'ms':>6} {'n':>4}"
    lines.append(header)
    lines.append("-" * 70)

    summary_rows = []
    for model_name, model_rows in sorted(by_model.items()):
        n = len(model_rows)
        def avg(col): return sum(float(r[col]) for r in model_rows if r[col]) / n

        json_pct     = sum(1 for r in model_rows if r["valid_json"] == "True") / n * 100
        cov          = avg("coverage_score")
        mood_pct     = sum(1 for r in model_rows if int(r.get("mood_count") or 0) > 0) / n * 100
        amb_pct      = sum(1 for r in model_rows if int(r.get("ambiente_count") or 0) > 0) / n * 100
        act_pct      = sum(1 for r in model_rows if int(r.get("actividad_count") or 0) > 0) / n * 100
        para_pct     = sum(1 for r in model_rows if int(r.get("ideal_para_count") or 0) > 0) / n * 100
        oov_avg      = avg("out_of_vocab_count")
        cost_avg     = avg("estimated_cost_usd")
        lat_avg      = avg("latency_ms")

        row_str = (f"{model_name:<22} {json_pct:5.1f}% {cov:5.3f} "
                   f"{mood_pct:4.0f}% {amb_pct:4.0f}% {act_pct:4.0f}% {para_pct:4.0f}% "
                   f"{oov_avg:4.1f} ${cost_avg:.6f} {lat_avg:5.0f}ms {n:4d}")
        lines.append(row_str)
        summary_rows.append((model_name, json_pct, cov, cost_avg, lat_avg))

    lines.append("-" * 70)
    lines.append("")
    lines.append("Columnas: JSON% = JSON válido | Cov = coverage_score (0-1) |")
    lines.append("  mood/amb/act/para = % eventos con ese campo lleno |")
    lines.append("  OoV = valores fuera del vocabulario (menor = mejor)")
    lines.append("  Avg$ = costo promedio por evento | ms = latencia promedio")
    lines.append("")

    # Análisis por estrata
    lines.append("═" * 70)
    lines.append("  COBERTURA POR TIPO DE EVENTO")
    lines.append("═" * 70)
    for model_name, model_rows in sorted(by_model.items()):
        lines.append(f"\n  {model_name}:")
        for strata in ["A_rich", "B_medium", "C_sparse"]:
            strata_rows = [r for r in model_rows if r["event_strata"] == strata]
            if not strata_rows:
                continue
            n_s  = len(strata_rows)
            cov  = sum(float(r["coverage_score"]) for r in strata_rows) / n_s
            json_ok = sum(1 for r in strata_rows if r["valid_json"] == "True") / n_s * 100
            lines.append(f"    {strata:<12} n={n_s:2d}  json={json_ok:4.0f}%  cov={cov:.3f}")

    # Proyección de costo a 1873 eventos
    lines.append("")
    lines.append("═" * 70)
    lines.append("  PROYECCIÓN DE COSTO — 1,873 EVENTOS (dataset completo)")
    lines.append("═" * 70)
    lines.append(f"  {'Modelo':<22} {'$/evento':>10} {'Total 1873':>12} {'Total 10k':>12}")
    lines.append("  " + "-" * 58)
    for model_name, _, _, cost_avg, _ in sorted(summary_rows, key=lambda x: x[3]):
        total_1873 = cost_avg * 1873
        total_10k  = cost_avg * 10000
        lines.append(f"  {model_name:<22} ${cost_avg:.6f} ${total_1873:>10.4f}   ${total_10k:>10.4f}")

    lines.append("")
    lines.append("  * Proyección basada en el promedio de la muestra del benchmark")
    lines.append("")

    # Recomendación automática
    lines.append("═" * 70)
    lines.append("  ANÁLISIS AUTOMÁTICO")
    lines.append("═" * 70)
    if summary_rows:
        best_cov   = max(summary_rows, key=lambda x: x[2])
        cheapest   = min(summary_rows, key=lambda x: x[3])
        fastest    = min(summary_rows, key=lambda x: x[4])
        lines.append(f"  Mayor cobertura:  {best_cov[0]}  (cov={best_cov[2]:.3f})")
        lines.append(f"  Menor costo:      {cheapest[0]}  (${cheapest[3]:.6f}/evento)")
        lines.append(f"  Menor latencia:   {fastest[0]}  ({fastest[4]:.0f}ms)")
        lines.append("")
        lines.append("  ⚠️  Esta es solo una guía automática. El análisis final requiere")
        lines.append("  revisión manual de la calidad real de las etiquetas generadas.")

    lines.append("")
    lines.append("═" * 70)

    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\n📄 Reporte guardado en: {report_path.name}")
    return report_path


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NEMI LLM Tagging Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models", nargs="+", default=["gpt-4o-mini"],
        help=f"Modelos a comparar. Opciones: {list(MODELS.keys())} o 'all'",
    )
    parser.add_argument(
        "--sample-size", type=int, default=90,
        help="Total de eventos en la muestra (default: 90 — 35 A + 35 B + 20 C)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Muestra la muestra de eventos sin llamar a ningún API",
    )
    parser.add_argument(
        "--report-only", type=str, default=None, metavar="CSV_PATH",
        help="Solo genera el reporte a partir de un CSV de resultados existente",
    )
    args = parser.parse_args()

    # Modo solo-reporte
    if args.report_only:
        generate_report(Path(args.report_only))
        return

    # Resolver modelos
    if args.models == ["all"]:
        selected_models = list(MODELS.keys())
    else:
        selected_models = []
        for m in args.models:
            if m not in MODELS:
                print(f"⚠️  Modelo '{m}' no reconocido. Opciones: {list(MODELS.keys())}")
            else:
                selected_models.append(m)

    if not selected_models:
        print("❌ Sin modelos válidos seleccionados.")
        return

    # Calcular distribución de muestra
    total = args.sample_size
    n_a   = int(total * 0.39)   # 39% tipo A
    n_b   = int(total * 0.39)   # 39% tipo B
    n_c   = total - n_a - n_b   # resto tipo C

    # Cargar muestra
    sample = load_and_sample_events(EVENTS_CSV, n_per_strata=(n_a, n_b, n_c))

    # Dry-run: mostrar muestra y salir
    if args.dry_run:
        print("\n--- MUESTRA (dry-run, sin llamar APIs) ---")
        for i, ev in enumerate(sample[:10], 1):
            desc = (ev.get("description") or "")[:80]
            print(f"  {i:2d}. [{ev['_strata']}] [{ev['source']}] {ev.get('title','')[:50]}")
            print(f"      Desc: {desc!r}")
        if len(sample) > 10:
            print(f"  ... y {len(sample) - 10} eventos más")
        print(f"\nModelos seleccionados: {selected_models}")
        print("\nEstimación de costo si se corre completo:")
        avg_input_tokens  = 400
        avg_output_tokens = 80
        for m in selected_models:
            cfg  = MODELS[m]
            cost = ((avg_input_tokens / 1e6) * cfg["cost_input_per_1m"] +
                    (avg_output_tokens / 1e6) * cfg["cost_output_per_1m"]) * len(sample)
            print(f"  {m:<22} ~${cost:.4f} USD ({len(sample)} eventos)")
        return

    # Mostrar resumen antes de empezar
    print