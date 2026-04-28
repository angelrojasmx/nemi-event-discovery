"""
╔══════════════════════════════════════════════════════════════════╗
║         NEMI — Embedding Pipeline                               ║
║                                                                  ║
║  Genera embeddings para eventos ya taggeados.                   ║
║  Modelo: text-embedding-3-small (OpenAI)                        ║
║  Dimensiones: 1536                                              ║
║  Costo estimado: ~$0.002 para 1,873 eventos                     ║
╚══════════════════════════════════════════════════════════════════╝

USO:
    # Paso 1: correr tagging_pipeline.py primero
    # Paso 2:
    python embedding_pipeline.py --input tagging_output_YYYYMMDD.csv
    python embedding_pipeline.py --input tagging_output_YYYYMMDD.csv --limit 50

OUTPUT:
    embeddings_YYYYMMDD.csv        — id + vector (formato Supabase pgvector)
    embeddings_full_YYYYMMDD.csv   — id + enriched_text + vector + métricas

CÓMO IMPORTAR A SUPABASE (pgvector):
    1. Asegúrate de que la tabla 'events' tenga columna: embedding vector(1536)
    2. Usa la columna 'embedding' del CSV de salida
    3. Supabase acepta vectores en formato "[0.1, 0.2, ...]"

NOTA SOBRE ENFOQUES DE EMBEDDING:
    Este script usa "enriched_text" — combina título, descripción y ai_tags
    para que el embedding capture tanto el contenido semántico del evento
    como su clasificación estructurada. Esto mejora la calidad de retrieval
    comparado con embeber solo el texto crudo.
"""

import os
import csv
import json
import time
import argparse
import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────

API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY", ""),
}

# Hardcodea aquí si prefieres:
# API_KEYS["openai"] = "sk-..."

EVENTS_DIR  = Path(__file__).parent.parent / "Events"
OUTPUT_DIR  = Path(__file__).parent

def _latest_events_csv() -> Path:
    import glob
    files = [f for f in glob.glob(str(EVENTS_DIR / "events_rows*.csv"))
             if Path(f).stat().st_size > 5_000]
    if not files:
        raise FileNotFoundError(f"No se encontró events_rows*.csv en {EVENTS_DIR}")
    return Path(max(files, key=lambda f: Path(f).stat().st_mtime))

def _load_venue_map() -> dict:
    venues_path = EVENTS_DIR / "venues_rows.csv"
    if not venues_path.exists():
        return {}
    with open(venues_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["id"]: r["name"] for r in rows if r.get("id") and r.get("name","").strip()}

EVENTS_CSV = _latest_events_csv()

EMBEDDING_MODEL    = "text-embedding-3-small"
EMBEDDING_DIMS     = 1536
COST_PER_1M_TOKENS = 0.020   # USD
BATCH_SIZE         = 50       # Eventos por batch (conservador para evitar timeouts)


# ─────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DEL ENRICHED TEXT
# ─────────────────────────────────────────────────────────────────

def build_enriched_text(event: dict, tags: dict, venue_map: dict = None) -> str:
    """
    Construye el texto enriquecido que se va a embeber.

    Estrategia: combinar el contenido semántico del evento (título + descripción)
    con su clasificación estructurada (ai_tags). Esto hace que el embedding
    capture ambas dimensiones — qué ES el evento y cómo CLASIFICA.

    Ejemplo de output:
        Título: Concierto de Jazz en el Parque
        Descripción: Una noche de jazz al aire libre...
        Venue: Parque Fundidora
        Mood: relajado, cultural
        Ambiente: al_aire_libre, casual
        Actividad: musica_en_vivo, concierto
        Ideal para: pareja, amigos
    """
    title       = (event.get("title") or "").strip()
    description = (event.get("description") or "").strip()
    # Resolver venue: texto legacy → join por venue_id → vacío
    venue = (event.get("venue") or event.get("location") or "").strip()
    if not venue and venue_map:
        venue = venue_map.get(event.get("venue_id", ""), "").strip()

    parts = []
    if title:
        parts.append(f"Título: {title}")
    if description:
        parts.append(f"Descripción: {description}")
    if venue:
        parts.append(f"Venue: {venue}")

    # Agregar tags estructurados (solo los que tienen valores)
    tag_lines = []
    for field in ["mood", "ambiente", "actividad", "ideal_para"]:
        vals = tags.get(field, [])
        if vals:
            tag_lines.append(f"{field.capitalize().replace('_', ' ')}: {', '.join(vals)}")

    if tags.get("dress_code"):
        tag_lines.append(f"Dress code: {tags['dress_code']}")

    if tag_lines:
        parts.extend(tag_lines)

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────
# LLAMADAS A LA API DE EMBEDDINGS
# ─────────────────────────────────────────────────────────────────

def embed_batch(texts: list[str], max_retries: int = 3) -> list[list[float]]:
    """Genera embeddings para un batch de textos. Retorna lista de vectores."""
    from openai import OpenAI
    client = OpenAI(api_key=API_KEYS["openai"])

    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model = EMBEDDING_MODEL,
                input = texts,
            )
            # Ordenar por índice para garantizar el orden correcto
            embeddings = sorted(response.data, key=lambda x: x.index)
            return [e.embedding for e in embeddings]
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Error al generar embeddings: {e}")
            wait = 2 ** attempt
            print(f"     ⚠️  Error en batch, reintentando en {wait}s: {e}")
            time.sleep(wait)

    return []


def estimate_embedding_cost(texts: list[str]) -> float:
    """Estimación de costo basada en caracteres (aprox. 4 chars = 1 token)."""
    total_chars  = sum(len(t) for t in texts)
    approx_tokens = total_chars / 4
    return round((approx_tokens / 1_000_000) * COST_PER_1M_TOKENS, 6)


# ─────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────

def _build_items_from_rows(tagged_rows: list[dict], all_events: dict, venue_map: dict = None) -> tuple[list[dict], int]:
    """
    Convierte filas del CSV (cualquier formato) a lista de items listos para embeber.
    Soporta dos formatos de entrada:
      - Formato completo (tagging_output): columnas event_id, valid_json, ai_tags_json, source, title, coverage_score
      - Formato supabase (tagging_supabase): columnas id, ai_tags
    """
    items = []
    skipped = 0

    for row in tagged_rows:
        # Detectar formato
        if "event_id" in row:
            # Formato completo (tagging_output)
            if row.get("valid_json") != "True":
                skipped += 1
                continue
            event_id   = str(row["event_id"])
            ai_tags_raw = row.get("ai_tags_json", "{}")
            source      = row.get("source", "")
            title_meta  = row.get("title", "")[:80]
            cov_score   = row.get("coverage_score", "0")
        else:
            # Formato supabase (id + ai_tags)
            event_id   = str(row["id"])
            ai_tags_raw = row.get("ai_tags", "{}")
            source      = ""
            title_meta  = ""
            cov_score   = "0"

        event = all_events.get(event_id, {})

        # Completar source y title desde el CSV de eventos si falta
        if not source:
            source = event.get("source", "")
        if not title_meta:
            title_meta = (event.get("title") or "")[:80]

        try:
            tags = json.loads(ai_tags_raw) if ai_tags_raw else {}
        except json.JSONDecodeError:
            tags = {}

        # Calcular coverage si no viene del CSV
        if cov_score == "0" and tags:
            filled = sum(1 for f in ["mood", "ambiente", "actividad", "ideal_para"]
                         if tags.get(f))
            cov_score = str(round(filled / 4, 3))

        enriched = build_enriched_text(event, tags, venue_map=venue_map)
        if not enriched.strip():
            skipped += 1
            continue

        items.append({
            "event_id":      event_id,
            "source":        source,
            "title":         title_meta,
            "enriched_text": enriched,
            "ai_tags_json":  ai_tags_raw,
            "coverage_score": cov_score,
        })

    return items, skipped


def run_embedding_pipeline(input_csv: Path, limit: int = None):
    # Cargar tagging output (formato completo o supabase)
    with open(input_csv, encoding="utf-8") as f:
        tagged_rows = list(csv.DictReader(f))

    if limit:
        tagged_rows = tagged_rows[:limit]

    # Cargar eventos originales para título, descripción y venue
    with open(EVENTS_CSV, encoding="utf-8") as f:
        all_events = {str(r["id"]): r for r in csv.DictReader(f)}

    # Cargar venue map (venue_id → nombre)
    venue_map = _load_venue_map()
    print(f"🗺️  Venues cargados: {len(venue_map)}")

    items, skipped = _build_items_from_rows(tagged_rows, all_events, venue_map=venue_map)

    total = len(items)
    print(f"\n🔢 Eventos a embeber: {total}  (saltados por JSON inválido: {skipped})")
    print(f"   Modelo:     {EMBEDDING_MODEL}  ({EMBEDDING_DIMS} dims)")
    est_cost = estimate_embedding_cost([i["enriched_text"] for i in items])
    print(f"   Costo est.: ~${est_cost:.4f} USD")

    # Output paths
    timestamp    = datetime.datetime.now().strftime("%Y%m%d")
    out_path     = OUTPUT_DIR / f"embeddings_{timestamp}.csv"
    out_full     = OUTPUT_DIR / f"embeddings_full_{timestamp}.csv"

    # Campos de salida
    fieldnames_full = ["event_id","source","title","coverage_score","enriched_text","ai_tags_json","embedding","token_estimate","cost_usd"]
    fieldnames_sb   = ["event_id","model_version","computed_at","embedding"]   # Schema exacto de event_embeddings

    total_cost = 0.0

    with open(out_full, "w", newline="", encoding="utf-8") as ff, \
         open(out_path, "w", newline="", encoding="utf-8") as fs:

        writer_full = csv.DictWriter(ff, fieldnames=fieldnames_full)
        writer_sb   = csv.DictWriter(fs, fieldnames=fieldnames_sb)
        writer_full.writeheader()
        writer_sb.writeheader()

        # Procesar en batches
        for batch_start in range(0, total, BATCH_SIZE):
            batch = items[batch_start : batch_start + BATCH_SIZE]
            texts = [item["enriched_text"] for item in batch]

            print(f"  Batch {batch_start//BATCH_SIZE + 1}/{-(-total//BATCH_SIZE)}"
                  f"  [{batch_start+1}-{min(batch_start+BATCH_SIZE, total)}/{total}]...",
                  end=" ", flush=True)

            t0 = time.time()
            try:
                vectors = embed_batch(texts)
            except Exception as e:
                print(f"❌ Error: {e}")
                continue

            elapsed = time.time() - t0
            batch_cost = estimate_embedding_cost(texts)
            total_cost += batch_cost
            print(f"✅  {elapsed:.1f}s  ${batch_cost:.5f}")

            for item, vector in zip(batch, vectors):
                token_est = len(item["enriched_text"]) // 4
                vec_str   = json.dumps(vector)

                writer_full.writerow({
                    "event_id":       item["event_id"],
                    "source":         item["source"],
                    "title":          item["title"],
                    "coverage_score": item["coverage_score"],
                    "enriched_text":  item["enriched_text"][:300],  # Truncado en CSV
                    "ai_tags_json":   item["ai_tags_json"],
                    "embedding":      vec_str,
                    "token_estimate": token_est,
                    "cost_usd":       round((token_est / 1_000_000) * COST_PER_1M_TOKENS, 8),
                })
                writer_sb.writerow({
                    "event_id":      item["event_id"],
                    "model_version": EMBEDDING_MODEL,
                    "computed_at":   datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                    "embedding":     vec_str,
                })

            ff.flush()
            fs.flush()
            time.sleep(0.1)

    print(f"\n{'═'*60}")
    print(f"  ✅ Embeddings generados — {total} eventos")
    print(f"  Costo total:   ${total_cost:.4f} USD")
    print(f"{'═'*60}")
    print(f"\n  📁 Supabase:    {out_path.name}  (id + embedding)")
    print(f"  📁 Completo:    {out_full.name}  (con enriched_text y métricas)")
    print(f"\n  Para agregar columna en Supabase:")
    print(f"    ALTER TABLE events ADD COLUMN embedding vector(1536);")
    print(f"  Luego importar {out_path.name} con upsert en Table Editor.")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEMI Embedding Pipeline")
    parser.add_argument("--input", required=True, help="CSV output de tagging_pipeline.py")
    parser.add_argument("--limit", type=int, default=None, help="Procesar solo los primeros N eventos")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = OUTPUT_DIR / input_path

    if not input_path.exists():
        print(f"❌ No se encontró el archivo: {input_path}")
        exit(1)

    run_embedding_pipeline(input_path, limit=args.limit)
