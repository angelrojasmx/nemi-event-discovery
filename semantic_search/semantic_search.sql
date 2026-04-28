-- ══════════════════════════════════════════════════════════════════
-- NEMI — Búsqueda semántica con pgvector
-- NEM-110
--
-- SETUP (correr una vez, en orden):
--   1. Habilitar extensión pgvector (ya debería estar activa en Supabase)
--   2. Crear índice HNSW para búsqueda rápida (~20ms con 865+ eventos)
--   3. Crear la función match_events
--
-- INVOCAR desde Python:
--   supabase.rpc('match_events', {
--     'query_embedding': [...],   # vector de 1536 dims
--     'query_text':      'rock en vivo',
--     'filter_city':     'Monterrey',
--     'match_count':     20,
--     'match_threshold': 0.3      # score mínimo (0-1), filtro de calidad
--   })
-- ══════════════════════════════════════════════════════════════════


-- ── 1. ÍNDICE HNSW (correr una sola vez) ─────────────────────────
-- Necesario para que la búsqueda sea < 50ms con miles de eventos.
-- m=16, ef_construction=64 es el balance recomendado por pgvector para
-- colecciones de menos de 100k vectores.

CREATE INDEX IF NOT EXISTS event_embeddings_hnsw_idx
ON event_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);


-- ── 2. FUNCIÓN PRINCIPAL DE BÚSQUEDA ─────────────────────────────

CREATE OR REPLACE FUNCTION match_events(
    query_embedding  vector(1536),
    query_text       text,
    filter_city      text    DEFAULT 'Monterrey',
    match_count      int     DEFAULT 20,
    match_threshold  float   DEFAULT 0.3
)
RETURNS TABLE (
    id              uuid,
    title           text,
    description     text,
    category        varchar,
    start_date      date,
    start_time      timetz,
    end_date        date,
    price_tier      varchar,
    price_min       float8,
    price_max       float8,
    currency        varchar,
    image_url       text,
    ai_tags         jsonb,
    venue_name      text,
    venue_city      text,
    source          varchar,
    source_url      text,
    semantic_score  float,
    text_score      float,
    final_score     float
)
LANGUAGE sql STABLE
AS $$
    -- Subquery: para eventos recurrentes (mismo título + venue) quedarse solo
    -- con la próxima ocurrencia antes de rankear por relevancia.
    -- DISTINCT ON requiere que ORDER BY empiece con las mismas columnas,
    -- por eso separamos deduplicación y ranking en dos pasos.
    SELECT
        deduped.id,
        deduped.title,
        deduped.description,
        deduped.category,
        deduped.start_date,
        deduped.start_time,
        deduped.end_date,
        deduped.price_tier,
        deduped.price_min,
        deduped.price_max,
        deduped.currency,
        deduped.image_url,
        deduped.ai_tags,
        deduped.venue_name,
        deduped.venue_city,
        deduped.source,
        deduped.source_url,
        deduped.semantic_score,
        deduped.text_score,
        deduped.final_score
    FROM (
        SELECT DISTINCT ON (e.title, e.venue_id)
            e.id,
            e.title,
            e.description,
            e.category,
            e.start_date,
            e.start_time,
            e.end_date,
            e.price_tier,
            e.price_min,
            e.price_max,
            e.currency,
            e.image_url,
            e.ai_tags,
            v.name AS venue_name,
            v.city AS venue_city,
            e.source,
            e.source_url,
            (1 - (ee.embedding <=> query_embedding))::float AS semantic_score,
            CASE
                WHEN e.title ILIKE '%' || query_text || '%'       THEN 0.40
                WHEN e.description ILIKE '%' || query_text || '%' THEN 0.20
                ELSE 0.0
            END::float AS text_score,
            (
                0.75 * (1 - (ee.embedding <=> query_embedding)) +
                0.25 * CASE
                    WHEN e.title ILIKE '%' || query_text || '%'       THEN 0.40
                    WHEN e.description ILIKE '%' || query_text || '%' THEN 0.20
                    ELSE 0.0
                END
            )::float AS final_score
        FROM events e
        JOIN event_embeddings ee ON e.id = ee.event_id
        JOIN venues           v  ON e.venue_id = v.id
        WHERE
            e.status = 'published'
            AND v.city = filter_city
            AND e.start_date >= CURRENT_DATE
            AND (1 - (ee.embedding <=> query_embedding)) >= match_threshold
        -- DISTINCT ON se queda con la fecha más próxima de cada título+venue
        ORDER BY e.title, e.venue_id, e.start_date ASC
    ) deduped
    -- Ranking final por relevancia
    ORDER BY deduped.final_score DESC
    LIMIT match_count;
$$;


-- ── 3. FUNCIÓN DE BÚSQUEDA SIN FILTRO DE CIUDAD ──────────────────
-- Versión fallback para cuando no se conoce la ciudad del usuario
-- o para búsquedas globales futuras.

CREATE OR REPLACE FUNCTION match_events_global(
    query_embedding  vector(1536),
    query_text       text,
    match_count      int   DEFAULT 20,
    match_threshold  float DEFAULT 0.3
)
RETURNS TABLE (
    id              uuid,
    title           text,
    description     text,
    category        varchar,
    start_date      date,
    start_time      timetz,
    price_tier      varchar,
    price_min       float8,
    image_url       text,
    ai_tags         jsonb,
    venue_name      text,
    venue_city      text,
    source          varchar,
    semantic_score  float,
    final_score     float
)
LANGUAGE sql STABLE
AS $$
    SELECT
        e.id,
        e.title,
        e.description,
        e.category,
        e.start_date,
        e.start_time,
        e.price_tier,
        e.price_min,
        e.image_url,
        e.ai_tags,
        v.name   AS venue_name,
        v.city   AS venue_city,
        e.source,
        (1 - (ee.embedding <=> query_embedding))::float AS semantic_score,
        (
            0.75 * (1 - (ee.embedding <=> query_embedding)) +
            0.25 * CASE
                WHEN e.title ILIKE '%' || query_text || '%' THEN 0.40
                ELSE 0.0
            END
        )::float AS final_score
    FROM events e
    JOIN event_embeddings ee ON e.id = ee.event_id
    JOIN venues           v  ON e.venue_id = v.id
    WHERE
        e.status = 'published'
        AND e.start_date >= CURRENT_DATE
        AND (1 - (ee.embedding <=> query_embedding)) >= match_threshold
    ORDER BY final_score DESC
    LIMIT match_count;
$$;


-- ── NOTAS DE CALIBRACIÓN ─────────────────────────────────────────
--
-- match_threshold = 0.3:
--   Score semántico mínimo aceptable. Con text-embedding-3-small y
--   enriched_text (título + descripción + ai_tags), un score de 0.3
--   ya indica cierta relevancia temática. Ajustar hacia 0.4 si hay
--   demasiado ruido en resultados, o hacia 0.2 si hay pocos resultados.
--
-- Pesos 75/25:
--   Favorece el match semántico sobre el textual. Esto permite que
--   "algo para bailar" encuentre eventos de salsa aunque no digan
--   exactamente "bailar". El 25% textual protege búsquedas de nombres
--   exactos ("Julieta Venegas", "Arena Monterrey").
--
-- HNSW vs IVFFlat:
--   HNSW tiene mejor recall y latencia. IVFFlat requiere mínimo
--   ~1000 vectores para ser efectivo. Con < 1000 eventos, HNSW
--   es la opción correcta.
