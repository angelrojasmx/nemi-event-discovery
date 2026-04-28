# NEMI — Event Discovery Backend

ML backend for an event discovery app in Monterrey. Handles automatic event tagging, semantic search, and personalized recommendations.

## What it does

Events published by partners get enriched automatically: an LLM extracts structured tags (category, vibe, price range, audience), then embeddings are generated and stored for semantic retrieval. Users get a ranked feed based on their interaction history.

## Architecture

```
partner publishes event
        │
        ▼
SQS: nemi-events-to-tag
        │
        ▼
Lambda: enrich_event  ──► Gemini 2.5 Flash (fallback: Claude Haiku)
        │                  extracts ai_tags → saves to Supabase
        ▼
SQS: nemi-events-to-embed
        │
        ▼
Lambda: embed_event  ──► OpenAI text-embedding-3-small
        │                 generates 1536-dim vector → pgvector (Supabase)
        ▼
Lambda: update_user_embedding  ──► updates user profile vector on interaction
```

Semantic search uses pgvector with HNSW indexing + trigram fallback for hybrid scoring. Retrieval is under 300ms in production.

## Structure

```
llm_tagging/
  benchmark_llm_tagging.py   # benchmarked 4 LLMs across coverage, consistency, cost
  tagging_pipeline.py        # batch tagging for existing events
  embedding_pipeline.py      # batch embedding generation
  test_semantic_search.py    # integration tests

semantic_search/
  semantic_search.py         # Lambda function (hybrid cosine + trigram scoring)
  semantic_search.sql        # pgvector HNSW index setup

sqs_pipeline/
  lambda_enrich_event.py     # tagging Lambda
  lambda_embed_event.py      # embedding Lambda
  lambda_update_user_embedding.py  # user profile update Lambda
  sqs_utils.py               # shared utilities

recommendation/
  recommendation_engine.py   # cold-start + warm user ranking (semantic/popularity/hybrid)
```

## LLM Benchmark results

Compared GPT-4o-mini, Gemini 2.5 Flash, Llama 3.3 70B, Claude Haiku 3.5 on a sample of real events.

Selected model: **Gemini 2.5 Flash** — ~85.5% tag coverage at ~$0.00013/event (7× below budget target). Anthropic Claude Haiku 3.5 used as fallback for reliability.

## Setup

```bash
# Required environment variables
OPENAI_API_KEY=...
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
SUPABASE_URL=...
SUPABASE_SECRET_KEY=...
AWS_REGION=us-east-1
SQS_EMBED_QUEUE_URL=...
```

Deploy Lambda functions with the included packaging scripts. Tested on Python 3.12.
