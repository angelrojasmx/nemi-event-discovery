from openai import OpenAI
from supabase import create_client

SUPABASE_URL = "https://imzjqgnlphbddlrfocei.supabase.co"   # <-- reemplaza
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImltempxZ25scGhiZGRscmZvY2VpIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3NDgwNzU1MCwiZXhwIjoyMDkwMzgzNTUwfQ.Uiz3XStWqsKySAVsHwXLXLi_rbRtPxDo_E_q0-5lg_g"              # <-- usa service_role, no anon
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

sb = create_client(SUPABASE_URL, SUPABASE_KEY)
oa = OpenAI(api_key=OPENAI_API_KEY)

queries = [
    "concierto de rock este fin de semana",
    "plan romántico para una cita",
    "actividad familiar con niños",
    "festival de música electrónica",
]

for query in queries:
    print(f"\n🔍 Query: '{query}'")

    # Generar embedding de la query
    resp = oa.embeddings.create(model="text-embedding-3-small", input=query)
    vector = resp.data[0].embedding

    # Llamar la función RPC de Supabase
    results = sb.rpc("match_events", {
        "query_embedding": vector,
        "query_text":      query,
        "match_threshold": 0.1,    # <-- baja esto
        "match_count":     5,
        "filter_city":     'Monterrey',
    }).execute()

    if not results.data:
        print("  (sin resultados)")
    else:
        for r in results.data:
            print(f"  • {r.get('title','?')[:60]}  [{r.get('final_score', 0):.3f}]")