# config.py — RAG Pipeline Konfigürasyon

# Reranker modu: "gemini" veya "local"
# "gemini" → Google Gemini API ile reranking (varsayılan)
# "local"  → cross-encoder/ms-marco-MiniLM-L-6-v2 (ücretsiz, offline)
RERANK_MODE = "local"

# Skip-rerank eşiği (%): top-1 skoru top-2'den bu kadar yüksekse reranker atlanır
SKIP_RERANK_THRESHOLD = 0.03
