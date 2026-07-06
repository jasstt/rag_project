"""
mcp_server.py — RAG Pipeline as MCP Server (FastMCP API, v1.28+)

Antigravity chat'te doğrudan araç olarak kullanım için.

Araçlar:
  - query_rag(question, top_k)  → Belgelerden cevap üretir
  - ingest_docs()               → Dokümanları yeniden index'ler
  - pipeline_status()           → Pipeline durumunu gösterir

Başlatmak için:
  python mcp_server.py

mcp_config.json:
  {
    "mcpServers": {
      "rag-pipeline": {
        "command": "python",
        "args": ["C:/Users/VICTUS/OneDrive/Desktop/proje/rag-project/mcp_server.py"],
        "cwd": "C:/Users/VICTUS/OneDrive/Desktop/proje/rag-project"
      }
    }
  }
"""

import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("rag-pipeline")


# ─── Tools ────────────────────────────────────────────────────────────────────

@mcp.tool()
def query_rag(question: str, top_k: int = 20) -> str:
    """
    Kullanıcının sorusunu RAG pipeline'ından geçirir.
    data/belgeler/ klasöründeki belgelerden ilgili chunk'ları bulur,
    rerank eder ve Gemini ile kaynak atıflı cevap üretir.
    Belgelere dair her soruyu bu araçla cevapla.
    """
    try:
        from retrieval.hybrid_search import search
        from llm.reranker import rerank, get_skip_stats
        from llm.gemini_client import generate

        if not question.strip():
            return "Soru boş olamaz."

        candidates = search(question, top_k=top_k)

        if not candidates:
            return (
                "❌ Hiçbir chunk bulunamadı.\n"
                "Önce 'ingest_docs' aracını çalıştır: python utils/helpers.py"
            )

        top_chunks = rerank(question, candidates)
        skip_stats = get_skip_stats()
        result = generate(question, top_chunks)

        answer = result["answer"]
        sources = result["sources"]

        sources_text = "\n".join([
            f"  [{s['number']}] {s['source']} — {s.get('text_preview','')[:100]}..."
            for s in sources
        ])

        return (
            f"{answer}\n\n"
            f"---\n"
            f"📚 Kaynaklar:\n{sources_text}\n\n"
            f"⚡ Skip-rerank: {skip_stats.get('skip_rate', 'N/A')}"
        )

    except FileNotFoundError as e:
        return f"❌ Veritabanı bulunamadı: {e}\nÖnce 'ingest_docs' çalıştır."
    except Exception as e:
        return f"❌ Hata: {type(e).__name__}: {e}"


@mcp.tool()
def ingest_docs() -> str:
    """
    data/belgeler/ klasöründeki tüm .txt ve .pdf dosyalarını
    okuyup chunk'a böler, embed eder ve ChromaDB'ye kaydeder.
    Yeni belge eklendikten sonra çalıştır.
    """
    try:
        from utils.helpers import run_ingest
        run_ingest()
        return "✅ Ingest tamamlandı. Belgeler ChromaDB'ye kaydedildi."
    except Exception as e:
        return f"❌ Ingest hatası: {type(e).__name__}: {e}"


@mcp.tool()
def pipeline_status() -> str:
    """
    RAG pipeline'ının durumunu kontrol eder:
    kaç chunk index'lenmiş, hangi model kullanılıyor,
    config parametreleri.
    """
    try:
        import yaml
        import chromadb

        config_path = os.path.join(ROOT, "config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        db_dir = os.path.join(ROOT, cfg["vectordb"]["db_dir"])
        collection_name = cfg["vectordb"]["collection_name"]
        chunks_json = os.path.join(ROOT, cfg["paths"]["chunks_json"])

        chunk_count = "?"
        try:
            client = chromadb.PersistentClient(path=db_dir)
            col = client.get_collection(collection_name)
            chunk_count = col.count()
        except Exception:
            chunk_count = "❌ DB yok — ingest_docs çalıştır"

        json_exists = "✅" if os.path.exists(chunks_json) else "❌ Yok"

        return (
            f"📊 RAG Pipeline Durumu\n"
            f"{'─'*40}\n"
            f"🗄️  ChromaDB chunks : {chunk_count}\n"
            f"📄 chunks.json     : {json_exists}\n"
            f"🔤 Embed model     : {cfg['embedding']['model']}\n"
            f"🎯 Reranker mode   : {cfg['reranker']['mode']}\n"
            f"⚡ Skip threshold  : {cfg['reranker']['skip_threshold']}\n"
            f"🤖 LLM model       : {cfg['llm']['model']}\n"
            f"📦 Chunk size      : {cfg['chunking']['chunk_size']} chars\n"
        )

    except Exception as e:
        return f"❌ Durum alınamadı: {type(e).__name__}: {e}"


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
