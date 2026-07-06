"""
mcp_server.py — RAG Pipeline as an MCP (Model Context Protocol) Server

Bu server, RAG pipeline'ınızı Antigravity chat'e doğrudan araç olarak entegre eder.
Başlatınca Antigravity bu araçları görebilir ve siz sormadan bile kullanabilir.

Araçlar:
  - query_rag(question)  → Belgelerden cevap üretir
  - ingest_docs()        → Dokümanları yeniden index'ler
  - pipeline_status()    → Pipeline durumunu kontrol eder

Başlatmak için:
  python mcp_server.py

Antigravity'ye eklemek için:
  Settings → MCP Servers → Add → Command: python
  Args: ["C:/Users/VICTUS/OneDrive/Desktop/proje/rag-project/mcp_server.py"]
"""

import sys
import os
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server
from mcp.server.models import InitializationOptions


# ─── MCP Server Setup ──────────────────────────────────────────────────────────

server = Server("rag-pipeline")


# ─── Tool Definitions ──────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="query_rag",
            description=(
                "Kullanıcının sorusunu RAG pipeline'ından geçirir. "
                "data/belgeler/ klasöründeki belgelerden ilgili chunk'ları bulur, "
                "rerank eder ve Gemini ile kaynak atıflı cevap üretir. "
                "Belgelere dair her soruyu bu araçla cevapla."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Kullanıcının sorusu"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Kaç aday chunk aranacak (varsayılan: 20)",
                        "default": 20
                    }
                },
                "required": ["question"]
            }
        ),
        types.Tool(
            name="ingest_docs",
            description=(
                "data/belgeler/ klasöründeki tüm .txt ve .pdf dosyalarını "
                "okuyup chunk'a böler, embed eder ve ChromaDB'ye kaydeder. "
                "Yeni belge eklendikten sonra çalıştır."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="pipeline_status",
            description=(
                "RAG pipeline'ının durumunu kontrol eder: "
                "kaç chunk index'lenmiş, hangi model kullanılıyor, "
                "skip-rerank istatistikleri."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
    ]


# ─── Tool Execution ────────────────────────────────────────────────────────────

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:

    if name == "query_rag":
        return await _query_rag(arguments)

    elif name == "ingest_docs":
        return await _ingest_docs()

    elif name == "pipeline_status":
        return await _pipeline_status()

    else:
        return [types.TextContent(type="text", text=f"Bilinmeyen araç: {name}")]


async def _query_rag(args: dict) -> list[types.TextContent]:
    """RAG pipeline'ını çalıştırır."""
    try:
        from retrieval.hybrid_search import search
        from llm.reranker import rerank, get_skip_stats
        from llm.gemini_client import generate

        question = args.get("question", "")
        top_k = args.get("top_k", 20)

        if not question:
            return [types.TextContent(type="text", text="Soru boş olamaz.")]

        # 1. Hybrid search
        candidates = search(question, top_k=top_k)

        if not candidates:
            return [types.TextContent(
                type="text",
                text="❌ Hiçbir chunk bulunamadı. Önce 'ingest_docs' aracını çalıştır."
            )]

        # 2. Rerank
        top_chunks = rerank(question, candidates)
        skip_stats = get_skip_stats()

        # 3. Generate
        result = generate(question, top_chunks)
        answer = result["answer"]
        sources = result["sources"]

        # Format response
        sources_text = "\n".join([
            f"  [{s['number']}] {s['source']} — {s['text_preview'][:100]}..."
            for s in sources
        ])

        output = (
            f"{answer}\n\n"
            f"---\n"
            f"📚 Kaynaklar:\n{sources_text}\n\n"
            f"⚡ Skip-rerank: {skip_stats['skip_rate']}"
        )

        return [types.TextContent(type="text", text=output)]

    except FileNotFoundError as e:
        return [types.TextContent(
            type="text",
            text=f"❌ Veritabanı bulunamadı: {e}\nÖnce 'ingest_docs' çalıştır."
        )]
    except Exception as e:
        return [types.TextContent(type="text", text=f"❌ Hata: {e}")]


async def _ingest_docs() -> list[types.TextContent]:
    """Doküman ingest işlemini başlatır."""
    try:
        from utils.helpers import run_ingest
        run_ingest()
        return [types.TextContent(
            type="text",
            text="✅ Ingest tamamlandı. Belgeler ChromaDB'ye kaydedildi."
        )]
    except Exception as e:
        return [types.TextContent(type="text", text=f"❌ Ingest hatası: {e}")]


async def _pipeline_status() -> list[types.TextContent]:
    """Pipeline durumunu kontrol eder."""
    try:
        import yaml
        import chromadb

        config_path = os.path.join(ROOT, "config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        db_dir = os.path.join(ROOT, cfg["vectordb"]["db_dir"])
        collection_name = cfg["vectordb"]["collection_name"]
        chunks_json = os.path.join(ROOT, cfg["paths"]["chunks_json"])

        # ChromaDB chunk count
        chunk_count = "?"
        try:
            client = chromadb.PersistentClient(path=db_dir)
            col = client.get_collection(collection_name)
            chunk_count = col.count()
        except Exception:
            chunk_count = "❌ DB bulunamadı — ingest gerekli"

        # chunks.json
        json_exists = "✅" if os.path.exists(chunks_json) else "❌ Yok"

        status = (
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

        return [types.TextContent(type="text", text=status)]

    except Exception as e:
        return [types.TextContent(type="text", text=f"❌ Durum alınamadı: {e}")]


# ─── Main ──────────────────────────────────────────────────────────────────────

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="rag-pipeline",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
