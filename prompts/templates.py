"""
prompts/templates.py — Prompt templates for the RAG pipeline.

Keeping prompt strings here (separate from LLM client logic) allows:
  - Easy versioning of prompts without touching model code.
  - A/B testing different templates.
  - Reuse across Gemini, local LLMs, or other providers.
"""


def build_system_instruction() -> str:
    """
    System instruction for the answer-generation step.
    Instructs the model to stay grounded and cite sources.
    """
    return (
        "Sen yardımcı bir Türkçe asistansın. "
        "Verilen bağlam parçalarını kullanarak soruyu yanıtla. "
        "Her bilgiyi [1], [2] gibi kaynak numarasıyla destekle. "
        "Bağlamda olmayan bilgileri uydurma, 'bu bilgi mevcut değil' de. "
        "Yanıtın sonunda 'Kaynaklar:' bölümü ekle."
    )


def build_user_prompt(query: str, context_chunks: list[dict]) -> str:
    """
    Build the user-facing prompt by injecting retrieved context.

    Args:
        query: User question.
        context_chunks: List of {text, metadata} dicts (from retrieval/reranker).

    Returns:
        Formatted prompt string ready to send to the LLM.
    """
    context_parts = []
    for i, chunk in enumerate(context_chunks, start=1):
        source_name = chunk["metadata"].get("source", "unknown")
        context_parts.append(f"[{i}] (Kaynak: {source_name})\n{chunk['text']}")

    context_text = "\n\n".join(context_parts)

    return (
        f"BAĞLAM PARÇALARI:\n{context_text}\n\n"
        f"SORU: {query}\n\n"
        "Yukarıdaki bağlam parçalarını kullanarak soruyu Türkçe yanıtla. "
        "Her bilgiye [N] formatında kaynak numarası ekle."
    )


def build_rerank_prompt(query: str, candidates: list[dict], top_n: int = 5) -> str:
    """
    Build the reranking prompt for Gemini-based reranker.

    Args:
        query: User question.
        candidates: List of candidate chunks from retrieval.
        top_n: How many top results to select.

    Returns:
        Formatted reranking prompt string.
    """
    chunks_text = ""
    for i, chunk in enumerate(candidates):
        preview = chunk["text"][:300].replace("\n", " ")
        chunks_text += f"\n[{i+1}] (Kaynak: {chunk['metadata']['source']})\n{preview}\n"

    return (
        f"Aşağıdaki kullanıcı sorusuna en alakalı {top_n} metin parçasını seç ve "
        f"sıralı indeks numaralarını JSON formatında döndür.\n\n"
        f"SORU: {query}\n\n"
        f"ADAY PARÇALAR ({len(candidates)} adet):\n{chunks_text}\n\n"
        f"SADECE şu JSON formatında yanıt ver, başka hiçbir şey yazma:\n"
        f'{{\"ranking\": [idx1, idx2, idx3, idx4, idx5]}}\n\n'
        f"İndeksler 1-tabanlı olmalı (1 ile {len(candidates)} arasında)."
    )
