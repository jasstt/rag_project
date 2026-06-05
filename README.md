# RAG Pipeline with Hybrid Search and Gemini 

This repository contains an advanced Retrieval-Augmented Generation (RAG) pipeline that combines hybrid search (Dense + Sparse) with Reciprocal Rank Fusion (RRF), and leverages Google's Gemini models for powerful reranking and response generation with precise source citations.

## 🌟 Key Features

* **Hybrid Retrieval System:**
  * **Dense Retrieval:** Uses `sentence-transformers` (`all-MiniLM-L6-v2`) and **ChromaDB** for semantic search.
  * **Sparse Retrieval:** Uses **BM25** (`rank_bm25`) for exact keyword matching.
* **Reciprocal Rank Fusion (RRF):** Combines the results of dense and sparse searches mathematically to get the best of both worlds.
* **LLM Reranking:** Sends the top 20 candidates to the **Gemini API** to select the absolute top 5 most contextually relevant chunks.
* **Citation-Backed Generation:** Uses the Gemini API to answer the user's query while strictly citing the sources inline (e.g., `[1]`, `[2]`).
* **Evaluation Module:** Built-in evaluation script to test the pipeline against a truth dataset and verify citation coverage and relevance.

## 📁 Project Structure

```
rag-project/
├── data/
│   └── belgeler/        # Put your .txt and .pdf files here
├── db/                  # Auto-generated ChromaDB storage and BM25 chunks
├── src/
│   ├── ingest.py        # Reads docs, chunks them, and stores embeddings
│   ├── search.py        # Performs Hybrid Search (ChromaDB + BM25) and RRF fusion
│   ├── rerank.py        # Sends top 20 to Gemini for top-5 reranking
│   ├── generate.py      # Generates the final answer with citations via Gemini
│   └── eval.py          # Runs tests from eval_set.json
├── eval_set.json        # Custom question-answer pairs for evaluation
├── main.py              # Interactive CLI application
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (API Keys)
└── .gitignore
```

## 🚀 Getting Started

### 1. Install Dependencies

Make sure you have Python installed, then install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys

This project uses the Google Gemini API for reranking and generating answers. 
1. Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).
2. Create a `.env` file in the root directory (already ignored by Git):
   ```env
   GEMINI_API_KEY=AIzaSy_YOUR_API_KEY_HERE
   ```

### 3. Ingest Documents

Drop your `.txt` or `.pdf` files into the `data/belgeler/` folder. 
Run the ingestion script to chunk the texts, create embeddings, and build the search index:

```bash
python src/ingest.py
```
*Note: The first time you run this, it will download the sentence-transformer model (~80MB).*

### 4. Run the RAG Pipeline

Start the interactive CLI to ask questions about your documents:

```bash
python main.py
```

Type your question and watch the pipeline search, rerank, and generate a well-cited answer!

## 🧪 Evaluation

To evaluate the system, add some questions and expected keywords into `eval_set.json`:

```json
[
  {
    "question": "What is machine learning?",
    "expected_keywords": ["learning", "data", "algorithms"]
  }
]
```

Then run the evaluation script to see how well the pipeline performs (verifying both citations and keywords):

```bash
python src/eval.py
```
*(Note: If you are using the free tier of the Gemini API, you may occasionally see a `503 UNAVAILABLE` error during heavy automated eval tests. Simply wait a minute and retry.)*

## 📊 Dense vs Hybrid Search Comparison

We ran a comparison test to demonstrate why Hybrid Search is necessary. When querying *"Transformer mimarisinin avantajları nelerdir?"* (What are the advantages of the Transformer architecture?):

**Only Dense Search (ChromaDB):**
1. `nlp_temelleri.txt` ✅ (Correctly identifies Transformers)
2. `veri_bilimi.txt` ❌ (Completely unrelated text about Data Science metrics)
3. `veri_bilimi.txt` ❌ (Unrelated text about Feature Engineering)
*Why? Dense search alone can sometimes be biased by the semantic structure of sentences rather than strict keyword matching.*

**Hybrid Search (Dense + BM25 + RRF):**
1. `nlp_temelleri.txt` ✅ (Correctly identifies Transformers)
2. `nlp_temelleri.txt` ✅ (Related context about NLP and Word Embeddings)
3. `veri_bilimi.txt` ❌
*Why? BM25 caught the exact keyword "Transformer" and "RNN" to boost the relevance of the NLP document, resulting in much richer and more accurate candidates for the LLM.*

## 🛠 Technologies Used
* **[ChromaDB](https://www.trychroma.com/):** Vector database for semantic search.
* **[Sentence-Transformers](https://sbert.net/):** Lightweight embedding generation.
* **[Rank-BM25](https://pypi.org/project/rank-bm25/):** Keyword-based sparse retrieval.
* **[Google GenAI SDK](https://github.com/google/genai-python):** LLM integration for reranking and generation.
* **[PyMuPDF (fitz)](https://pymupdf.readthedocs.io/):** High-speed PDF parsing.
