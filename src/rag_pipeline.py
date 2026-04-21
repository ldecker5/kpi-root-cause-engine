"""
rag_pipeline.py
---------------
MILESTONE 7: RAG Pipeline

Implements document ingestion, chunking, embedding, vector retrieval,
and RAG-augmented generation for root-cause analysis.

Usage:
    from src.rag_pipeline import RAGPipeline

    rag = RAGPipeline()
    rag.build_or_load()
    response = rag.generate_rag_response(anomaly_summary)
"""

import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

from src.performance import EMBEDDING_CACHE, COST_TRACKER

# ---------------------------------------------------------------------------
# Paths — resolve relative to this file so imports work from anywhere
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).parent
_PROJECT_ROOT = _SRC_DIR.parent

DOCS_DIR = _PROJECT_ROOT / "knowledge_base"
PERSIST_DIR = str(_PROJECT_ROOT / "chroma_db")

EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4.1-nano"

# Matryoshka-style "quantization" for embeddings. Dropping from 1536 → 512 dims
# cuts vector-store RAM/disk ~3x with <2% recall loss on most domains (per
# OpenAI). Set to None to keep the full native dimensionality.
EMBEDDING_DIMENSIONS: int | None = 512

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_id(doc) -> str:
    """Stable string ID for a chunk: source::page=N::start=N."""
    m = doc.metadata
    return f"{m.get('source')}::page={m.get('page')}::start={m.get('start_index')}"


def _format_context(docs, max_chars: int = 6000) -> str:
    """Format retrieved docs into a numbered block for the prompt."""
    blocks = []
    total = 0
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source")
        page = d.metadata.get("page")
        block = f"[{i}] Source: {src} (page {page})\n{d.page_content}".strip()
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Caching embeddings shim — LangChain-compatible
# ---------------------------------------------------------------------------

class _CachingEmbeddings:
    """
    Wraps a LangChain Embeddings object with a hash-keyed cache + cost tracking.

    Implements just `embed_documents` and `embed_query`, which is all Chroma
    needs. Cache hits skip the API entirely; misses go through the underlying
    embeddings object in one batched call.
    """

    def __init__(self, inner, *, model: str, dimensions: int | None):
        self._inner = inner
        self._model = model
        self._dimensions = dimensions

    # LangChain expects this exact signature on both methods.
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float] | None] = [None] * len(texts)
        pending_idx: list[int] = []
        pending_texts: list[str] = []

        for i, t in enumerate(texts):
            hit = EMBEDDING_CACHE.get(self._model, self._dimensions, t)
            if hit is not None:
                out[i] = hit
            else:
                pending_idx.append(i)
                pending_texts.append(t)

        if pending_texts:
            vecs = self._inner.embed_documents(pending_texts)
            # OpenAI usage isn't exposed through LangChain here, so approximate:
            # ~1 token per 4 chars is the standard heuristic for embeddings.
            approx_tokens = sum(max(1, len(t) // 4) for t in pending_texts)
            COST_TRACKER.record_embed(self._model, approx_tokens,
                                      tag="rag:doc")
            for idx, text, vec in zip(pending_idx, pending_texts, vecs):
                out[idx] = vec
                EMBEDDING_CACHE.put(self._model, self._dimensions, text, vec)

        return [v if v is not None else [] for v in out]

    def embed_query(self, text: str) -> list[float]:
        hit = EMBEDDING_CACHE.get(self._model, self._dimensions, text)
        if hit is not None:
            # Log a $0 entry so cache hits are observable in telemetry.
            COST_TRACKER.record_embed(self._model, 0, cached=True, tag="rag:query")
            return hit
        vec = self._inner.embed_query(text)
        approx_tokens = max(1, len(text) // 4)
        COST_TRACKER.record_embed(self._model, approx_tokens, tag="rag:query")
        EMBEDDING_CACHE.put(self._model, self._dimensions, text, vec)
        return vec


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def precision_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
    retrieved_k = retrieved_ids[:k]
    tp = len(set(retrieved_k) & set(relevant_ids))
    return tp / k


def recall_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
    retrieved_k = retrieved_ids[:k]
    tp = len(set(retrieved_k) & set(relevant_ids))
    return tp / len(relevant_ids) if relevant_ids else 0.0


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Encapsulates the full RAG pipeline:
      - Load PDFs from knowledge_base/
      - Chunk, embed, and store in Chroma
      - Retrieve relevant chunks for a query
      - Generate RAG-augmented or baseline responses
      - Evaluate retrieval quality (precision@k, recall@k)
    """

    def __init__(
        self,
        docs_dir: str | Path = DOCS_DIR,
        persist_dir: str = PERSIST_DIR,
        embedding_model: str = EMBEDDING_MODEL,
        generation_model: str = GENERATION_MODEL,
        dimensions: int | None = EMBEDDING_DIMENSIONS,
    ):
        self.docs_dir = Path(docs_dir)
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self.dimensions = dimensions

        # Base LangChain embeddings; wrapped below with caching + cost tracking.
        base_kwargs = {"model": embedding_model}
        if dimensions:
            base_kwargs["dimensions"] = dimensions
        self.embeddings = _CachingEmbeddings(
            OpenAIEmbeddings(**base_kwargs),
            model=embedding_model,
            dimensions=dimensions,
        )

        self.llm = ChatOpenAI(model=generation_model)
        self.vector_db = None

    # ------------------------------------------------------------------
    # 1. Document processing
    # ------------------------------------------------------------------

    def _load_documents(self) -> list:
        """Load all PDFs from docs_dir, one Document per page."""
        if not self.docs_dir.exists():
            raise FileNotFoundError(
                f"knowledge_base/ not found at {self.docs_dir}. "
                "Create the directory and add your PDF files."
            )

        pdf_files = list(self.docs_dir.rglob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(
                f"No PDF files found in {self.docs_dir}. "
                "Add your knowledge-base PDFs and re-run build_or_load()."
            )

        docs = []
        for fp in pdf_files:
            loader = PyPDFLoader(str(fp))
            pages = loader.load()
            for d in pages:
                d.metadata.update({
                    "source": fp.name,
                    "source_path": str(fp),
                    "source_type": "kpi_kb_pdf",
                })
            docs.extend(pages)

        print(f"Loaded {len(docs)} pages from {len(pdf_files)} PDF(s).")
        return docs

    def _split_documents(self, docs: list) -> list:
        """Split pages into overlapping chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True,
        )
        splits = splitter.split_documents(docs)
        print(f"Created {len(splits)} chunks.")
        return splits

    # ------------------------------------------------------------------
    # 2. Vector store
    # ------------------------------------------------------------------

    def build_or_load(self, force_rebuild: bool = False) -> None:
        """
        Build the Chroma vector DB from PDFs or load from disk if it exists.

        Args:
            force_rebuild: If True, re-index even if a persisted DB exists.
        """
        db_exists = Path(self.persist_dir).exists() and any(
            Path(self.persist_dir).iterdir()
        )

        if db_exists and not force_rebuild:
            print(f"Loading existing vector DB from {self.persist_dir} ...")
            self.vector_db = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
            )
            print(f"Vector DB loaded — {self.vector_db._collection.count()} chunks.")
        else:
            print("Building vector DB from documents ...")
            docs = self._load_documents()
            splits = self._split_documents(docs)
            self.vector_db = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self.persist_dir,
            )
            print(f"Vector DB built — {self.vector_db._collection.count()} chunks.")

    def _check_ready(self):
        if self.vector_db is None:
            raise RuntimeError("Call build_or_load() before using the pipeline.")

    # ------------------------------------------------------------------
    # 3. Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 5) -> list:
        """Dense similarity search — returns top-k Documents."""
        self._check_ready()
        return self.vector_db.similarity_search(query, k=k)

    def retrieve_with_scores(self, query: str, k: int = 5) -> list[tuple]:
        """Similarity search with cosine distance scores (lower = more similar)."""
        self._check_ready()
        return self.vector_db.similarity_search_with_score(query, k=k)

    def retrieve_mmr(self, query: str, k: int = 5, fetch_k: int = 20) -> list:
        """Maximal Marginal Relevance — balances relevance and diversity."""
        self._check_ready()
        return self.vector_db.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)

    # ------------------------------------------------------------------
    # 4. Generation
    # ------------------------------------------------------------------

    def generate_rag_response(self, anomaly_summary: str, k: int = 5) -> dict:
        """
        Generate an executive root-cause explanation augmented with
        retrieved knowledge-base context.

        Returns a dict with keys: response, retrieved_docs, context.
        """
        self._check_ready()
        retrieved = self.retrieve(anomaly_summary, k=k)
        context = _format_context(retrieved)

        prompt = f"""You are an analytics assistant generating executive-facing explanations.

Rules:
- Use ONLY the evidence in (1) the anomaly summary and (2) the retrieved context.
- Clearly separate evidence from hypotheses.
- Do not introduce causes not supported by the anomaly summary or retrieved context.
- If the knowledge base does not support a claim, say: "Not supported by retrieved context."
- When referencing retrieved info, cite the bracketed source number(s) like [1], [2].

Retrieved context:
{context}

Anomaly summary:
{anomaly_summary}

Output format:
Executive summary (2-3 sentences)
Key evidence
Plausible explanations (clearly labeled as hypotheses)
Suggested next analysis
"""
        response = self.llm.invoke(prompt).content
        return {
            "response": response,
            "retrieved_docs": retrieved,
            "context": context,
        }

    def generate_baseline_response(self, anomaly_summary: str) -> str:
        """
        Generate a response using ONLY the anomaly summary (no retrieval).
        Used as a comparison baseline.
        """
        prompt = f"""You are an analytics assistant generating executive-facing explanations.

Rules:
- Base claims strictly on the anomaly summary only.
- Clearly separate evidence from hypotheses.
- Do not introduce causes not supported by the anomaly summary.
- If unsure, say "I don't know."

Anomaly summary:
{anomaly_summary}

Output format:
Executive summary (2-3 sentences)
Key evidence
Plausible explanations (clearly labeled as hypotheses)
Suggested next analysis
"""
        return self.llm.invoke(prompt).content

    # ------------------------------------------------------------------
    # 5. Evaluation
    # ------------------------------------------------------------------

    def evaluate_retrieval(
        self,
        test_set: list[dict],
        k: int = 5,
        use_mmr: bool = False,
    ) -> dict:
        """
        Evaluate retrieval quality against a labeled test set.

        Each item in test_set must have:
            - "query": str
            - "relevant_chunk_ids": list[str]  (use chunk_id format)

        Returns:
            {
                "avg_precision": float,
                "avg_recall": float,
                "per_query": list[dict],
            }
        """
        self._check_ready()
        scores = []

        for case in test_set:
            rel = case["relevant_chunk_ids"]
            if not rel:
                continue

            if use_mmr:
                hits = self.retrieve_mmr(case["query"], k=k)
            else:
                hits = self.retrieve(case["query"], k=k)

            retrieved_ids = [_chunk_id(h) for h in hits]
            p = precision_at_k(retrieved_ids, rel, k)
            r = recall_at_k(retrieved_ids, rel, k)
            scores.append({"query": case["query"], "precision": p, "recall": r})

        if not scores:
            return {"avg_precision": 0.0, "avg_recall": 0.0, "per_query": []}

        avg_p = sum(s["precision"] for s in scores) / len(scores)
        avg_r = sum(s["recall"] for s in scores) / len(scores)
        return {"avg_precision": avg_p, "avg_recall": avg_r, "per_query": scores}

    def print_evaluation(self, test_set: list[dict], k: int = 5) -> None:
        """Convenience method: run and print similarity vs MMR comparison."""
        sim = self.evaluate_retrieval(test_set, k=k, use_mmr=False)
        mmr = self.evaluate_retrieval(test_set, k=k, use_mmr=True)

        print(f"Similarity:  Precision@{k}={sim['avg_precision']:.4f}  Recall@{k}={sim['avg_recall']:.4f}")
        print(f"MMR:         Precision@{k}={mmr['avg_precision']:.4f}  Recall@{k}={mmr['avg_recall']:.4f}")

        print("\nPer-query (Similarity):")
        for row in sim["per_query"]:
            print(f"  P={row['precision']:.2f}  R={row['recall']:.2f}  | {row['query']}")

        print("\nPer-query (MMR):")
        for row in mmr["per_query"]:
            print(f"  P={row['precision']:.2f}  R={row['recall']:.2f}  | {row['query']}")


# ---------------------------------------------------------------------------
# Labeled test set (mirrors notebook cell-45)
# ---------------------------------------------------------------------------

DEFAULT_TEST_SET = [
    {
        "query": "What is conversion_rate and what drives changes in it?",
        "relevant_chunk_ids": [
            "kpi_schema_and_definitions.pdf::page=1::start=0",
            "mobile_conversion_patterns.pdf::page=0::start=0",
            "past_rca_scenarios.pdf::page=0::start=0",
        ],
    },
    {
        "query": "How can marketing_spend changes (especially regional decreases) affect revenue?",
        "relevant_chunk_ids": [
            "marketing_spend_impact_framework.pdf::page=0::start=0",
            "kpi_schema_and_definitions.pdf::page=2::start=0",
            "past_rca_scenarios.pdf::page=0::start=0",
            "kpi_schema_and_definitions.pdf::page=1::start=0",
        ],
    },
    {
        "query": "If avg_order_value is stable but revenue drops, what should we check?",
        "relevant_chunk_ids": [
            "kpi_schema_and_definitions.pdf::page=2::start=0",
            "kpi_schema_and_definitions.pdf::page=0::start=0",
            "anomaly_investigation_runbook.pdf::page=0::start=0",
            "past_rca_scenarios.pdf::page=0::start=0",
            "kpi_schema_and_definitions.pdf::page=1::start=0",
        ],
    },
]


# ---------------------------------------------------------------------------
# Quick smoke-test: python src/rag_pipeline.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    rag = RAGPipeline()
    rag.build_or_load()

    anomaly_summary = (
        "Dataset columns: date, revenue, orders, traffic, conversion_rate, "
        "avg_order_value, region, device_type, marketing_spend, product_category.\n\n"
        "Anomaly: Revenue dropped 15% starting 2024-04-20.\n"
        "Findings: Primary driver: Mobile conversion_rate dropped 22% (r=0.71, p<0.01). "
        "Secondary: marketing_spend in the West region decreased 35%. "
        "Ruled out: avg_order_value (no significant change), product_category mix (stable)."
    )

    print("\n=== WITHOUT RAG ===\n")
    print(rag.generate_baseline_response(anomaly_summary))

    print("\n=== WITH RAG ===\n")
    result = rag.generate_rag_response(anomaly_summary)
    print(result["response"])
    print("\nRetrieved sources:")
    for i, d in enumerate(result["retrieved_docs"], 1):
        print(f"  [{i}] {d.metadata.get('source')} (page {d.metadata.get('page')})")

    print("\n=== RETRIEVAL EVALUATION ===\n")
    rag.print_evaluation(DEFAULT_TEST_SET, k=5)
