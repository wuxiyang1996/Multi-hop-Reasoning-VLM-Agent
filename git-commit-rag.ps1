# Commit RAG module and related changes (run from Game-AI-Agent root).
# Ensure git user is set first, e.g.:
#   git config user.email "your@email.com"
#   git config user.name "Your Name"

git commit -m "Add RAG module: embeddings and top-k retrieval

- Text embedding: Qwen/Qwen3-Embedding-0.6B (default), configurable via RAG_EMBEDDING_MODEL
- Multimodal embedding: Qwen/Qwen3-VL-Embedding-2B (default), configurable via MULTIMODAL_EMBEDDING_MODEL
- MemoryStore and rank_memories() for comparing query to stored embeddings, returning top-k (k hyperparameter)
- README and requirements for rag/; quick link in main readme"
