"""
RAG utilities - Simplified version
ChromaDB + GROQ + Sentence-Transformers + BM25 Hybrid Search + LangChain SemanticChunker
"""

import os
import re
import numpy as np
from typing import List, Tuple

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from rank_bm25 import BM25Okapi
from groq import Groq
import streamlit as st


def clean_transcript(text: str) -> str:
    """Clean transcript by removing timestamps and VTT/SRT formatting."""
    if not text:
        return ""
    
    # Remove VTT header and metadata
    text = re.sub(r'^WEBVTT.*?\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Kind:.*?\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Language:.*?\n', '', text, flags=re.MULTILINE)
    
    # Remove timestamps
    text = re.sub(r'\d{1,2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[.,]\d{3}', '', text)
    text = re.sub(r'[\[\(]?\d{1,2}:\d{2}:\d{2}[\]\)]?', '', text)
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Remove VTT tags
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\b(align|position|line|size):[^\s]+', '', text)
    
    # Clean whitespace
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r' +', ' ', text)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return ' '.join(lines).strip()


# Cache expensive model loading - runs ONCE
@st.cache_resource
def load_hf_embeddings():
    """Load HuggingFace embeddings once - used for both LangChain and ChromaDB."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def load_semantic_chunker(_hf_embeddings):
    """Load semantic chunker once."""
    return SemanticChunker(_hf_embeddings)


@st.cache_resource
def get_rag_system(groq_api_key: str, model: str = "llama-3.1-8b-instant"):
    """Get or create a cached RAG system instance."""
    return RAGSystem(groq_api_key=groq_api_key, model=model)


class RAGSystem:
    """Simple RAG system with LangChain SemanticChunker and hybrid search."""
    
    PROMPT_TEMPLATE = """You are a helpful assistant answering questions about video content.

Context from transcript:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the provided context
- Be concise and clear
- If info not available, say so and answer based on your knowledge

Answer:"""

    def __init__(self, groq_api_key: str = None, model: str = "llama-3.1-8b-instant"):
        # GROQ setup
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY required")
        self.groq = Groq(api_key=self.groq_api_key)
        self.llm_model = model
        
        # Use cached embeddings (loaded ONCE) - same model for both purposes
        self.embeddings = load_hf_embeddings()
        self.chunker = load_semantic_chunker(self.embeddings)
        
        # ChromaDB - persistent client
        self.chroma = chromadb.Client()
        self.collection = None
        
        # BM25
        self.bm25 = None
        self.corpus = []
        self.chunk_ids = []
        
        # State
        self.indexed_video_id = None
        self.chunks = []

    def index(self, transcript: str, video_id: str) -> int:
        """Index transcript with LangChain SemanticChunker."""
        # Clean transcript
        cleaned = clean_transcript(transcript)
        if not cleaned or len(cleaned) < 50:
            cleaned = transcript[:1000] if transcript else "No transcript"
        
        # Semantic chunking using LangChain
        docs = self.chunker.create_documents([cleaned])
        self.chunks = [{"id": f"{video_id}_{i}", "text": doc.page_content} for i, doc in enumerate(docs)]
        
        # Fallback if no chunks
        if not self.chunks:
            self.chunks = [{"id": f"{video_id}_0", "text": cleaned[:500]}]
        
        # Setup ChromaDB
        collection_name = f"yt_{re.sub(r'[^a-zA-Z0-9_]', '_', video_id)}"[:63]
        try:
            self.chroma.delete_collection(collection_name)
        except:
            pass
        self.collection = self.chroma.create_collection(collection_name, metadata={"hnsw:space": "cosine"})
        
        # Index in ChromaDB
        ids = [c["id"] for c in self.chunks]
        texts = [c["text"] for c in self.chunks]
        embeddings = self.embeddings.embed_documents(texts)
        self.collection.add(ids=ids, embeddings=embeddings, documents=texts)
        
        # Build BM25 index
        self.corpus = texts
        self.chunk_ids = ids
        tokenized = [re.findall(r'\b\w+\b', t.lower()) for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        
        self.indexed_video_id = video_id
        return len(self.chunks)

    def _semantic_search(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """Search using embeddings."""
        if not self.collection:
            return []
        emb = self.embeddings.embed_query(query)
        results = self.collection.query(query_embeddings=[emb], n_results=min(top_k, len(self.chunks)), include=["documents", "distances"])
        if not results['ids'][0]:
            return []
        return [(results['ids'][0][i], results['documents'][0][i], 1 - results['distances'][0][i]) 
                for i in range(len(results['ids'][0]))]

    def _bm25_search(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """Search using BM25 keyword matching."""
        if not self.bm25 or not self.corpus:
            return []
        tokens = re.findall(r'\b\w+\b', query.lower())
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        max_score = max(scores) if max(scores) > 0 else 1
        return [(self.chunk_ids[i], self.corpus[i], scores[i]/max_score) for i in top_idx if scores[i] > 0]

    def _hybrid_search(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """Combine semantic + BM25 using Reciprocal Rank Fusion."""
        semantic = self._semantic_search(query, top_k * 2)
        bm25 = self._bm25_search(query, top_k * 2)
        
        # RRF scoring
        scores = {}
        k = 60
        for rank, (cid, text, _) in enumerate(semantic):
            scores[cid] = scores.get(cid, {"text": text, "score": 0})
            scores[cid]["score"] += 0.7 / (k + rank + 1)
        for rank, (cid, text, _) in enumerate(bm25):
            scores[cid] = scores.get(cid, {"text": text, "score": 0})
            scores[cid]["score"] += 0.3 / (k + rank + 1)
        
        if not scores:
            return []
        
        sorted_results = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)[:top_k]
        return [(cid, data["text"], data["score"]) for cid, data in sorted_results]

    def search(self, query: str, top_k: int = 3, method: str = "hybrid") -> List[Tuple[str, str, float]]:
        """Search with specified method: 'semantic', 'bm25', or 'hybrid'."""
        if method == "semantic":
            return self._semantic_search(query, top_k)
        elif method == "bm25":
            return self._bm25_search(query, top_k)
        return self._hybrid_search(query, top_k)

    def generate(self, query: str, context: str, max_context_chars: int = 14000) -> str:
        """Generate response using GROQ."""
        if len(context) > max_context_chars:
            context = context[:max_context_chars] + "\n[truncated]"
        
        prompt = self.PROMPT_TEMPLATE.format(context=context, question=query)
        
        try:
            response = self.groq.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

    def query(self, question: str, top_k: int = 3, method: str = "hybrid") -> Tuple[str, List]:
        """Full RAG: search + generate."""
        results = self.search(question, top_k, method)
        if not results:
            return "No relevant content found.", []
        
        context = "\n\n".join([text for _, text, _ in results])
        response = self.generate(question, context)
        return response, results

    @staticmethod
    def available_models() -> List[str]:
        return ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
