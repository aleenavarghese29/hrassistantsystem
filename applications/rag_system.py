import os
import re
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Protocol
from pathlib import Path
from enum import Enum

import numpy as np
import faiss
import pickle
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from pathlib import Path
from rank_bm25 import BM25Okapi


# CONFIGURATION


@dataclass
class RAGConfig:
    """Central configuration for the RAG system."""
    
    # Paths
    faiss_index_path: str = "./index/faiss_index.bin"
    metadata_path: str = "./index/chunk_metadata.pkl"
    embedding_model_name: str = "./models/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Retrieval settings
    top_k: int = 5
    base_similarity_threshold: float = 0.35
    
    # Reranking
    enable_reranking: bool = True
    reranker_model_name: str = "./models/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 5  # Rerank top N candidates
    
    # LLM settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3:8b"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 120
    llm_timeout: int = 20
    llm_max_retries: int = 3
    llm_top_p: float = 0.1
    llm_presence_penalty: float = 0.0
    llm_frequency_penalty: float = 0.0
    
    # Context management
    max_context_tokens: int = 2048
    max_chunks_in_context: int = 3
    
    # Safety
    max_query_length: int = 500
    enable_pii_redaction: bool = False
    
    # Observability
    log_level: str = "INFO"
    enable_debug_mode: bool = False
    
    def __post_init__(self) -> None:
        """Apply environment-variable overrides at instantiation time."""
        if os.getenv("FAISS_INDEX_PATH"):
            self.faiss_index_path = os.environ["FAISS_INDEX_PATH"]
        if os.getenv("METADATA_PATH"):
            self.metadata_path = os.environ["METADATA_PATH"]
        if os.getenv("OLLAMA_BASE_URL"):
            self.ollama_base_url = os.environ["OLLAMA_BASE_URL"]
        if os.getenv("OLLAMA_MODEL"):
            self.ollama_model = os.environ["OLLAMA_MODEL"]
        if os.getenv("LOG_LEVEL"):
            self.log_level = os.environ["LOG_LEVEL"]

    
    def validate(self) -> None:
        """Validate configuration on initialization."""
        if not Path(self.faiss_index_path).exists():
            raise FileNotFoundError(f"FAISS index not found: {self.faiss_index_path}")
        if not Path(self.metadata_path).exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not 0 <= self.base_similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be in [0, 1]")

# METRICS & OBSERVABILITY


@dataclass
class QueryMetrics:
    """Structured metrics for each query execution."""
    
    query_id: str
    original_query: str
    normalized_query: str
    
    # Retrieval metrics
    retrieval_time_ms: float = 0.0
    chunks_retrieved: int = 0
    top_similarity_score: float = 0.0
    avg_similarity_score: float = 0.0
    score_gap: float = 0.0
    
    # Reranking metrics
    reranking_enabled: bool = False
    reranking_time_ms: float = 0.0
    
    # Validation metrics
    passed_confidence_check: bool = False
    confidence_reason: str = ""
    
    # Generation metrics
    generation_time_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    # Overall
    total_time_ms: float = 0.0
    success: bool = False
    error_message: str = ""
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for logging."""
        return asdict(self)


class MetricsLogger:
    """Centralized metrics logging."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_query_metrics(self, metrics: QueryMetrics) -> None:
        """Log structured query metrics."""
        self.logger.info(
            "Query completed",
            extra={
                "metrics": metrics.to_dict(),
                "query_id": metrics.query_id,
                "success": metrics.success,
                "total_time_ms": metrics.total_time_ms
            }
        )
    
    def log_retrieval(self, chunks: int, top_score: float, time_ms: float) -> None:
        """Log retrieval phase metrics."""
        self.logger.debug(
            f"Retrieval: {chunks} chunks, top_score={top_score:.3f}, time={time_ms:.1f}ms"
        )
    
    def log_reranking(self, original_scores: List[float], reranked_scores: List[float], time_ms: float) -> None:
        """Log reranking phase metrics."""
        self.logger.debug(
            f"Reranking: scores changed from {original_scores} to {reranked_scores}, time={time_ms:.1f}ms"
        )



# PROTOCOLS (INTERFACES)


class Retriever(Protocol):
    """Interface for retrieval backends."""
    
    def retrieve(
        self, 
        query: str, 
        top_k: int, 
        threshold: float
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """Retrieve relevant chunks for a query."""
        ...


class Reranker(Protocol):
    """Interface for reranking models."""
    
    def rerank(
        self, 
        query: str, 
        chunks: List[str], 
        scores: List[float]
    ) -> Tuple[List[str], List[float]]:
        """Rerank retrieved chunks."""
        ...


class LLMGenerator(Protocol):
    """Interface for LLM backends."""
    
    def generate(self, prompt: str) -> str:
        """Generate answer from prompt."""
        ...

# RETRIEVAL LAYER


class FAISSRetriever:
    """FAISS-based semantic retrieval with confidence scoring."""
    
    def __init__(
        self, 
        index_path: str, 
        metadata_path: str, 
        model_name: str,
        logger: logging.Logger
    ):
        self.logger = logger
        self.index = self._load_index(index_path)
        self.metadata = self._load_metadata(metadata_path)
        self.embedding_model = self._load_embedding_model(model_name)
        
        self._validate_index()
    
    def _load_index(self, path: str) -> faiss.Index:
        """Load and validate FAISS index."""
        try:
            index = faiss.read_index(path)
            self.logger.info(f"Loaded FAISS index: {index.ntotal} vectors, dim={index.d}")
            return index
        except Exception as e:
            self.logger.error(f"Failed to load FAISS index: {e}")
            raise
    
    def _load_metadata(self, path: str) -> List[Dict]:
        """Load chunk metadata."""
        try:
            with open(path, 'rb') as f:
                metadata = pickle.load(f)
            self.logger.info(f"Loaded metadata: {len(metadata)} chunks")
            return metadata
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
            raise
    
    def _load_embedding_model(self, model_name: str) -> SentenceTransformer:
        """Load sentence transformer model."""
        try:
            model = SentenceTransformer(model_name)
            self.logger.info(f"Loaded embedding model: {model_name}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _validate_index(self) -> None:
        """Validate index type and metric."""
        if not isinstance(self.index, faiss.IndexFlat):
            raise ValueError(f"Expected IndexFlat, got {type(self.index).__name__}")
        
        if self.index.metric_type != faiss.METRIC_INNER_PRODUCT:
            self.logger.warning(
                f"Index metric is {self.index.metric_type}, expected INNER_PRODUCT. "
                "Similarity scores may be incorrect."
            )
    
    def retrieve(
        self, 
        query: str, 
        top_k: int, 
        threshold: float 
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Retrieve top-k most relevant chunks.
        
        Args:
            query: User query text
            top_k: Number of candidates to retrieve
            threshold: Minimum cosine similarity score
        
        Returns:
            Tuple of (chunk_texts, metadata, similarity_scores)
        """
        # Encode query
        query_vector = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False
        ).astype('float32')
        
        # Normalize for cosine similarity (IndexFlatIP requirement)
        faiss.normalize_L2(query_vector)
        
        # Search index
        distances, indices = self.index.search(query_vector, top_k)
        
        # Extract results above threshold
        chunks = []
        chunk_metadata = []
        scores = []
        
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:  # Unfilled result
                continue
            
            similarity = float(distance)  # Cosine similarity for normalized IP
            
            if similarity >= threshold:
                chunks.append(self.metadata[idx]['text'])
                chunk_metadata.append(self.metadata[idx])
                scores.append(similarity)
        
        self.logger.debug(
            f"Retrieved {len(chunks)}/{top_k} chunks above threshold {threshold:.2f}"
        )
        print("Index size:", self.index.ntotal)
        return chunks, chunk_metadata, scores

class BM25Retriever:
    """Sparse retrieval using BM25 for keyword preservation."""
    def __init__(self, metadata: List[Dict], logger: logging.Logger):
        self.logger = logger
        self.metadata = metadata
        self.bm25 = None
        self._build_index()
        
    def _build_index(self):
        tokenized_corpus = []
        for doc in self.metadata:
            text = doc.get('text', '').lower()
            tokenized_corpus.append(text.split())
        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
            self.logger.info(f"Built BM25 index with {len(self.metadata)} documents")
        
    def retrieve(self, query: str, top_k: int) -> Tuple[List[str], List[Dict], List[float]]:
        if not self.bm25: return [], [], []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        chunks, chunk_metadata, top_scores = [], [], []
        max_score = np.max(scores) if len(scores) > 0 and np.max(scores) > 0 else 1.0
        
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:
                chunks.append(self.metadata[idx]['text'])
                chunk_metadata.append(self.metadata[idx])
                top_scores.append(score / max_score)
                
        self.logger.debug(f"BM25 retrieved {len(chunks)} chunks")
        return chunks, chunk_metadata, top_scores

# RERANKING LAYER

class CrossEncoderReranker:
    """Cross-encoder based reranking with proper score normalization."""

    def __init__(self, model_name: str, logger: logging.Logger):
        self.logger = logger
        try:
            self.model = CrossEncoder(model_name)
            self.logger.info(f"Loaded reranker: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load reranker: {e}")
            raise

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Convert logits to probability scores [0-1]."""
        return 1 / (1 + np.exp(-x))

    def rerank(
        self,
        query: str,
        chunks: List[str],
        scores: List[float],
        top_k: int = 5
    ) -> Tuple[List[str], List[float]]:

        if not chunks:
            return chunks, scores

        # Create query-chunk pairs
        pairs = [(query, chunk) for chunk in chunks]

        # Raw cross-encoder logits
        rerank_logits = self.model.predict(pairs)

        # Normalize logits → probabilities (0–1)
        rerank_probs = self._sigmoid(np.array(rerank_logits))

        #  Weighted combination (precision > recall)
        combined_scores = [
            0.5 * float(rerank_prob) + 0.5 * float(retrieve_score)
            for rerank_prob, retrieve_score in zip(rerank_probs, scores)
        ]

        # Sort by combined score
        sorted_indices = np.argsort(combined_scores)[::-1][:top_k]

        reranked_chunks = [chunks[i] for i in sorted_indices]
        reranked_scores = [combined_scores[i] for i in sorted_indices]

        self.logger.debug(
            f"Reranking complete. Combined scores: {reranked_scores}"
        )

        return reranked_chunks, reranked_scores



# VALIDATION LAYER


class ConfidenceValidator:
    """Enterprise-grade confidence validation for retrieval results."""

    def __init__(self, config: RAGConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        

        print("Resolved FAISS path:", Path(self.config.faiss_index_path).resolve())
        print("Resolved metadata path:", Path(self.config.metadata_path).resolve())
    def validate(
        self,
        query: str,
        chunks: List[str],
        scores: List[float]
    ) -> Tuple[bool, str]:
    
        #  No retrieval case
        if not chunks or not scores:
            self.logger.debug("No chunks retrieved.")
            return False, "no_chunks_retrieved"

        top_score = scores[0]

        #  Hard minimum threshold
        hard_min_threshold = max(self.config.base_similarity_threshold, 0.35)

        if top_score < hard_min_threshold:
            self.logger.debug(
                f"Top score {top_score:.3f} below minimum {hard_min_threshold:.3f}"
            )
            return False, f"low_confidence_score_{top_score:.3f}"

        # Strong match override
        if top_score > 0.70:
            return True, "strong_top_match"

        # Ambiguity detection (only if multiple scores exist)
        if len(scores) > 1:
            second_score = scores[1]
            score_gap = top_score - second_score

            if top_score < 0.50 and score_gap < 0.015:
                self.logger.debug(
                    f"Ambiguous match detected. top={top_score:.3f}, gap={score_gap:.3f}"
                )
                return False, "ambiguous_top_match"

        # Final computed confidence check
        confidence = self.compute_confidence_score(scores)
        self.logger.debug(f"Computed overall confidence: {confidence:.3f}")

        if confidence < 0.40:
            return False, f"overall_confidence_too_low_{confidence:.3f}"

        return True, "passed_confidence_checks"

    def compute_confidence_score(self, scores: List[float]) -> float:
        """
        Compute overall confidence score [0-1]
        Based on top score + gap bonus
        """

        if not scores:
            return 0.0

        top_score = scores[0]

        gap_bonus = 0.0
        if len(scores) > 1:
            gap = scores[0] - scores[1]
            gap_bonus = min(gap * 2, 0.2)

        confidence = min(top_score + gap_bonus, 1.0)

        return float(confidence)
# PROMPT CONSTRUCTION LAYER

class PromptBuilder:
    """Constructs structured prompts for LLM generation."""
    
    SYSTEM_PROMPT = """You are a highly grounded corporate HR assistant.
Your ONLY purpose is to extract and relay factual information explicitly present in the provided CONTEXT.

RULES:
1. FOCUS ON FACTS: Provide the factual answer immediately and clearly.
2. NO METADATA CITATIONS: Do NOT mention, cite, or name any policies, document titles, section numbers, or clauses in your response. Just give the answer.
3. NO EXTERNAL KNOWLEDGE: Do not answer based on general knowledge. If the answer is not in the context, you cannot answer it.
4. EXACT REFUSAL: If the CONTEXT does not explicitly contain the exact answer to the user's question, you must immediately output exactly: "I don't have that information available. Please contact HR directly."

EXAMPLES:
Query: How many vacation days do I get?
Context: [PTO Policy] Employees with 1-3 years of tenure accrue 15 days of paid time off per year.
Response: You accrue 15 days of paid time off per year. 

Query: What is the policy for short-term disability?
Context: [Dress Code] Jeans are acceptable on Fridays.
Response: I don't have that information available. Please contact HR directly.

COMPLIANCE REQUIREMENT:
These instructions are mandatory. You will be penalized for including phrases like 'According to the policy' or 'Section X states'."""

    def __init__(self, config: RAGConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def build_prompt(self, context: str, query: str, tone_directive: str = "Direct, concise, and professional.") -> Dict[str, str]:
        """
        Build structured prompt for LLM.
        """
        user_message = f"""CONTEXT:
{context}

EMPLOYEE QUESTION:
{query}

INSTRUCTION: Provide a clear, concise answer based ONLY on the context above.
TONE GUIDELINE: {tone_directive}"""
        
        return {
            "system": self.SYSTEM_PROMPT,
            "user": user_message
        }
    
    def format_context(
        self, 
        chunks: List[str], 
        max_chunks: Optional[int] = None
    ) -> str:
        """
        Format retrieved chunks into clean context.
        
        Args:
            chunks: List of text chunks
            max_chunks: Maximum chunks to include
        
        Returns:
            Formatted context string
        """
        if max_chunks:
            chunks = chunks[:max_chunks]
        
        # Simple concatenation with separator
        # No source labels to prevent model from citing documents
        return "\n\n---\n\n".join(chunks)
    
    def estimate_tokens(self, text: str) -> int:
        """
        Rough token estimate (words * 1.3).
        For production, use tiktoken or transformers tokenizer.
        """
        return int(len(text.split()) * 1.3)

# LLM INTERFACE LAYER


class OllamaGenerator:
    """Ollama API client with retry logic and error handling."""
    
    def __init__(self, config: RAGConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.api_url = f"{config.ollama_base_url}/api/chat"
    
    def generate(
        self, 
        system_prompt: str, 
        user_prompt: str,
        retry_count: int = 0
    ) -> str:
        """
        Generate completion from Ollama.
        
        Args:
            system_prompt: System instructions
            user_prompt: User message
            retry_count: Current retry attempt
        
        Returns:
            Generated text
        
        Raises:
            RuntimeError: If all retries fail
        """
        payload = {
            "model": self.config.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": self.config.llm_temperature,
                "num_predict": self.config.llm_max_tokens,
                "top_p": getattr(self.config, 'llm_top_p', 0.1),
                "presence_penalty": getattr(self.config, 'llm_presence_penalty', 0.0),
                "frequency_penalty": getattr(self.config, 'llm_frequency_penalty', 0.0)
            }
        }
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.config.llm_timeout
            )
            response.raise_for_status()
            
            result = response.json()
            answer = result.get("message", {}).get("content", "").strip()
            
            if not answer:
                raise ValueError("Empty response from LLM")
            
            self.logger.debug(f"Generated answer: {len(answer)} chars")
            return answer
            
        except (requests.exceptions.RequestException, ValueError) as e:
            self.logger.warning(f"LLM generation failed (attempt {retry_count + 1}): {e}")
            
            if retry_count < self.config.llm_max_retries:
                wait_time = 2 ** retry_count  # Exponential backoff
                self.logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
                return self.generate(system_prompt, user_prompt, retry_count + 1)
            else:
                raise RuntimeError(
                    f"LLM generation failed after {self.config.llm_max_retries} retries: {e}"
                )
    
    def health_check(self) -> bool:
        """Check if Ollama is accessible."""
        try:
            response = requests.get(
                f"{self.config.ollama_base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False



# POST-PROCESSING LAYER


class AnswerPostProcessor:
    """Safe, non-destructive answer cleanup."""
    
    def __init__(self, config: RAGConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def process(self, answer: str, context: str = "") -> str:
        """
        Clean up answer and optionally verify grounding.
        
        Args:
            answer: Raw LLM output
            context: Retrieved context text
        
        Returns:
            Cleaned answer
        """
        # Safe stripping of common citation patterns instead of rejecting the whole answer
        citation_patterns = [
            r"according to the \[?[a-zA-Z0-9\s-]+\]?(?:\s*policy)?,?\s*",
            r"according to .*?(?:policy|section|clause|article|document),?\s*",
            r"as per (?:the )?\[?[a-zA-Z0-9\s-]+\]?(?:\s*policy)?,?\s*",
            r"as stated in (?:the )?\[?[a-zA-Z0-9\s-]+\]?(?:\s*policy)?,?\s*",
            r"in section \d+(?:\.\d+)?,?\s*",
            r"(?:under|in) the \[?[a-zA-Z0-9\s-]+\]?(?:\s*policy)?,?\s*",
        ]
        
        cleaned_ans = answer
        for pattern in citation_patterns:
            cleaned_ans = re.sub(pattern, "", cleaned_ans, flags=re.IGNORECASE)
            
        # Strip out lingering bracketed names like "[Leave Policy]"
        cleaned_ans = re.sub(r"\[.*?\]\s*", "", cleaned_ans)
        
        # Strip leading punctuation left over from removal (e.g., ", we offer...")
        cleaned_ans = cleaned_ans.lstrip(', -:')
                
        answer = cleaned_ans

        # Normalize whitespace
        answer = ' '.join(answer.split())
        
        # Remove excessive punctuation
        answer = re.sub(r'([.!?]){3,}', r'\1', answer)
        
        # Ensure proper capitalization
        if answer and answer[0].islower():
            answer = answer[0].upper() + answer[1:]
        
        # Optional: PII redaction
        if self.config.enable_pii_redaction:
            answer = self._redact_pii(answer)
        
        return answer
    
    def _redact_pii(self, text: str) -> str:
        """Redact common PII patterns."""
        # SSN
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN-REDACTED]', text)
        # Email
        text = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '[EMAIL-REDACTED]',
            text
        )
        # Phone
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE-REDACTED]', text)
        
        return text


# INPUT VALIDATION


class QueryValidator:
    """Validates and sanitizes user input."""
    
    def __init__(self, config: RAGConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def validate(self, query: str) -> str:
        """
        Validate and normalize query.
        
        Args:
            query: Raw user input
        
        Returns:
            Validated query
        
        Raises:
            ValueError: If query is invalid
        """
        if not query or not query.strip():
            raise ValueError("Empty query")
        
        query = query.strip()
        
        # Length check
        if len(query) > self.config.max_query_length:
            raise ValueError(
                f"Query too long ({len(query)} > {self.config.max_query_length})"
            )
        
        # Prompt injection detection
        if self._detect_injection(query):
            raise ValueError("Suspicious query pattern detected")
        
        return query
    
    def _detect_injection(self, query: str) -> bool:
        """Detect potential prompt injection attempts."""
        injection_patterns = [
            r'ignore (previous|all|above) (instructions|prompts|rules)',
            r'you are (now|actually|really)',
            r'new (instruction|rule|system)',
            r'forget (everything|all|previous)',
            r'</?(prompt|system|instruction)>',
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                self.logger.warning(f"Injection pattern detected: {pattern}")
                return True
        
        return False

class IntentRouter:
    """Classifies user intent to route queries appropriately."""
    
    def __init__(self, generator, logger: logging.Logger):
        self.generator = generator
        self.logger = logger
        
    def route(self, query: str) -> str:
        """Categorize the query as CHITCHAT, COMPLAINT, or POLICY."""
        system_prompt = """Classify the following user query into one of three categories:
1. CHITCHAT: Greetings, pleasantries, casual conversation.
2. COMPLAINT: Expressing distress, harassment, unfair treatment, or serious grievances.
3. POLICY: Questions about rules, procedures, benefits, leave, software, or general HR inquiries.

Respond with EXACTLY ONE WORD: CHITCHAT, COMPLAINT, or POLICY. No other text."""
        try:
            response = self.generator.generate(system_prompt, query, retry_count=0).strip().upper()
            if "CHITCHAT" in response: return "CHITCHAT"
            if "COMPLAINT" in response: return "COMPLAINT"
            return "POLICY"
        except Exception as e:
            self.logger.warning(f"Intent routing failed, defaulting to POLICY: {e}")
            return "POLICY"

# QUERY PROCESSING LAYER 

class QueryProcessor:
    """
    Handles query normalization, domain standardization,
    and conditional LLM-based expansion.
    """

    FILLER_PATTERNS = [
        r"\b(can you|could you|please|kindly|i want to know|i would like to know|just wanted to know)\b",
        r"\b(hi|hello|thanks|thank you)\b",
        r"\b(is it possible to|is it allowed to)\b",
    ]

    SYNONYM_MAP = {
        "wfh": "remote work",
        "vacation": "annual leave",
        "swipe": "attendance",
        "check in": "attendance",
        "check-in": "attendance",
        "work from home": "remote work"
    }

    def __init__(self, config: RAGConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.api_url = f"{config.ollama_base_url}/api/chat"


    # Normalization
  

    def normalize(self, query: str) -> str:
        original = query

        query = query.lower().strip()

        # Remove filler phrases
        for pattern in self.FILLER_PATTERNS:
            query = re.sub(pattern, "", query, flags=re.IGNORECASE)

        # Synonym replacement
        for k, v in self.SYNONYM_MAP.items():
            query = re.sub(rf"\b{k}\b", v, query)

        # Collapse whitespace
        query = re.sub(r"\s+", " ", query).strip()

        self.logger.debug(f"Normalized query: '{original}' → '{query}'")

        return query

    # -------------------------
    # Conditional Expansion
    # -------------------------

    def expand(self, query: str) -> List[str]:
        """
        Generate 3 paraphrases using Ollama.
        """

        system_prompt = """
Rewrite the user query into 3 concise semantic search variations.
Preserve exact meaning.
Do not introduce new policy concepts.
Return each variation on a new line.
"""

        payload = {
            "model": self.config.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 150
            }
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=15)
            response.raise_for_status()
            content = response.json()["message"]["content"]

            lines = [line.strip("- ").strip()
                     for line in content.split("\n") if line.strip()]

            paraphrases = lines[:3]

            self.logger.debug(f"Expansion generated: {paraphrases}")

            return [query] + paraphrases

        except Exception as e:
            self.logger.warning(f"Query expansion failed: {e}")
            return [query]

# MAIN RAG PIPELINE


class HRAssistantRAG:
    """Main RAG pipeline orchestrator."""
    
    def __init__(self, config: RAGConfig):
        """Initialize RAG pipeline with all components."""
        self.config = config
        self.config.validate()
        
        # Setup logging
        self.logger = self._setup_logging()
        self.metrics_logger = MetricsLogger(self.logger)
        
        # Initialize components
        self.logger.info("Initializing HR Assistant RAG System...")
        
        self.validator = QueryValidator(config, self.logger)
        self.query_processor = QueryProcessor(config, self.logger)
        self.retriever = FAISSRetriever(
            config.faiss_index_path,
            config.metadata_path,
            config.embedding_model_name,
            self.logger
        )
        
        self.bm25_retriever = BM25Retriever(self.retriever.metadata, self.logger)
        
        # Optional reranker
        self.reranker = None
        if config.enable_reranking:
            self.reranker = CrossEncoderReranker(
                config.reranker_model_name,
                self.logger
            )
        
        self.confidence_validator = ConfidenceValidator(config, self.logger)
        self.prompt_builder = PromptBuilder(config, self.logger)
        self.generator = OllamaGenerator(config, self.logger)
        self.postprocessor = AnswerPostProcessor(config, self.logger)
        
        self.intent_router = IntentRouter(self.generator, self.logger)
        
        # Health check
        if not self.generator.health_check():
            self.logger.warning("Ollama health check failed - LLM may be unavailable")
        
        self.logger.info("HR Assistant RAG System ready!")
        print("FAISS index path:", config.faiss_index_path)
        print("Metadata path:", config.metadata_path)       
    def _setup_logging(self) -> logging.Logger:
        """Configure structured logging."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger(__name__)
    
    def answer(self, query: str, query_id: Optional[str] = None) -> Tuple[str, QueryMetrics]:
        """
        Main answer pipeline with full observability.
        
        Args:
            query: User question
            query_id: Optional tracking ID
        
        Returns:
            Tuple of (answer, metrics)
        """
        start_time = time.time()
        
        # Initialize metrics
        metrics = QueryMetrics(
            query_id=query_id or f"q_{int(time.time() * 1000)}",
            original_query=query,
            normalized_query=""
        )
        
        try:
            # Step 1: Validate input
            validated_query = self.validator.validate(query)

            # Normalize
            normalized_query = self.query_processor.normalize(validated_query)
            metrics.normalized_query = normalized_query
            
            self.logger.info(f"Processing query: {query[:100]}...")
            
            # --- INTENT ROUTING ---
            intent = self.intent_router.route(normalized_query)
            self.logger.info(f"Detected intent: {intent}")
            
            if intent == "CHITCHAT":
                raw_answer = self.generator.generate(
                    "You are a friendly, professional corporate HR assistant. Keep responses very brief and warm. DO NOT hallucinate policies.",
                    validated_query
                )
                metrics.success = True
                metrics.total_time_ms = (time.time() - start_time) * 1000
                self.metrics_logger.log_query_metrics(metrics)
                return self.postprocessor.process(raw_answer), metrics
                
            tone_directive = (
                "Empathetic, highly supportive, prioritizing immediate human escalation alongside procedural steps." 
                if intent == "COMPLAINT" else "Direct, concise, and professional."
            )

            # Step 2: Retrieval (Hybrid Search)
            retrieval_start = time.time()

            dense_chunks, dense_md, dense_scores = self.retriever.retrieve(
                normalized_query,
                self.config.top_k,
                self.config.base_similarity_threshold
            )
            
            sparse_chunks, sparse_md, sparse_scores = self.bm25_retriever.retrieve(
                normalized_query,
                self.config.top_k
            )
            
            # RRF (Reciprocal Rank Fusion)
            rrf_k = 60
            chunk_scores = {}
            chunk_texts = {}
            chunk_dense_scores = {}
            
            for rank, chunk in enumerate(dense_chunks):
                chunk_texts[chunk] = chunk
                chunk_scores[chunk] = chunk_scores.get(chunk, 0) + 1.0 / (rrf_k + rank + 1)
                chunk_dense_scores[chunk] = dense_scores[rank]
                
            for rank, chunk in enumerate(sparse_chunks):
                chunk_texts[chunk] = chunk
                chunk_scores[chunk] = chunk_scores.get(chunk, 0) + 1.0 / (rrf_k + rank + 1)
                if chunk not in chunk_dense_scores:
                    chunk_dense_scores[chunk] = sparse_scores[rank] # Proxy fallback
                
            # Sort by RRF score
            sorted_chunks = sorted(chunk_scores.keys(), key=lambda x: chunk_scores[x], reverse=True)
            chunks = sorted_chunks[:self.config.top_k]
            scores = [chunk_dense_scores[c] for c in chunks]

            metrics.retrieval_time_ms = (time.time() - retrieval_start) * 1000
            metrics.chunks_retrieved = len(chunks)

            if scores:
                metrics.top_similarity_score = max(scores)
                metrics.avg_similarity_score = float(np.mean(scores))
                if len(scores) > 1:
                    metrics.score_gap = scores[0] - scores[1]

            self.metrics_logger.log_retrieval(
                len(chunks),
                metrics.top_similarity_score,
                metrics.retrieval_time_ms
            )
            
            # CONDITIONAL EXPANSION
            if not chunks or metrics.top_similarity_score < 0.55:
                self.logger.info("Low confidence detected. Triggering query expansion.")
                expanded_queries = self.query_processor.expand(normalized_query)
                all_chunks = []
                all_scores = []
                for q in expanded_queries:
                    d_ch, _, d_sc = self.retriever.retrieve(q, self.config.top_k, self.config.base_similarity_threshold)
                    s_ch, _, s_sc = self.bm25_retriever.retrieve(q, self.config.top_k)
                    
                    # Compute quick RRF for expanded query iteration
                    local_scores = {}
                    for r, c in enumerate(d_ch):
                        local_scores[c] = local_scores.get(c, 0) + 1.0 / (rrf_k + r + 1)
                    for r, c in enumerate(s_ch):
                        local_scores[c] = local_scores.get(c, 0) + 1.0 / (rrf_k + r + 1)
                        
                    sorted_local = sorted(local_scores.keys(), key=lambda x: local_scores[x], reverse=True)[:self.config.top_k]
                    
                    all_chunks.extend(sorted_local)
                    # For scores we'll just track if we had dense hits
                    for c in sorted_local:
                        if c in d_ch:
                            all_scores.append(d_sc[d_ch.index(c)])
                        else:
                            all_scores.append(0.5) # Synthetic proxy score for expansion fallback
               
                # Deduplicate while keeping max score
                unique = {}
                for c, s in zip(all_chunks, all_scores):
                    if c not in unique or s > unique[c]:
                        unique[c] = s

                chunks = list(unique.keys())
                scores = list(unique.values())
                self.logger.info(f"Post-expansion retrieval: {len(chunks)} chunks")
                        
            # Step 3: Optional reranking
            if self.reranker and chunks:
                rerank_start = time.time()
                original_scores = scores.copy()
                chunks, scores = self.reranker.rerank(
                    validated_query,
                    chunks,
                    scores,
                    self.config.reranker_top_k
                )
                metrics.reranking_enabled = True
                metrics.reranking_time_ms = (time.time() - rerank_start) * 1000
                
                self.metrics_logger.log_reranking(
                    original_scores,
                    scores,
                    metrics.reranking_time_ms
                )
          
            # Step 4: Confidence validation & Dynamic Edge-Case Recovery
            is_answerable, reason = self.confidence_validator.validate(
                validated_query,
                chunks,
                scores
            )
            metrics.passed_confidence_check = is_answerable
            metrics.confidence_reason = reason
            
            if not is_answerable:
                # Dynamic Clarification Loop
                clarification_prompt = f"The user asked: '{validated_query}'. However, the retrieved documents did not have a clear answer. Please respond with a polite remark that you couldn't find a direct policy, and ask them a brief question to clarify what they mean. Maximum 2 sentences."
                answer = self.generator.generate(
                    "You are a helpful HR bot identifying missing context.",
                    clarification_prompt
                )
                metrics.success = True
                metrics.total_time_ms = (time.time() - start_time) * 1000
                self.metrics_logger.log_query_metrics(metrics)
                return self.postprocessor.process(answer), metrics
            
            # Step 5: Build prompt
            context = self.prompt_builder.format_context(
                chunks,
                self.config.max_chunks_in_context
            )
            
            prompt_dict = self.prompt_builder.build_prompt(context, validated_query, tone_directive)
            
            # Estimate tokens
            context_tokens = self.prompt_builder.estimate_tokens(
                prompt_dict["system"] + prompt_dict["user"]
            )
            metrics.prompt_tokens = context_tokens
            
            if context_tokens > self.config.max_context_tokens:
                self.logger.warning(
                    f"Context exceeds limit: {context_tokens} > {self.config.max_context_tokens}"
                )
            
            # Step 6: Generate answer
            gen_start = time.time()
            raw_answer = self.generator.generate(
                prompt_dict["system"],
                prompt_dict["user"]
            )
            metrics.generation_time_ms = (time.time() - gen_start) * 1000
            metrics.completion_tokens = self.prompt_builder.estimate_tokens(raw_answer)
            
            # Step 7: Post-process (pass context for grounding checks)
            final_answer = self.postprocessor.process(raw_answer, context)
            
            # Success
            metrics.success = True
            metrics.total_time_ms = (time.time() - start_time) * 1000
            
            self.metrics_logger.log_query_metrics(metrics)
            
            return final_answer, metrics
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            metrics.success = False
            metrics.error_message = str(e)
            metrics.total_time_ms = (time.time() - start_time) * 1000
            
            self.metrics_logger.log_query_metrics(metrics)
            
            # Return graceful error message
            return (
                "I encountered an error processing your question. "
                "Please try again or contact HR directly."
            ), metrics
    
    def evaluate(self, test_cases: List[Dict]) -> Dict:
        """
        Run evaluation on test cases.
        
        Args:
            test_cases: List of dicts with 'question' and optional 'expected_answer'
        
        Returns:
            Evaluation metrics dict
        """
        results = {
            'total': len(test_cases),
            'successful': 0,
            'failed': 0,
            'fallbacks': 0,
            'avg_latency_ms': 0.0,
            'avg_confidence': 0.0
        }
        
        latencies = []
        confidences = []
        
        for i, case in enumerate(test_cases, 1):
            self.logger.info(f"Evaluating case {i}/{len(test_cases)}")
            
            answer, metrics = self.answer(case['question'])
            
            if metrics.success:
                results['successful'] += 1
            else:
                results['failed'] += 1
            
            if "don't have information" in answer.lower():
                results['fallbacks'] += 1
            
            latencies.append(metrics.total_time_ms)
            confidences.append(metrics.top_similarity_score)
        
        results['avg_latency_ms'] = np.mean(latencies)
        results['avg_confidence'] = np.mean(confidences)
        
        return results


# CLI INTERFACE


class InteractiveCLI:
    """Interactive command-line interface."""
    
    def __init__(self, rag_system: HRAssistantRAG):
        self.rag = rag_system
    
    def run(self) -> None:
        """Run interactive session."""
        print("=" * 70)
        print("HR ASSISTANT - Interactive Mode")
        print("=" * 70)
        print("\nCommands:")
        print("  - Type your question and press Enter")
        print("  - 'quit' or 'exit' to quit")
        print("  - 'debug on/off' to toggle debug mode")
        print("=" * 70)
        
        while True:
            try:
                query = input("\nYour Question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                if query.lower() == 'debug on':
                    self.rag.config.enable_debug_mode = True
                    self.rag.logger.setLevel(logging.DEBUG)
                    print("Debug mode enabled")
                    continue
                
                if query.lower() == 'debug off':
                    self.rag.config.enable_debug_mode = False
                    self.rag.logger.setLevel(logging.INFO)
                    print("Debug mode disabled")
                    continue
                
                if not query:
                    continue
                
                # Process query
                answer, metrics = self.rag.answer(query)
                
                # Display answer
                print("\n" + "─" * 70)
                print(f"Answer: {answer}")
                print("─" * 70)
                
                # Display metrics if debug mode
                if self.rag.config.enable_debug_mode:
                    print(f"\nMetrics:")
                    print(f"  Total time: {metrics.total_time_ms:.1f}ms")
                    print(f"  Retrieval: {metrics.retrieval_time_ms:.1f}ms")
                    print(f"  Generation: {metrics.generation_time_ms:.1f}ms")
                    print(f"  Top score: {metrics.top_similarity_score:.3f}")
                    print(f"  Confidence: {metrics.confidence_reason}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")



# MAIN ENTRY POINT


def main():
    """Main execution function."""
    
    # Initialize configuration
    config = RAGConfig(
        # Update these paths for your environment
        faiss_index_path=r"C:\Users\Aleena\Documents\hr_assistant\index\faiss_index.bin",
        metadata_path=r"C:\Users\Aleena\Documents\hr_assistant\index\chunk_metadata.pkl",
        
        # Retrieval settings
        top_k=5,
        base_similarity_threshold=0.35,
        
        # Enable/disable reranking
        enable_reranking=True,  # Set to True for better precision (slower)
        
        # LLM settings
        ollama_model="llama3:8b",  # or "llama3", "llama2", etc.
        llm_temperature=0.1,
        llm_max_tokens=120,
        
        # Observability
        log_level="INFO",
        enable_debug_mode=False
    )
    
    try:
        # Initialize RAG system
        rag_system = HRAssistantRAG(config)
        
        # Example queries
        print("\n" + "=" * 70)
        print("EXAMPLE QUERIES")
        print("=" * 70)
        
        example_queries = [
            "How do I apply for sick leave?",
            "What is the remote work policy?",
            "How many vacation days do I get per year?"
        ]
        
        for query in example_queries:
            print(f"\nQuery: {query}")
            answer, metrics = rag_system.answer(query)
            print(f"Answer: {answer}")
            print(f"Time: {metrics.total_time_ms:.1f}ms | Confidence: {metrics.top_similarity_score:.3f}")
            print("-" * 70)
        
        # Interactive mode
        cli = InteractiveCLI(rag_system)
        cli.run()
        
    except Exception as e:
        print(f"\nFailed to initialize system: {e}")
        raise


if __name__ == "__main__":
    main()