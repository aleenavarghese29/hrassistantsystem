"""
This module handles:
1. Loading chunked HR policy documents
2. Generating embeddings using Sentence Transformers
3. Building and saving FAISS index for efficient similarity search
4. Retrieving relevant chunks for RAG-based query answering
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class HRPolicyRAG:
    """
    RAG system for HR policy document retrieval using FAISS and Sentence Transformers.

    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the RAG system with a specified embedding model.
        
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        
        self.index = None
        self.metadata = []
        
    def load_chunks(self, json_path: str) -> List[Dict[str, Any]]:
        """
        Load chunked HR policy documents from JSON file.
        
        """
        print(f"\nLoading chunks from: {json_path}")
        json_path = Path(json_path)
        
        if not json_path.exists():
            raise FileNotFoundError(f"Chunk file not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        print(f"Loaded {len(chunks)} chunks successfully")
        return chunks
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]], 
                          batch_size: int = 32) -> np.ndarray:
        """
        Generate contextual embeddings for all chunks using policy and section combined.
        
        Args:
            chunks: List of chunk dictionaries
            batch_size: Number of texts to process simultaneously
            
        Returns:
            Numpy array of embeddings (shape: [num_chunks, embedding_dim])
        """
        print(f"\nGenerating embeddings for {len(chunks)} chunks...")
        
        # Extract and contextualize text content from chunks
        texts = [
            f"[{chunk.get('policy_name', '')} - {chunk.get('section_title', '')}]\n{chunk.get('text', '')}"
            for chunk in chunks
        ]
        
        # Generate embeddings in batches with progress bar
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray, 
                         chunks: List[Dict[str, Any]]) -> None:
        """
        Build FAISS index optimized for CPU-based similarity search.
        
        Args:
            embeddings: Numpy array of embeddings
            chunks: Original chunk dictionaries for metadata storage
        """
        print("\nBuilding FAISS index...")
        
        # Convert embeddings to float32 (FAISS requirement)
        embeddings = embeddings.astype('float32')
        
        # Create FAISS index
        # Using IndexFlatIP for Inner Product (equivalent to cosine similarity with normalized vectors)
        # This is optimal for CPU and provides exact search results
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Store metadata separately (preserving all fields including contextualized text)
        self.metadata = []
        for chunk in chunks:
            contextual_text = f"[{chunk.get('policy_name', '')} - {chunk.get('section_title', '')}]\n{chunk.get('text', '')}"
            metadata_entry = {
                'policy_name': chunk.get('policy_name', ''),
                'section_id': chunk.get('section_id', ''),
                'section_title': chunk.get('section_title', ''),
                'chunk_id': chunk.get('chunk_id', ''),
                'global_id': chunk.get('global_id', ''),
                'text': contextual_text  # Include contextualized text for retrieval
            }
            self.metadata.append(metadata_entry)
        
        print(f"FAISS index built with {self.index.ntotal} vectors")
        print(f"Metadata stored for {len(self.metadata)} chunks")
    
    def save_index(self, index_path: str = "faiss_index.bin", 
                   metadata_path: str = "chunk_metadata.pkl") -> None:
        """
        Save FAISS index and metadata to disk.
        
        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata (pickle format)
        """
        print("\nSaving index and metadata...")
        
        # Save FAISS index
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        print(f"FAISS index saved to: {index_path}")
        
        # Save metadata
        metadata_path = Path(metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        print(f"Metadata saved to: {metadata_path}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Index Summary:")
        print(f"  Total vectors: {self.index.ntotal}")
        print(f"  Embedding dimension: {self.embedding_dim}")
        print(f"  Metadata entries: {len(self.metadata)}")
        print(f"{'='*60}")
    
    def load_index(self, index_path: str = "faiss_index.bin", 
                   metadata_path: str = "chunk_metadata.pkl") -> None:
        """
        Load FAISS index and metadata from disk.
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata pickle file
        """
        print("\nLoading existing index and metadata...")
        
        # Load FAISS index
        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        self.index = faiss.read_index(str(index_path))
        print(f"FAISS index loaded from: {index_path}")
        
        # Load metadata
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        print(f"Metadata loaded from: {metadata_path}")
        
        print(f"Index contains {self.index.ntotal} vectors")
        print(f"Metadata contains {len(self.metadata)} entries")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant chunks for a given query.
        
        Args:
            query: User query string
            top_k: Number of most relevant chunks to retrieve
            
        Returns:
            List of dictionaries containing chunk text, metadata, and similarity scores
        """
        if self.index is None or not self.metadata:
            raise ValueError("Index not loaded. Call load_index() or build_faiss_index() first.")
        
        # Generate query embedding
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype('float32')
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results with metadata
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):  # Ensure valid index
                result = {
                    'score': float(score),
                    'text': self.metadata[idx]['text'],
                    'policy_name': self.metadata[idx]['policy_name'],
                    'section_id': self.metadata[idx]['section_id'],
                    'section_title': self.metadata[idx]['section_title'],
                    'chunk_id': self.metadata[idx]['chunk_id'],
                    'global_id': self.metadata[idx]['global_id']
                }
                results.append(result)
        
        return results
    
    def format_retrieval_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into a context string for RAG prompt injection.
        
        Args:
            results: List of retrieval results from retrieve() method
            
        Returns:
            Formatted context string ready for LLM prompt
        """
        context_parts = []
        
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Document {i}]\n"
                f"Policy: {result['policy_name']}\n"
                f"Section: {result['section_title']}\n"
                f"Content: {result['text']}\n"
                f"Relevance Score: {result['score']:.4f}\n"
            )
        
        return "\n".join(context_parts)


def build_index_pipeline(chunk_json_path: str, 
                        output_dir: str = "./hr_rag_index",
                        model_name: str = "all-MiniLM-L6-v2") -> HRPolicyRAG:
    """
    Complete pipeline to build and save RAG index from scratch.
    
    Args:
        chunk_json_path: Path to the chunked JSON file
        output_dir: Directory to save index and metadata
        model_name: Sentence Transformer model name
        
    Returns:
        Initialized HRPolicyRAG instance with built index
    """
    print("\n" + "="*60)
    print("HR POLICY RAG INDEX BUILD PIPELINE")
    print("="*60)
    
    # Initialize RAG system
    rag = HRPolicyRAG(model_name=model_name)
    
    # Load chunks
    chunks = rag.load_chunks(chunk_json_path)
    
    # Generate embeddings
    embeddings = rag.generate_embeddings(chunks)
    
    # Build FAISS index
    rag.build_faiss_index(embeddings, chunks)
    
    # Save to disk
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    index_path = output_dir / "faiss_index.bin"
    metadata_path = output_dir / "chunk_metadata.pkl"
    
    rag.save_index(str(index_path), str(metadata_path))

    print("\nPipeline completed successfully!")
    return rag


def demo_retrieval(rag: HRPolicyRAG, query: str, top_k: int = 3) -> None:
    """
    Demonstrate retrieval functionality with a sample query.
    
    Args:
        rag: Initialized HRPolicyRAG instance
        query: Sample query to test
        top_k: Number of results to retrieve
    """
    print("\n" + "="*60)
    print(f"DEMO RETRIEVAL")
    print("="*60)
    print(f"Query: {query}")
    print(f"Retrieving top {top_k} results...\n")
    
    results = rag.retrieve(query, top_k=top_k)
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} (Score: {result['score']:.4f}) ---")
        print(f"Policy: {result['policy_name']}")
        print(f"Section: {result['section_title']}")
        print(f"Text Preview: {result['text'][:200]}...")
        print(f"Chunk ID: {result['chunk_id']}")
    
    print("\n" + "="*60)
    print("Context for RAG Prompt:")
    print("="*60)
    context = rag.format_retrieval_context(results)
    print(context)


if __name__ == "__main__":
    """
    Main execution block - Build index and demonstrate retrieval
    """

    BASE_DIR = Path(__file__).resolve().parent

    CHUNK_JSON_PATH = BASE_DIR / "chunks" / "policy_chunks.json"
    OUTPUT_DIR = BASE_DIR / "index"
    MODEL_NAME = BASE_DIR / "models" / "all-MiniLM-L6-v2"
        
   
    try:
        # Build the index
        rag = build_index_pipeline(
            chunk_json_path=CHUNK_JSON_PATH,
            output_dir=OUTPUT_DIR,
            model_name=MODEL_NAME
        )
        
        # Demo retrieval with sample queries
        demo_retrieval(rag, "What is the annual leave policy?", top_k=3)
        demo_retrieval(rag, "How do I request sick leave?", top_k=3)
        
        print("\nAll operations completed successfully!")
        print(f"\nTo use the index in production:")
        print(f"  1. Initialize: rag = HRPolicyRAG()")
        print(f"  2. Load index: rag.load_index('{OUTPUT_DIR}/faiss_index.bin', '{OUTPUT_DIR}/chunk_metadata.pkl')")
        print(f"  3. Query: results = rag.retrieve('your query', top_k=5)")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure the chunk JSON file exists at the specified path.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()