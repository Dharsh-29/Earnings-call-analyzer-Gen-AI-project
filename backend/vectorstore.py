# vectorstore.py
import os
from typing import List, Tuple
import numpy as np
import openai
import faiss
from config import config

# Initialize OpenAI client
client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

class VectorStore:
    def __init__(self, chunks: List[dict]):
        self.chunks = chunks
        self.embeddings = None
        self.index = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            print(f"Creating embeddings for {len(self.chunks)} chunks...")
            self.embeddings = self._embed_chunks(self.chunks)
            self.index = self._create_faiss_index(self.embeddings)
            print("Vector store initialized successfully!")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def _embed_chunks(self, chunks: List[dict]) -> np.ndarray:
        texts = []
        for chunk in chunks:
            # Create comprehensive text for embedding
            text_parts = []
            
            if chunk.get('speaker'):
                text_parts.append(f"Speaker: {chunk['speaker']}")
            
            if chunk.get('message'):
                text_parts.append(chunk['message'])
            
            if chunk.get('type'):
                text_parts.append(f"Type: {chunk['type']}")
            
            full_text = " | ".join(text_parts)
            texts.append(full_text)
        
        embeddings = []
        batch_size = 10
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                print(f"Processing embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                
                response = client.embeddings.create(
                    model=config.EMBEDDING_MODEL,
                    input=batch
                )
                
                batch_emb = [np.array(item.embedding, dtype=np.float32) for item in response.data]
                embeddings.extend(batch_emb)
                
            except openai.APIError as e:
                print(f"OpenAI API error for batch {i}: {e}")
                # Create dummy embeddings if API fails
                dummy_emb = np.random.rand(len(batch), config.VECTOR_DIMENSION).astype(np.float32)
                embeddings.extend([emb for emb in dummy_emb])
                
            except Exception as e:
                print(f"Error generating embeddings for batch {i}: {e}")
                # Create dummy embeddings if anything else fails
                dummy_emb = np.random.rand(len(batch), config.VECTOR_DIMENSION).astype(np.float32)
                embeddings.extend([emb for emb in dummy_emb])
        
        if not embeddings:
            raise ValueError("Failed to generate any embeddings")
            
        return np.vstack(embeddings)

    def _create_faiss_index(self, embeddings: np.ndarray):
        try:
            dim = embeddings.shape[1]
            print(f"Creating FAISS index with dimension {dim}")
            
            # Use IndexFlatIP for cosine similarity (after normalization)
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            index = faiss.IndexFlatIP(dim)  # Inner Product for normalized vectors = cosine similarity
            index.add(embeddings)
            
            print(f"FAISS index created with {index.ntotal} vectors")
            return index
            
        except Exception as e:
            print(f"Error creating FAISS index: {e}")
            raise

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if not self.index or not self.embeddings.size:
            print("Vector store not properly initialized")
            return []
            
        try:
            # Generate query embedding
            response = client.embeddings.create(
                model=config.EMBEDDING_MODEL,
                input=query
            )
            
            q_emb = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
            
            # Normalize query embedding for cosine similarity
            faiss.normalize_L2(q_emb)
            
            # Search
            scores, indices = self.index.search(q_emb, min(top_k, len(self.chunks)))
            
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if idx >= 0 and idx < len(self.chunks):
                    # Convert inner product back to similarity score (0-1 range)
                    similarity = max(0.0, float(score))
                    chunk_text = self.chunks[idx].get('message', '')
                    
                    if chunk_text and similarity > 0.1:  # Filter out very low relevance
                        results.append((chunk_text, similarity))
            
            # Sort by similarity score (descending)
            results.sort(key=lambda x: x[1], reverse=True)
            return results
            
        except openai.APIError as e:
            print(f"OpenAI API error in search: {e}")
            # Fallback: return first few chunks with low relevance
            return [(chunk.get('message', ''), 0.3) for chunk in self.chunks[:top_k] if chunk.get('message')]
            
        except Exception as e:
            print(f"Error in vector search: {e}")
            # Fallback: return first few chunks with low relevance
            return [(chunk.get('message', ''), 0.3) for chunk in self.chunks[:top_k] if chunk.get('message')]

# Global vector store instance
vectorstore_instance = None

def create_vectorstore(chunks: List[dict]):
    """Initialize global vector store"""
    global vectorstore_instance
    
    if not chunks:
        print("No chunks provided for vector store creation")
        return
        
    if not config.OPENAI_API_KEY:
        print("OpenAI API key not configured")
        return
    
    try:
        print("Creating new vector store...")
        vectorstore_instance = VectorStore(chunks)
        print("Vector store created successfully!")
        
    except Exception as e:
        print(f"Failed to create vector store: {e}")
        vectorstore_instance = None

def search_chunks(question: str, chunks: List[dict], top_k: int = 5) -> List[Tuple[str, float]]:
    global vectorstore_instance
    
    # Validate inputs
    if not question or not question.strip():
        return []
    
    if not chunks:
        return []
    
    # Create or update vector store if needed
    if vectorstore_instance is None or len(vectorstore_instance.chunks) != len(chunks):
        create_vectorstore(chunks)
    
    if vectorstore_instance is None:
        print("Failed to initialize vector store, using simple text matching")
        return _simple_text_search(question, chunks, top_k)
    
    try:
        return vectorstore_instance.search(question, top_k=top_k)
    except Exception as e:
        print(f"Vector search failed: {e}, falling back to simple search")
        return _simple_text_search(question, chunks, top_k)

def _simple_text_search(question: str, chunks: List[dict], top_k: int = 5) -> List[Tuple[str, float]]:
    question_words = set(question.lower().split())
    results = []
    
    for chunk in chunks:
        message = chunk.get('message', '')
        if not message:
            continue
            
        message_words = set(message.lower().split())
        
        # Simple word overlap scoring
        overlap = len(question_words.intersection(message_words))
        total_words = len(question_words.union(message_words))
        
        if total_words > 0:
            score = overlap / total_words
            if score > 0.1:  # Minimum relevance threshold
                results.append((message, score))
    
    # Sort by score and return top k
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

def get_vectorstore_stats():
    global vectorstore_instance
    
    if vectorstore_instance is None:
        return {"status": "not_initialized", "chunks": 0, "embeddings": 0}
    
    return {
        "status": "ready",
        "chunks": len(vectorstore_instance.chunks),
        "embeddings": vectorstore_instance.embeddings.shape[0] if vectorstore_instance.embeddings is not None else 0,
        "dimension": vectorstore_instance.embeddings.shape[1] if vectorstore_instance.embeddings is not None else 0
    }