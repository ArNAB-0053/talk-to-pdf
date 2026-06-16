import logging
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from vectordb.chroma_store import ChromaStore
from embeddings.embedding_manager import EmbeddingManager
import config

logger = logging.getLogger(__name__)

class RAGRetriever:
    """Handles query-based retrieval from the vector store"""

    def __init__(self, vector_store: ChromaStore, embedding_manager: EmbeddingManager):
        """
        Initialize the retriever.

        Args:
            vector_store: ChromaStore instance.
            embedding_manager: EmbeddingManager instance.
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = None, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The search query.
            top_k: Number of top results to return (optional, defaults to config.RETRIEVAL_TOP_K).
            score_threshold: Minimum similarity score threshold.

        Returns:
            List of dictionaries containing retrieved documents and metadata.
        """
        if top_k is None:
            top_k = config.RETRIEVAL_TOP_K

        # logger.info(f"Retrieving documents for query: '{query}'")
        # logger.info(f"Top K: {top_k}, Score threshold: {score_threshold}")

        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        try:
            # Query the underlying collection
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            retrieved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1 - distance

                    # logger.info(
                    #     f"Rank={i+1} "
                    #     f"Distance={distance:.4f} "
                    #     f"Similarity={similarity_score:.4f}"
                    # )

                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })

                # logger.info(f"Retrieved {len(retrieved_docs)} documents (after filtering).")
            # else:
                # logger.info("No documents found in store.")

            return retrieved_docs

        except Exception as e:
            # logger.error(f"Error during retrieval: {e}")
            return []


class RAGPipeline:
    """Orchestrates document retrieval and LLM context generation"""

    def __init__(
        self,
        retriever: RAGRetriever,
        model_name: str = None,
        api_key: str = None,
        base_url: str = None
    ):
        """
        Initialize the generation pipeline.
        
        Args:
            retriever: RAGRetriever instance.
            model_name: Name of model to use.
            api_key: API Key for LLM service.
            base_url: Base URL for LLM provider.
        """
        self.retriever = retriever
        self.model_name = model_name or config.OPEN_ROUTER_MODEL_NAME
        self.api_key = api_key or config.OPEN_ROUTER_API_KEY
        self.base_url = base_url or config.OPEN_ROUTER_BASE_URL
        self._llm = None
        self._chain = None

    def _initialize_llm(self):
        """Dynamically instantiates the LLM client and template chain to avoid import-time side effects."""
        if self._llm is None:
            # logger.info(f"Initializing ChatOpenAI client with model: {self.model_name}...")
            self._llm = ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            prompt_template = ChatPromptTemplate.from_template("""
                Use the following question to answer the question concisely.

                Context:
                {context}

                Question: {query}

                Answer:
                """)
            self._chain = prompt_template | self._llm | StrOutputParser()

    def run(self, query: str, top_k: int = None, min_score: float = 0.2, return_context: bool = False) -> str:
        """
        Execute RAG generation process.

        Args:
            query: The question query.
            top_k: Number of documents to retrieve.
        """
        if top_k is None:
            top_k = config.RETRIEVAL_TOP_K

        # 1. Retrieve context
        results = self.retriever.retrieve(query=query, top_k=top_k, score_threshold=min_score)

        if not results:
            return {'answer': 'No relevant context found.', 'sources': [], 'confidence': 0.0, 'context': ''}

        context = "\n\n".join([doc['content'] for doc in results]) if results else ""

        sources = [{
            'source_file': doc['metadata'].get('source_file', 'unknown'),
            'source_path': doc['metadata'].get('source_path', 'unknown'),
            'page_number': doc['metadata'].get('page_number', 'unknown'),
            'score': doc['similarity_score'],
            'preview': doc['content'][:300] + '...'
        } for doc in results]
        confidence = max([doc['similarity_score'] for doc in results])

        if not context:
            # logger.warning(f"No context found for query: '{query}'")
            return "No context found"

        # 2. Lazy-initialize LLM chain
        self._initialize_llm()

        # 3. Generate response
        # logger.info(f"Invoking LLM for query: '{query}'...")
        response = self._chain.invoke({"context": context, "query": query})

        output = {
            'answer': response,
            'sources': sources,
            'confidence': confidence
        }
        
        if return_context:
            output['context'] = context

        return output


def RAG_output(q: str, rag_retriever: RAGRetriever, llm=None, top_k: int = None, min_score: float = 0.2, return_context: bool = False) -> str:
    """
    Backward-compatible helper function for RAG execution.
    
    Args:
        q: The search query.
        rag_retriever: RAGRetriever instance.
        llm: Ignored (kept for backward-compatible interface signature).
        top_k: Top K retrieved results (defaults to config.RETRIEVAL_TOP_K).
    """
    pipeline = RAGPipeline(retriever=rag_retriever)
    return pipeline.run(query=q, top_k=top_k, min_score=min_score, return_context=return_context)