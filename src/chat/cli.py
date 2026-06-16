import json
import logging
from rag.pipeline import RAG_output

logger = logging.getLogger(__name__)

def chat_loop(rag_retriever):
    """
    Starts an interactive CLI chat loop. Keep prompting the user for questions
    until they enter 'exit' or 'quit'.
    """
    print("\n" + "=" * 15 + " PDF RAG CHAT LOOP STARTED " + "=" * 15)
    print("Type your question below. Type 'exit' or 'quit' to end the session.")
    print("=" * 59 + "\n")
    
    while True:
        try:
            print("Ask a question:")
            query = input("> ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                print("Exiting chat. Goodbye!")
                break
                
            logger.info(f"Processing query: '{query}'")
            # Query the RAG pipeline using RAG_output
            answer = RAG_output(query, rag_retriever, min_score=0.1, return_context=True)
            
            print("\n" + "=" * 20 + " RAG ANSWER " + "=" * 20)
            print(json.dumps(answer, indent=4))
            print("=" * 52 + "\n")
            
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Goodbye!")
            break
