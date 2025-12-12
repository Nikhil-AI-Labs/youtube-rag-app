"""
Query Engine Module
Handles RAG chain creation and query processing with multilingual support
"""

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Try to import streamlit for secrets management
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

load_dotenv()

def get_api_token():
    """Get API token from Streamlit secrets or environment variable"""
    if HAS_STREAMLIT:
        try:
            return st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
        except:
            pass
    return os.getenv("HUGGINGFACEHUB_API_TOKEN")

class QueryEngine:
    """Handles query processing using RAG with multilingual support"""
    
    def __init__(self, vector_store, k: int = 4):
        """
        Initialize the query engine
        
        Args:
            vector_store: FAISS vector store
            k: Number of chunks to retrieve
        """
        self.vector_store = vector_store
        self.k = k
        self.model = None
        self.rag_chain = None
        self.retriever = None
        
        self._initialize_llm()
        self._build_rag_chain()
    
    def _initialize_llm(self):
        """Initialize the language model"""
        try:
            llm = HuggingFaceEndpoint(
                repo_id="meta-llama/Llama-3.1-8B-Instruct",
                task="text-generation",
                max_new_tokens=512,
                temperature=0.2,
                top_k=50,
                repetition_penalty=1.03,
                huggingfacehub_api_token=get_api_token()
            )
            
            self.model = ChatHuggingFace(llm=llm)
            print("✓ LLM initialized (supports multilingual responses)")
            
        except Exception as e:
            print(f"✗ Error initializing LLM: {e}")
            raise
    
    def _build_rag_chain(self):
        """Build the RAG chain with English-only responses"""
        try:
            # Create retriever
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.k}
            )
            
            # Create prompt template - ENGLISH ONLY
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful AI assistant that answers questions based on YouTube video transcripts.

    IMPORTANT INSTRUCTIONS:
    - Answer ONLY using information from the provided context
    - The context may be in Hindi, English, or any other language
    - **ALWAYS respond in ENGLISH, regardless of the question language or context language**
    - If the context is in Hindi or another language, translate the information to English in your response
    - If the context doesn't contain enough information, say: "I don't have enough information in the transcript to answer that"
    - Be concise, clear, and specific
    - Use direct quotes when relevant (translate them to English if needed)
    - Maintain a conversational but informative tone
    - Never respond in Hindi or any language other than English"""),
                ("user", """Context from video transcript (may be in any language):
    {context}

    Question: {question}

    Remember: Your response MUST be in ENGLISH only.

    Answer:""")
            ])
            
            # Helper function to format documents
            def format_docs(retrieved_docs):
                return "\n\n".join(doc.page_content for doc in retrieved_docs)
            
            # Build parallel chain
            parallel_chain = RunnableParallel({
                'context': self.retriever | RunnableLambda(format_docs),
                'question': RunnablePassthrough()
            })
            
            # Complete RAG chain
            self.rag_chain = parallel_chain | prompt | self.model | StrOutputParser()
            
            print("✓ RAG chain built (English-only responses)")
            
        except Exception as e:
            print(f"✗ Error building RAG chain: {e}")
            raise

    
    def query(self, question: str) -> dict:
        """
        Query the video content (supports Hindi and English questions)
        
        Args:
            question: User's question in Hindi or English
            
        Returns:
            Dictionary with answer and sources
        """
        try:
            # Get answer
            answer = self.rag_chain.invoke(question)
            
            # Get source documents
            source_docs = self.retriever.invoke(question)
            sources = [doc.page_content for doc in source_docs]
            
            return {
                'success': True,
                'question': question,
                'answer': answer,
                'sources': sources,
                'num_sources': len(sources),
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'question': question,
                'answer': None,
                'sources': [],
                'num_sources': 0,
                'error': f"Error processing query: {str(e)}"
            }
    
    def similarity_search(self, query: str, k: int = 5) -> list:
        """
        Search for similar content with scores
        
        Args:
            query: Search query (Hindi or English)
            k: Number of results
            
        Returns:
            List of (document, score) tuples
        """
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
    def update_k(self, new_k: int):
        """
        Update the number of chunks to retrieve
        
        Args:
            new_k: New k value
        """
        self.k = new_k
        self.retriever.search_kwargs["k"] = new_k
    
    def batch_query(self, questions: list) -> list:
        """
        Process multiple questions (Hindi or English)
        
        Args:
            questions: List of questions
            
        Returns:
            List of result dictionaries
        """
        results = []
        for question in questions:
            result = self.query(question)
            results.append(result)
        return results


# Testing function
if __name__ == "__main__":
    from vector_store import VectorStoreManager
    
    # Create vector store
    manager = VectorStoreManager()
    result = manager.process_video("https://www.youtube.com/watch?v=Gfr50f6ZBvo")
    
    if result['success']:
        # Create query engine
        engine = QueryEngine(result['vector_store'])
        
        # Test query
        response = engine.query("What is the main topic of this video?")
        
        if response['success']:
            print(f"\nQuestion: {response['question']}")
            print(f"Answer: {response['answer']}")
            print(f"Sources used: {response['num_sources']}")
        else:
            print(f"Error: {response['error']}")
