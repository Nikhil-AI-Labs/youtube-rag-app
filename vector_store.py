"""
Vector Store Module
Handles transcript fetching and vector store creation with multi-language support
"""

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from youtube_transcript_api._errors import NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from urllib.parse import parse_qs, urlparse
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


class VectorStoreManager:
    """Manages vector store creation and transcript processing"""
    
    def __init__(self):
        """Initialize the vector store manager"""
        self.embedding_model = None
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize the multilingual embedding model"""
        try:
            self.embedding_model = HuggingFaceEndpointEmbeddings(
                model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                huggingfacehub_api_token=get_api_token()  # Changed this line
            )
            print("✓ Multilingual embedding model initialized (supports Hindi + English)")
        except Exception as e:
            print(f"✗ Error initializing embeddings: {e}")
            raise
    
    @staticmethod
    def extract_video_id(url: str) -> str:
        """
        Extract YouTube video ID from various URL formats
        
        Args:
            url: YouTube URL or video ID
            
        Returns:
            Video ID string or None
        """
        try:
            parsed = urlparse(url)
            
            # Handle youtu.be links
            if parsed.netloc.lower() == 'youtu.be':
                return parsed.path.split('/')[-1]
            
            # Handle youtube.com/shorts links
            if parsed.path.lower().startswith('/shorts'):
                return parsed.path.split('/')[-1]
            
            # Handle youtube.com/watch links
            if parsed.query:
                video_id = parse_qs(parsed.query).get('v')
                if video_id:
                    return video_id[0]
            
            # If just video ID is provided
            if len(url) == 11 and '/' not in url:
                return url
                
            return None
        except Exception as e:
            print(f"Error extracting video ID: {e}")
            return None
    
    @staticmethod
    def fetch_transcript(video_id: str) -> tuple:
        """
        Fetch transcript for a YouTube video in Hindi or English
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Tuple of (transcript_text, language, is_generated, error_message)
        """
        try:
            ytt_api = YouTubeTranscriptApi()
            
            # Try to fetch transcript in Hindi or English (priority order)
            # Supports: Hindi (hi), English (en), and auto-generated versions
            fetched_transcript = ytt_api.fetch(
                video_id, 
                languages=['hi', 'en', 'en-IN', 'hi-IN']  # Hindi and English variants
            )
            
            # Extract text from transcript
            transcript = " ".join(snippet.text for snippet in fetched_transcript)
            
            return (
                transcript,
                fetched_transcript.language,
                fetched_transcript.is_generated,
                None
            )
            
        except TranscriptsDisabled:
            return None, None, None, "Transcripts are disabled for this video"
        except NoTranscriptFound:
            # Try to get available transcripts and translate if needed
            try:
                transcript_list = ytt_api.list(video_id)
                
                # Get the first available transcript
                available_transcripts = transcript_list.transcripts
                if available_transcripts:
                    # Get first available transcript
                    first_transcript = list(available_transcripts.values())[0]
                    
                    # If it's translatable, translate to English
                    if first_transcript.is_translatable:
                        print(f"Translating from {first_transcript.language_code} to English...")
                        translated = first_transcript.translate('en')
                        fetched_transcript = translated.fetch()
                        transcript = " ".join(snippet.text for snippet in fetched_transcript)
                        
                        return (
                            transcript,
                            f"{first_transcript.language} (translated to English)",
                            first_transcript.is_generated,
                            None
                        )
                
                return None, None, None, "No Hindi or English transcript found, and translation not available"
                
            except Exception as e:
                return None, None, None, f"No transcript available: {str(e)}"
                
        except Exception as e:
            return None, None, None, f"Error: {str(e)}"
    
    def create_vector_store(
        self,
        transcript: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> tuple:
        """
        Create FAISS vector store from transcript
        
        Args:
            transcript: Video transcript text (Hindi or English)
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            Tuple of (vector_store, num_chunks, error_message)
        """
        try:
            # Split transcript into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            chunks = splitter.create_documents([transcript])
            
            # Create vector store with multilingual embeddings
            vector_store = FAISS.from_documents(chunks, self.embedding_model)
            
            return vector_store, len(chunks), None
            
        except Exception as e:
            return None, 0, f"Error creating vector store: {str(e)}"
    
    def process_video(self, video_url: str) -> dict:
        """
        Complete video processing pipeline for Hindi/English videos
        
        Args:
            video_url: YouTube video URL or ID
            
        Returns:
            Dictionary with processing results
        """
        result = {
            'success': False,
            'video_id': None,
            'vector_store': None,
            'metadata': {},
            'error': None
        }
        
        # Extract video ID
        video_id = self.extract_video_id(video_url)
        if not video_id:
            result['error'] = "Invalid YouTube URL"
            return result
        
        result['video_id'] = video_id
        
        # Fetch transcript (Hindi or English)
        transcript, language, is_generated, error = self.fetch_transcript(video_id)
        if error:
            result['error'] = error
            return result
        
        # Create vector store
        vector_store, num_chunks, error = self.create_vector_store(transcript)
        if error:
            result['error'] = error
            return result
        
        # Success
        result['success'] = True
        result['vector_store'] = vector_store
        result['metadata'] = {
            'video_id': video_id,
            'language': language,
            'is_generated': is_generated,
            'num_chunks': num_chunks,
            'transcript_length': len(transcript)
        }
        
        return result


# Testing function
if __name__ == "__main__":
    manager = VectorStoreManager()
    
    # Test with English video
    # test_url_en = "https://www.youtube.com/watch?v=Gfr50f6ZBvo"
    # print("\n" + "="*80)
    # print("=== Testing English Video ===")
    # print("="*80)
    # result_en = manager.process_video(test_url_en)
    
    # if result_en['success']:
    #     print(f"✓ English video processed successfully!")
    #     print(f"  Video ID: {result_en['metadata']['video_id']}")
    #     print(f"  Language: {result_en['metadata']['language']}")
    #     print(f"  Chunks: {result_en['metadata']['num_chunks']}")
    #     print(f"  Transcript Length: {result_en['metadata']['transcript_length']} characters")
    # else:
    #     print(f"✗ Error: {result_en['error']}")
    
    # Test with Hindi video
    test_url_hi = "https://www.youtube.com/watch?v=X0btK9X0Xnk"
    print("\n" + "="*80)
    print("=== Testing Hindi Video ===")
    print("="*80)
    result_hi = manager.process_video(test_url_hi)
    
    if result_hi['success']:
        print(f"✓ Hindi video processed successfully!")
        print(f"  Video ID: {result_hi['metadata']['video_id']}")
        print(f"  Language: {result_hi['metadata']['language']}")
        print(f"  Chunks: {result_hi['metadata']['num_chunks']}")
        print(f"  Transcript Length: {result_hi['metadata']['transcript_length']} characters")
    else:
        print(f"✗ Error: {result_hi['error']}")
    
    print("\n" + "="*80)
    print("Testing completed!")
    print("="*80)
