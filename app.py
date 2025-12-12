"""
Main Streamlit Application
YouTube Video Q&A Interface
"""

import streamlit as st
from vector_store import VectorStoreManager
from query_engine import QueryEngine
import time
from datetime import datetime
import json

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="YouTube Video Q&A",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - FIXED VERSION
# ============================================================================

st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #FF0000;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    
    /* Fix chat message visibility - IMPORTANT */
    .stChatMessage {
        background-color: transparent !important;
        padding: 10px;
        margin: 5px 0;
    }
    
    /* User messages */
    [data-testid="stChatMessageContent"] {
        background-color: rgba(240, 242, 246, 0.8);
        border-radius: 10px;
        padding: 15px;
        color: #1e1e1e;
    }
    
    /* Make chat text visible */
    .stMarkdown p {
        color: #1e1e1e !important;
    }
    
    /* Info boxes */
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        color: #155724;
        margin: 1rem 0;
    }
    
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        color: #0c5460;
        margin: 1rem 0;
    }
    
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        color: #856404;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        background-color: rgba(248, 249, 250, 0.5);
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 5px;
        font-weight: 500;
    }
    
    /* Remove white background from main area */
    .main .block-container {
        background-color: transparent;
        padding-top: 2rem;
    }
    
    /* Chat input styling */
    .stChatInputContainer {
        border-top: 1px solid rgba(250, 250, 250, 0.2);
        padding-top: 1rem;
    }
    
    /* Expander styling for sources */
    .streamlit-expanderHeader {
        background-color: rgba(240, 242, 246, 0.5);
        border-radius: 5px;
    }
    
    /* Text area for sources */
    .stTextArea textarea {
        background-color: rgba(255, 255, 255, 0.05);
        color: #e0e0e0;
        border: 1px solid rgba(250, 250, 250, 0.2);
    }
    
    /* FOOTER STYLING - NEW */
    .footer-container {
        text-align: center;
        padding: 1.5rem;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .footer-container p {
        color: rgb(255, 255, 255) !important;
        margin: 0.5rem 0;
    }
    
    .footer-container span {
        color: rgb(255, 255, 255) !important;
    }
    
    .footer-container strong {
        color: rgb(255, 255, 255) !important;
    }
    
    .footer-container b {
        color: rgb(255, 255, 255) !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "vector_store_manager" not in st.session_state:
        st.session_state.vector_store_manager = None
    
    if "query_engine" not in st.session_state:
        st.session_state.query_engine = None
    
    if "current_video_id" not in st.session_state:
        st.session_state.current_video_id = None
    
    if "video_metadata" not in st.session_state:
        st.session_state.video_metadata = {}
    
    if "processing" not in st.session_state:
        st.session_state.processing = False

initialize_session_state()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def export_chat_history():
    """Export chat history as downloadable formats"""
    if not st.session_state.messages:
        return None, None
    
    # JSON format
    chat_data = {
        "video_id": st.session_state.current_video_id,
        "export_date": datetime.now().isoformat(),
        "messages": st.session_state.messages
    }
    json_data = json.dumps(chat_data, indent=2)
    
    # Text format
    text_lines = [
        f"YouTube Video Q&A - Chat History",
        f"Video ID: {st.session_state.current_video_id}",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "="*80,
        ""
    ]
    
    for msg in st.session_state.messages:
        role = "You" if msg["role"] == "user" else "Assistant"
        text_lines.append(f"{role}: {msg['content']}")
        text_lines.append("-"*80)
        text_lines.append("")
    
    text_data = "".join(text_lines)
    
    return json_data, text_data

def reset_app():
    """Reset the entire application state"""
    st.session_state.messages = []
    st.session_state.vector_store_manager = None
    st.session_state.query_engine = None
    st.session_state.current_video_id = None
    st.session_state.video_metadata = {}
    st.session_state.processing = False

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<div class="main-header">🎬 YouTube Video Q&A Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by AI • Ask anything about any YouTube video!</div>', unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Video Input Section
    with st.container():
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("📹 Video Input")
        
        video_url = st.text_input(
            "Enter YouTube URL:",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste any YouTube video URL (supports watch, youtu.be, shorts)",
            key="video_url_input"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            process_button = st.button("🚀 Process", type="primary", use_container_width=True)
        with col2:
            example_button = st.button("💡 Example", use_container_width=True)
        
        if example_button:
            st.code("https://www.youtube.com/watch?v=Gfr50f6ZBvo")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Settings Section
    with st.container():
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("🎛️ Settings")
        
        k_value = st.slider(
            "Context chunks",
            min_value=1,
            max_value=10,
            value=4,
            help="Number of relevant chunks to retrieve"
        )
        
        show_sources = st.checkbox("Show source chunks", value=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Video Info Section
    # Video Info Section
    if st.session_state.current_video_id:
        st.subheader("📊 Current Video")
        
        metadata = st.session_state.video_metadata
        
        st.success(f"**Video ID:** `{st.session_state.current_video_id}`")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chunks", metadata.get('num_chunks', 0))
        with col2:
            st.metric("Messages", len(st.session_state.messages))
        
        # Enhanced language display (NEW CODE STARTS HERE)
        language = metadata.get('language', 'N/A')
        if 'Hindi' in language or 'hi' in language.lower():
            st.info(f"🇮🇳 **Language:** {language}")
        else:
            st.info(f"🌐 **Language:** {language}")
        # (NEW CODE ENDS HERE)
        
        if metadata.get('is_generated'):
            st.warning("⚠️ Auto-generated captions")

    
    st.markdown("---")
    
    # Chat Controls Section
    st.subheader("💬 Chat Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset All", use_container_width=True):
            reset_app()
            st.rerun()
    
    # Download chat history
    if st.session_state.messages and st.session_state.current_video_id:
        st.markdown("---")
        st.subheader("💾 Export Chat")
        
        json_data, text_data = export_chat_history()
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📄 TXT",
                data=text_data,
                file_name=f"chat_{st.session_state.current_video_id}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            st.download_button(
                label="📋 JSON",
                data=json_data,
                file_name=f"chat_{st.session_state.current_video_id}.json",
                mime="application/json",
                use_container_width=True
            )
    
    st.markdown("---")
    
    # Quick Questions Section
    st.subheader("⚡ Quick Questions")
    
    quick_questions = [
        "📝 Summarize the video",
        "🔑 What are the key points?",
        "👥 Who is mentioned?",
        "📚 What topics are discussed?"
    ]
    
    for question in quick_questions:
        display_text = question.split(" ", 1)[1]  # Remove emoji for clean question
        if st.button(question, use_container_width=True, key=f"quick_{display_text}"):
            st.session_state.messages.append({"role": "user", "content": display_text})
            st.rerun()
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: #999; font-size: 0.8rem; padding: 1rem 0;'>
        <p>Built with ❤️ using</p>
        <p>Streamlit • LangChain • HuggingFace</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# VIDEO PROCESSING
# ============================================================================

if process_button and video_url:
    # Initialize vector store manager if needed
    if st.session_state.vector_store_manager is None:
        with st.spinner("🔧 Initializing system..."):
            try:
                st.session_state.vector_store_manager = VectorStoreManager()
                st.success("✓ System initialized!")
            except Exception as e:
                st.error(f"❌ Failed to initialize system: {e}")
                st.stop()
    
    # Extract video ID to check if it's new
    video_id = VectorStoreManager.extract_video_id(video_url)
    
    if video_id:
        # Check if it's a new video
        if video_id != st.session_state.current_video_id:
            st.session_state.processing = True
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Fetch transcript
            status_text.markdown("### 📥 Fetching transcript...")
            progress_bar.progress(25)
            time.sleep(0.5)
            
            result = st.session_state.vector_store_manager.process_video(video_url)
            
            if result['success']:
                # Step 2: Creating vector store
                status_text.markdown("### 🧠 Creating knowledge base...")
                progress_bar.progress(50)
                time.sleep(0.5)
                
                # Step 3: Initialize query engine
                status_text.markdown("### 🔗 Building query system...")
                progress_bar.progress(75)
                
                try:
                    st.session_state.query_engine = QueryEngine(
                        result['vector_store'],
                        k=k_value
                    )
                    
                    # Update session state
                    st.session_state.current_video_id = video_id
                    st.session_state.video_metadata = result['metadata']
                    st.session_state.messages = []  # Clear chat for new video
                    
                    # Complete
                    status_text.markdown("### ✅ Ready to answer questions!")
                    progress_bar.progress(100)
                    time.sleep(1)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Success message
                    st.success(f"🎉 Video processed successfully! Video ID: **{video_id}**")
                    st.balloons()
                    
                    st.session_state.processing = False
                    
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Error initializing query engine: {e}")
                    st.session_state.processing = False
            else:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ {result['error']}")
                st.session_state.processing = False
        else:
            st.info("ℹ️ This video is already loaded. Start asking questions below!")
    else:
        st.error("❌ Invalid YouTube URL. Please check and try again.")

# Update k value if changed
if st.session_state.query_engine and k_value:
    st.session_state.query_engine.update_k(k_value)

# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================

# Welcome message if no video loaded
# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================

# Welcome message if no video loaded
if not st.session_state.current_video_id:
    st.markdown("""
    <div class="info-box">
        <h3>👋 Welcome to YouTube Video Q&A!</h3>
        <p><strong>Get started in 3 easy steps:</strong></p>
        <ol>
            <li>📹 Enter a YouTube video URL in the sidebar</li>
            <li>🚀 Click "Process" to analyze the video</li>
            <li>💬 Ask any question about the video content!</li>
        </ol>
        <p><strong>Supported URL formats:</strong></p>
        <ul>
            <li>https://www.youtube.com/watch?v=VIDEO_ID</li>
            <li>https://youtu.be/VIDEO_ID</li>
            <li>https://www.youtube.com/shorts/VIDEO_ID</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Show example
    with st.expander("💡 See an example"):
        st.code("https://www.youtube.com/watch?v=Gfr50f6ZBvo", language="text")
        st.markdown("This is a sample video you can try!")

else:
    # Display chat messages from history
    for msg_idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available and enabled
            if message["role"] == "assistant" and show_sources and "sources" in message:
                with st.expander("📄 View Source Chunks"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.text_area(
                            f"Source {i}",
                            value=source,
                            height=150,
                            key=f"source_msg{msg_idx}_chunk{i}",  # FIXED: Unique key with message index
                            label_visibility="collapsed"
                        )
                        if i < len(message["sources"]):
                            st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("💭 Ask a question about the video...", key="chat_input"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    # Query the video
                    result = st.session_state.query_engine.query(prompt)
                    
                    if result['success']:
                        answer = result['answer']
                        sources = result['sources']
                        
                        # Display answer
                        st.markdown(answer)
                        
                        # Show sources if enabled
                        if show_sources and sources:
                            with st.expander("📄 View Source Chunks"):
                                for i, source in enumerate(sources, 1):
                                    st.markdown(f"**Chunk {i}:**")
                                    st.text_area(
                                        f"Source {i}",
                                        value=source,
                                        height=150,
                                        key=f"source_current_chunk{i}_{time.time()}",  # FIXED: Unique key with timestamp
                                        label_visibility="collapsed"
                                    )
                                    if i < len(sources):
                                        st.markdown("---")
                        
                        # Add to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                    else:
                        error_msg = f"❌ {result['error']}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })
                
                except Exception as e:
                    error_msg = f"❌ Unexpected error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
# ============================================================================
# FOOTER - UPDATED
# ============================================================================

# ============================================================================
# FOOTER - UPDATED
# ============================================================================

st.markdown("---")
st.markdown("""
<div class="footer-container">
    <p style='font-size: 1rem;'>
        <span>💡 <b>Tip:</b> Your chat history is preserved during this session</span>
    </p>
    <p style='font-size: 0.9rem;'>
        <span>Made with ❤️ by Nikhil</span>
    </p>
</div>
""", unsafe_allow_html=True)






