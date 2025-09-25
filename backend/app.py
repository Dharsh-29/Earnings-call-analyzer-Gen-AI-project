#app.py
import os
import time
import pandas as pd
import streamlit as st
from transcript_processor import process_pdf
from chatbot import ask_question, get_sample_questions
from summarizer import generate_topics, generate_summary
from vectorstore import create_vectorstore, search_chunks
from config import config

# Set Streamlit page config
st.set_page_config(
    page_title="Earnings Call Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI matching demo
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E7D32;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        text-align: center;
        color: #546E7A;
        font-size: 1.3rem;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    .info-card {
        background: linear-gradient(135deg, #E8F5E8 0%, #F1F8E9 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        border: 1px solid #C8E6C9;
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.15);
    }
    .info-card h4 {
        color: #2E7D32;
        margin-bottom: 0.5rem;
        font-size: 1.2rem;
        font-weight: 600;
    }
    .topic-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    .topic-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .topic-card h4 {
        color: #2E7D32;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .chunk-display {
        background: linear-gradient(135deg, #F3F4F6 0%, #FFFFFF 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #2196F3;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
    }
    .chunk-id {
        font-weight: 700;
        color: #1976D2;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .speaker-name {
        font-weight: 600;
        color: #424242;
        font-size: 1.05rem;
    }
    .sample-question-btn {
        margin: 0.4rem;
        border-radius: 25px !important;
        font-weight: 500;
    }
    .success-banner {
        background: linear-gradient(135deg, #D4EDDA 0%, #C3E6CB 100%);
        border: 1px solid #28A745;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        color: #155724;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.2);
    }
    .sidebar-section {
        background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        border: 1px solid #DEE2E6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .nav-button {
        width: 100%;
        margin: 0.3rem 0;
        padding: 0.8rem;
        border-radius: 10px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .nav-button:hover {
        transform: translateX(5px);
    }
    .metrics-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
        flex-wrap: wrap;
    }
    .metric-card {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        min-width: 180px;
        margin: 0.5rem;
        border: 1px solid #90CAF9;
        box-shadow: 0 4px 12px rgba(33, 150, 243, 0.2);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1565C0;
    }
    .metric-label {
        color: #1976D2;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    .chat-container {
        background: #FFFFFF;
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid #E0E0E0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 1rem 0;
    }
    .question-user {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        padding: 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196F3;
    }
    .answer-bot {
        background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C9 100%);
        padding: 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }
    .processing-spinner {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 2rem;
        color: #2E7D32;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# File paths - use config
DEMO_PDF_PATH = config.DEMO_PDF_PATH

# Session state initialization
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "processed_data": None,
        "vectorstore_ready": False,
        "chat_history": [],
        "opening_topics": [],
        "qa_topics": [],
        "selected_opening_topics": [],
        "selected_qa_topics": [],
        "opening_summaries": {},
        "qa_summaries": {},
        "current_question": "",
        "processing_stage": "ready",  # ready, processing, complete
        "current_page": "Welcome & Upload"
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Helper functions
def load_and_process_pdf():
    """Load and process the demo PDF"""
    if not os.path.exists(DEMO_PDF_PATH):
        st.error(f"❌ Demo file not found: {DEMO_PDF_PATH}")
        return False
    
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔄 Reading PDF file...")
        progress_bar.progress(20)
        
        # Process the PDF
        status_text.text("🔄 Processing transcript content...")
        progress_bar.progress(40)
        processed_data = process_pdf(DEMO_PDF_PATH)
        st.session_state.processed_data = processed_data
        
        # Create vectorstore
        status_text.text("🔄 Creating vector database...")
        progress_bar.progress(70)
        all_chunks = []
        all_chunks.extend(processed_data.get("opening_chunks", []))
        all_chunks.extend(processed_data.get("qa_chunks", []))
        
        if all_chunks:
            create_vectorstore(all_chunks)
            st.session_state.vectorstore_ready = True
        
        status_text.text("✅ Processing complete!")
        progress_bar.progress(100)
        
        # Clear progress indicators
        import time
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        
        return True
            
    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")
        return False

def display_sidebar():
    """Enhanced sidebar with navigation and status"""
    st.sidebar.markdown("# 📊 Navigation")
    
    pages = [
        ("🏠", "Welcome & Upload"),
        ("🎤", "Opening Remarks"),
        ("❓", "Q&A Session"),
        ("🤖", "AI Assistant")
    ]
    
    # Create navigation buttons
    for icon, page in pages:
        if st.sidebar.button(f"{icon} {page}", key=f"nav_{page}", use_container_width=True):
            st.session_state.current_page = page
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Processing Status Section
    st.sidebar.markdown("## 📈 Processing Status")
    
    if st.session_state.processed_data:
        metadata = st.session_state.processed_data.get("concall_metadata", {})
        
        # Company info
        st.sidebar.markdown(f"""
        <div class="sidebar-section">
            <strong>🏢 Company:</strong><br>
            {metadata.get('company_name', 'Unknown')}<br>
            <strong>📅 Date:</strong><br>
            {metadata.get('date', 'Unknown')}
        </div>
        """, unsafe_allow_html=True)
        
        # Processing metrics
        opening_count = len(st.session_state.processed_data.get("opening_chunks", []))
        qa_count = len(st.session_state.processed_data.get("qa_chunks", []))
        
        st.sidebar.markdown(f"""
        <div class="sidebar-section">
            <strong>📄 Content Processed:</strong><br>
            Opening: {opening_count} chunks<br>
            Q&A: {qa_count} chunks<br>
            Vector DB: {'✅ Ready' if st.session_state.vectorstore_ready else '❌ Not ready'}
        </div>
        """, unsafe_allow_html=True)
        
        # Topic generation status
        st.sidebar.markdown(f"""
        <div class="sidebar-section">
            <strong>🎯 Analysis Status:</strong><br>
            Opening Topics: {'✅' if st.session_state.opening_topics else '⭕'} {len(st.session_state.opening_topics)}<br>
            Q&A Topics: {'✅' if st.session_state.qa_topics else '⭕'} {len(st.session_state.qa_topics)}<br>
            Summaries: {'✅' if st.session_state.opening_summaries or st.session_state.qa_summaries else '⭕'}
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.sidebar.markdown(f"""
        <div class="sidebar-section">
            ⚠️ No transcript loaded<br>
            <small>Upload a transcript to begin analysis</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick actions
    st.sidebar.markdown("## ⚡ Quick Actions")
    if st.sidebar.button("🔄 Reset All Data", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key != "current_page":
                del st.session_state[key]
        init_session_state()
        st.rerun()
    
    return st.session_state.current_page

# Main content functions
def show_welcome_page():
    """Enhanced welcome page"""
    st.markdown('<h1 class="main-header">📊 Earnings Call Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered transcript analysis with topic extraction, summarization, and intelligent Q&A</p>', unsafe_allow_html=True)
    
    # Demo button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 **Try Demo with Laurus Labs Q2 FY24**", type="primary", use_container_width=True):
            with st.spinner("Processing earnings call transcript..."):
                if load_and_process_pdf():
                    st.balloons()
                    st.success("🎉 Transcript processed successfully!")
                    time.sleep(2)
                    st.rerun()
    
    # Show results if available
    if st.session_state.processed_data:
        st.markdown('<div class="success-banner">✅ Transcript processed successfully! Navigate using the sidebar to explore insights.</div>', unsafe_allow_html=True)
        
        metadata = st.session_state.processed_data.get("concall_metadata", {})
        
        # Metrics display
        st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">🏢</div>
                <div class="metric-label">Company</div>
                <div style="font-weight: 600; margin-top: 0.5rem;">{metadata.get('company_name', 'Unknown')}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">📅</div>
                <div class="metric-label">Call Date</div>
                <div style="font-weight: 600; margin-top: 0.5rem;">{metadata.get('date', 'Unknown')}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            opening_count = len(st.session_state.processed_data.get("opening_chunks", []))
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{opening_count}</div>
                <div class="metric-label">Opening Remarks</div>
                <div style="font-weight: 600; margin-top: 0.5rem;">Chunks</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            qa_count = len(st.session_state.processed_data.get("qa_chunks", []))
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{qa_count}</div>
                <div class="metric-label">Q&A Session</div>
                <div style="font-weight: 600; margin-top: 0.5rem;">Chunks</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Next steps
        st.markdown("### 🎯 Next Steps")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎤 Analyze Opening Remarks", use_container_width=True, type="secondary"):
                st.session_state.current_page = "Opening Remarks"
                st.rerun()
        
        with col2:
            if st.button("❓ Analyze Q&A Session", use_container_width=True, type="secondary"):
                st.session_state.current_page = "Q&A Session"
                st.rerun()

def show_opening_remarks_page():
    """Enhanced opening remarks analysis page"""
    st.header("🎤 Opening Remarks Analysis")
    
    if not st.session_state.processed_data:
        st.warning("⚠️ No transcript loaded. Please go to Welcome & Upload and try the demo first.")
        return
    
    opening_chunks = st.session_state.processed_data.get("opening_chunks", [])
    
    if not opening_chunks:
        st.warning("⚠️ No opening remarks found in the transcript.")
        return
    
    # Content overview
    st.markdown(f"**📊 Content Overview:** {len(opening_chunks)} chunks processed from management presentations")
    
    # View chunks section with improved UI
    st.subheader("📄 Raw Content")
    with st.expander("View Opening Remarks Chunks", expanded=False):
        for i, chunk in enumerate(opening_chunks):
            chunk_id = chunk.get("id", f"Chunk {i+1}")
            speaker = chunk.get("speaker", "Unknown Speaker")
            message = chunk.get("message", "")
            
            st.markdown(f"""
            <div class="chunk-display">
                <span class="chunk-id">{chunk_id}</span> |
                <span class="speaker-name">👤 {speaker}</span><br><br>
                {message}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Topic generation section
    st.subheader("🎯 AI Topic Extraction")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("Generate AI-powered topics from the opening remarks to identify key business themes and strategic insights.")
    
    with col2:
        if st.button("🚀 Generate Topics", type="primary", use_container_width=True):
            with st.spinner("🤖 Analyzing content and extracting key topics..."):
                try:
                    topics = generate_topics(opening_chunks, max_topics=5, regenerate=False)
                    st.session_state.opening_topics = topics
                    st.success("✅ Topics generated successfully!")
                except Exception as e:
                    st.error(f"❌ Error generating topics: {str(e)}")
    
    with col3:
        if st.session_state.opening_topics:
            if st.button("🔄 New Topics", use_container_width=True):
                with st.spinner("🤖 Generating fresh topic perspectives..."):
                    try:
                        topics = generate_topics(opening_chunks, max_topics=5, regenerate=True)
                        st.session_state.opening_topics = topics
                        # Clear existing summaries since topics changed
                        st.session_state.opening_summaries = {}
                        st.success("✅ New topics generated!")
                    except Exception as e:
                        st.error(f"❌ Error generating new topics: {str(e)}")
    
    # Display topics and enable selection
    if st.session_state.opening_topics:
        st.markdown("---")
        st.subheader("📝 Select Topics for Detailed Analysis")
        
        selected_topics = []
        for i, topic_data in enumerate(st.session_state.opening_topics):
            topic_name = topic_data.get("topic", f"Topic {i+1}")
            topic_desc = topic_data.get("description", "")
            
            col1, col2 = st.columns([1, 10])
            
            with col1:
                is_selected = st.checkbox("", key=f"opening_topic_select_{i}")
            
            with col2:
                if is_selected:
                    selected_topics.append(topic_name)
                st.markdown(f"**🎯 {topic_name}**")
                if topic_desc:
                    st.caption(f"💡 {topic_desc}")
        
        # Generate summaries
        if selected_topics:
            st.markdown(f"**Selected for analysis:** {len(selected_topics)} topics")
            
            col1, col2 = st.columns([2, 1])
            
            with col2:
                if st.button("📊 Generate Summaries", type="secondary", use_container_width=True):
                    progress_bar = st.progress(0)
                    with st.spinner("🤖 Creating detailed summaries..."):
                        try:
                            for i, topic in enumerate(selected_topics):
                                progress_bar.progress((i + 1) / len(selected_topics))
                                summary = generate_summary(opening_chunks, topic)
                                st.session_state.opening_summaries[topic] = summary
                            st.success("✅ Summaries generated successfully!")
                        except Exception as e:
                            st.error(f"❌ Error generating summaries: {str(e)}")
    
    # Display summaries
    if st.session_state.opening_summaries:
        st.markdown("---")
        st.subheader("📋 Generated Analysis")
        
        for topic, summary in st.session_state.opening_summaries.items():
            st.markdown(f"""
            <div class="topic-card">
                <h4>🎯 {topic}</h4>
                <p style="line-height: 1.6; color: #424242;">{summary}</p>
            </div>
            """, unsafe_allow_html=True)

def show_qa_session_page():
    """Enhanced Q&A session analysis page"""
    st.header("❓ Q&A Session Analysis")
    
    if not st.session_state.processed_data:
        st.warning("⚠️ No transcript loaded. Please go to Welcome & Upload and try the demo first.")
        return
    
    qa_chunks = st.session_state.processed_data.get("qa_chunks", [])
    
    if not qa_chunks:
        st.warning("⚠️ No Q&A session found in the transcript.")
        return
    
    # Content overview with Q&A specific metrics
    questions = [chunk for chunk in qa_chunks if chunk.get("type") == "question"]
    answers = [chunk for chunk in qa_chunks if chunk.get("type") == "answer"]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("❓ Questions", len(questions))
    with col2:
        st.metric("💬 Answers", len(answers))
    with col3:
        st.metric("📊 Total Exchanges", len(qa_chunks))
    
    # View chunks section
    st.subheader("📄 Q&A Content")
    with st.expander("View Q&A Session Chunks", expanded=False):
        for chunk in qa_chunks:
            chunk_id = chunk.get("id", "Unknown")
            speaker = chunk.get("speaker", "Unknown Speaker")
            message = chunk.get("message", "")
            chunk_type = chunk.get("type", "unknown")
            
            type_emoji = "❓" if chunk_type == "question" else "💬"
            type_color = "#1976D2" if chunk_type == "question" else "#388E3C"
            
            st.markdown(f"""
            <div class="chunk-display" style="border-left-color: {type_color};">
                <span class="chunk-id">{chunk_id}</span> |
                {type_emoji} <span class="speaker-name">{speaker}</span><br><br>
                {message}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Topic generation
    st.subheader("🎯 AI Topic Extraction")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("Extract key themes from analyst questions and management responses to understand market concerns and strategic directions.")
    
    with col2:
        if st.button("🚀 Generate Topics", type="primary", use_container_width=True, key="qa_gen_topics"):
            with st.spinner("🤖 Analyzing Q&A content and extracting key topics..."):
                try:
                    topics = generate_topics(qa_chunks, max_topics=5, regenerate=False)
                    st.session_state.qa_topics = topics
                    st.success("✅ Topics generated successfully!")
                except Exception as e:
                    st.error(f"❌ Error generating topics: {str(e)}")
    
    with col3:
        if st.session_state.qa_topics:
            if st.button("🔄 New Topics", use_container_width=True, key="qa_new_topics"):
                with st.spinner("🤖 Generating fresh topic perspectives..."):
                    try:
                        topics = generate_topics(qa_chunks, max_topics=5, regenerate=True)
                        st.session_state.qa_topics = topics
                        # Clear existing summaries since topics changed
                        st.session_state.qa_summaries = {}
                        st.success("✅ New topics generated!")
                    except Exception as e:
                        st.error(f"❌ Error generating new topics: {str(e)}")
    
    # Topic selection and summary generation
    if st.session_state.qa_topics:
        st.markdown("---")
        st.subheader("📝 Select Topics for Detailed Analysis")
        
        selected_topics = []
        for i, topic_data in enumerate(st.session_state.qa_topics):
            topic_name = topic_data.get("topic", f"Topic {i+1}")
            topic_desc = topic_data.get("description", "")
            
            col1, col2 = st.columns([1, 10])
            
            with col1:
                is_selected = st.checkbox("", key=f"qa_topic_select_{i}")
            
            with col2:
                if is_selected:
                    selected_topics.append(topic_name)
                st.markdown(f"**🎯 {topic_name}**")
                if topic_desc:
                    st.caption(f"💡 {topic_desc}")
        
        if selected_topics:
            st.markdown(f"**Selected for analysis:** {len(selected_topics)} topics")
            
            col1, col2 = st.columns([2, 1])
            
            with col2:
                if st.button("📊 Generate Summaries", type="secondary", use_container_width=True):
                    progress_bar = st.progress(0)
                    with st.spinner("🤖 Creating detailed summaries..."):
                        try:
                            for i, topic in enumerate(selected_topics):
                                progress_bar.progress((i + 1) / len(selected_topics))
                                summary = generate_summary(qa_chunks, topic)
                                st.session_state.qa_summaries[topic] = summary
                            st.success("✅ Summaries generated successfully!")
                        except Exception as e:
                            st.error(f"❌ Error generating summaries: {str(e)}")
    
    # Display summaries
    if st.session_state.qa_summaries:
        st.markdown("---")
        st.subheader("📋 Generated Analysis")
        
        for topic, summary in st.session_state.qa_summaries.items():
            st.markdown(f"""
            <div class="topic-card">
                <h4>🎯 {topic}</h4>
                <p style="line-height: 1.6; color: #424242;">{summary}</p>
            </div>
            """, unsafe_allow_html=True)

def show_ai_assistant_page():
    """Enhanced AI assistant page with chat interface"""
    st.header("🤖 AI Assistant - Ask Questions")
    
    if not st.session_state.processed_data or not st.session_state.vectorstore_ready:
        st.warning("⚠️ No transcript loaded or vector database not ready. Please process the transcript first.")
        return
    
    # Sample questions section
    st.subheader("💡 Sample Questions")
    st.markdown("*Click any question below to try it out, or type your own question in the chat box.*")
    
    sample_questions = get_sample_questions()
    
    # Display sample questions in a grid
    cols = st.columns(2)
    for i, question in enumerate(sample_questions):
        col = cols[i % 2]
        with col:
            if st.button(f"💭 {question}", key=f"sample_q_{i}", use_container_width=True):
                st.session_state.current_question = question
                st.rerun()
    
    st.markdown("---")
    
    # Chat interface
    st.subheader("💬 Chat with the Transcript")
    
    # Question input
    question = st.text_area(
        "**Your Question:**",
        value=st.session_state.current_question,
        placeholder="e.g., What was the revenue growth this quarter? What challenges did management discuss?",
        height=100,
        key="question_input"
    )
    
    # Action buttons
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        ask_button = st.button("🚀 Ask Question", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.current_question = ""
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.current_question = ""
            st.rerun()
    
    # Process question
    if ask_button and question.strip():
        with st.spinner("🔍 Searching transcript and generating answer..."):
            try:
                # Get all chunks for search
                all_chunks = []
                all_chunks.extend(st.session_state.processed_data.get("opening_chunks", []))
                all_chunks.extend(st.session_state.processed_data.get("qa_chunks", []))
                
                # Ask question
                answer, chunks_used = ask_question(question, all_chunks)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer,
                    "chunks": chunks_used,
                    "timestamp": pd.Timestamp.now().strftime("%H:%M:%S")
                })
                
                # Clear current question
                st.session_state.current_question = ""
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error generating answer: {str(e)}")
    
    elif ask_button and not question.strip():
        st.warning("⚠️ Please enter a question.")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("💬 Conversation History")
        
        # Display conversations in reverse order (most recent first)
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            question = chat["question"]
            answer = chat["answer"]
            chunks_used = chat.get("chunks", [])
            timestamp = chat.get("timestamp", "")
            
            # Create chat container
            with st.container():
                # User question
                st.markdown(f"""
                <div class="question-user">
                    <strong>🙋 You asked:</strong> <span style="color: #1565C0;">{question}</span>
                    <div style="font-size: 0.8em; color: #666; margin-top: 0.5rem;">⏰ {timestamp}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # AI answer
                st.markdown(f"""
                <div class="answer-bot">
                    <strong>🤖 AI Assistant:</strong><br><br>
                    <div style="line-height: 1.6;">{answer}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Source information
                if chunks_used:
                    with st.expander(f"📚 Source Information ({len(chunks_used)} sources used)", expanded=False):
                        st.markdown("**Most relevant sources that informed this answer:**")
                        
                        for j, (chunk_text, score) in enumerate(chunks_used[:5]):
                            relevance_color = "#4CAF50" if score > 0.8 else "#FF9800" if score > 0.6 else "#F44336"
                            relevance_text = "High" if score > 0.8 else "Medium" if score > 0.6 else "Low"
                            
                            st.markdown(f"""
                            <div class="chunk-display">
                                <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 0.5rem;">
                                    <strong>📄 Source {j+1}</strong>
                                    <span style="background: {relevance_color}; color: white; padding: 0.2rem 0.5rem; border-radius: 10px; font-size: 0.8rem;">
                                        {relevance_text} ({score:.2f})
                                    </span>
                                </div>
                                <div style="color: #424242; line-height: 1.5;">
                                    {chunk_text[:400]}{"..." if len(chunk_text) > 400 else ""}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                
                st.markdown("---")
    
    # Tips for better questions
    if not st.session_state.chat_history:
        st.markdown("### 💡 Tips for Better Questions")
        
        tips_col1, tips_col2 = st.columns(2)
        
        with tips_col1:
            st.markdown("""
            **🎯 Specific Questions Work Best:**
            - What was the revenue for Q2?
            - What challenges were discussed?
            - What is the company's outlook?
            - Who are the key management personnel?
            """)
        
        with tips_col2:
            st.markdown("""
            **📊 Financial & Strategic Focus:**
            - Ask about financial metrics and KPIs
            - Inquire about strategic initiatives
            - Question market conditions and outlook
            - Explore operational updates
            """)

# Main application logic
def main():
    """Main application function"""
    # Display sidebar and get current page
    current_page = display_sidebar()
    
    # Route to appropriate page
    if current_page == "Welcome & Upload":
        show_welcome_page()
    elif current_page == "Opening Remarks":
        show_opening_remarks_page()
    elif current_page == "Q&A Session":
        show_qa_session_page()
    elif current_page == "AI Assistant":
        show_ai_assistant_page()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 1rem;">
            <small>🔬 Earnings Call Analyzer | Powered by AI & Vector Search |
            <a href="https://github.com" style="color: #2E7D32;">View Source Code</a></small>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()