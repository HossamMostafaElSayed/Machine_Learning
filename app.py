import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
import yt_dlp
import json
import tempfile
import base64
import streamlit.components.v1 as components

# --- UTILITY IMPORTS ---
from utilities import (
    get_video_id,
    extract_transcript_details,
    generate_gemini_content,
    generate_structured_quiz,
    display_quiz
)
from whisper_utils import transcribe_with_whisper
from slide_utils import extract_slides_from_url

# Load API key
load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))


# ---------------- Streamlit App ----------------

# Custom CSS for better styling
st.set_page_config(page_title="YouTube Assistant Pro", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .stButton>button {
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .feature-icon {
        font-size: 2rem;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🎬 YouTube Assistant Pro</h1><p>⚡ AI-Powered Video Analysis & Transcription</p></div>', unsafe_allow_html=True)

# Input method selector with better styling
st.markdown("### 📥 Choose Your Source")
input_method = st.radio(
    "",
    ["🔗 YouTube URL", "📁 Upload Local Video/Audio"],
    horizontal=True,
    label_visibility="collapsed"
)

youtube_link = None
local_video_path = None
video_id = None
video_title = "Video"

if input_method == "🔗 YouTube URL":
    # Input YouTube link
    youtube_link = st.text_input(
        "🔗 Enter YouTube Video Link",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Make sure the video has English subtitles"
    )
    if youtube_link:
        video_id = get_video_id(youtube_link)
else:
    # Upload local video or audio
    uploaded_file = st.file_uploader(
        "📁 Upload Video or Audio File",
        type=["mp4", "avi", "mov", "mkv", "webm", "flv", "mp3", "wav", "m4a", "aac", "ogg", "flac"],
        help="Upload a video or audio file to transcribe"
    )
    if uploaded_file:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.read())
            local_video_path = tmp_file.name
        video_title = uploaded_file.name
        video_id = "local_video"

if youtube_link or local_video_path:
    st.markdown("---")
    
    # Display video info and thumbnail (YouTube only)
    col1, col2 = st.columns([1, 2])
    with col1:
        if youtube_link and video_id:
            st.image(f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg")
        else:
            st.info("📹 Local video uploaded")
    with col2:
        st.markdown("### 🎯 Choose Your Action")
        mode = st.radio(
            "",
            ["📝 Summarize", "❓ Ask Questions", "🎯 Key Points", "🧠 Generate Quiz"],
            horizontal=True,
            label_visibility="collapsed"
        )
    
    # Get video info (YouTube only)
    if youtube_link:
        try:
            ydl_opts = {'skip_download': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_link, download=False)
                video_title = info.get('title', 'YouTube Video')
        except:
            video_title = "YouTube Video"
    
    st.subheader(f"Video: {video_title}")
    
    # Initialize Whisper settings with defaults
    prefer_whisper = False
    whisper_model = "tiny"
    
    # Whisper settings UI with expander for cleaner look
    with st.expander("⚙️ Transcript Settings", expanded=False):
        if youtube_link:
            prefer_whisper = st.checkbox("🎤 Use Whisper even if captions exist", value=False)
        else:
            prefer_whisper = True  # Always use Whisper for local files
            st.info("📹 Local files will be transcribed using Whisper")
        
        col_a, col_b = st.columns(2)
        with col_a:
            whisper_model = st.selectbox("Model", ["tiny", "small"], index=0, help="'tiny' is fastest; 'small' is more accurate.")
        with col_b:
            st.metric("Speed", "⚡ Fast" if whisper_model == "tiny" else "🎯 Accurate")

    # Create a unique key for caching
    video_key = youtube_link if youtube_link else local_video_path

    # Process transcript extraction
    if 'transcript' not in st.session_state or st.session_state.get('last_video') != video_key or st.session_state.get('whisper_model') != whisper_model or st.session_state.get('prefer_whisper') != prefer_whisper:
        with st.spinner("📥 Extracting transcript..."):
            if youtube_link:
                st.session_state.transcript = extract_transcript_details(youtube_link, prefer_whisper=prefer_whisper, whisper_model=whisper_model)
            else:
                # For local files, directly use Whisper
                st.session_state.transcript = transcribe_with_whisper(local_video_path, model_size=whisper_model, language="en")
            st.session_state.last_video = video_key
            st.session_state.whisper_model = whisper_model
            st.session_state.prefer_whisper = prefer_whisper

    # Display transcript and features (after extraction)
    if st.session_state.transcript == "Couldn't find transcript":
        st.warning("Transcript not available for this video. Please try another video with English subtitles.")
    else:
        transcript_text = st.session_state.transcript
        
        # Full transcript view/download with better UI
        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            show_full = st.button("📜 Show Full Transcript", key="show_full_transcript_btn", use_container_width=True)
        with col_t2:
            st.download_button("⬇️ Download Transcript", transcript_text, file_name=f"transcript_{video_id}.txt", key="dl_full_transcript", use_container_width=True)
        
        if show_full:
            st.text_area("Full Transcript", transcript_text, height=300)
        
        # ========== SUMMARIZE SECTION ==========
        if "Summarize" in mode:
            st.subheader("Video Summary")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Get Brief Summary", use_container_width=True, key="brief_summary_btn"):
                    with st.spinner("Generating summary..."):
                        summary_prompt = (
                            "You are a YouTube video summarizer. "
                            "Summarize the transcript below into clear, important points within 200 words:\n\n"
                            + transcript_text
                        )
                        summary = generate_gemini_content(summary_prompt)
                        st.session_state.brief_summary = summary
                        st.markdown("### Brief Summary")
                        st.markdown(summary)
                        st.download_button("Download Brief Summary", summary, 
                                         file_name="brief_summary.txt", key="dl_brief_summary")
            
            with col2:
                if st.button("Get Detailed Notes", use_container_width=True, key="detailed_notes_btn"):
                    with st.spinner("Generating detailed notes..."):
                        detailed_prompt = (
                            "Create a detailed summary with these sections:\n"
                            "1. Main Topic\n2. Key Arguments/Points\n"
                            "3. Supporting Evidence\n4. Conclusion\n5. Key Takeaways\n\n"
                            + transcript_text
                        )
                        detailed = generate_gemini_content(detailed_prompt)
                        st.session_state.detailed_notes = detailed
                        st.markdown("### Detailed Notes")
                        st.markdown(detailed)
                        st.download_button("Download Detailed Notes", detailed, 
                                         file_name="detailed_notes.txt", key="dl_detailed_notes")
            
            # Show cached summaries if they exist
            if 'brief_summary' in st.session_state:
                with st.expander("📋 Show Brief Summary (Cached)"):
                    st.markdown(st.session_state.brief_summary)
            
            if 'detailed_notes' in st.session_state:
                with st.expander("📋 Show Detailed Notes (Cached)"):
                    st.markdown(st.session_state.detailed_notes)
            
            # Slide extraction - single button with auto download
            st.markdown("---")
            if st.button("📸 Extract Slides & Download PPT", key="extract_slides_btn", use_container_width=True):
                with st.spinner("Analyzing video stream..."):
                    source = youtube_link if youtube_link else local_video_path
                    pptx_file, error = extract_slides_from_url(source)
                    if error:
                        st.error(error)
                        st.session_state.pop('slides_pptx', None)
                    else:
                        st.success("Slides extracted successfully! Starting download...")
                        st.session_state.slides_pptx = pptx_file
                        # Prepare base64 and auto-trigger download via JS
                        pptx_file.seek(0)
                        b64_pptx = base64.b64encode(pptx_file.read()).decode()
                        filename = f"slides_{video_id}.pptx"
                        html = f"""
                        <html>
                        <body>
                          <a id='dl' href='data:application/vnd.openxmlformats-officedocument.presentationml.presentation;base64,{b64_pptx}' download='{filename}'>download</a>
                          <script>
                            document.getElementById('dl').click();
                          </script>
                        </body>
                        </html>
                        """
                        components.html(html, height=0)
        
        # ========== ASK QUESTIONS SECTION ==========
        elif "Ask Questions" in mode:
            st.subheader("Ask Questions About the Video")
            
            # Custom question input
            user_question = st.text_input("Enter your question:", key="user_question_input")
            
            if st.button("Get Answer", key="get_answer_btn") and user_question:
                with st.spinner("Generating answer..."):
                    qna_prompt = (
                        "You are a YouTube Q&A assistant. "
                        "Based on the transcript below, answer the question concisely:\n\n"
                        f"{transcript_text}\n\nQuestion: {user_question}"
                    )
                    answer = generate_gemini_content(qna_prompt)
                    st.markdown("### Answer")
                    st.markdown(answer)
                    
                    # Store in conversation history
                    if 'conversation' not in st.session_state:
                        st.session_state.conversation = []
                    st.session_state.conversation.append({
                        'question': user_question,
                        'answer': answer
                    })
            
            # Quick sample questions
            st.markdown("*Try these sample questions:*")
            sample_col1, sample_col2 = st.columns(2)
            with sample_col1:
                if st.button("What's the main idea?", use_container_width=True, key="main_idea_btn"):
                    with st.spinner("Finding answer..."):
                        prompt = f"What is the main idea of this video?\n\n{transcript_text}"
                        answer = generate_gemini_content(prompt)
                        st.markdown("*Main Idea:*")
                        st.markdown(answer)
            
            with sample_col2:
                if st.button("Key takeaways?", use_container_width=True, key="key_takeaways_btn"):
                    with st.spinner("Finding answer..."):
                        prompt = f"What are the key takeaways from this video?\n\n{transcript_text}"
                        answer = generate_gemini_content(prompt)
                        st.markdown("*Key Takeaways:*")
                        st.markdown(answer)
            
            # Conversation history
            if 'conversation' in st.session_state and st.session_state.conversation:
                with st.expander("💬 Conversation History"):
                    for i, qa in enumerate(st.session_state.conversation):
                        st.markdown(f"*Q{i+1}:* {qa['question']}")
                        st.markdown(f"*A:* {qa['answer']}")
                        st.divider()
        
        # ========== KEY POINTS SECTION ==========
        elif "Key Points" in mode:
            st.subheader("Key Points & Insights")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Extract Key Points", use_container_width=True, key="key_points_btn"):
                    with st.spinner("Extracting key points..."):
                        points_prompt = (
                            "Extract the 7-10 most important points from this video. "
                            "Format as numbered bullet points with brief explanations.\n\n"
                            + transcript_text
                        )
                        points = generate_gemini_content(points_prompt)
                        st.session_state.key_points = points
                        st.markdown("### Key Points")
                        st.markdown(points)
            
            with col2:
                if st.button("Actionable Insights", use_container_width=True, key="insights_btn"):
                    with st.spinner("Finding insights..."):
                        insights_prompt = (
                            "What are the actionable takeaways from this video? "
                            "What can viewers actually DO after watching?\n\n"
                            + transcript_text
                        )
                        insights = generate_gemini_content(insights_prompt)
                        st.session_state.insights = insights
                        st.markdown("### Actionable Insights")
                        st.markdown(insights)
            
            with col3:
                if st.button("Timeline of Ideas", use_container_width=True, key="timeline_btn"):
                    with st.spinner("Creating timeline..."):
                        timeline_prompt = (
                            "Create a chronological timeline of the main ideas presented in this video. "
                            "Show how concepts build upon each other.\n\n"
                            + transcript_text
                        )
                        timeline = generate_gemini_content(timeline_prompt)
                        st.session_state.timeline = timeline
                        st.markdown("### Timeline of Ideas")
                        st.markdown(timeline)
            
            # Download buttons for generated content
            if 'key_points' in st.session_state:
                st.download_button("Download Key Points", st.session_state.key_points, 
                                 file_name="key_points.txt", key="dl_key_points")
            
            if 'insights' in st.session_state:
                st.download_button("Download Insights", st.session_state.insights, 
                                 file_name="actionable_insights.txt", key="dl_insights")
            
            if 'timeline' in st.session_state:
                st.download_button("Download Timeline", st.session_state.timeline, 
                                 file_name="timeline.txt", key="dl_timeline")
        
        # ========== GENERATE QUIZ SECTION ==========
        elif "Generate Quiz" in mode:
            st.subheader("Learning Quiz Generator")
            
            # Quiz configuration
            col1, col2, col3 = st.columns(3)
            with col1:
                num_questions = st.slider("Number of Questions", 3, 15, 5, key="num_questions_slider")
            with col2:
                question_type = st.selectbox("Question Type", 
                    ["Multiple Choice", "True/False", "Mix of Both"], key="question_type_select")
            with col3:
                difficulty = st.select_slider("Difficulty", 
                    ["Easy", "Medium", "Hard"], value="Medium", key="difficulty_slider")
            
            # Additional options
            with st.expander("📋 Advanced Options"):
                col_a, col_b = st.columns(2)
                with col_a:
                    include_explanations = st.checkbox("Include Answer Explanations", value=True, key="include_explanations")
                with col_b:
                    export_format = st.selectbox(
                        "Export Format",
                        ["Text", "JSON"],
                        index=0,
                        key="export_format_select"
                    )
            
            # Generate quiz button
            if st.button("🎯 Generate Quiz", type="primary", use_container_width=True, key="generate_quiz_btn"):
                with st.spinner(f"Generating {num_questions} questions..."):
                    # Generate structured quiz data
                    quiz_data = generate_structured_quiz(
                        transcript_text,
                        num_questions,
                        question_type,
                        difficulty
                    )
                    
                    # Store in session state
                    st.session_state.quiz_data = quiz_data
            
            # Display quiz if available
            if 'quiz_data' in st.session_state:
                st.markdown("### 📝 Generated Quiz")
                
                # Quiz mode selector
                quiz_mode = st.radio(
                    "Quiz Mode:",
                    ["📚 Study Mode (Show Answers)", "✏ Test Mode (Hide Answers)"],
                    horizontal=True,
                    key="quiz_mode_selector"
                )
                
                show_answers = (quiz_mode == "📚 Study Mode (Show Answers)")
                
                # Display the quiz
                display_quiz(st.session_state.quiz_data, show_answers)
                
                # Answer sheet for test mode
                if not show_answers:
                    with st.expander("🔓 Show Answer Sheet", expanded=False):
                        st.markdown("### Answer Sheet")
                        for q in st.session_state.quiz_data['questions']:
                            if q['correct_answer'] is not None:
                                correct_letter = chr(65 + q['correct_answer'])
                                st.write(f"*Q{q['id']}:* {correct_letter}) {q['options'][q['correct_answer']]}")
                
                # Download options
                st.markdown("### 📥 Download Options")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download as JSON
                    quiz_json = json.dumps(st.session_state.quiz_data, indent=2)
                    st.download_button(
                        "📊 Download as JSON",
                        quiz_json,
                        file_name=f"quiz_{video_id}.json",
                        mime="application/json",
                        key="dl_quiz_json"
                    )
                
                with col2:
                    # Download as text
                    text_output = f"Quiz for: {video_title}\n\n"
                    for q in st.session_state.quiz_data['questions']:
                        text_output += f"Question {q['id']}: {q['question']}\n"
                        for i, opt in enumerate(q['options']):
                            text_output += f"  {chr(65 + i)}) {opt}\n"
                        if q['correct_answer'] is not None:
                            text_output += f"Correct Answer: {chr(65 + q['correct_answer'])}\n"
                        if q.get('explanation'):
                            text_output += f"Explanation: {q['explanation']}\n"
                        text_output += "\n"
                    
                    st.download_button(
                        "📄 Download as Text",
                        text_output,
                        file_name=f"quiz_{video_id}.txt",
                        mime="text/plain",
                        key="dl_quiz_text"
                    )
                
                # Regenerate button
                if st.button("🔄 Generate New Quiz", use_container_width=True, key="regenerate_quiz_btn"):
                    if 'quiz_data' in st.session_state:
                        del st.session_state.quiz_data
                    st.rerun()

# Enhanced sidebar with better organization
with st.sidebar:
    st.markdown("# 🎯 Feature Guide")
    st.markdown("---")
    
    # Feature sections with expanders
    with st.expander("📝 Summarize", expanded=False):
        st.markdown("""
        - 📄 **Brief Summary** (200 words)
        - 📋 **Detailed Notes** (structured sections)        - 🖼️ **Extract Slides** (PPT generation)        -  Download all as text files
        """)
    
    with st.expander("❓ Ask Questions", expanded=False):
        st.markdown("""
        - 💬 **Custom Q&A** with AI
        - ⚡ **Quick Questions** (presets)
        - 📜 **Conversation History** tracking
        - 🎯 Context-aware responses
        """)
    
    with st.expander("🎯 Key Points", expanded=False):
        st.markdown("""
        - 📌 **7-10 Key Points** extraction
        - 💡 **Actionable Insights** (what to do)
        - ⏰ **Timeline of Ideas** (concept flow)
        - 📥 Download all formats
        """)
    
    with st.expander("🧠 Generate Quiz", expanded=False):
        st.markdown("""
        - 🎲 **3-15 Questions** (customizable)
        - 🎯 **Multiple Choice** or True/False
        - 📚 **Study Mode** (with answers)
        - ✏️ **Test Mode** (practice)
        - 📊 Export as JSON/Text
        """)
    
    st.markdown("---")
    
    # Stats section
    st.markdown("### 📊 Session Stats")
    stats_col1, stats_col2 = st.columns(2)
    with stats_col1:
        cached_items = len([k for k in st.session_state.keys() if k in ['brief_summary', 'detailed_notes', 'key_points', 'insights', 'timeline', 'conversation', 'quiz_data']])
        st.metric("Cached", cached_items)
    with stats_col2:
        total_keys = len(st.session_state.keys())
        st.metric("Total", total_keys)
    
    st.markdown("---")
    
    # Action buttons
    if st.button("🧹 Clear Cache", key="clear_cache_btn", use_container_width=True, type="primary"):
        keys_to_clear = ['brief_summary', 'detailed_notes', 'key_points', 
                        'insights', 'timeline', 'conversation', 'quiz_data', 'slides_pptx']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.success("✅ Cache cleared!")
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.info("""
    - Use **tiny model** for speed
    - Use **small model** for accuracy
    - Supports MP3, MP4, and more
    - Works offline with local files
    """)
    
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit & OpenAI Whisper & Gemini API & OpenCVs")