import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
import yt_dlp
import requests
from urllib.parse import urlparse, parse_qs
import json

# --- NEW IMPORTS ---
from whisper_utils import transcribe_with_whisper
from slide_utils import extract_slides_from_url

# Load API key
load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


# ---------------- Helper Functions ----------------

def get_video_id(url):
    """Extract YouTube video ID from URL"""
    if "youtu.be" in url:
        return url.split("/")[-1]
    query = parse_qs(urlparse(url).query)
    return query.get("v", [None])[0]

def extract_transcript_details(url):
    """Extract transcript text from YouTube video"""
    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    captions = info.get("requested_subtitles") or info.get("automatic_captions")
    if captions and "en" in captions:
        subtitle_url = captions["en"]["url"]
        response = requests.get(subtitle_url)
        if response.status_code == 200:
            return response.text
    st.info("⚠️ No official subtitles found. Using Whisper AI to generate them...")
    st.caption("⏳ This downloads the audio and processes it. Please wait...")
    return transcribe_with_whisper(url)

def generate_gemini_content(prompt_text):
    """Generate content using Google Gemini"""
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt_text)
    return response.text

def generate_structured_quiz(transcript_text, num_questions=5, question_type="Multiple Choice", difficulty="Medium"):
    """Generate a structured quiz that can be displayed in different modes"""
    
    # Create a prompt that forces JSON output
    quiz_prompt = f"""
    Create a {num_questions}-question quiz based on this YouTube video transcript.
    
    Requirements:
    - Question Type: {question_type}
    - Difficulty: {difficulty}
    - Return ONLY valid JSON format
    - Each question must have: question_text, options (list), correct_answer, explanation
    
    Format your response as this exact JSON structure:
    {{
      "quiz_title": "Video Comprehension Quiz",
      "num_questions": {num_questions},
      "questions": [
        {{
          "id": 1,
          "question": "What is the main topic?",
          "options": ["Option A", "Option B", "Option C", "Option D"],
          "correct_answer": 0,
          "explanation": "This is correct because..."
        }}
      ]
    }}
    
    IMPORTANT: 
    - Return ONLY the JSON, no other text
    - For correct_answer, use 0, 1, 2, or 3 (index of correct option)
    - For True/False questions, options should be ["True", "False"]
    
    Video Transcript:
    {transcript_text[:3000]}  # Limit transcript length
    """
    
    try:
        # Get the quiz JSON from Gemini
        quiz_json_str = generate_gemini_content(quiz_prompt)
        
        # Clean the response - extract JSON if there's extra text
        start_idx = quiz_json_str.find('{')
        end_idx = quiz_json_str.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            quiz_json_str = quiz_json_str[start_idx:end_idx]
        
        # Parse the JSON
        quiz_data = json.loads(quiz_json_str)
        return quiz_data
        
    except json.JSONDecodeError as e:
        # Fallback: generate text quiz and parse it
        st.warning("Could not parse JSON. Using fallback method...")
        return generate_fallback_quiz(transcript_text, num_questions, question_type, difficulty)

def generate_fallback_quiz(transcript_text, num_questions, question_type, difficulty):
    """Fallback method if JSON parsing fails"""
    quiz_prompt = f"""
    Create a {num_questions}-question {difficulty.lower()} {question_type.lower()} quiz based on this video.
    
    Format each question like this:
    
    Question 1: [Question text?]
    A) [Option A]
    B) [Option B]
    C) [Option C]
    D) [Option D]
    Correct Answer: [Letter of correct option]
    Explanation: [Brief explanation]
    
    Video transcript:
    {transcript_text[:2000]}
    """
    
    quiz_text = generate_gemini_content(quiz_prompt)
    
    # Parse the text into structured format
    questions = []
    lines = quiz_text.split('\n')
    current_question = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('Question'):
            # Start new question
            if current_question:
                questions.append(current_question)
            
            current_question = {
                'id': len(questions) + 1,
                'question': line.split(':', 1)[1].strip() if ':' in line else line,
                'options': [],
                'correct_answer': None,
                'explanation': ''
            }
        
        elif line.startswith(('A)', 'B)', 'C)', 'D)')):
            if current_question:
                current_question['options'].append(line)
        
        elif 'Correct Answer:' in line:
            if current_question:
                ans = line.split(':')[1].strip().upper()
                # Convert letter to index
                if ans == 'A': current_question['correct_answer'] = 0
                elif ans == 'B': current_question['correct_answer'] = 1
                elif ans == 'C': current_question['correct_answer'] = 2
                elif ans == 'D': current_question['correct_answer'] = 3
        
        elif 'Explanation:' in line and current_question:
            current_question['explanation'] = line.split(':', 1)[1].strip() if ':' in line else line
    
    # Add the last question
    if current_question and current_question['options']:
        questions.append(current_question)
    
    return {
        'quiz_title': 'Video Comprehension Quiz',
        'num_questions': len(questions),
        'questions': questions
    }

def display_quiz(quiz_data, show_answers=True):
    """Display quiz in the selected mode"""
    if not quiz_data or 'questions' not in quiz_data:
        st.error("No quiz data available")
        return
    
    st.markdown(f"## {quiz_data.get('quiz_title', 'Quiz')}")
    
    for q in quiz_data['questions']:
        st.markdown(f"### Question {q['id']}")
        st.markdown(f"{q['question']}")
        
        # Display options
        for i, option in enumerate(q['options']):
            col1, col2 = st.columns([1, 20])
            with col1:
                option_letter = chr(65 + i)  # A, B, C, D
                st.write(f"{option_letter})")
            with col2:
                st.write(option)
        
        # Show answer if in study mode
        if show_answers and q['correct_answer'] is not None:
            correct_letter = chr(65 + q['correct_answer'])
            st.success(f"✅ *Correct Answer: {correct_letter}) {q['options'][q['correct_answer']]}*")
            if q.get('explanation'):
                st.info(f"💡 *Explanation:* {q['explanation']}")
        
        st.markdown("---")

# ---------------- Streamlit App ----------------

st.set_page_config(page_title="YouTube Assistant Pro", layout="wide")
st.title("🎬 YouTube Assistant Pro")
st.markdown("⚡ *Fast processing for YouTube videos with subtitles*")

# Input YouTube link
youtube_link = st.text_input(
    "🔗 Enter YouTube Video Link",
    placeholder="https://www.youtube.com/watch?v=...",
    help="Make sure the video has English subtitles"
)

if youtube_link:
    video_id = get_video_id(youtube_link)
    if video_id:
        # Display video thumbnail
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg")
        with col2:
            mode = st.radio(
                "Choose what you want to do:",
                ["📝 Summarize", "❓ Ask Questions", "🎯 Key Points", "🧠 Generate Quiz"]
            )
        
        # Get video info
        try:
            ydl_opts = {'skip_download': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_link, download=False)
                video_title = info.get('title', 'YouTube Video')
            st.subheader(f"Video: {video_title}")
        except:
            st.subheader("YouTube Video")
        
        # Process based on mode
        if 'transcript' not in st.session_state or st.session_state.get('last_video') != youtube_link:
            with st.spinner("📥 Extracting transcript..."):
                st.session_state.transcript = extract_transcript_details(youtube_link)
                st.session_state.last_video = youtube_link

        if st.session_state.transcript == "Couldn't find transcript":
            st.warning("Transcript not available for this video. Please try another video with English subtitles.")
        else:
            transcript_text = st.session_state.transcript
            
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

            st.markdown("---")
            with st.expander("🖼️ Extract Slides from Video (Beta)"):
                st.info("This feature scans the video to find unique slides and creates a PowerPoint.")

                # Check if we have the video link from earlier in the code
                if st.button("📸 Extract Slides & Create PPT", key="extract_slides_btn"):
                    with st.spinner("Analyzing video stream... (Do not close tab)"):
                        # Call the function we imported from slide_utils.py
                        pptx_file, error = extract_slides_from_url(youtube_link)

                        if error:
                            st.error(error)
                        else:
                            st.success("Slides extracted successfully!")
                            st.download_button(
                                label="📥 Download Slides (.pptx)",
                                data=pptx_file,
                                file_name=f"slides_{video_id}.pptx",
                                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                                key="dl_slides_btn"
                            )
# Add complete sidebar help
with st.sidebar:
    st.markdown("### 🎯 All Features")
    st.info("""
    *📝 Summarize:*
    - Brief summary (200 words)
    - Detailed notes with sections
    - Download as text
    
    *❓ Ask Questions:*
    - Ask custom questions
    - Quick sample questions
    - Conversation history
    
    *🎯 Key Points:*
    - 7-10 key points
    - Actionable insights
    - Timeline of ideas
    - Download all content
    
    *🧠 Generate Quiz:*
    - Custom number of questions
    - Multiple choice/True False
    - Study/Test modes
    - Download as JSON/text
    """)
    
    # Clear cache button
    if st.button("🧹 Clear All Cache", key="clear_cache_btn"):
        keys_to_clear = ['brief_summary', 'detailed_notes', 'key_points', 
                        'insights', 'timeline', 'conversation', 'quiz_data']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.success("Cache cleared!")
        st.rerun()