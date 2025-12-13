"""
Utility functions for YouTube Assistant Pro
Contains all helper functions for transcript extraction, content generation, and quiz management.
"""

import streamlit as st
import os
import google.generativeai as genai
import yt_dlp
import requests
from urllib.parse import urlparse, parse_qs
import json
from whisper_utils import transcribe_with_whisper


def get_video_id(url):
    """Extract YouTube video ID from URL"""
    if "youtu.be" in url:
        return url.split("/")[-1]
    query = parse_qs(urlparse(url).query)
    return query.get("v", [None])[0]


def extract_transcript_details(url, prefer_whisper: bool = False, whisper_model: str = "tiny"):
    """Extract transcript text from YouTube video.
    If no official captions or prefer_whisper=True, use local Whisper with selected model.
    """
    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    captions = info.get("requested_subtitles") or info.get("automatic_captions")
    if not prefer_whisper and captions and "en" in captions:
        subtitle_url = captions["en"]["url"]
        response = requests.get(subtitle_url)
        if response.status_code == 200:
            return response.text
    st.info("⚠️ Using Whisper to generate English transcript...")
    st.caption(f"⏳ Model: {whisper_model}. Downloading audio and transcribing...")
    return transcribe_with_whisper(url, model_size=whisper_model, language="en")


def generate_gemini_content(prompt_text: str) -> str:
    """Generate content using Google Gemini."""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        response = model.generate_content(prompt_text)
        return response.text
    except Exception as e:
        return f"Error generating content: {e}"


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
