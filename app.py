import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
import yt_dlp
import requests
from urllib.parse import urlparse, parse_qs

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
    return "Couldn’t find transcript"

def generate_gemini_content(prompt_text):
    """Generate content using Google Gemini Pro"""
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt_text)
    return response.text

# ---------------- Streamlit App ----------------

st.set_page_config(page_title="YouTube Assistant", layout="wide")
st.title("📺 YouTube Video Assistant")

# Input YouTube link
youtube_link = st.text_input("Enter YouTube Video Link")

if youtube_link:
    video_id = get_video_id(youtube_link)
    if video_id:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(f"https://img.youtube.com/vi/{video_id}/0.jpg")
        with col2:
            mode = st.radio("Choose what you want to do:", ["Summarize", "Ask Questions"])
    else:
        st.error("Invalid YouTube link. Please try again.")

# Process based on mode
if youtube_link and video_id:
    if 'transcript' not in st.session_state:
        with st.spinner("Extracting transcript..."):
            st.session_state.transcript = extract_transcript_details(youtube_link)

    if st.session_state.transcript == "Couldn’t find transcript":
        st.warning("Transcript not available for this video.")
    else:
        transcript_text = st.session_state.transcript

        if mode == "Summarize":
            if st.button("Get Detailed Notes"):
                with st.spinner("Generating summary..."):
                    summary_prompt = (
                        "You are a YouTube video summarizer. "
                        "Summarize the transcript below into clear, important points within 250 words:\n\n"
                        + transcript_text
                    )
                    summary = generate_gemini_content(summary_prompt)
                    st.markdown("### Summary")
                    st.markdown(summary)
                    st.download_button("Download Summary", summary, file_name="summary.txt")

        elif mode == "Ask Questions":
            user_question = st.text_input("Ask a question about this video:")
            if st.button("Get Answer") and user_question:
                with st.spinner("Generating answer..."):
                    qna_prompt = (
                        "You are a YouTube Q&A assistant. "
                        "Based on the transcript below, answer the question concisely:\n\n"
                        f"{transcript_text}\n\nQuestion: {user_question}"
                    )
                    answer = generate_gemini_content(qna_prompt)
                    st.markdown("### Answer")
                    st.markdown(answer)
