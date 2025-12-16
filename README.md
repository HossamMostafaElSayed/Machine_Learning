# 🎬 YouTube Assistant Pro

> AI-Powered Video Analysis & Transcription Tool

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Overview

YouTube Assistant Pro is a comprehensive AI-powered application that extracts insights from video content. It supports YouTube URLs and local video/audio files, providing transcription, summarization, quiz generation, and intelligent Q&A capabilities.

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎤 **Transcription** | Speech-to-text using OpenAI Whisper |
| 📝 **Summarization** | Brief & detailed summaries via Google Gemini |
| 🎯 **Key Points** | Extract main ideas and actionable insights |
| 🧠 **Quiz Generation** | Auto-generate MCQ/True-False quizzes |
| 🤖 **RAG Chat** | Intelligent Q&A with hybrid search |
| 🖼️ **Slide Extraction** | Extract slides to PowerPoint format |

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│                    (Streamlit Web App)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT PROCESSING                           │
│         ┌──────────────────┬──────────────────┐                │
│         │  YouTube URL     │  Local File      │                │
│         │  (yt-dlp)        │  Upload          │                │
│         └──────────────────┴──────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   TRANSCRIPTION ENGINE                          │
│    ┌────────────────────┬────────────────────┐                 │
│    │  YouTube Captions  │  OpenAI Whisper    │                 │
│    │  (if available)    │  (fallback/forced) │                 │
│    └────────────────────┴────────────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AI PROCESSING LAYER                          │
│  ┌───────────┬───────────┬───────────┬───────────────────┐     │
│  │ Summarize │ Key Points│   Quiz    │     RAG Chat      │     │
│  │ (Gemini)  │ (Gemini)  │ (Gemini)  │ (ChromaDB+GROQ)   │     │
│  └───────────┴───────────┴───────────┴───────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       OUTPUT LAYER                              │
│    Summaries │ Notes │ Quizzes │ Chat │ Slides (PPTX)          │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Transcription**: OpenAI Whisper, yt-dlp, FFmpeg
- **LLMs**: Google Gemini 2.5, GROQ (Llama 3.1)
- **RAG**: ChromaDB, Sentence-Transformers, BM25
- **Media**: OpenCV, python-pptx

## 📦 Installation

### Prerequisites
- Python 3.11+
- FFmpeg installed and in PATH
- API Keys (Google Gemini, GROQ)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/HossamMostafaElSayed/Machine_Learning.git
   cd Machine_Learning
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_google_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 🚀 Usage

1. **Select Input Source**
   - Paste a YouTube URL, or
   - Upload a local video/audio file

2. **Choose Action**
   - 📝 **Summarize**: Get brief or detailed summaries
   - 🎯 **Key Points**: Extract main ideas and insights
   - 🧠 **Generate Quiz**: Create study quizzes
   - 🤖 **RAG Chat**: Ask questions about the video

3. **Configure Settings** (Optional)
   - Select Whisper model (tiny/small)
   - Adjust quiz difficulty and question count

## 📁 Project Structure

```
Machine_Learning/
├── app.py              # Main Streamlit application
├── utilities.py        # Content generation & quiz functions
├── whisper_utils.py    # Transcription with OpenAI Whisper
├── rag_utils.py        # RAG system (ChromaDB + GROQ)
├── slide_utils.py      # Slide extraction to PowerPoint
├── requirements.txt    # Python dependencies
└── .env                # API keys (not tracked)
```

## 📋 Requirements

```txt
streamlit
google-generativeai
python-dotenv
yt-dlp
requests
opencv-python
numpy
python-pptx
torch
openai-whisper
chromadb
langchain-huggingface
langchain-experimental
sentence-transformers
rank-bm25
groq
```

## 🔑 API Keys

| Service | Purpose | Get Key |
|---------|---------|---------|
| Google Gemini | Content generation | [Google AI Studio](https://makersuite.google.com/app/apikey) |
| GROQ | RAG chat responses | [GROQ Console](https://console.groq.com/keys) |

## 💡 Tips

- Use **tiny model** for faster transcription
- Use **small model** for better accuracy
- Hybrid search combines semantic + keyword matching for best results
- Supported formats: MP4, AVI, MOV, MKV, MP3, WAV, M4A, and more

## 👥 Team

Machine Learning Project - Senior 2, Faculty of Engineering

## 📄 License

This project is for educational purposes.

---

Built with ❤️ using Streamlit, OpenAI Whisper, Google Gemini, GROQ & ChromaDB
