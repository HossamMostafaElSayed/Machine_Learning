import os
import shutil
import tempfile
import subprocess
import yt_dlp
from faster_whisper import WhisperModel


def detect_gpu():
    """
    Detect GPU availability (NVIDIA CUDA) without importing PyTorch.
    Returns: (has_gpu: bool, gpu_info: str)
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().split('\n')[0]
            return True, gpu_name
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    
    return False, "None (CPU mode)"


def transcribe_with_whisper(video_url: str, use_gpu: bool = True) -> str:
    """
    Downloads audio from a YouTube URL and transcribes it using Faster Whisper with optional GPU support.
    
    Faster Whisper with ONNX Runtime (lightweight, ~200MB vs 2.4GB PyTorch).
    Uses NVIDIA GPU if available (via CUDA/cuDNN), falls back to CPU.

    Args:
        video_url: YouTube video URL.
        use_gpu: If True, attempt to use GPU via ONNX Runtime.

    Returns the transcript text, or an error message string on failure.
    """
    # Check for ffmpeg/ffprobe
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        return (
            "❌ Error: ffmpeg/ffprobe not found on system. "
            "Please install FFmpeg:\n"
            "  - Windows (winget): winget install --id=Gyan.FFmpeg -e\n"
            "  - Windows (choco): choco install ffmpeg -y\n"
            "  - macOS (brew): brew install ffmpeg\n"
            "  - Linux (apt): sudo apt-get install ffmpeg\n"
            "After install, restart your terminal/Streamlit app."
        )
    
    # Detect GPU
    has_gpu, gpu_info = detect_gpu()
    device = "cuda" if (use_gpu and has_gpu) else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    print(f"🎤 Transcription Device: {device.upper()} | GPU: {gpu_info}")
    
    # 1. Create a temporary filename for output
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_path = tmp_file.name
    tmp_file.close()

    # Use yt_dlp to extract audio and convert to mp3
    outtmpl = tmp_path.rsplit(".", 1)[0] + ".%(ext)s"
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': outtmpl,
        'quiet': True,
    }

    audio_path = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        audio_path = outtmpl.replace('%(ext)s', 'mp3')
        if not os.path.exists(audio_path):
            return "Error: audio file not created after download."
    except Exception as e:
        # cleanup tmp if exists
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return f"Error downloading audio: {e}"

    # 2. Run Whisper transcription
    try:
        # Load model once (base model is ~140MB with ONNX)
        # Faster Whisper automatically uses ONNX Runtime for inference
        model = WhisperModel(
            "base",
            device=device,
            compute_type=compute_type,
            cpu_threads=4,  # Optimize CPU threads for faster inference
            num_workers=1   # Parallel processing
        )
        
        # Transcribe with optimized settings
        segments, info = model.transcribe(
            audio_path,
            beam_size=5,           # Beam search for accuracy
            best_of=1,             # Single pass (faster)
            patience=1.0,          # Early stopping patience
            language="en"          # Force English (faster)
        )

        transcript_text = ""
        for segment in segments:
            transcript_text += segment.text + " "

        return transcript_text.strip()

    except Exception as e:
        error_msg = str(e)
        if "CUDA" in error_msg or "cuda" in error_msg:
            return (
                f"⚠️ GPU error: {error_msg}\n"
                "Ensure NVIDIA drivers and CUDA are installed. "
                "CPU fallback available — try re-running."
            )
        return f"Error transcribing: {error_msg}"

    finally:
        # 3. Cleanup
        try:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass