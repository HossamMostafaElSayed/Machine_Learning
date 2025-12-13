import os
import shutil
import tempfile
import subprocess
import yt_dlp
import threading
import torch
try:
	import whisper
except Exception as e:
	raise RuntimeError(
		"Failed to import OpenAI Whisper. Ensure 'openai-whisper' is installed via pip (not the 'whisper' package)."
	) from e

_torch_device = "cuda" if torch.cuda.is_available() else "cpu"
_model_cache = {}
_model_lock = threading.Lock()


def _ensure_ffmpeg():
	if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
		raise RuntimeError(
			"ffmpeg/ffprobe not found. Install FFmpeg and restart."
		)


def _download_audio_best(video_url: str, tmp_base: str) -> str:
	outtmpl = tmp_base + ".%(ext)s"
	ydl_opts = {
		'format': 'bestaudio[ext=m4a]/bestaudio/best',
		'outtmpl': outtmpl,
		'quiet': False,
		'no_warnings': False,
		'socket_timeout': 30,
		'retries': 3,
	}
	try:
		with yt_dlp.YoutubeDL(ydl_opts) as ydl:
			info = ydl.extract_info(video_url, download=True)
			if info:
				print(f"Downloaded: {info.get('ext', 'unknown')} format")
	except Exception as e:
		raise RuntimeError(f"yt-dlp download failed: {e}")
	
	# Check for downloaded file with common extensions
	for ext in ("m4a", "webm", "opus", "mp3", "wav", "aac", "ogg"):
		candidate = outtmpl.replace('%(ext)s', ext)
		if os.path.exists(candidate):
			print(f"Found audio file: {candidate}")
			return candidate
	
	# List files in directory for debugging
	dir_path = os.path.dirname(tmp_base)
	files = [f for f in os.listdir(dir_path) if os.path.basename(tmp_base) in f]
	raise FileNotFoundError(f"Audio file not created. Found in temp dir: {files}")


def _is_audio_file(path: str) -> bool:
	"""Check if file is an audio file based on extension."""
	audio_exts = ('.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac', '.opus', '.wma')
	return path.lower().endswith(audio_exts)


def _load_model(model_size: str):
	key = f"{model_size}-{_torch_device}"
	with _model_lock:
		if key in _model_cache:
			return _model_cache[key]
		model = whisper.load_model(model_size, device=_torch_device)
		_model_cache[key] = model
		return model


def transcribe_with_whisper(video_url: str, model_size: str = "tiny", language: str = "en") -> str:
	"""
	Transcribe audio from YouTube URL or local audio/video file using OpenAI Whisper.

	- Supports YouTube URLs and local files (audio or video)
	- Uses tiny/small models for speed
	- English-only for faster decoding
	"""
	_ensure_ffmpeg()

	# Check if input is a local file
	is_local_file = os.path.exists(video_url)
	audio_path = None
	temp_files = []
	
	try:
		if is_local_file:
			# Handle local audio or video file
			if _is_audio_file(video_url):
				# Direct audio file - use as-is
				print(f"Using local audio file directly: {video_url}")
				audio_path = video_url
			else:
				# Video file - extract audio
				print(f"Extracting audio from local video: {video_url}")
				tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".m4a")
				audio_path = tmp_audio.name
				tmp_audio.close()
				temp_files.append(audio_path)
				
				# Extract audio with ffmpeg
				cmd = ['ffmpeg', '-y', '-i', video_url, '-vn', '-acodec', 'copy', audio_path]
				try:
					subprocess.run(cmd, capture_output=True, check=True)
				except subprocess.CalledProcessError:
					# Fallback: re-encode to AAC
					cmd = ['ffmpeg', '-y', '-i', video_url, '-vn', '-c:a', 'aac', '-b:a', '128k', audio_path]
					subprocess.run(cmd, capture_output=True, check=True)
		else:
			# YouTube URL - download audio
			print(f"Downloading audio from YouTube: {video_url}")
			tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tmp")
			tmp_base = tmp_file.name.rsplit('.', 1)[0]
			tmp_file.close()
			temp_files.append(tmp_base + '.tmp')
			audio_path = _download_audio_best(video_url, tmp_base)
			temp_files.append(audio_path)
	except Exception as e:
		for f in temp_files:
			try:
				if os.path.exists(f):
					os.remove(f)
			except Exception:
				pass
		return f"Error extracting/downloading audio: {e}"

	try:
		model = _load_model(model_size)
		# Use fp16 on CUDA, otherwise default
		fp16 = _torch_device == "cuda"
		print(f"Transcribing with model={model_size}, device={_torch_device}, fp16={fp16}")
		result = model.transcribe(
			audio_path,
			language=language,
			task="transcribe",
			fp16=fp16,
			verbose=False,
			temperature=0.0,
			no_speech_threshold=0.7,
			logprob_threshold=-1.0,
			compression_ratio_threshold=2.4,
		)
		text = result.get("text", "").strip()
		return text if text else ""
	except Exception as e:
		return f"Error transcribing: {e}"
	finally:
		# Only delete temporary files, not user's original audio files
		for f in temp_files:
			try:
				if f and os.path.exists(f):
					os.remove(f)
			except Exception:
				pass
