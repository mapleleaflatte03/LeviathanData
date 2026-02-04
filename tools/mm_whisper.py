"""Tool: mm_whisper
OpenAI Whisper speech-to-text transcription.

Supported operations:
- transcribe: Transcribe audio to text
- translate: Translate audio to English
- detect_language: Detect spoken language
- segments: Get timestamped segments
"""
from typing import Any, Dict, List, Optional
import json
import os


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None


whisper = _optional_import("whisper")
torch = _optional_import("torch")

_loaded_model = {}


def _get_model(model_name: str = "base"):
    """Load or retrieve cached Whisper model."""
    if whisper is None:
        raise ImportError("whisper is not installed. Run: pip install openai-whisper")
    
    if model_name not in _loaded_model:
        device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        _loaded_model[model_name] = whisper.load_model(model_name, device=device)
    
    return _loaded_model[model_name]


def _transcribe(
    audio_path: str,
    model_name: str = "base",
    language: Optional[str] = None,
    task: str = "transcribe",
    verbose: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """Transcribe or translate audio file."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    model = _get_model(model_name)
    
    options = {
        "task": task,
        "verbose": verbose,
    }
    if language:
        options["language"] = language
    
    # Add any additional options
    options.update(kwargs)
    
    result = model.transcribe(audio_path, **options)
    
    return {
        "text": result["text"].strip(),
        "language": result.get("language", "unknown"),
        "segments": [
            {
                "id": seg["id"],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
            }
            for seg in result.get("segments", [])
        ],
    }


def _detect_language(audio_path: str, model_name: str = "base") -> Dict[str, Any]:
    """Detect the language spoken in an audio file."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    model = _get_model(model_name)
    
    # Load audio and pad/trim to 30 seconds
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    
    # Make log-mel spectrogram
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    
    # Detect language
    _, probs = model.detect_language(mel)
    
    # Get top 5 languages by probability
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        "detected_language": sorted_probs[0][0],
        "confidence": sorted_probs[0][1],
        "top_languages": [
            {"language": lang, "probability": prob}
            for lang, prob in sorted_probs
        ],
    }


def _get_segments(
    audio_path: str,
    model_name: str = "base",
    language: Optional[str] = None,
    word_timestamps: bool = False,
) -> Dict[str, Any]:
    """Get detailed timestamped segments."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    model = _get_model(model_name)
    
    options = {"word_timestamps": word_timestamps}
    if language:
        options["language"] = language
    
    result = model.transcribe(audio_path, **options)
    
    segments = []
    for seg in result.get("segments", []):
        segment_data = {
            "id": seg["id"],
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
            "avg_logprob": seg.get("avg_logprob"),
            "no_speech_prob": seg.get("no_speech_prob"),
        }
        
        if word_timestamps and "words" in seg:
            segment_data["words"] = [
                {
                    "word": w["word"],
                    "start": w["start"],
                    "end": w["end"],
                    "probability": w.get("probability"),
                }
                for w in seg["words"]
            ]
        
        segments.append(segment_data)
    
    return {
        "text": result["text"].strip(),
        "language": result.get("language", "unknown"),
        "segments": segments,
        "duration": segments[-1]["end"] if segments else 0,
    }


def _list_models() -> List[str]:
    """List available Whisper models."""
    return ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run Whisper speech-to-text operations.
    
    Args:
        args: Dictionary with:
            - operation: "transcribe", "translate", "detect_language", "segments", "models"
            - audio_path: Path to audio file
            - model: Whisper model size (tiny, base, small, medium, large)
            - language: Optional language code (e.g., "en", "es", "fr")
            - word_timestamps: Include word-level timestamps (for segments)
    
    Returns:
        Result dictionary with transcription/translation data
    """
    args = args or {}
    operation = args.get("operation", "transcribe")
    audio_path = args.get("audio_path", "")
    model_name = args.get("model", "base")
    language = args.get("language")
    
    if whisper is None:
        return {
            "tool": "mm_whisper",
            "status": "error",
            "error": "whisper not installed. Run: pip install openai-whisper",
        }
    
    try:
        if operation == "models":
            return {
                "tool": "mm_whisper",
                "status": "ok",
                "models": _list_models(),
            }
        
        elif operation == "transcribe":
            result = _transcribe(audio_path, model_name, language, task="transcribe")
            return {
                "tool": "mm_whisper",
                "status": "ok",
                **result,
            }
        
        elif operation == "translate":
            result = _transcribe(audio_path, model_name, language, task="translate")
            return {
                "tool": "mm_whisper",
                "status": "ok",
                **result,
            }
        
        elif operation == "detect_language":
            result = _detect_language(audio_path, model_name)
            return {
                "tool": "mm_whisper",
                "status": "ok",
                **result,
            }
        
        elif operation == "segments":
            word_timestamps = args.get("word_timestamps", False)
            result = _get_segments(audio_path, model_name, language, word_timestamps)
            return {
                "tool": "mm_whisper",
                "status": "ok",
                **result,
            }
        
        else:
            return {"tool": "mm_whisper", "status": "error", "error": f"Unknown operation: {operation}"}
    
    except Exception as e:
        return {"tool": "mm_whisper", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "transcribe": {
            "operation": "transcribe",
            "audio_path": "audio.mp3",
            "model": "base",
            "language": "en",
        },
        "translate": {
            "operation": "translate",
            "audio_path": "audio_spanish.mp3",
            "model": "small",
        },
        "detect_language": {
            "operation": "detect_language",
            "audio_path": "audio.mp3",
            "model": "base",
        },
        "segments_with_words": {
            "operation": "segments",
            "audio_path": "audio.mp3",
            "model": "base",
            "word_timestamps": True,
        },
    }
