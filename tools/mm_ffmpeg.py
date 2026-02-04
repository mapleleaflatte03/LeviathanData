"""Tool: mm_ffmpeg
FFmpeg for audio/video processing.

Supported operations:
- info: Get media file info
- convert: Convert media format
- extract_audio: Extract audio from video
- trim: Trim media file
- merge: Merge multiple files
- thumbnail: Extract thumbnail from video
- transcode: Transcode with specific settings
- compress: Compress video
"""
from typing import Any, Dict, List, Optional, Union
import json
import subprocess
import shutil
import os


def _optional_import(module_name: str):
    try:
        return __import__(module_name)
    except Exception:
        return None


def _check_ffmpeg() -> bool:
    """Check if ffmpeg is installed."""
    return shutil.which("ffmpeg") is not None


def _check_ffprobe() -> bool:
    """Check if ffprobe is installed."""
    return shutil.which("ffprobe") is not None


def _run_command(cmd: List[str], capture_output: bool = True) -> Dict[str, Any]:
    """Run a command and return result."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Command timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _info(input_path: str) -> Dict[str, Any]:
    """Get media file information."""
    if not _check_ffprobe():
        raise RuntimeError("ffprobe not found. Install FFmpeg.")
    
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        input_path,
    ]
    
    result = _run_command(cmd)
    
    if result["success"]:
        info = json.loads(result["stdout"])
        
        # Extract key information
        format_info = info.get("format", {})
        streams = info.get("streams", [])
        
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
        
        return {
            "filename": format_info.get("filename"),
            "format": format_info.get("format_name"),
            "duration": float(format_info.get("duration", 0)),
            "size": int(format_info.get("size", 0)),
            "bit_rate": int(format_info.get("bit_rate", 0)),
            "video_streams": len(video_streams),
            "audio_streams": len(audio_streams),
            "video": {
                "codec": video_streams[0].get("codec_name") if video_streams else None,
                "width": video_streams[0].get("width") if video_streams else None,
                "height": video_streams[0].get("height") if video_streams else None,
                "fps": eval(video_streams[0].get("r_frame_rate", "0/1")) if video_streams else None,
            } if video_streams else None,
            "audio": {
                "codec": audio_streams[0].get("codec_name") if audio_streams else None,
                "sample_rate": audio_streams[0].get("sample_rate") if audio_streams else None,
                "channels": audio_streams[0].get("channels") if audio_streams else None,
            } if audio_streams else None,
        }
    
    return {"error": result.get("stderr", "Unknown error")}


def _convert(
    input_path: str,
    output_path: str,
    video_codec: Optional[str] = None,
    audio_codec: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Convert media format."""
    if not _check_ffmpeg():
        raise RuntimeError("ffmpeg not found")
    
    cmd = ["ffmpeg", "-i", input_path, "-y"]
    
    if video_codec:
        cmd.extend(["-c:v", video_codec])
    if audio_codec:
        cmd.extend(["-c:a", audio_codec])
    if extra_args:
        cmd.extend(extra_args)
    
    cmd.append(output_path)
    
    result = _run_command(cmd)
    
    if result["success"]:
        return {
            "output": output_path,
            "converted": True,
        }
    return {"error": result.get("stderr", "Conversion failed")}


def _extract_audio(
    input_path: str,
    output_path: str,
    audio_codec: str = "aac",
    bitrate: str = "192k",
) -> Dict[str, Any]:
    """Extract audio from video."""
    if not _check_ffmpeg():
        raise RuntimeError("ffmpeg not found")
    
    cmd = [
        "ffmpeg", "-i", input_path,
        "-vn",  # No video
        "-c:a", audio_codec,
        "-b:a", bitrate,
        "-y", output_path,
    ]
    
    result = _run_command(cmd)
    
    if result["success"]:
        return {"output": output_path, "extracted": True}
    return {"error": result.get("stderr", "Extraction failed")}


def _trim(
    input_path: str,
    output_path: str,
    start: str = "00:00:00",
    duration: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, Any]:
    """Trim media file."""
    if not _check_ffmpeg():
        raise RuntimeError("ffmpeg not found")
    
    cmd = ["ffmpeg", "-i", input_path, "-ss", start]
    
    if duration:
        cmd.extend(["-t", duration])
    elif end:
        cmd.extend(["-to", end])
    
    cmd.extend(["-c", "copy", "-y", output_path])
    
    result = _run_command(cmd)
    
    if result["success"]:
        return {"output": output_path, "trimmed": True}
    return {"error": result.get("stderr", "Trim failed")}


def _thumbnail(
    input_path: str,
    output_path: str,
    time: str = "00:00:01",
    width: int = 320,
    height: int = -1,
) -> Dict[str, Any]:
    """Extract thumbnail from video."""
    if not _check_ffmpeg():
        raise RuntimeError("ffmpeg not found")
    
    cmd = [
        "ffmpeg", "-i", input_path,
        "-ss", time,
        "-vframes", "1",
        "-vf", f"scale={width}:{height}",
        "-y", output_path,
    ]
    
    result = _run_command(cmd)
    
    if result["success"]:
        return {"output": output_path, "thumbnail": True}
    return {"error": result.get("stderr", "Thumbnail extraction failed")}


def _merge(
    input_paths: List[str],
    output_path: str,
    method: str = "concat",
) -> Dict[str, Any]:
    """Merge multiple files."""
    if not _check_ffmpeg():
        raise RuntimeError("ffmpeg not found")
    
    if method == "concat":
        # Create concat file
        concat_file = "/tmp/ffmpeg_concat.txt"
        with open(concat_file, "w") as f:
            for path in input_paths:
                f.write(f"file '{path}'\n")
        
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            "-y", output_path,
        ]
        
        result = _run_command(cmd)
        os.remove(concat_file)
    else:
        return {"error": f"Unknown merge method: {method}"}
    
    if result["success"]:
        return {"output": output_path, "merged": True, "input_count": len(input_paths)}
    return {"error": result.get("stderr", "Merge failed")}


def _compress(
    input_path: str,
    output_path: str,
    crf: int = 23,
    preset: str = "medium",
    max_width: Optional[int] = None,
) -> Dict[str, Any]:
    """Compress video."""
    if not _check_ffmpeg():
        raise RuntimeError("ffmpeg not found")
    
    cmd = [
        "ffmpeg", "-i", input_path,
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-c:a", "aac",
        "-b:a", "128k",
    ]
    
    if max_width:
        cmd.extend(["-vf", f"scale={max_width}:-2"])
    
    cmd.extend(["-y", output_path])
    
    result = _run_command(cmd)
    
    if result["success"]:
        input_size = os.path.getsize(input_path)
        output_size = os.path.getsize(output_path)
        return {
            "output": output_path,
            "compressed": True,
            "input_size": input_size,
            "output_size": output_size,
            "compression_ratio": round(input_size / output_size, 2),
        }
    return {"error": result.get("stderr", "Compression failed")}


def run(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run FFmpeg operations."""
    args = args or {}
    operation = args.get("operation", "info")
    
    if not _check_ffmpeg() and operation != "info":
        return {
            "tool": "mm_ffmpeg",
            "status": "error",
            "error": "FFmpeg is not installed. Install with: apt install ffmpeg",
        }
    
    try:
        if operation == "info":
            result = _info(input_path=args.get("input_path", ""))
        
        elif operation == "convert":
            result = _convert(
                input_path=args.get("input_path", ""),
                output_path=args.get("output_path", ""),
                video_codec=args.get("video_codec"),
                audio_codec=args.get("audio_codec"),
                extra_args=args.get("extra_args"),
            )
        
        elif operation == "extract_audio":
            result = _extract_audio(
                input_path=args.get("input_path", ""),
                output_path=args.get("output_path", ""),
                audio_codec=args.get("audio_codec", "aac"),
                bitrate=args.get("bitrate", "192k"),
            )
        
        elif operation == "trim":
            result = _trim(
                input_path=args.get("input_path", ""),
                output_path=args.get("output_path", ""),
                start=args.get("start", "00:00:00"),
                duration=args.get("duration"),
                end=args.get("end"),
            )
        
        elif operation == "thumbnail":
            result = _thumbnail(
                input_path=args.get("input_path", ""),
                output_path=args.get("output_path", ""),
                time=args.get("time", "00:00:01"),
                width=args.get("width", 320),
                height=args.get("height", -1),
            )
        
        elif operation == "merge":
            result = _merge(
                input_paths=args.get("input_paths", []),
                output_path=args.get("output_path", ""),
                method=args.get("method", "concat"),
            )
        
        elif operation == "compress":
            result = _compress(
                input_path=args.get("input_path", ""),
                output_path=args.get("output_path", ""),
                crf=args.get("crf", 23),
                preset=args.get("preset", "medium"),
                max_width=args.get("max_width"),
            )
        
        else:
            return {"tool": "mm_ffmpeg", "status": "error", "error": f"Unknown operation: {operation}"}
        
        return {"tool": "mm_ffmpeg", "status": "ok", **result}
    
    except Exception as e:
        return {"tool": "mm_ffmpeg", "status": "error", "error": str(e)}


def example():
    """Example usage payloads."""
    return {
        "info": {
            "operation": "info",
            "input_path": "video.mp4",
        },
        "convert": {
            "operation": "convert",
            "input_path": "video.avi",
            "output_path": "video.mp4",
            "video_codec": "libx264",
            "audio_codec": "aac",
        },
        "extract_audio": {
            "operation": "extract_audio",
            "input_path": "video.mp4",
            "output_path": "audio.mp3",
            "audio_codec": "libmp3lame",
        },
        "trim": {
            "operation": "trim",
            "input_path": "video.mp4",
            "output_path": "clip.mp4",
            "start": "00:01:00",
            "duration": "00:00:30",
        },
        "compress": {
            "operation": "compress",
            "input_path": "video.mp4",
            "output_path": "compressed.mp4",
            "crf": 28,
            "preset": "fast",
        },
    }
