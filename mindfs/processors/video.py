"""Video intelligence processor for container inspection and sparse temporal keyframe sampling."""

import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from mindfs.artifacts.models import SemanticArtifact
from mindfs.config.settings import MindFSConfig
from mindfs.identification.models import FileCategory, FileInfo
from mindfs.processors.base import FileProcessor


class VideoProcessor(FileProcessor):
    """Processes video containers, extracting stream properties, codecs, and sparse temporal metadata."""

    name: str = "video"
    version: str = "1.0.0"
    supported_categories: List[FileCategory] = [FileCategory.VIDEO]
    supported_mimes: List[str] = [
        "video/mp4", "video/x-matroska", "video/quicktime",
        "video/x-msvideo", "video/webm"
    ]
    supported_extensions: List[str] = [
        ".mp4", ".mkv", ".mov", ".avi", ".webm"
    ]

    def _probe_video(self, path: Path) -> Dict[str, Any]:
        ffprobe_bin = shutil.which("ffprobe")
        if ffprobe_bin:
            try:
                cmd = [
                    ffprobe_bin,
                    "-v", "quiet",
                    "-print_format", "json",
                    "-show_format",
                    "-show_streams",
                    str(path)
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
                if res.returncode == 0:
                    data = json.loads(res.stdout)
                    fmt = data.get("format", {})
                    streams = data.get("streams", [])
                    
                    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
                    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})
                    
                    width = int(video_stream.get("width", 0) or 0)
                    height = int(video_stream.get("height", 0) or 0)
                    duration = float(fmt.get("duration", 0.0) or video_stream.get("duration", 0.0) or 0.0)
                    codec = video_stream.get("codec_name", "unknown")
                    
                    # Compute fps
                    r_fps = video_stream.get("r_frame_rate", "0/1")
                    try:
                        num, den = map(int, r_fps.split("/"))
                        fps = round(num / den, 2) if den else 0.0
                    except Exception:
                        fps = 0.0

                    return {
                        "duration_sec": round(duration, 2),
                        "width": width,
                        "height": height,
                        "fps": fps,
                        "video_codec": codec,
                        "audio_codec": audio_stream.get("codec_name", "none"),
                        "bitrate_bps": int(fmt.get("bit_rate", 0) or 0),
                        "format_name": fmt.get("format_name", ""),
                    }
            except Exception:
                pass

        return {
            "duration_sec": 0.0,
            "width": 0,
            "height": 0,
            "fps": 0.0,
            "video_codec": "unknown",
            "audio_codec": "none",
            "bitrate_bps": 0,
            "format_name": path.suffix.lstrip("."),
        }

    def inspect(self, file_info: FileInfo) -> Dict[str, Any]:
        path = Path(file_info.canonical_path)
        return self._probe_video(path)

    @staticmethod
    def _format_time(seconds: float) -> str:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        hrs = int(mins // 60)
        mins = mins % 60
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"

    def extract(self, file_info: FileInfo) -> List[SemanticArtifact]:
        path = Path(file_info.canonical_path)
        meta = self._probe_video(path)

        duration_sec = meta.get("duration_sec", 0.0)
        duration_str = self._format_time(duration_sec)
        width = meta.get("width", 0)
        height = meta.get("height", 0)
        fps = meta.get("fps", 0.0)
        v_codec = meta.get("video_codec", "unknown")
        a_codec = meta.get("audio_codec", "none")

        lines = [
            f"Video File: {file_info.filename}",
            f"Resolution: {width}x{height} pixels",
            f"Framerate: {fps} fps",
            f"Video Codec: {v_codec}",
            f"Audio Track: {a_codec}",
            f"Total Duration: {duration_str} ({duration_sec}s)",
        ]

        # Generate sparse temporal keyframe timestamps
        interval = self.config.media.video_frame_interval_seconds
        sample_points = []
        if duration_sec > 0 and interval > 0:
            current_t = 0.0
            while current_t < duration_sec:
                sample_points.append(self._format_time(current_t))
                current_t += interval
            lines.append(f"Sparse Keyframe Sample Timestamps ({len(sample_points)} intervals): {', '.join(sample_points[:10])}")

        text_content = "\n".join(lines)
        summary = f"Video '{file_info.filename}' [{width}x{height} {v_codec}, {duration_str}, {fps} fps]"

        art = SemanticArtifact(
            file_id=file_info.file_id,
            artifact_type="video_metadata",
            source_path=file_info.canonical_path,
            source_offset={"duration_sec": duration_sec, "resolution": f"{width}x{height}", "timestamps": sample_points},
            text=text_content,
            summary=summary,
            metadata=meta,
            entities=[v_codec, a_codec],
            processor=self.name,
            processor_version=self.version,
        )

        return [art]

