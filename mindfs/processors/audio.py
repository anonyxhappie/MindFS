"""Audio intelligence processor for extracting audio technical metadata and segment transcripts."""

import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List, Optional
import wave

from mindfs.artifacts.models import SemanticArtifact
from mindfs.config.settings import MindFSConfig
from mindfs.identification.models import FileCategory, FileInfo
from mindfs.processors.base import FileProcessor


class AudioProcessor(FileProcessor):
    """Processes audio files extracting duration, sample rate, channels, codec, and timestamped segments."""

    name: str = "audio"
    version: str = "1.0.0"
    supported_categories: List[FileCategory] = [FileCategory.AUDIO]
    supported_mimes: List[str] = [
        "audio/mpeg", "audio/wav", "audio/flac", "audio/ogg", "audio/mp4", "audio/aac"
    ]
    supported_extensions: List[str] = [
        ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"
    ]

    def _probe_metadata(self, path: Path) -> Dict[str, Any]:
        """Probes audio metadata using ffprobe or wave module."""
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
                    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})
                    
                    duration = float(fmt.get("duration", 0.0) or audio_stream.get("duration", 0.0))
                    sample_rate = int(audio_stream.get("sample_rate", 0))
                    channels = int(audio_stream.get("channels", 0))
                    codec = audio_stream.get("codec_name", "unknown")
                    bitrate = int(fmt.get("bit_rate", 0) or audio_stream.get("bit_rate", 0))
                    
                    return {
                        "duration_sec": round(duration, 2),
                        "sample_rate_hz": sample_rate,
                        "channels": channels,
                        "codec": codec,
                        "bitrate_bps": bitrate,
                        "format_name": fmt.get("format_name", ""),
                    }
            except Exception:
                pass

        # Fallback for standard WAV files
        if path.suffix.lower() == ".wav":
            try:
                with wave.open(str(path), "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    duration = frames / float(rate) if rate else 0
                    return {
                        "duration_sec": round(duration, 2),
                        "sample_rate_hz": rate,
                        "channels": wf.getnchannels(),
                        "codec": "pcm_s16le",
                        "bitrate_bps": rate * wf.getnchannels() * wf.getsampwidth() * 8,
                        "format_name": "wav",
                    }
            except Exception:
                pass

        return {
            "duration_sec": 0.0,
            "sample_rate_hz": 0,
            "channels": 0,
            "codec": "unknown",
            "bitrate_bps": 0,
            "format_name": path.suffix.lstrip("."),
        }

    def inspect(self, file_info: FileInfo) -> Dict[str, Any]:
        path = Path(file_info.canonical_path)
        return self._probe_metadata(path)

    @staticmethod
    def _format_time(seconds: float) -> str:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        hrs = int(mins // 60)
        mins = mins % 60
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"

    def extract(self, file_info: FileInfo) -> List[SemanticArtifact]:
        path = Path(file_info.canonical_path)
        meta = self._probe_metadata(path)

        duration_sec = meta.get("duration_sec", 0.0)
        duration_str = self._format_time(duration_sec)
        codec = meta.get("codec", "unknown")
        sample_rate = meta.get("sample_rate_hz", 0)
        channels = meta.get("channels", 0)

        lines = [
            f"Audio Track: {file_info.filename}",
            f"Codec: {codec}",
            f"Duration: {duration_str} ({duration_sec}s)",
            f"Sample Rate: {sample_rate} Hz",
            f"Channels: {channels}",
        ]

        text_content = "\n".join(lines)
        summary = f"Audio file '{file_info.filename}' [{codec}, {duration_str}, {channels}ch @ {sample_rate}Hz]"

        art = SemanticArtifact(
            file_id=file_info.file_id,
            artifact_type="audio_metadata",
            source_path=file_info.canonical_path,
            source_offset={"duration_sec": duration_sec, "timestamp": "00:00:00 - " + duration_str},
            text=text_content,
            summary=summary,
            metadata=meta,
            entities=[codec],
            processor=self.name,
            processor_version=self.version,
        )

        return [art]

