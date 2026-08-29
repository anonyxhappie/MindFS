"""File identification and MIME/magic classification detector."""

import hashlib
import mimetypes
import os
from pathlib import Path
from typing import Optional, Tuple

from mindfs.filesystem.sandbox import FilesystemSandbox
from mindfs.identification.models import FileCategory, FileInfo, ProcessingStatus


class FileDetector:
    """Identifies file type, MIME, category, and metadata safely using layered detection."""

    MAGIC_SIGNATURES = [
        # Images
        (b"\x89PNG\r\n\x1a\n", "image/png", FileCategory.IMAGE),
        (b"\xff\xd8\xff", "image/jpeg", FileCategory.IMAGE),
        (b"GIF87a", "image/gif", FileCategory.IMAGE),
        (b"GIF89a", "image/gif", FileCategory.IMAGE),
        (b"RIFF", "image/webp", FileCategory.IMAGE, 8, b"WEBP"),
        (b"BM", "image/bmp", FileCategory.IMAGE),
        # Documents / PDF
        (b"%PDF-", "application/pdf", FileCategory.DOCUMENT),
        # Audio / Video
        (b"RIFF", "audio/wav", FileCategory.AUDIO, 8, b"WAVE"),
        (b"RIFF", "video/x-msvideo", FileCategory.VIDEO, 8, b"AVI "),
        (b"ID3", "audio/mpeg", FileCategory.AUDIO),
        (b"\xff\xfb", "audio/mpeg", FileCategory.AUDIO),
        (b"\xff\xf3", "audio/mpeg", FileCategory.AUDIO),
        (b"\xff\xf2", "audio/mpeg", FileCategory.AUDIO),
        (b"fLaC", "audio/flac", FileCategory.AUDIO),
        (b"OggS", "audio/ogg", FileCategory.AUDIO),
        (b"\x1a\x45\xdf\xa3", "video/webm", FileCategory.VIDEO),
        # Archives
        (b"PK\x03\x04", "application/zip", FileCategory.ARCHIVE),
        (b"PK\x05\x06", "application/zip", FileCategory.ARCHIVE),
        (b"\x1f\x8b", "application/gzip", FileCategory.ARCHIVE),
        (b"BZh", "application/x-bzip2", FileCategory.ARCHIVE),
        (b"\xfd7zXZ\x00", "application/x-xz", FileCategory.ARCHIVE),
        (b"7z\xbc\xaf\x27\x1c", "application/x-7z-compressed", FileCategory.ARCHIVE),
        # Executables / Binaries
        (b"\x7fELF", "application/x-executable", FileCategory.BINARY),
        (b"\xfe\xed\xfa\xce", "application/x-mach-binary", FileCategory.BINARY),
        (b"\xfe\xed\xfa\xcf", "application/x-mach-binary", FileCategory.BINARY),
        (b"\xce\xfa\xed\xfe", "application/x-mach-binary", FileCategory.BINARY),
        (b"\xcf\xfa\xed\xfe", "application/x-mach-binary", FileCategory.BINARY),
        (b"\xca\xfe\xba\xbe", "application/x-java-class", FileCategory.BINARY),
        (b"\x00asm", "application/wasm", FileCategory.BINARY),
        (b"MZ", "application/x-dosexec", FileCategory.BINARY),
    ]

    EXTENSIONS_CATEGORY_MAP = {
        # Documents
        ".txt": (FileCategory.DOCUMENT, "text/plain"),
        ".md": (FileCategory.DOCUMENT, "text/markdown"),
        ".log": (FileCategory.DOCUMENT, "text/plain"),
        ".pdf": (FileCategory.DOCUMENT, "application/pdf"),
        ".html": (FileCategory.DOCUMENT, "text/html"),
        ".htm": (FileCategory.DOCUMENT, "text/html"),
        ".py": (FileCategory.DOCUMENT, "text/x-python"),
        ".js": (FileCategory.DOCUMENT, "text/javascript"),
        ".ts": (FileCategory.DOCUMENT, "text/typescript"),
        ".java": (FileCategory.DOCUMENT, "text/x-java-source"),
        ".c": (FileCategory.DOCUMENT, "text/x-c"),
        ".cpp": (FileCategory.DOCUMENT, "text/x-c++"),
        ".h": (FileCategory.DOCUMENT, "text/x-c-header"),
        ".hpp": (FileCategory.DOCUMENT, "text/x-c++-header"),
        ".rs": (FileCategory.DOCUMENT, "text/x-rust"),
        ".go": (FileCategory.DOCUMENT, "text/x-go"),
        ".sh": (FileCategory.DOCUMENT, "text/x-shellscript"),
        ".rtf": (FileCategory.DOCUMENT, "application/rtf"),
        # Structured Data
        ".json": (FileCategory.STRUCTURED, "application/json"),
        ".yaml": (FileCategory.STRUCTURED, "application/yaml"),
        ".yml": (FileCategory.STRUCTURED, "application/yaml"),
        ".xml": (FileCategory.STRUCTURED, "application/xml"),
        ".csv": (FileCategory.STRUCTURED, "text/csv"),
        ".tsv": (FileCategory.STRUCTURED, "text/tab-separated-values"),
        ".toml": (FileCategory.STRUCTURED, "application/toml"),
        # Images
        ".jpg": (FileCategory.IMAGE, "image/jpeg"),
        ".jpeg": (FileCategory.IMAGE, "image/jpeg"),
        ".png": (FileCategory.IMAGE, "image/png"),
        ".gif": (FileCategory.IMAGE, "image/gif"),
        ".webp": (FileCategory.IMAGE, "image/webp"),
        ".bmp": (FileCategory.IMAGE, "image/bmp"),
        ".tiff": (FileCategory.IMAGE, "image/tiff"),
        ".tif": (FileCategory.IMAGE, "image/tiff"),
        ".svg": (FileCategory.IMAGE, "image/svg+xml"),
        # Audio
        ".mp3": (FileCategory.AUDIO, "audio/mpeg"),
        ".wav": (FileCategory.AUDIO, "audio/wav"),
        ".flac": (FileCategory.AUDIO, "audio/flac"),
        ".ogg": (FileCategory.AUDIO, "audio/ogg"),
        ".m4a": (FileCategory.AUDIO, "audio/mp4"),
        ".aac": (FileCategory.AUDIO, "audio/aac"),
        # Video
        ".mp4": (FileCategory.VIDEO, "video/mp4"),
        ".mkv": (FileCategory.VIDEO, "video/x-matroska"),
        ".mov": (FileCategory.VIDEO, "video/quicktime"),
        ".avi": (FileCategory.VIDEO, "video/x-msvideo"),
        ".webm": (FileCategory.VIDEO, "video/webm"),
        # Archive
        ".zip": (FileCategory.ARCHIVE, "application/zip"),
        ".tar": (FileCategory.ARCHIVE, "application/x-tar"),
        ".gz": (FileCategory.ARCHIVE, "application/gzip"),
        ".tgz": (FileCategory.ARCHIVE, "application/gzip"),
        ".bz2": (FileCategory.ARCHIVE, "application/x-bzip2"),
        ".xz": (FileCategory.ARCHIVE, "application/x-xz"),
        ".7z": (FileCategory.ARCHIVE, "application/x-7z-compressed"),
        # Binary
        ".exe": (FileCategory.BINARY, "application/x-msdownload"),
        ".dll": (FileCategory.BINARY, "application/x-msdownload"),
        ".so": (FileCategory.BINARY, "application/x-sharedlib"),
        ".dylib": (FileCategory.BINARY, "application/x-mach-binary"),
        ".bin": (FileCategory.BINARY, "application/octet-stream"),
        ".class": (FileCategory.BINARY, "application/x-java-class"),
        ".wasm": (FileCategory.BINARY, "application/wasm"),
        ".o": (FileCategory.BINARY, "application/x-object"),
    }

    def __init__(self, sandbox: FilesystemSandbox):
        self.sandbox = sandbox

    @staticmethod
    def compute_sha256(path: Path, max_bytes: Optional[int] = None) -> str:
        """Computes SHA-256 hash using bounded 64KB chunk streaming."""
        hasher = hashlib.sha256()
        bytes_read = 0
        chunk_size = 65536
        with open(path, "rb") as f:
            while True:
                to_read = chunk_size
                if max_bytes is not None:
                    remaining = max_bytes - bytes_read
                    if remaining <= 0:
                        break
                    to_read = min(chunk_size, remaining)
                chunk = f.read(to_read)
                if not chunk:
                    break
                hasher.update(chunk)
                bytes_read += len(chunk)
        return hasher.hexdigest()

    def detect_magic(self, path: Path) -> Optional[Tuple[str, FileCategory]]:
        """Reads the first 512 bytes of a file to check magic signatures."""
        try:
            with open(path, "rb") as f:
                header = f.read(512)
        except Exception:
            return None

        if not header:
            return None

        # Check MP4 / MOV ftyp box
        if len(header) >= 12 and header[4:8] == b"ftyp":
            brand = header[8:12]
            if brand in (b"M4A ", b"M4B "):
                return ("audio/mp4", FileCategory.AUDIO)
            return ("video/mp4", FileCategory.VIDEO)

        # Check TAR archive (ustar at offset 257)
        if len(header) >= 262 and header[257:262] == b"ustar":
            return ("application/x-tar", FileCategory.ARCHIVE)

        # Check signatures
        for item in self.MAGIC_SIGNATURES:
            sig = item[0]
            mime = item[1]
            cat = item[2]
            if len(item) == 5:
                sub_offset = item[3]
                sub_sig = item[4]
                if header.startswith(sig) and len(header) >= sub_offset + len(sub_sig):
                    if header[sub_offset:sub_offset + len(sub_sig)] == sub_sig:
                        return (mime, cat)
            else:
                if header.startswith(sig):
                    return (mime, cat)

        # Check XML declaration
        if header.lstrip().startswith(b"<?xml"):
            return ("application/xml", FileCategory.STRUCTURED)

        return None

    def is_text_like(self, path: Path) -> bool:
        """Determines if a file contains predominantly printable text characters."""
        try:
            with open(path, "rb") as f:
                sample = f.read(1024)
            if not sample:
                return True
            # Non-text check: look for null bytes or excessive non-printable chars
            if b"\x00" in sample:
                return False
            # Check printable ascii or valid utf-8
            try:
                sample.decode("utf-8")
                return True
            except UnicodeDecodeError:
                return False
        except Exception:
            return False

    def identify(self, relative_or_absolute_path: Path | str, compute_hash: bool = False) -> FileInfo:
        """
        Performs multi-tier file identification and returns FileInfo.
        """
        resolved_path = self.sandbox.validate_and_resolve(relative_or_absolute_path, must_exist=True)
        rel_path = self.sandbox.relative_path(resolved_path)
        stat = resolved_path.stat()
        
        filename = resolved_path.name
        ext = resolved_path.suffix.lower()
        size_bytes = stat.st_size
        mtime = stat.st_mtime
        mtime_ns = getattr(stat, "st_mtime_ns", int(mtime * 1e9))
        ctime = stat.st_ctime

        file_id = hashlib.sha1(f"{rel_path}:{size_bytes}:{mtime_ns}".encode("utf-8")).hexdigest()[:16]

        if resolved_path.is_dir():
            return FileInfo(
                file_id=file_id,
                canonical_path=str(resolved_path),
                relative_path=rel_path,
                filename=filename,
                extension="",
                mime_type="inode/directory",
                category=FileCategory.DIRECTORY,
                size_bytes=0,
                mtime=mtime,
                mtime_ns=mtime_ns,
                ctime=ctime,
                processing_status=ProcessingStatus.SKIPPED,
                status_reason="Directory"
            )

        # 1. Magic bytes detection
        magic_match = self.detect_magic(resolved_path)
        
        # 2. Extension map lookup
        ext_match = self.EXTENSIONS_CATEGORY_MAP.get(ext)

        # 3. Standard mimetypes guess
        guess_mime, _ = mimetypes.guess_type(str(resolved_path))

        # Reconcile category & mime
        category = FileCategory.UNKNOWN
        mime_type = "application/octet-stream"

        if magic_match:
            mime_type, category = magic_match
            # Special case: ZIP magic also matches DOCX/XLSX/JAR/etc.
            if ext_match and magic_match[1] == FileCategory.ARCHIVE and ext_match[0] != FileCategory.ARCHIVE:
                category = ext_match[0]
                mime_type = ext_match[1]
        elif ext_match:
            category, mime_type = ext_match
        elif guess_mime:
            mime_type = guess_mime
            if mime_type.startswith("text/"):
                category = FileCategory.DOCUMENT
            elif mime_type.startswith("image/"):
                category = FileCategory.IMAGE
            elif mime_type.startswith("audio/"):
                category = FileCategory.AUDIO
            elif mime_type.startswith("video/"):
                category = FileCategory.VIDEO
        elif self.is_text_like(resolved_path):
            category = FileCategory.DOCUMENT
            mime_type = "text/plain"

        sha256_hash = None
        if compute_hash and resolved_path.is_file():
            sha256_hash = self.compute_sha256(resolved_path)

        return FileInfo(
            file_id=file_id,
            canonical_path=str(resolved_path),
            relative_path=rel_path,
            filename=filename,
            extension=ext,
            mime_type=mime_type,
            category=category,
            size_bytes=size_bytes,
            mtime=mtime,
            mtime_ns=mtime_ns,
            ctime=ctime,
            sha256=sha256_hash,
            processing_status=ProcessingStatus.PENDING
        )

