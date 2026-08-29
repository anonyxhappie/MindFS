"""Archive intelligence processor with zip-bomb protection and manifest inspection."""

from pathlib import Path
import tarfile
from typing import Any, Dict, List, Tuple
import zipfile

from mindfs.artifacts.models import SemanticArtifact
from mindfs.config.settings import MindFSConfig
from mindfs.identification.models import FileCategory, FileInfo
from mindfs.processors.base import FileProcessor


class ArchiveProcessor(FileProcessor):
    """Safely inspects archive structures, files, sizes, and extensions without extraction."""

    name: str = "archive"
    version: str = "1.0.0"
    supported_categories: List[FileCategory] = [FileCategory.ARCHIVE]
    supported_mimes: List[str] = [
        "application/zip",
        "application/x-tar",
        "application/gzip",
        "application/x-bzip2",
        "application/x-xz",
        "application/x-7z-compressed",
    ]
    supported_extensions: List[str] = [
        ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z"
    ]

    def _inspect_zip(self, path: Path) -> Dict[str, Any]:
        members = []
        total_uncompressed_size = 0
        ext_dist: Dict[str, int] = {}
        nested_archives = 0
        archive_bomb_warning = False

        max_members = self.config.archive.max_members
        max_size_bytes = int(self.config.archive.max_expanded_size_mb * 1024 * 1024)
        max_member_bytes = int(self.config.archive.max_member_size_mb * 1024 * 1024)

        with zipfile.ZipFile(str(path), "r") as zf:
            infolist = zf.infolist()
            for info in infolist:
                if len(members) >= max_members:
                    archive_bomb_warning = True
                    break
                
                size = info.file_size
                if size > max_member_bytes:
                    archive_bomb_warning = True

                total_uncompressed_size += size
                if total_uncompressed_size > max_size_bytes:
                    archive_bomb_warning = True
                    break

                ext = Path(info.filename).suffix.lower()
                ext_dist[ext or "none"] = ext_dist.get(ext or "none", 0) + 1
                if ext in (".zip", ".tar", ".gz", ".tgz", ".bz2", ".7z"):
                    nested_archives += 1

                members.append({
                    "filename": info.filename,
                    "size_bytes": size,
                    "is_dir": info.is_dir(),
                })

        return {
            "format": "zip",
            "member_count": len(members),
            "total_uncompressed_bytes": total_uncompressed_size,
            "extension_distribution": ext_dist,
            "nested_archives_count": nested_archives,
            "members_sample": members[:25],
            "archive_bomb_warning": archive_bomb_warning,
        }

    def _inspect_tar(self, path: Path) -> Dict[str, Any]:
        members = []
        total_uncompressed_size = 0
        ext_dist: Dict[str, int] = {}
        nested_archives = 0
        archive_bomb_warning = False

        max_members = self.config.archive.max_members
        max_size_bytes = int(self.config.archive.max_expanded_size_mb * 1024 * 1024)

        mode = "r:*"
        with tarfile.open(str(path), mode) as tf:
            for member in tf:
                if len(members) >= max_members:
                    archive_bomb_warning = True
                    break

                size = member.size
                total_uncompressed_size += size
                if total_uncompressed_size > max_size_bytes:
                    archive_bomb_warning = True
                    break

                ext = Path(member.name).suffix.lower()
                ext_dist[ext or "none"] = ext_dist.get(ext or "none", 0) + 1
                if ext in (".zip", ".tar", ".gz", ".tgz", ".bz2", ".7z"):
                    nested_archives += 1

                members.append({
                    "filename": member.name,
                    "size_bytes": size,
                    "is_dir": member.isdir(),
                })

        return {
            "format": "tar",
            "member_count": len(members),
            "total_uncompressed_bytes": total_uncompressed_size,
            "extension_distribution": ext_dist,
            "nested_archives_count": nested_archives,
            "members_sample": members[:25],
            "archive_bomb_warning": archive_bomb_warning,
        }

    def inspect(self, file_info: FileInfo) -> Dict[str, Any]:
        path = Path(file_info.canonical_path)
        try:
            if zipfile.is_zipfile(str(path)):
                return self._inspect_zip(path)
            elif tarfile.is_tarfile(str(path)):
                return self._inspect_tar(path)
        except Exception as exc:
            return {"error": f"Failed to inspect archive: {str(exc)}"}
        return {"format": file_info.extension, "size_bytes": file_info.size_bytes}

    def extract(self, file_info: FileInfo) -> List[SemanticArtifact]:
        path = Path(file_info.canonical_path)
        meta = {}
        try:
            if zipfile.is_zipfile(str(path)):
                meta = self._inspect_zip(path)
            elif tarfile.is_tarfile(str(path)):
                meta = self._inspect_tar(path)
        except Exception as exc:
            meta = {"error": str(exc), "format": file_info.extension}

        member_count = meta.get("member_count", 0)
        uncomp_size = meta.get("total_uncompressed_bytes", 0)
        ext_dist = meta.get("extension_distribution", {})
        samples = meta.get("members_sample", [])
        bomb_warn = meta.get("archive_bomb_warning", False)

        lines = [
            f"Archive Manifest: {file_info.filename}",
            f"Format: {meta.get('format', file_info.extension)}",
            f"Total Contained Files/Folders: {member_count}",
            f"Expanded Size: {round(uncomp_size / (1024*1024), 2)} MB ({uncomp_size} bytes)",
            f"Nested Archives: {meta.get('nested_archives_count', 0)}",
        ]

        if bomb_warn:
            lines.append("WARNING: Potential archive bomb or large expansion limits exceeded.")

        if ext_dist:
            lines.append("File Type Distribution:")
            for ext, count in ext_dist.items():
                lines.append(f"  - {ext}: {count} files")

        if samples:
            lines.append("\nContained Member Files (Sample):")
            for m in samples:
                kind = "dir" if m.get("is_dir") else "file"
                lines.append(f"  - [{kind}] {m.get('filename')} ({m.get('size_bytes', 0)} bytes)")

        text_content = "\n".join(lines)
        summary = f"Archive '{file_info.filename}' containing {member_count} members ({round(uncomp_size/(1024*1024), 2)} MB expanded)."

        art = SemanticArtifact(
            file_id=file_info.file_id,
            artifact_type="archive_manifest",
            source_path=file_info.canonical_path,
            source_offset={"member_count": member_count},
            text=text_content,
            summary=summary,
            metadata=meta,
            entities=[m.get("filename") for m in samples[:15]],
            processor=self.name,
            processor_version=self.version,
        )

        return [art]

