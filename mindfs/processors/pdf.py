import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List
import pypdf

from mindfs.artifacts.models import SemanticArtifact
from mindfs.config.settings import MindFSConfig
from mindfs.identification.models import FileCategory, FileInfo
from mindfs.processors.base import FileProcessor


class PDFProcessor(FileProcessor):
    """Processes PDF files page-by-page, preserving page numbers and provenance."""

    name: str = "pdf"
    version: str = "1.0.0"
    supported_categories: List[FileCategory] = []
    supported_mimes: List[str] = ["application/pdf"]
    supported_extensions: List[str] = [".pdf"]

    def can_handle(self, file_info: FileInfo) -> bool:
        return file_info.mime_type == "application/pdf" or file_info.extension.lower() == ".pdf"

    def _run_ocr_on_images(self, page: pypdf.PageObject) -> str:
        """Extracts text from images on a PDF page using tesseract if available."""
        if not self.config.media.enable_ocr:
            return ""
        tesseract_bin = shutil.which("tesseract")
        if not tesseract_bin:
            return ""

        ocr_parts = []
        try:
            images = list(page.images)
            for idx, img_obj in enumerate(images[:5]):
                if len(img_obj.data) < 1000:
                    continue
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(img_obj.data)
                    tmp_path = Path(tmp.name)
                try:
                    res = subprocess.run(
                        [tesseract_bin, str(tmp_path), "stdout", "--oem", "1", "-l", "eng"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=False,
                    )
                    if res.returncode == 0 and res.stdout.strip():
                        ocr_parts.append(res.stdout.strip())
                finally:
                    tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

        return "\n".join(ocr_parts)

    def inspect(self, file_info: FileInfo) -> Dict[str, Any]:
        path = Path(file_info.canonical_path)
        try:
            reader = pypdf.PdfReader(str(path))
            num_pages = len(reader.pages)
            meta = reader.metadata
            pdf_info = {
                "pages": num_pages,
                "title": getattr(meta, "title", None) or "",
                "author": getattr(meta, "author", None) or "",
                "creator": getattr(meta, "creator", None) or "",
                "is_encrypted": reader.is_encrypted,
            }
            if num_pages > 0:
                first_page_text = reader.pages[0].extract_text() or ""
                pdf_info["page_1_preview"] = first_page_text[:300].strip()
            return pdf_info
        except Exception as exc:
            return {"error": f"Failed to inspect PDF: {str(exc)}"}

    def extract(self, file_info: FileInfo) -> List[SemanticArtifact]:
        path = Path(file_info.canonical_path)
        artifacts: List[SemanticArtifact] = []

        try:
            reader = pypdf.PdfReader(str(path))
        except Exception as exc:
            raise RuntimeError(f"Cannot open PDF: {str(exc)}")

        num_pages = len(reader.pages)
        max_file_size_bytes = int(self.config.index.max_file_size_mb * 1024 * 1024)

        if file_info.size_bytes > max_file_size_bytes:
            pages_to_process = min(num_pages, 5)
        else:
            pages_to_process = num_pages

        for page_idx in range(pages_to_process):
            page_num = page_idx + 1
            try:
                page = reader.pages[page_idx]
                page_text = page.extract_text() or ""
            except Exception as page_exc:
                page_text = f"[Error reading page {page_num}: {str(page_exc)}]"

            clean_text = page_text.strip()
            if len(clean_text) < 30:
                ocr_text = self._run_ocr_on_images(page)
                if ocr_text:
                    clean_text = ocr_text

            is_image_only = len(clean_text) < 20

            summary = f"Page {page_num} of {file_info.filename}"
            if clean_text:
                summary += f": {clean_text[:120]}..."

            art = SemanticArtifact(
                file_id=file_info.file_id,
                artifact_type="pdf_page",
                source_path=file_info.canonical_path,
                source_offset={"page": page_num, "total_pages": num_pages},
                text=clean_text if clean_text else f"[Image-only or empty page {page_num}]",
                summary=summary,
                metadata={
                    "page": page_num,
                    "total_pages": num_pages,
                    "is_image_only": is_image_only,
                    "ocr_applied": bool(clean_text and len(page_text.strip()) < 30),
                },
                entities=[],
                processor=self.name,
                processor_version=self.version,
            )
            artifacts.append(art)

        return artifacts
