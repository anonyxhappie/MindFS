"""Binary intelligence processor for inspecting compiled executables, bytecode, and object files without execution."""

import os
from pathlib import Path
import re
import struct
from typing import Any, Dict, List, Optional

from mindfs.artifacts.models import SemanticArtifact
from mindfs.config.settings import MindFSConfig
from mindfs.identification.models import FileCategory, FileInfo
from mindfs.processors.base import FileProcessor


class BinaryProcessor(FileProcessor):
    """Safely extracts headers, architecture, endianness, and interesting strings from binaries."""

    name: str = "binary"
    version: str = "1.0.0"
    supported_categories: List[FileCategory] = [FileCategory.BINARY]
    supported_mimes: List[str] = [
        "application/x-executable",
        "application/x-mach-binary",
        "application/x-dosexec",
        "application/x-sharedlib",
        "application/x-java-class",
        "application/wasm",
        "application/x-object",
        "application/octet-stream",
    ]
    supported_extensions: List[str] = [
        ".exe", ".dll", ".so", ".dylib", ".bin", ".class", ".wasm", ".o"
    ]

    def _extract_printable_strings(self, path: Path, max_bytes: int = 131072, min_len: int = 4, max_strings: int = 50) -> List[str]:
        """Extracts bounded printable strings from the binary head."""
        interesting = []
        try:
            with open(path, "rb") as f:
                raw = f.read(max_bytes)
            
            pattern = re.compile(rb"[\x20-\x7e]{" + str(min_len).encode() + rb",}")
            matches = pattern.findall(raw)
            for m in matches:
                try:
                    s = m.decode("ascii")
                    # Filter for interesting strings (paths, URLs, function-like names, versions)
                    if any(c in s for c in ("/", ".", "_", ":", "-", "http", "lib", "version", "API", "Error")):
                        if len(s) < 120:
                            interesting.append(s)
                            if len(interesting) >= max_strings:
                                break
                except Exception:
                    continue
        except Exception:
            pass
        return interesting

    def _parse_elf_header(self, raw: bytes) -> Dict[str, Any]:
        """Parses ELF 32/64 bit header."""
        if len(raw) < 52 or not raw.startswith(b"\x7fELF"):
            return {}
        
        ei_class = raw[4]  # 1 = 32-bit, 2 = 64-bit
        ei_data = raw[5]   # 1 = Little-endian, 2 = Big-endian
        is_64 = (ei_class == 2)
        endian = "<" if ei_data == 1 else ">"
        
        arch_code = struct.unpack(f"{endian}H", raw[18:20])[0] if len(raw) >= 20 else 0
        arch_map = {
            0x03: "x86",
            0x3E: "x86_64 / AMD64",
            0x28: "ARM",
            0xB7: "AArch64 (ARM64)",
            0xF3: "RISC-V",
        }
        
        return {
            "binary_format": "ELF",
            "bitness": 64 if is_64 else 32,
            "endianness": "little-endian" if ei_data == 1 else "big-endian",
            "architecture": arch_map.get(arch_code, f"Unknown ({hex(arch_code)})"),
        }

    def _parse_macho_header(self, raw: bytes) -> Dict[str, Any]:
        """Parses Mach-O 32/64 bit / Universal header."""
        if len(raw) < 8:
            return {}
        magic = struct.unpack(">I", raw[:4])[0]
        
        if magic in (0xFEEDFACE, 0xCEFAEDFE):
            return {"binary_format": "Mach-O", "bitness": 32, "endianness": "big" if magic == 0xFEEDFACE else "little", "architecture": "32-bit Mach-O"}
        elif magic in (0xFEEDFACF, 0xCFFAEDFE):
            cputype = struct.unpack("<I" if magic == 0xCFFAEDFE else ">I", raw[4:8])[0]
            arch = "ARM64" if cputype == (0x01000000 | 12) else ("x86_64" if cputype == (0x01000000 | 7) else "64-bit Mach-O")
            return {"binary_format": "Mach-O", "bitness": 64, "endianness": "little" if magic == 0xCFFAEDFE else "big", "architecture": arch}
        elif magic == 0xCAFEBABE:
            # Java class or Mach-O Fat Binary
            if len(raw) >= 8 and raw[4:8] == b"\x00\x00\x00\x02":
                return {"binary_format": "Mach-O Universal Binary", "architecture": "Universal Fat Binary"}
            return {"binary_format": "Java Class Bytecode", "architecture": "JVM Bytecode"}
        return {}

    def _parse_pe_header(self, raw: bytes) -> Dict[str, Any]:
        """Parses Windows PE header."""
        if len(raw) < 64 or not raw.startswith(b"MZ"):
            return {}
        pe_offset = struct.unpack("<I", raw[60:64])[0]
        if len(raw) >= pe_offset + 6 and raw[pe_offset:pe_offset+4] == b"PE\x00\x00":
            machine = struct.unpack("<H", raw[pe_offset+4:pe_offset+6])[0]
            machine_map = {
                0x14C: "i386 (x86)",
                0x8664: "AMD64 (x64)",
                0xAA64: "ARM64",
                0x1C0: "ARM",
            }
            return {
                "binary_format": "PE (Portable Executable)",
                "architecture": machine_map.get(machine, f"Unknown ({hex(machine)})"),
            }
        return {"binary_format": "DOS Executable / PE Stub"}

    def _parse_wasm_header(self, raw: bytes) -> Dict[str, Any]:
        if raw.startswith(b"\x00asm"):
            version = struct.unpack("<I", raw[4:8])[0] if len(raw) >= 8 else 1
            return {"binary_format": "WebAssembly (WASM)", "version": version, "architecture": "WASM Virtual Machine"}
        return {}

    def _inspect_headers(self, path: Path) -> Dict[str, Any]:
        try:
            with open(path, "rb") as f:
                header = f.read(1024)
        except Exception as exc:
            return {"error": str(exc)}

        if header.startswith(b"\x7fELF"):
            return self._parse_elf_header(header)
        elif header.startswith((b"\xfe\xed\xfa", b"\xce\xfa\xed", b"\xcf\xfa\xed", b"\xca\xfe\xba")):
            return self._macho_or_class(header)
        elif header.startswith(b"MZ"):
            return self._parse_pe_header(header)
        elif header.startswith(b"\x00asm"):
            return self._parse_wasm_header(header)

        return {"binary_format": "Generic Binary", "architecture": "Unknown"}

    def _macho_or_class(self, raw: bytes) -> Dict[str, Any]:
        return self._parse_macho_header(raw)

    def inspect(self, file_info: FileInfo) -> Dict[str, Any]:
        path = Path(file_info.canonical_path)
        info = self._inspect_headers(path)
        info["size_bytes"] = file_info.size_bytes
        return info

    def extract(self, file_info: FileInfo) -> List[SemanticArtifact]:
        path = Path(file_info.canonical_path)
        header_info = self._inspect_headers(path)
        strings = self._extract_printable_strings(path)

        fmt = header_info.get("binary_format", "Generic Binary")
        arch = header_info.get("architecture", "Unknown")
        bitness = header_info.get("bitness", "")
        endian = header_info.get("endianness", "")

        lines = [
            f"Binary Inspection: {file_info.filename}",
            f"Format: {fmt}",
            f"Architecture: {arch}",
        ]
        if bitness:
            lines.append(f"Bitness: {bitness}-bit")
        if endian:
            lines.append(f"Endianness: {endian}")
        lines.append(f"Size: {round(file_info.size_bytes / (1024*1024), 2)} MB ({file_info.size_bytes} bytes)")

        if strings:
            lines.append(f"\nExtracted Identifier / Path Strings ({len(strings)}):")
            for s in strings:
                lines.append(f"  - {s}")

        text_content = "\n".join(lines)
        summary = f"Binary '{file_info.filename}' [{fmt} {arch}, {file_info.size_bytes} bytes] with {len(strings)} identifiable strings."

        metadata = {
            **header_info,
            "strings_count": len(strings),
            "size_bytes": file_info.size_bytes,
        }

        art = SemanticArtifact(
            file_id=file_info.file_id,
            artifact_type="binary_inspection",
            source_path=file_info.canonical_path,
            source_offset={"architecture": arch, "format": fmt},
            text=text_content,
            summary=summary,
            metadata=metadata,
            entities=strings[:15],
            processor=self.name,
            processor_version=self.version,
        )

        return [art]

