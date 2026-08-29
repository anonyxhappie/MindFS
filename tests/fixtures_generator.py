"""Deterministic test corpus fixture generator for MindFS tests."""

import io
import json
import os
from pathlib import Path
import struct
import wave
import zipfile
from typing import Any, Dict
from PIL import Image, ImageDraw
import pypdf
import yaml


def generate_test_corpus(base_dir: Path) -> Dict[str, Path]:
    """Generates all deterministic test files required by Section 39."""
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    created: Dict[str, Path] = {}

    # 1. plain.txt
    p_txt = base_dir / "plain.txt"
    p_txt.write_text("MindFS is a local privacy-preserving filesystem intelligence engine.\nIt operates under 2GB RAM budget.", encoding="utf-8")
    created["plain.txt"] = p_txt

    # 2. markdown.md
    p_md = base_dir / "markdown.md"
    p_md.write_text("# Project Apollo\n\nDatabase migration planned for Q3.\nPostgreSQL will be the primary database backend.\n\n## Team\nLead: Alice\nSecurity: Bob\n", encoding="utf-8")
    created["markdown.md"] = p_md

    # 3. large.txt (6 MB to test oversized boundary)
    p_large = base_dir / "large.txt"
    with open(p_large, "w", encoding="utf-8") as f:
        chunk = "This is a large file line testing bounded chunking and memory limits in MindFS.\n" * 50
        # Write ~5.5 MB
        for _ in range(1400):
            f.write(chunk)
    created["large.txt"] = p_large

    # 4. sample.pdf
    p_pdf = base_dir / "sample.pdf"
    writer = pypdf.PdfWriter()
    page1 = writer.add_blank_page(width=612, height=792)
    # Write synthetic PDF with PDF page text stream
    # Note: creating a clean multi-page PDF with annotations/text
    stream_content = b"BT /F1 12 Tf 72 712 Td (Project Apollo Architecture Report Page 1) Tj ET"
    p_pdf.write_bytes(
        b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length 75 >>\nstream\n"
        + stream_content +
        b"\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000266 00000 n \ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n392\n%%EOF\n"
    )
    created["sample.pdf"] = p_pdf

    # 5. image.jpg
    p_jpg = base_dir / "image.jpg"
    img = Image.new("RGB", (200, 200), color=(73, 109, 137))
    img.save(p_jpg, "JPEG")
    created["image.jpg"] = p_jpg

    # 6. image_with_text.png
    p_png = base_dir / "image_with_text.png"
    img_png = Image.new("RGB", (300, 100), color=(255, 255, 255))
    d = ImageDraw.Draw(img_png)
    d.text((20, 40), "MindFS OCR Engine Test", fill=(0, 0, 0))
    img_png.save(p_png, "PNG")
    created["image_with_text.png"] = p_png

    # 7. audio.wav
    p_wav = base_dir / "audio.wav"
    with wave.open(str(p_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        # 1 second of silence/sine
        data = struct.pack("<h", 0) * 16000
        wf.writeframes(data)
    created["audio.wav"] = p_wav

    # 8. video.mp4 (synthetic MP4 container header)
    p_mp4 = base_dir / "video.mp4"
    # Write valid MP4 header box (ftypisom)
    mp4_bytes = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41\x00\x00\x00\x08free\x00\x00\x00\x10mdat\x00\x00\x00\x00"
    p_mp4.write_bytes(mp4_bytes)
    created["video.mp4"] = p_mp4

    # 9. data.csv
    p_csv = base_dir / "data.csv"
    p_csv.write_text(
        "customer_id,name,plan,monthly_fee,status\n"
        "101,Acme Corp,Enterprise,499.00,Active\n"
        "102,Globex,Standard,99.00,Active\n"
        "103,Soylent,Starter,29.00,Churned\n"
        "104,Initech,Enterprise,499.00,Active\n",
        encoding="utf-8"
    )
    created["data.csv"] = p_csv

    # 10. data.json
    p_json = base_dir / "data.json"
    json_obj = {
        "service": "MindFS Server",
        "version": "2.0.0",
        "clusters": ["us-east-1", "eu-west-1"],
        "metrics": {"requests": 45000, "latency_ms": 12.4},
        "tags": ["production", "filesystem", "ai"]
    }
    p_json.write_text(json.dumps(json_obj, indent=2), encoding="utf-8")
    created["data.json"] = p_json

    # 11. data.yaml
    p_yaml = base_dir / "data.yaml"
    p_yaml.write_text("database:\n  host: db.internal\n  port: 5432\n  name: mindfs_db\n  ssl: true\n", encoding="utf-8")
    created["data.yaml"] = p_yaml

    # 12. data.xml
    p_xml = base_dir / "data.xml"
    p_xml.write_text('<?xml version="1.0" encoding="UTF-8"?>\n<catalog>\n  <book id="bk101">\n    <author>Gambardella, Matthew</author>\n    <title>XML Developer\'s Guide</title>\n    <price>44.95</price>\n  </book>\n</catalog>', encoding="utf-8")
    created["data.xml"] = p_xml

    # 13. archive.zip
    p_zip = base_dir / "archive.zip"
    with zipfile.ZipFile(p_zip, "w") as zf:
        zf.writestr("internal_report.txt", "Quarterly audit results: All systems compliant.")
        zf.writestr("config.ini", "[server]\nport=8080\n")
    created["archive.zip"] = p_zip

    # 14. nested_archive.zip
    p_nested_zip = base_dir / "nested_archive.zip"
    with zipfile.ZipFile(p_nested_zip, "w") as zf:
        zf.write(p_zip, arcname="inner.zip")
        zf.writestr("readme.txt", "Outer archive containing inner.zip")
    created["nested_archive.zip"] = p_nested_zip

    # 15. sample_elf (ELF64 x86_64 synthetic binary)
    p_elf = base_dir / "sample_elf"
    # ELF Magic + 64bit + little endian + EV_CURRENT + x86_64 arch (0x3E)
    elf_hdr = bytearray(b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00")
    elf_hdr += struct.pack("<H", 2)  # ET_EXEC
    elf_hdr += struct.pack("<H", 0x3E)  # x86_64
    elf_hdr += b"\x00" * 40
    elf_hdr += b"/usr/lib/libc.so.6\x00SSL_CTX_new\x00Version 1.2.3\x00"
    p_elf.write_bytes(bytes(elf_hdr))
    created["sample_elf"] = p_elf

    # 16. empty.file
    p_empty = base_dir / "empty.file"
    p_empty.write_bytes(b"")
    created["empty.file"] = p_empty

    # 17. unknown.dat
    p_unk = base_dir / "unknown.dat"
    p_unk.write_bytes(bytes([x % 256 for x in range(500)]))
    created["unknown.dat"] = p_unk

    # 18. symlink
    p_symlink = base_dir / "plain_symlink.txt"
    if p_symlink.exists() or p_symlink.is_symlink():
        p_symlink.unlink()
    p_symlink.symlink_to(p_txt.name)
    created["plain_symlink.txt"] = p_symlink

    return created
