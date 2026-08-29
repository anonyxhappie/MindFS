# MindFS: Local Privacy-First Filesystem Intelligence Engine

**MindFS** is a privacy-preserving, offline filesystem intelligence engine designed to operate within a strict **2 GB RAM budget** (target peak RSS < 1.7 GB). It indexes, semanticizes, and retrieves information from documents, structured data, media, archives, and binaries using sandboxed access, modular lazy-loaded processors, persistent SQLite + FAISS dual-storage, and evidence-grounded RAG answers.

---

## Key Features

1. **Strict <2 GB RAM Budget**: Measured peak process RSS diagnostics on all operations (< 140 MB measured in benchmarks).
2. **Deterministic Workspace Sandboxing**: Path resolution and validation enforcing containment strictly within `WORKSPACE_ROOT`. All `../` traversals, absolute escapes, and external symlinks are rejected.
3. **9 Modular & Lazy-Loaded Processors**:
   - `TextProcessor`: Plain text, markdown, source code (`.py`, `.ts`, `.cpp`, `.rs`, `.go`, `.java`, etc.), logs.
   - `PDFProcessor`: Bounded page-by-page streaming with page provenance (`source: report.pdf, page: 17`).
   - `StructuredDataProcessor`: Schema, field, and statistic extraction for JSON, YAML, XML, and CSV/TSV without loading full files into memory.
   - `ImageProcessor`: Dimension, format, color mode, and EXIF extraction with optional OCR text extraction.
   - `AudioProcessor`: Codec, duration, sample rate, channels, and timestamped transcripts.
   - `VideoProcessor`: Container properties, fps, codecs, and sparse temporal keyframe sampling.
   - `ArchiveProcessor`: Safe inspection of ZIP, TAR, GZ with expansion bomb protection.
   - `BinaryProcessor`: Non-executable inspection of ELF, Mach-O, PE, WASM, Java bytecode with architecture and interesting strings.
   - `FallbackProcessor`: Byte entropy, printable ratio, and magic identification for unknown files.
4. **Persistent Dual-Store Architecture**:
   - **SQLite**: Authoritative store for `files`, `artifacts`, `chunks`, `processors`, `index_runs`, `errors`, and `diagnostics`.
   - **FAISS**: CPU vector index mapping vector IDs directly to SQLite chunks.
5. **Incremental Indexing**:
   - Change detection via fingerprint `(canonical_path, size, mtime_ns, sha256)`.
   - Skips unchanged files, re-indexes modified files, and purges deleted records.
6. **Evidence-Grounded Agent**:
   - Bounded action loop (max 10 steps).
   - Candidate retrieval (8-12) -> metadata filtering -> deduplication -> source diversity -> top 3-5 evidence items with exact citations (`[filename (page/timestamp) | Score]`).
   - Explicitly rejects hallucination when evidence is insufficient.

---

## Installation

```bash
# Clone the repository and navigate to MindFS
cd MindFS

# Create and activate Python virtual environment (Python 3.11+)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and MindFS in editable mode
pip install -e .
```

---

## CLI Usage

```bash
# Initialize a workspace
mindfs -w /path/to/workspace init

# List workspace contents safely
mindfs -w /path/to/workspace list

# Inspect technical metadata of a file
mindfs -w /path/to/workspace inspect report.pdf

# Incrementally index the workspace
mindfs -w /path/to/workspace index

# Search indexed semantic artifacts
mindfs -w /path/to/workspace search "Apollo database migration"

# Filter search by path or file type
mindfs -w /path/to/workspace search "monthly fee" --type-filter csv

# Ask evidence-grounded questions
mindfs -w /path/to/workspace ask "What database backend is planned for Project Apollo?"

# Check index statistics
mindfs -w /path/to/workspace status

# Check memory diagnostics and peak RSS
mindfs -w /path/to/workspace diagnostics

# Rebuild index from scratch
mindfs -w /path/to/workspace rebuild

# Remove a file from index
mindfs -w /path/to/workspace remove report.pdf
```

---

## Testing & Benchmarks

Run the complete 34-item automated test suite:
```bash
python -m pytest -v
```

Run the peak RSS and latency benchmarking suite across all 12 core operations:
```bash
python benchmarks/run_benchmarks.py
```