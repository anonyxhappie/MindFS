"""Structured data processor (JSON, YAML, XML, CSV, TSV, TOML)."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
import yaml

from mindfs.artifacts.models import SemanticArtifact
from mindfs.config.settings import MindFSConfig
from mindfs.identification.models import FileCategory, FileInfo
from mindfs.processors.base import FileProcessor


class StructuredDataProcessor(FileProcessor):
    """Processes structured formats into schema-aware, searchable representations."""

    name: str = "structured"
    version: str = "1.0.0"
    supported_categories: List[FileCategory] = [FileCategory.STRUCTURED]
    supported_mimes: List[str] = [
        "application/json",
        "application/yaml",
        "application/xml",
        "text/csv",
        "text/tab-separated-values",
        "application/toml",
    ]
    supported_extensions: List[str] = [
        ".json", ".yaml", ".yml", ".xml", ".csv", ".tsv", ".toml"
    ]

    def _inspect_csv(self, path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            sample_rows = []
            row_count = 0
            for i, row in enumerate(reader):
                if i < 5:
                    sample_rows.append(row)
                row_count += 1
            return {
                "format": "csv",
                "columns": header,
                "column_count": len(header),
                "sample_row_count": len(sample_rows),
                "total_rows_estimate": row_count,
                "sample_rows": sample_rows,
            }

    def _inspect_json(self, path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            # Read first 32KB to inspect structure safely
            sample = f.read(32768)
        try:
            data = json.loads(sample)
            if isinstance(data, dict):
                return {
                    "format": "json_object",
                    "top_keys": list(data.keys())[:20],
                    "total_keys": len(data.keys()),
                }
            elif isinstance(data, list):
                return {
                    "format": "json_array",
                    "array_length": len(data),
                    "first_element_type": type(data[0]).__name__ if data else "empty",
                }
        except Exception:
            pass
        return {"format": "json", "preview": sample[:300]}

    def inspect(self, file_info: FileInfo) -> Dict[str, Any]:
        path = Path(file_info.canonical_path)
        ext = file_info.extension.lower()
        if ext in (".csv", ".tsv"):
            return self._inspect_csv(path)
        elif ext == ".json":
            return self._inspect_json(path)
        return {"format": ext.lstrip("."), "size_bytes": file_info.size_bytes}

    def _extract_csv(self, file_info: FileInfo, path: Path) -> SemanticArtifact:
        delimiter = "\t" if file_info.extension.lower() == ".tsv" else ","
        header: List[str] = []
        samples: List[List[str]] = []
        row_count = 0
        numeric_stats: Dict[str, Dict[str, float]] = {}

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter=delimiter)
            try:
                header = next(reader, [])
            except StopIteration:
                header = []

            for col in header:
                numeric_stats[col] = {"min": float("inf"), "max": float("-inf"), "count": 0}

            for row in reader:
                row_count += 1
                if len(samples) < 10:
                    samples.append(row)
                
                # Bounded stats on first 5000 rows
                if row_count <= 5000:
                    for i, val in enumerate(row):
                        if i < len(header):
                            col_name = header[i]
                            try:
                                num = float(val.strip())
                                numeric_stats[col_name]["min"] = min(numeric_stats[col_name]["min"], num)
                                numeric_stats[col_name]["max"] = max(numeric_stats[col_name]["max"], num)
                                numeric_stats[col_name]["count"] += 1
                            except (ValueError, TypeError):
                                pass

        lines = [
            f"CSV Document: {file_info.filename}",
            f"Columns ({len(header)}): {', '.join(header)}",
            f"Estimated Rows: {row_count}",
            "",
            "Column Statistics:",
        ]
        for col, st in numeric_stats.items():
            if st["count"] > 0:
                lines.append(f"  - {col}: Numeric [Min: {st['min']}, Max: {st['max']}, Valid: {st['count']}]")
            else:
                lines.append(f"  - {col}: Text / Non-numeric")

        lines.append("")
        lines.append("Sample Data:")
        for idx, srow in enumerate(samples, 1):
            row_repr = ", ".join(f"{h}={v}" for h, v in zip(header, srow))
            lines.append(f"  Row {idx}: {row_repr}")

        text_content = "\n".join(lines)
        summary = f"CSV dataset '{file_info.filename}' with {len(header)} columns and ~{row_count} rows. Columns: {', '.join(header[:8])}"

        return SemanticArtifact(
            file_id=file_info.file_id,
            artifact_type="structured_csv",
            source_path=file_info.canonical_path,
            source_offset={"row_count": row_count, "columns": header},
            text=text_content,
            summary=summary,
            metadata={"columns": header, "row_count": row_count},
            entities=header[:30],
            processor=self.name,
            processor_version=self.version,
        )

    def _extract_json_yaml(self, file_info: FileInfo, path: Path) -> SemanticArtifact:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw_content = f.read(int(self.config.index.max_file_size_mb * 1024 * 1024))

        try:
            if file_info.extension.lower() == ".json":
                parsed = json.loads(raw_content)
            else:
                parsed = yaml.safe_load(raw_content)
        except Exception:
            parsed = None

        lines = [f"Structured File: {file_info.filename}"]
        entities: List[str] = []

        if isinstance(parsed, dict):
            keys = list(parsed.keys())
            entities.extend(keys)
            lines.append(f"Root Type: Object (Keys: {len(keys)})")
            lines.append(f"Top-level keys: {', '.join(keys)}")
            lines.append("\nKey-Value Summary:")
            for k, v in list(parsed.items())[:25]:
                v_str = str(v)
                if len(v_str) > 150:
                    v_str = v_str[:150] + "..."
                lines.append(f"  - {k}: {v_str}")
        elif isinstance(parsed, list):
            lines.append(f"Root Type: Array (Items: {len(parsed)})")
            if parsed and isinstance(parsed[0], dict):
                sub_keys = list(parsed[0].keys())
                entities.extend(sub_keys)
                lines.append(f"Array element schema keys: {', '.join(sub_keys)}")
            lines.append("\nSample elements:")
            for idx, item in enumerate(parsed[:5], 1):
                lines.append(f"  Item {idx}: {str(item)[:200]}")
        else:
            lines.append(raw_content[:2000])

        text_content = "\n".join(lines)
        summary = f"Structured document '{file_info.filename}' containing {len(entities)} identified keys/fields."

        return SemanticArtifact(
            file_id=file_info.file_id,
            artifact_type="structured_data",
            source_path=file_info.canonical_path,
            source_offset=None,
            text=text_content,
            summary=summary,
            metadata={"entities_count": len(entities)},
            entities=entities[:30],
            processor=self.name,
            processor_version=self.version,
        )

    def _extract_xml(self, file_info: FileInfo, path: Path) -> SemanticArtifact:
        try:
            tree = ET.parse(str(path))
            root = tree.getroot()
            tag_counts: Dict[str, int] = {}
            for elem in root.iter():
                tag_counts[elem.tag] = tag_counts.get(elem.tag, 0) + 1

            top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:15]
            lines = [
                f"XML Document: {file_info.filename}",
                f"Root Element: <{root.tag}>",
                f"Distinct Tags ({len(tag_counts)}):",
            ]
            for tag, count in top_tags:
                lines.append(f"  - <{tag}>: {count} occurrences")

            text_content = "\n".join(lines)
            summary = f"XML document '{file_info.filename}' with root tag <{root.tag}> and {len(tag_counts)} tags."

            return SemanticArtifact(
                file_id=file_info.file_id,
                artifact_type="structured_xml",
                source_path=file_info.canonical_path,
                source_offset={"root_tag": root.tag},
                text=text_content,
                summary=summary,
                metadata={"root_tag": root.tag, "tag_counts": tag_counts},
                entities=[t for t, _ in top_tags],
                processor=self.name,
                processor_version=self.version,
            )
        except Exception as exc:
            return self._extract_json_yaml(file_info, path)

    def extract(self, file_info: FileInfo) -> List[SemanticArtifact]:
        path = Path(file_info.canonical_path)
        ext = file_info.extension.lower()
        if ext in (".csv", ".tsv"):
            return [self._extract_csv(file_info, path)]
        elif ext == ".xml":
            return [self._extract_xml(file_info, path)]
        else:
            return [self._extract_json_yaml(file_info, path)]

