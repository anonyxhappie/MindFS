"""Workspace sandbox and path security validator for MindFS."""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union


class SandboxSecurityError(Exception):
    """Raised when a path breaches the workspace sandbox boundary."""
    pass


class PathNotFoundError(Exception):
    """Raised when a requested path does not exist."""
    pass


class FilesystemSandbox:
    """Enforces sandbox containment within configured workspace roots or allowed folders."""

    def __init__(self, workspace_root: Union[str, Path], allowed_roots: Optional[List[Union[str, Path]]] = None):
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        if not self.workspace_root.exists():
            self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.allowed_roots: List[Path] = [self.workspace_root]
        if allowed_roots:
            for r in allowed_roots:
                rp = Path(r).expanduser().resolve()
                if rp.exists() and rp not in self.allowed_roots:
                    self.allowed_roots.append(rp)

    def add_allowed_root(self, root: Union[str, Path]) -> Path:
        """Adds a directory to the allowed sandbox roots."""
        rp = Path(root).expanduser().resolve()
        if not rp.exists():
            rp.mkdir(parents=True, exist_ok=True)
        if rp not in self.allowed_roots:
            self.allowed_roots.append(rp)
        return rp

    def remove_allowed_root(self, root: Union[str, Path]) -> bool:
        """Removes a directory from allowed sandbox roots."""
        rp = Path(root).expanduser().resolve()
        if rp in self.allowed_roots and rp != self.workspace_root:
            self.allowed_roots.remove(rp)
            return True
        return False

    def get_allowed_roots(self) -> List[Path]:
        """Returns all registered sandbox roots."""
        return list(self.allowed_roots)

    def validate_and_resolve(self, relative_or_absolute_path: Union[str, Path], must_exist: bool = False) -> Path:
        """
        Resolves a path and verifies that it is strictly contained within at least one allowed root.
        Rejects symlinks pointing outside the allowed boundaries, parent directory traversals, etc.
        """
        raw_path = Path(relative_or_absolute_path)
        
        # If absolute path is provided, check against all allowed roots
        if raw_path.is_absolute():
            resolved = raw_path.resolve()
        else:
            resolved = (self.workspace_root / raw_path).resolve()

        real_path_str = os.path.realpath(str(resolved))

        # Check if realpath is within ANY of the allowed roots
        in_sandbox = False
        for root in self.allowed_roots:
            root_real_str = os.path.realpath(str(root))
            try:
                Path(real_path_str).relative_to(root_real_str)
                in_sandbox = True
                break
            except ValueError:
                continue

        if not in_sandbox:
            raise SandboxSecurityError(
                f"Security violation: Path '{relative_or_absolute_path}' escapes sandbox boundaries {[str(r) for r in self.allowed_roots]}"
            )

        if must_exist and not resolved.exists():
            raise PathNotFoundError(f"Path does not exist: '{relative_or_absolute_path}'")

        return resolved

    def relative_path(self, target_path: Union[str, Path]) -> str:
        """Returns relative path string from the best matching sandbox root."""
        resolved = self.validate_and_resolve(target_path)
        for root in self.allowed_roots:
            try:
                rel = resolved.relative_to(root)
                if root == self.workspace_root:
                    return str(rel)
                return f"{root.name}/{rel}" if str(rel) != "." else root.name
            except ValueError:
                continue
        return str(resolved)

    def is_safe_path(self, target_path: Union[str, Path]) -> bool:
        """Checks whether a path is valid and within the sandbox without raising."""
        try:
            self.validate_and_resolve(target_path)
            return True
        except (SandboxSecurityError, Exception):
            return False

    def list_dir(self, subpath: Optional[Union[str, Path]] = None) -> List[Tuple[str, str, int]]:
        """
        Lists directory contents safely within sandbox.
        Returns list of tuples: (relative_path_name, entry_type, size_bytes)
        """
        target = self.validate_and_resolve(subpath or "", must_exist=True)
        if not target.is_dir():
            raise ValueError(f"Path '{subpath}' is not a directory.")

        results = []
        for entry in sorted(os.scandir(target), key=lambda e: e.name):
            entry_path = Path(entry.path)
            # Skip .mindfs internal directory
            if entry.name == ".mindfs" or entry.name == ".git":
                continue

            try:
                # Validate that entry doesn't symlink escape
                self.validate_and_resolve(entry_path)
                entry_type = "directory" if entry.is_dir() else "file"
                size = entry.stat().st_size if entry.is_file() else 0
                results.append((entry.name, entry_type, size))
            except SandboxSecurityError:
                # Symlink escape, skip or flag
                results.append((entry.name, "symlink_escape", 0))
            except Exception:
                continue

        return results

