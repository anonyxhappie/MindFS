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
    """Enforces sandbox containment within a configured workspace root."""

    def __init__(self, workspace_root: Union[str, Path]):
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        if not self.workspace_root.exists():
            self.workspace_root.mkdir(parents=True, exist_ok=True)

    def validate_and_resolve(self, relative_or_absolute_path: Union[str, Path], must_exist: bool = False) -> Path:
        """
        Resolves a path and verifies that it is strictly contained within the workspace root.
        Rejects symlinks pointing outside the workspace, parent directory traversals, etc.
        """
        raw_path = Path(relative_or_absolute_path)
        
        # If absolute path is provided, ensure it starts within workspace
        if raw_path.is_absolute():
            resolved = raw_path.resolve()
        else:
            resolved = (self.workspace_root / raw_path).resolve()

        # Check realpath for symlink traversal
        real_path_str = os.path.realpath(str(resolved))
        workspace_real_str = os.path.realpath(str(self.workspace_root))

        # Check if realpath is within workspace_root
        try:
            Path(real_path_str).relative_to(workspace_real_str)
        except ValueError:
            raise SandboxSecurityError(
                f"Security violation: Path '{relative_or_absolute_path}' escapes workspace boundary '{self.workspace_root}'"
            )

        if must_exist and not resolved.exists():
            raise PathNotFoundError(f"Path does not exist: '{relative_or_absolute_path}'")

        return resolved

    def relative_path(self, target_path: Union[str, Path]) -> str:
        """Returns relative path string from workspace root."""
        resolved = self.validate_and_resolve(target_path)
        try:
            return str(resolved.relative_to(self.workspace_root))
        except ValueError:
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

