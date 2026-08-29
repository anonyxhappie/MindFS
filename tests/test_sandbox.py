"""Tests for filesystem sandboxing and security."""

from pathlib import Path
import tempfile
import pytest

from mindfs.filesystem.sandbox import FilesystemSandbox, PathNotFoundError, SandboxSecurityError


def test_sandbox_containment(temp_workspace):
    ws_root, fixtures = temp_workspace
    sandbox = FilesystemSandbox(ws_root)

    # Valid relative file
    p = sandbox.validate_and_resolve("plain.txt", must_exist=True)
    assert p.exists()
    assert sandbox.relative_path(p) == "plain.txt"

    # Valid subfolder file
    sub = ws_root / "subdir"
    sub.mkdir()
    sub_file = sub / "subfile.txt"
    sub_file.write_text("hello")
    p_sub = sandbox.validate_and_resolve("subdir/subfile.txt", must_exist=True)
    assert p_sub == sub_file.resolve()


def test_sandbox_rejects_parent_traversal(temp_workspace):
    ws_root, fixtures = temp_workspace
    sandbox = FilesystemSandbox(ws_root)

    with pytest.raises(SandboxSecurityError):
        sandbox.validate_and_resolve("../outside.txt")

    with pytest.raises(SandboxSecurityError):
        sandbox.validate_and_resolve("../../etc/passwd")


def test_sandbox_rejects_symlink_escape(temp_workspace):
    ws_root, fixtures = temp_workspace
    sandbox = FilesystemSandbox(ws_root)

    # Create external target
    external_dir = Path(tempfile.mkdtemp(prefix="mindfs_outside_"))
    external_file = external_dir / "secret.txt"
    external_file.write_text("secret outside workspace")

    # Create symlink inside workspace pointing outside
    escape_link = ws_root / "escape_link.txt"
    escape_link.symlink_to(external_file)

    with pytest.raises(SandboxSecurityError):
        sandbox.validate_and_resolve("escape_link.txt")


def test_sandbox_missing_file_error(temp_workspace):
    ws_root, fixtures = temp_workspace
    sandbox = FilesystemSandbox(ws_root)

    with pytest.raises(PathNotFoundError):
        sandbox.validate_and_resolve("nonexistent_file_xyz.txt", must_exist=True)


def test_sandbox_list_dir(temp_workspace):
    ws_root, fixtures = temp_workspace
    sandbox = FilesystemSandbox(ws_root)

    entries = sandbox.list_dir()
    names = [name for name, kind, size in entries]
    assert "plain.txt" in names
    assert "data.json" in names
    assert ".mindfs" not in names
