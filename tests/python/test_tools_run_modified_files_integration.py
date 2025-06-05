import subprocess
import os
from pathlib import Path

import pytest

from tools import run_modified_files


@pytest.fixture
def git_repo(tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    subprocess.run(["git", "init", "-b", "main"], check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], check=True)
    # Add a remote origin (fake, but needed for HEAD branch detection)
    subprocess.run(["git", "remote", "add", "origin", "."], check=True)
    yield tmp_path
    os.chdir(cwd)


def test_get_changed_files_added_file(git_repo):
    # Create a file and add it
    f = git_repo / "foo.py"
    f.write_text("print('hi')")
    subprocess.run(["git", "add", "foo.py"], check=True)
    subprocess.run(["git", "commit", "-m", "add foo.py"], check=True)
    # Modify file
    f.write_text("print('hello')")
    # Should show as changed
    files = run_modified_files.get_changed_files("*.py")
    assert "foo.py" in files or str(f) in files


def test_get_changed_files_untracked(git_repo):
    f = git_repo / "bar.py"
    f.write_text("print('bar')")
    # Not added to git
    files = run_modified_files.get_changed_files("*.py")
    assert "bar.py" in files or str(f) in files


def test_get_changed_files_pattern(git_repo):
    (git_repo / "a.py").write_text("a")
    (git_repo / "b.txt").write_text("b")
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", "add files"], check=True)
    # Change a.py
    (git_repo / "a.py").write_text("changed")
    files = run_modified_files.get_changed_files("*.py")
    assert any(f.endswith("a.py") for f in files)
    assert not any(f.endswith("b.txt") for f in files)


def test_get_changed_files_none(git_repo):
    # No changes
    files = run_modified_files.get_changed_files("*.md")
    assert files == []


@pytest.mark.parametrize(
    "filenames,pattern,expected",
    [
        (["foo.py"], "*.py", ["foo.py"]),
        (["foo.py", "bar.txt"], "*.py", ["foo.py"]),
        (["foo.py", "bar.py"], "*.py", ["foo.py", "bar.py"]),
        (["foo.cpp", "bar.h"], "*.py", []),
        (["foo.py", "foo.pyc"], "foo.*", ["foo.py", "foo.pyc"]),
        (["foo.md"], "*.md", ["foo.md"]),
        ([], "*.py", []),
    ]
)
def test_get_changed_files_parametrize(git_repo, filenames, pattern, expected):
    # Create files and add/commit them
    for fname in filenames:
        f = git_repo / fname
        f.write_text("content")
        subprocess.run(["git", "add", fname], check=True)
    if filenames:
        subprocess.run(["git", "commit", "-m", "add files"], check=True)
    # Modify all files to make them "changed"
    for fname in filenames:
        f = git_repo / fname
        f.write_text("changed content")
    files = run_modified_files.get_changed_files(pattern)
    # Only compare basenames for assertion
    found = {Path(f).name for f in files}
    assert set(expected) == found


@pytest.mark.parametrize(
    "untracked,pattern,expected",
    [
        (["baz.py"], "*.py", ["baz.py"]),
        (["baz.txt"], "*.py", []),
        (["baz.py", "baz.txt"], "*.txt", ["baz.txt"]),
    ]
)
def test_get_changed_files_untracked_parametrize(git_repo, untracked, pattern, expected):
    # Create and do not add to git
    for fname in untracked:
        (git_repo / fname).write_text("untracked")
    files = run_modified_files.get_changed_files(pattern)
    found = {Path(f).name for f in files}
    assert set(expected) == found
