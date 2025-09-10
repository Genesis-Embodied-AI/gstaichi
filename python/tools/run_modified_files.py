# type: ignore

#!/usr/bin/env python3

# compares with origin/main, so make sure you have origin/main available

import argparse
import fnmatch
import subprocess
import sys
from pathlib import Path
from typing import List


def get_changed_files(include_pattern: str, cwd: str = ".") -> List[str]:
    """Get all changed files matching glob pattern"""
    include_pattern = include_pattern

    commands = [
        ["git", "diff", "--name-only", "--diff-filter=AM", f"origin/main...HEAD"],
        ["git", "diff", "--name-only", "--diff-filter=AM", "--cached"],
        ["git", "diff", "--name-only", "--diff-filter=AM", "--"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    ]

    files = set()
    for cmd in commands:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
        files.update(result.stdout.splitlines())
    return [f for f in files if fnmatch.fnmatch(f, include_pattern) and (Path(cwd) / f).is_file()]


def main():
    parser = argparse.ArgumentParser(
        description="Run command against modified files", usage="%(prog)s [--include PATTERN] -- COMMAND [ARGS...]"
    )
    parser.add_argument("--include", default="*", help="File pattern to include (glob, e.g. '*.py')")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run")

    args = parser.parse_args()

    # Skip the leading '--' if present
    command = args.command
    if command and command[0] == "--":
        command = command[1:]

    if not command:
        parser.print_help()
        sys.exit(1)

    files = get_changed_files(args.include)
    if not files:
        print("No files changed matching pattern")
        return

    print(f"Running on {len(files)} files:")
    for f in files:
        print(f"  {f}")

    exit_codes = []
    for file in files:
        cmd = command + [file]
        result = subprocess.run(cmd)
        exit_codes.append(result.returncode)

    sys.exit(max(exit_codes))


if __name__ == "__main__":
    main()
