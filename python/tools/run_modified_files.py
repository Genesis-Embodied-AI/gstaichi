#!/usr/bin/env python3

# compares with origin/main, so make sure you have origin/main available

import subprocess
import re
import sys
import argparse
from typing import List
from pathlib import Path
import fnmatch

def get_main_branch() -> str:
    """Detect the main branch name (main/master)"""
    result = subprocess.run(
        ["git", "remote", "show", "origin"],
        capture_output=True,
        text=True
    )
    match = re.search(r"HEAD branch: (\w+)", result.stdout)
    return match.group(1) if match else "main"

def get_changed_files(include_pattern: str) -> List[str]:
    """Get all changed files matching glob pattern"""
    main_branch = get_main_branch()
    include_pattern = include_pattern
    
    commands = [
        ["git", "diff", "--name-only", "--diff-filter=AM", f"origin/{main_branch}...HEAD"],
        ["git", "diff", "--name-only", "--diff-filter=AM", "--cached"],
        ["git", "diff", "--name-only", "--diff-filter=AM", "--"],
        ["git", "ls-files", "--others", "--exclude-standard"]
    ]
    
    files = set()
    for cmd in commands:
        result = subprocess.run(cmd, capture_output=True, text=True)
        files.update(result.stdout.splitlines())
    
    return [f for f in files 
            if fnmatch.fnmatch(f, include_pattern) and Path(f).is_file()]

def main():
    parser = argparse.ArgumentParser(
        description="Run command against modified files",
        usage="%(prog)s [--include PATTERN] -- COMMAND [ARGS...]"
    )
    parser.add_argument("--include", default="*", 
                       help="File pattern to include (glob, e.g. '*.py')")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                       help="Command to run")
    
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
