import re
import os
import pathlib
from urllib.parse import urlparse
import argparse

error_found = False  # Track if any errors are found


def check_markdown_links(file_path, base_dir=None):
    """
    Check all links in a Markdown file, including anchor references.

    Args:
        file_path: Path to the Markdown file
        base_dir: Base directory for relative links (defaults to file's directory)
    """
    global error_found
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(file_path))

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find all links and image references
    link_pattern = r"\[.*?\]\((.*?)\)|!\[.*?\]\((.*?)\)"
    matches = re.findall(link_pattern, content)

    # Combine both capturing groups (links and images)
    links = [match[0] or match[1] for match in matches if match[0] or match[1]]

    for link in links:
        parsed = urlparse(link)

        # Skip mailto and external links
        if parsed.scheme in ("http", "https", "mailto"):
            print(f"[-] External link (not checked): {link}")
            continue

        # Handle anchor-only links
        if not parsed.path and parsed.fragment:
            check_anchor(file_path, parsed.fragment)
            continue

        # Handle relative paths
        if not parsed.scheme and not parsed.netloc:
            full_path = os.path.normpath(os.path.join(base_dir, parsed.path))

            # Check if file exists
            if not os.path.exists(full_path):
                print(f"❌ Broken link: {link} (File not found: {full_path})")
                error_found = True
                continue

            # Check anchor in local file
            if parsed.fragment:
                if full_path.endswith(".md"):
                    check_anchor(full_path, parsed.fragment)
                else:
                    # For non-markdown files, we can't check anchors
                    print(f"⚠️ Anchor in non-Markdown file (not checked): {link}")


def check_anchor(md_file_path, anchor):
    """
    Check if an anchor exists in a Markdown file.

    Args:
        md_file_path: Path to the Markdown file
        anchor: Anchor to check (without #)
    """
    global error_found
    try:
        with open(md_file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Improved anchor cleaning: remove non-alphanum except hyphens, collapse multiple hyphens, strip hyphens
        def clean_anchor(s):
            s = s.lower().replace(" ", "-")
            s = re.sub(r"[^a-z0-9\-]", "", s)
            s = re.sub(r"-+", "-", s)
            s = s.strip("-")
            return s

        normalized_anchor = clean_anchor(anchor)

        # Pattern for Markdown headers
        header_pattern = r"^#+\s+(.*)$"

        found = False
        available_anchors = []
        for line in content.split("\n"):
            match = re.match(header_pattern, line)
            if match:
                header_text = match.group(1)
                anchor_dash = clean_anchor(header_text)
                anchor_underscore = re.sub(r"[^a-z0-9\-]", "", header_text.lower().replace(" ", "_"))
                anchor_nospace = re.sub(r"[^a-z0-9\-]", "", header_text.replace(" ", ""))
                anchor_raw = re.sub(r"[^a-z0-9\-]", "", header_text)
                possible_anchors = [anchor_dash, anchor_underscore, anchor_nospace, anchor_raw]
                available_anchors.append(anchor_dash)
                if normalized_anchor in possible_anchors:
                    found = True
                    break

        if not found:
            print(f"❌ Broken anchor: #{anchor} in {md_file_path}")
            print(f"   Available anchors in this file:")
            for a in available_anchors:
                print(f"     - {a}")
            error_found = True
    except Exception as e:
        print(f"⚠️ Error checking anchor #{anchor} in {md_file_path}: {str(e)}")


def find_markdown_files(root_dir):
    """
    Recursively find all .md files under root_dir.
    """
    md_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".md"):
                md_files.append(os.path.join(dirpath, filename))
    return md_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check Markdown links in a directory or a single Markdown file.")
    parser.add_argument("path", help="Path to the root directory or a Markdown file")
    args = parser.parse_args()

    input_path = os.path.abspath(args.path)
    md_files = []

    if os.path.isdir(input_path):
        md_files = find_markdown_files(input_path)
        if not md_files:
            print(f"No Markdown files found in {input_path}")
            exit(0)
    elif os.path.isfile(input_path) and input_path.lower().endswith(".md"):
        md_files = [input_path]
    else:
        print(f"Error: {input_path} is not a directory or a Markdown (.md) file.")
        exit(1)

    for md_file in md_files:
        print(f"\nChecking: {md_file}")
        check_markdown_links(md_file, base_dir=os.path.dirname(md_file))
    if error_found:
        exit(2)
