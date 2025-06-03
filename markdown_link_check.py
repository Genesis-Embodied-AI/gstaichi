import re
import os
import pathlib
from urllib.parse import urlparse
import argparse

def check_markdown_links(file_path, base_dir=None):
    """
    Check all links in a Markdown file, including anchor references.
    
    Args:
        file_path: Path to the Markdown file
        base_dir: Base directory for relative links (defaults to file's directory)
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(file_path))
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all links and image references
    link_pattern = r'\[.*?\]\((.*?)\)|!\[.*?\]\((.*?)\)'
    matches = re.findall(link_pattern, content)
    
    # Combine both capturing groups (links and images)
    links = [match[0] or match[1] for match in matches if match[0] or match[1]]
    
    for link in links:
        parsed = urlparse(link)
        
        # Skip mailto and external links
        if parsed.scheme in ('http', 'https', 'mailto'):
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
                continue
            
            # Check anchor in local file
            if parsed.fragment:
                if full_path.endswith('.md'):
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
    try:
        with open(md_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for GitHub-style anchors (spaces become dashes, lowercase)
        normalized_anchor = anchor.lower().replace(' ', '-')
        
        # Pattern for Markdown headers
        header_pattern = r'^#+\s+(.*)$'
        
        found = False
        for line in content.split('\n'):
            match = re.match(header_pattern, line)
            if match:
                header_text = match.group(1)
                # Generate possible anchor formats
                possible_anchors = [
                    header_text.lower().replace(' ', '-'),
                    header_text.lower().replace(' ', '_'),
                    header_text.replace(' ', ''),
                    header_text
                ]
                if normalized_anchor in possible_anchors:
                    found = True
                    break
        
        if not found:
            print(f"❌ Broken anchor: #{anchor} in {md_file_path}")
    except Exception as e:
        print(f"⚠️ Error checking anchor #{anchor} in {md_file_path}: {str(e)}")

def find_markdown_files(root_dir):
    """
    Recursively find all .md files under root_dir.
    """
    md_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.md'):
                md_files.append(os.path.join(dirpath, filename))
    return md_files

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Check Markdown links in a directory recursively.")
    parser.add_argument("directory", help="Path to the root directory containing Markdown files")
    args = parser.parse_args()

    root_dir = os.path.abspath(args.directory)
    if not os.path.isdir(root_dir):
        print(f"Error: {root_dir} is not a directory.")
        exit(1)

    md_files = find_markdown_files(root_dir)
    if not md_files:
        print(f"No Markdown files found in {root_dir}")
        exit(0)

    for md_file in md_files:
        print(f"\nChecking: {md_file}")
        check_markdown_links(md_file, base_dir=os.path.dirname(md_file))
