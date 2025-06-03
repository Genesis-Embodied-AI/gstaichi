import re
import os
import pathlib
from urllib.parse import urlparse

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
            print(f"⚠️ External link (not checked): {link}")
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

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python markdown_link_checker.py <markdown_file> [base_dir]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    base_dir = sys.argv[2] if len(sys.argv) > 2 else None
    check_markdown_links(file_path, base_dir)
    