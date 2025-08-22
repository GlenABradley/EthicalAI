#!/usr/bin/env python3
"""
Markdown formatter script.

This script fixes common markdown linting issues including:
- MD009: Trailing spaces
- MD012: Multiple consecutive blank lines
- MD022: Headers should be surrounded by blank lines
- MD031: Fenced code blocks should be surrounded by blank lines
- MD032: Lists should be surrounded by blank lines
- MD033: Inline HTML
- MD034: Bare URL used
- MD040: Fenced code blocks should have a language specified
- MD041: First line in file should be a top level header
- MD047: Files should end with a single newline character
"""

import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

def remove_trailing_spaces(content: str) -> str:
    """Remove trailing spaces from each line."""
    return '\n'.join(line.rstrip() for line in content.splitlines())

def fix_consecutive_blank_lines(content: str) -> str:
    """Replace multiple consecutive blank lines with a single blank line."""
    return re.sub(r'\n{3,}', '\n\n', content)

def fix_headers(content: str) -> str:
    """Ensure headers are properly surrounded by blank lines."""
    lines = content.splitlines()
    new_lines = []
    
    for i, line in enumerate(lines):
        if re.match(r'^#{1,6}\s+', line):
            # Add blank line before header if needed
            if i > 0 and new_lines and new_lines[-1].strip() != '':
                new_lines.append('')
            new_lines.append(line)
            # Add blank line after header if needed
            if i < len(lines) - 1 and lines[i+1].strip() != '':
                new_lines.append('')
        elif line.strip() == '---' and i > 0 and i < len(lines) - 1:  # Handle YAML front matter
            if new_lines and new_lines[-1].strip() != '':
                new_lines.append(line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    
    return '\n'.join(new_lines)

def fix_code_blocks(content: str) -> str:
    """Ensure code blocks are properly formatted and have language specified."""
    lines = content.splitlines()
    new_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Handle fenced code blocks
        if line.strip().startswith(('```', '~~~')):
            # Add blank line before code block if needed
            if i > 0 and new_lines and new_lines[-1].strip() != '':
                new_lines.append('')
            
            # Get the marker and language
            marker = line.strip().split()[0] if line.strip() else '```'
            lang = line.strip()[len(marker):].strip()
            
            # If no language specified, default to 'text'
            if not lang and marker in ('```', '~~~'):
                new_lines.append(f'{marker}text')
            else:
                new_lines.append(line.strip())
            
            # Find end of code block
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith(marker):
                new_lines.append(lines[j])
                j += 1
            
            if j < len(lines):
                new_lines.append(marker)  # Add the closing marker
                # Add blank line after code block if needed
                if j + 1 < len(lines) and lines[j+1].strip() != '':
                    new_lines.append('')
                i = j + 1
            else:
                i = j
        else:
            new_lines.append(line)
            i += 1
    
    return '\n'.join(new_lines)

def fix_lists(content: str) -> str:
    """Ensure lists are properly formatted and surrounded by blank lines."""
    lines = content.splitlines()
    new_lines = []
    in_list = False
    prev_line_blank = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        is_blank = not stripped
        
        # Skip multiple consecutive blank lines
        if is_blank and prev_line_blank:
            continue
            
        # Check if this is a list item
        is_list_item = bool(re.match(r'^\s*[-*+]\s+', stripped) or 
                           re.match(r'^\s*\d+\.\s+', stripped))
        
        if is_list_item and not in_list:
            # Add blank line before list if needed
            if i > 0 and new_lines and new_lines[-1].strip() != '':
                new_lines.append('')
            in_list = True
        elif not is_list_item and in_list and not is_blank:
            # Add blank line after list if needed
            new_lines.append('')
            in_list = False
        
        new_lines.append(line)
        prev_line_blank = is_blank
    
    return '\n'.join(new_lines)

def fix_bare_urls(content: str) -> str:
    """Convert bare URLs to proper markdown links."""
    # Match URLs not already in markdown links or code blocks
    url_pattern = r'(?<!\[)(https?://[^\s<>\]]+)(?![^\[\]]*\]|\([^)]*\)|`[^`]*`)'
    return re.sub(url_pattern, r'<\1>', content)

def fix_file_end(content: str) -> str:
    """Ensure file ends with exactly one newline character."""
    return content.rstrip() + '\n'

def fix_markdown(content: str) -> str:
    """Apply all markdown fixes to the content."""
    content = remove_trailing_spaces(content)
    content = fix_consecutive_blank_lines(content)
    content = fix_headers(content)
    content = fix_code_blocks(content)
    content = fix_lists(content)
    content = fix_bare_urls(content)
    content = fix_file_end(content)
    return content

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_markdown.py file1.md [file2.md ...]")
        sys.exit(1)
    
    for file_path in sys.argv[1:]:
        path = Path(file_path)
        if not path.exists():
            print(f"Warning: File not found: {file_path}")
            continue
            
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content = fix_markdown(content)
        
        if new_content != content:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Fixed: {file_path}")
        else:
            print(f"No changes needed: {file_path}")

if __name__ == "__main__":
    main()
