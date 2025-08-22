#!/usr/bin/env python3
"""
Markdown Lint Fixer

Fixes common markdown linting issues:
- Ensures blank lines around headers, lists, and code blocks
- Removes trailing whitespace
- Ensures consistent list indentation
- Fixes code block formatting
- Handles bare URLs
"""

import re
import sys
from pathlib import Path
from typing import List, Optional

def fix_trailing_whitespace(content: str) -> str:
    """Remove trailing whitespace from each line."""
    return '\n'.join(line.rstrip() for line in content.splitlines())

def fix_consecutive_blank_lines(content: str) -> str:
    """Replace multiple consecutive blank lines with a single blank line."""
    return re.sub(r'\n{3,}', '\n\n', content)

def fix_headers(content: str) -> str:
    """Ensure headers have blank lines before and after."""
    lines = content.splitlines()
    new_lines = []
    
    for i, line in enumerate(lines):
        if re.match(r'^#{1,6}\s+', line):
            # Add blank line before header if needed
            if i > 0 and new_lines and new_lines[-1].strip() != '':
                new_lines.append('')
            new_lines.append(line)
            # Add blank line after header if needed
            if i + 1 < len(lines) and lines[i+1].strip() != '':
                new_lines.append('')
        else:
            new_lines.append(line)
    
    return '\n'.join(new_lines)

def fix_code_blocks(content: str) -> str:
    """Ensure code blocks have proper formatting and language specified."""
    lines = content.splitlines()
    new_lines = []
    in_code_block = False
    code_block_marker = ''
    
    for line in lines:
        stripped = line.strip()
        
        # Check for code block start
        if not in_code_block and (stripped.startswith('```') or stripped.startswith('~~~')):
            in_code_block = True
            code_block_marker = stripped[:3]
            # Add blank line before code block if needed
            if new_lines and new_lines[-1].strip() != '':
                new_lines.append('')
            # Ensure language is specified
            if len(stripped) <= 3:  # No language specified
                new_lines.append(f'{code_block_marker}text')
            else:
                new_lines.append(line)
        # Check for code block end
        elif in_code_block and stripped.startswith(code_block_marker):
            in_code_block = False
            new_lines.append(line)
            # Add blank line after code block if needed
            if new_lines and new_lines[-1].strip() != '':
                new_lines.append('')
        else:
            new_lines.append(line)
    
    return '\n'.join(new_lines)

def fix_lists(content: str) -> str:
    """Ensure lists have proper spacing and indentation."""
    lines = content.splitlines()
    new_lines = []
    in_list = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        is_list_item = bool(re.match(r'^\s*[-*+]\s+', stripped) or 
                          re.match(r'^\s*\d+\.\s+', stripped))
        
        if is_list_item and not in_list:
            # Add blank line before list if needed
            if i > 0 and new_lines and new_lines[-1].strip() != '':
                new_lines.append('')
            in_list = True
        elif not is_list_item and in_list and stripped != '':
            # Add blank line after list if needed
            new_lines.append('')
            in_list = False
        
        new_lines.append(line)
    
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
    content = fix_trailing_whitespace(content)
    content = fix_consecutive_blank_lines(content)
    content = fix_headers(content)
    content = fix_code_blocks(content)
    content = fix_lists(content)
    content = fix_bare_urls(content)
    content = fix_file_end(content)
    return content

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_md_lint.py file1.md [file2.md ...]")
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
