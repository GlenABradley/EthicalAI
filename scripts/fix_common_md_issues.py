#!/usr/bin/env python3
"""
Fix common markdown linting issues.

1. Ensures blank lines around headers, lists, and code blocks
2. Adds language to code blocks
3. Removes trailing whitespace
4. Fixes bare URLs
"""

import re
import sys
from pathlib import Path

def fix_file(file_path: str) -> bool:
    """Fix common markdown issues in a file."""
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found: {file_path}")
        return False
    
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.splitlines()
    new_lines = []
    in_code_block = False
    in_list = False
    
    for i, line in enumerate(lines):
        stripped = line.rstrip()
        
        # Skip processing inside code blocks
        if stripped.startswith(('```', '~~~')):
            if not in_code_block:
                # Ensure blank line before code block
                if new_lines and new_lines[-1].strip() != '':
                    new_lines.append('')
                # Add language if missing
                if len(stripped) <= 3:
                    stripped = stripped + 'text'
            else:
                # Ensure blank line after code block
                if i + 1 < len(lines) and lines[i+1].strip() != '':
                    new_lines.append('')
            in_code_block = not in_code_block
            new_lines.append(stripped)
            continue
        
        if in_code_block:
            new_lines.append(line)
            continue
            
        # Fix trailing whitespace
        line = line.rstrip()
        
        # Fix bare URLs (not in code blocks or links)
        if not any(c in line for c in ['`', '[', ']']):
            line = re.sub(r'(?<!\[)(https?://[^\s<>\)\]`]+)(?![^\[\]]*\]|\([^)]*\)|`[^`]*`)', r'<\1>', line)
        
        # Check for headers
        if re.match(r'^#{1,6}\s+', line):
            # Ensure blank line before header
            if new_lines and new_lines[-1].strip() != '':
                new_lines.append('')
            new_lines.append(line)
            # Ensure blank line after header
            if i + 1 < len(lines) and lines[i+1].strip() != '':
                new_lines.append('')
            continue
            
        # Check for list items
        is_list_item = bool(re.match(r'^\s*[-*+]\s+', line) or 
                          re.match(r'^\s*\d+\.\s+', line))
        
        if is_list_item and not in_list:
            # Ensure blank line before list
            if new_lines and new_lines[-1].strip() != '':
                new_lines.append('')
            in_list = True
        elif not is_list_item and in_list and line.strip() != '':
            # Ensure blank line after list
            new_lines.append('')
            in_list = False
        
        new_lines.append(line)
    
    # Ensure file ends with exactly one newline
    while new_lines and new_lines[-1].strip() == '':
        new_lines.pop()
    new_content = '\n'.join(new_lines) + '\n'
    if new_content != content:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_common_md_issues.py file1.md [file2.md ...]")
        sys.exit(1)
    
    for file_path in sys.argv[1:]:
        if fix_file(file_path):
            print(f"Fixed: {file_path}")
        else:
            print(f"No changes needed: {file_path}")

if __name__ == "__main__":
    main()
