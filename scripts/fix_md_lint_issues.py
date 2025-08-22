#!/usr/bin/env python3
"""
Fix specific markdown linting issues:
- MD012: No multiple consecutive blank lines
- MD022: Headers should be surrounded by blank lines
- MD029: Ordered list item prefix
- MD032: Lists should be surrounded by blank lines
"""

import re
import sys
from pathlib import Path

def fix_consecutive_blank_lines(content: str) -> str:
    """Fix MD012: No multiple consecutive blank lines."""
    return re.sub(r'\n{3,}', '\n\n', content)

def fix_headers(content: str) -> str:
    """Fix MD022: Headers should be surrounded by blank lines."""
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

def fix_ordered_lists(content: str) -> str:
    """Fix MD029: Ordered list item prefix."""
    lines = content.splitlines()
    new_lines = []
    in_ordered_list = False
    list_counter = 1
    
    for line in lines:
        # Check for ordered list items
        match = re.match(r'^(\s*)(\d+)\.(\s+)', line)
        if match:
            if not in_ordered_list:
                in_ordered_list = True
                list_counter = 1
            # Replace with sequential number
            new_line = f"{match.group(1)}{list_counter}.{match.group(3)}{line[match.end(0):]}"
            list_counter += 1
            new_lines.append(new_line)
        else:
            in_ordered_list = False
            new_lines.append(line)
    
    return '\n'.join(new_lines)

def fix_lists(content: str) -> str:
    """Fix MD032: Lists should be surrounded by blank lines."""
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

def fix_file(file_path: str) -> bool:
    """Fix all markdown linting issues in a file."""
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found: {file_path}")
        return False
    
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Apply all fixes
    fixed_content = content
    fixed_content = fix_consecutive_blank_lines(fixed_content)
    fixed_content = fix_headers(fixed_content)
    fixed_content = fix_ordered_lists(fixed_content)
    fixed_content = fix_lists(fixed_content)
    
    # Ensure file ends with exactly one newline
    fixed_content = fixed_content.rstrip() + '\n'
    if fixed_content != content:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        return True
    return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_md_lint_issues.py file1.md [file2.md ...]")
        sys.exit(1)
    
    for file_path in sys.argv[1:]:
        if fix_file(file_path):
            print(f"Fixed: {file_path}")
        else:
            print(f"No changes needed: {file_path}")

if __name__ == "__main__":
    main()
