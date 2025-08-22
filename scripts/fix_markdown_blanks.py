from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

LIST_MARKERS = ("- ", "* ")


def is_heading(line: str) -> bool:
    return line.lstrip().startswith("#") and not line.lstrip().startswith("#######!")


def is_fence(line: str) -> bool:
    s = line.strip()
    return s.startswith("```") or s.startswith("~~~")


def is_list_item(line: str) -> bool:
    ls = line.lstrip()
    if any(ls.startswith(m) for m in LIST_MARKERS):
        return True
    # ordered list: 1. foo
    i = 0
    while i < len(ls) and ls[i].isdigit():
        i += 1
    return i > 0 and i + 1 <= len(ls) and ls[i:i+2] == ". "


def ensure_blank(lines: List[str], idx: int, before: bool) -> None:
    if before:
        if idx > 0 and lines[idx-1].strip() != "":
            lines.insert(idx, "\n")
    else:
        if idx < len(lines) - 1 and lines[idx+1].strip() != "":
            lines.insert(idx + 1, "\n")
        line = lines[i]
        
        # Handle fenced code blocks
        if line.strip().startswith(('```', '~~~')):
            # Add blank line before code block if needed
            if i > 0 and new_lines and new_lines[-1].strip() != '':
                new_lines.append('')
            
            # Check if language is specified
            if line.strip() in ('```', '~~~'):
                # Default to 'text' if no language specified
                new_lines.append(line)
                if i + 1 < len(lines) and not lines[i+1].strip().startswith(('```', '~~~')):
                    new_lines.append(lines[i+1])
                    i += 1
            else:
                new_lines.append(line)
            
            # Find end of code block
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith(('```', '~~~')):
                new_lines.append(lines[j])
                j += 1
            
            if j < len(lines):
                new_lines.append(lines[j])
                # Add blank line after code block if needed
                if j + 1 < len(lines) and lines[j+1].strip() != '':
                    new_lines.append('')
                i = j + 1
        else:
            new_lines.append(line)
    return '\n'.join(new_lines)

def fix_code_blocks(content: str) -> str:
    """Ensure fenced code blocks are surrounded by blank lines."""
    lines = content.splitlines()
    new_lines = []
    in_code_block = False
    for i, line in enumerate(lines):
        if line.strip().startswith(('```', '~~~')):
            if not in_code_block:
                if i > 0 and new_lines and new_lines[-1].strip() != '':
                    new_lines.append('')
                in_code_block = True
            else:
                if i + 1 < len(lines) and lines[i+1].strip() != '':
                    new_lines.append('')
                in_code_block = False
        new_lines.append(line)
    return '\n'.join(new_lines)

def fix_lists(content: str) -> str:
    """Ensure lists are surrounded by blank lines."""
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
    content = fix_file_end(content)
    return content

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_markdown_blanks.py file1.md [file2.md ...]")
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
