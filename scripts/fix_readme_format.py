#!/usr/bin/env python3
"""
Fix formatting issues in README.md
"""

import re
from pathlib import Path

def fix_readme_format(file_path):
    """Fix formatting issues in README.md"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix code blocks
    content = re.sub(r'```bash\n\s*```text\n', '```bash\n', content)
    content = re.sub(r'```text\n\s*```', '```', content)
    
    # Fix ordered list items
    lines = content.splitlines()
    new_lines = []
    list_counter = 1
    
    for line in lines:
        # Check for list items that need renumbering
        if re.match(r'^\d+\.\s+', line):
            line = re.sub(r'^\d+\.', f'{list_counter}.', line, 1)
            list_counter += 1
        elif line.strip() == '':
            list_counter = 1  # Reset counter on blank lines
        new_lines.append(line)
    
    # Join lines and ensure proper spacing
    content = '\n'.join(new_lines)
    
    # Fix multiple consecutive blank lines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Ensure file ends with exactly one newline
    content = content.rstrip() + '\n'
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    fix_readme_format("README.md")
