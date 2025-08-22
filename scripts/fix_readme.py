#!/usr/bin/env python3
"""
Fix formatting issues in README.md
"""

import re
from pathlib import Path

def fix_readme(file_path):
    """Fix formatting issues in README.md"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the first paragraph that got merged
    content = re.sub(
        r'ts, and a runnable API\.at token, span, frame, and frame-span levels',
        'puts, and a runnable API. It outputs per-token, span, frame, and frame-span vectors',
        content
    )
    
    # Fix the second paragraph
    content = re.sub(
        r'## Goalsository follows the authoritative build plan provided\. We will implement the system milestone-by-milestone with concrete artifacts, test',
        '## Goals',
        content
    )
    
    # Fix code block formatting
    content = re.sub(r'```text\n\s*```', '```', content)
    
    # Fix ordered list items
    content = re.sub(r'^1\.\s+(.*?)\n1\.', '1. \\1\\n2.', content, flags=re.MULTILINE)
    
    # Remove extra blank lines around code blocks
    content = re.sub(r'\n{3,}```', '\n```', content)
    content = re.sub(r'```\n{3,}', '```\n\n', content)
    
    # Ensure exactly one blank line before and after headers
    content = re.sub(r'([^\n])\n\n#', '\\1\n\n#', content)
    content = re.sub(r'#.*\n\n{2,}', lambda m: m.group(0).rstrip('\n') + '\n', content)
    
    # Write the fixed content back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    fix_readme("README.md")
