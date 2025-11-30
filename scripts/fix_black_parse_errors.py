#!/usr/bin/env python3
"""
Fix syntax errors in files that prevent Black from formatting them.

This script fixes common indentation issues that cause Black parsing errors.
"""

import re
import sys
from pathlib import Path

# Files that failed Black formatting
FAILED_FILES = [
    "corerec/engines/contentFilterEngine/tfidf_recommender.py",
    "corerec/engines/unionizedFilterEngine/bayesian_method_base/vmf_base.py",
    "corerec/engines/unionizedFilterEngine/graph_based_base/geoimc_base.py",
    "corerec/engines/gnnrec.py",
    "corerec/engines/unionizedFilterEngine/graph_based_base/lightgcn_base.py",
    "corerec/engines/unionizedFilterEngine/mf_base/A2SVD_base.py",
    "corerec/engines/unionizedFilterEngine/mf_base/svd_base.py",
    "corerec/engines/unionizedFilterEngine/mf_base/factorization_machine_base.py",
    "corerec/engines/unionizedFilterEngine/mf_base/ALS_base.py",
    "corerec/engines/mind.py",
    "corerec/engines/unionizedFilterEngine/mf_base/matrix_factorization.py",
    "corerec/engines/unionizedFilterEngine/mf_base/matrix_factorization_base.py",
    "corerec/engines/unionizedFilterEngine/mf_base/user_based_uf.py",
    "corerec/engines/unionizedFilterEngine/rbm.py",
    "corerec/engines/unionizedFilterEngine/rlrmc.py",
    "corerec/engines/deepfm.py",
    "corerec/engines/unionizedFilterEngine/sar.py",
    "corerec/engines/unionizedFilterEngine/sli.py",
    "corerec/engines/unionizedFilterEngine/sum.py",
    "corerec/engines/nasrec.py",
    "corerec/engines/unionizedFilterEngine/bayesian_method_base/bpr_base.py",
    "corerec/engines/unionizedFilterEngine/graph_based_base/lightgcn.py",
    "corerec/engines/unionizedFilterEngine/nn_base/DeepRec_base.py",
    "corerec/engines/unionizedFilterEngine/bayesian_method_base/bprmf_base.py",
    "corerec/engines/sasrec.py",
]


def fix_orphaned_methods(content: str) -> str:
    """Remove orphaned methods and docstrings that appear before class definitions."""
    lines = content.split('\n')
    fixed_lines = []
    in_orphaned_block = False
    orphaned_indent = None
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this is an orphaned docstring (triple quotes with too much indentation)
        docstring_match = re.match(r'^(\s+)"""', line)
        if docstring_match and not any('class ' in prev_line for prev_line in lines[max(0, i-10):i]):
            indent = len(docstring_match.group(1))
            if indent >= 8:  # Too much indentation for module level
                # Skip this docstring block
                orphaned_indent = indent
                in_orphaned_block = True
                i += 1
                # Skip until we find the closing triple quotes
                while i < len(lines) and '"""' not in lines[i]:
                    i += 1
                if i < len(lines):
                    i += 1  # Skip the closing line
                in_orphaned_block = False
                orphaned_indent = None
                continue
        
        # Check if this is an orphaned method definition (indented def at module level)
        match = re.match(r'^(\s+)def\s+\w+\(self', line)
        if match and not any('class ' in prev_line for prev_line in lines[max(0, i-10):i]):
            # This looks like an orphaned method
            indent = len(match.group(1))
            if indent >= 8:  # Too much indentation for module level
                # Skip this method until we find a class or another top-level definition
                orphaned_indent = indent
                in_orphaned_block = True
                i += 1
                continue
        
        # If we're in an orphaned block, skip until dedent
        if in_orphaned_block:
            if line.strip() == '' or (line.startswith(' ') and len(line) - len(line.lstrip()) < orphaned_indent - 4) or not line.startswith(' '):
                in_orphaned_block = False
                orphaned_indent = None
            else:
                i += 1
                continue
        
        fixed_lines.append(line)
        i += 1
    
    return '\n'.join(fixed_lines)


def fix_indentation_issues(content: str) -> str:
    """Fix common indentation issues."""
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Fix methods that are incorrectly indented inside classes
        # Look for def statements with too much indentation (likely should be 4 spaces)
        if re.match(r'^        def\s+\w+\(', line):  # 8 spaces - likely should be 4
            # Check if we're inside a class
            # Look backwards for class definition
            for j in range(i-1, max(0, i-20), -1):
                if 'class ' in lines[j] and ':' in lines[j]:
                    # We're in a class, fix indentation
                    line = '    ' + line.lstrip()
                    break
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)


def fix_file(filepath: Path) -> bool:
    """Fix a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        
        # Remove orphaned methods at the top
        content = fix_orphaned_methods(content)
        
        # Fix indentation issues
        content = fix_indentation_issues(content)
        
        if content != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error fixing {filepath}: {e}", file=sys.stderr)
        return False


def main():
    """Main function."""
    repo_root = Path(__file__).parent.parent
    
    fixed_count = 0
    for rel_path in FAILED_FILES:
        filepath = repo_root / rel_path
        if filepath.exists():
            if fix_file(filepath):
                print(f"Fixed: {rel_path}")
                fixed_count += 1
            else:
                print(f"No changes needed: {rel_path}")
        else:
            print(f"File not found: {rel_path}", file=sys.stderr)
    
    print(f"\nFixed {fixed_count} files")
    return 0


if __name__ == '__main__':
    sys.exit(main())

