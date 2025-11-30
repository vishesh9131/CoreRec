#!/usr/bin/env python3
"""
Aggressively remove all orphaned code blocks that prevent Black formatting.
"""

import re
import sys
from pathlib import Path

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


def remove_orphaned_code(content: str) -> str:
    """Remove all orphaned code blocks before class definitions."""
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Find the first class definition
        if re.match(r'^\s*class\s+\w+', line):
            # We found a class - keep everything from here
            fixed_lines.extend(lines[i:])
            break
        
        # Before class definition, only keep imports and top-level statements
        stripped = line.strip()
        
        # Keep imports, comments, blank lines, and module-level docstrings
        if (stripped.startswith('import ') or 
            stripped.startswith('from ') or
            stripped.startswith('#') or
            stripped == '' or
            (stripped.startswith('"""') and i < 10)):  # Module docstring at top
            fixed_lines.append(line)
            i += 1
        else:
            # Skip orphaned code (indented statements before class)
            if line.startswith(' ') and len(line) - len(line.lstrip()) >= 8:
                # Skip this orphaned block
                indent_level = len(line) - len(line.lstrip())
                i += 1
                # Skip until we find something at a lower indent or class
                while i < len(lines):
                    next_line = lines[i]
                    if re.match(r'^\s*class\s+\w+', next_line):
                        # Found class, break and process it
                        break
                    if next_line.strip() == '':
                        i += 1
                        continue
                    next_indent = len(next_line) - len(next_line.lstrip())
                    if next_indent < indent_level - 4:
                        # Dedented enough, might be valid code
                        break
                    i += 1
            else:
                # Not heavily indented, might be valid - but skip if it's clearly orphaned
                if any(keyword in stripped for keyword in ['def ', 'if ', 'for ', 'while ', 'try:', 'except', 'return ', 'raise ']):
                    # Looks like code, skip it if we haven't seen a class yet
                    i += 1
                    continue
                fixed_lines.append(line)
                i += 1
    
    return '\n'.join(fixed_lines)


def fix_file(filepath: Path) -> bool:
    """Fix a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        content = remove_orphaned_code(content)
        
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
            print(f"File not found: {rel_path}", file=sys.stderr)
    
    print(f"\nFixed {fixed_count} files")
    return 0


if __name__ == '__main__':
    sys.exit(main())

