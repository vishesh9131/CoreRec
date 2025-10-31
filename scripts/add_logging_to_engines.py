"""
Script to replace print() statements with proper logging in all engine files.

This script:
1. Finds all print() statements in engine files
2. Replaces them with logger.info() or logger.debug()
3. Adds logging import if missing
4. Preserves the original message format

Run: python scripts/add_logging_to_engines.py
"""

import os
import re
from pathlib import Path
from typing import List, Tuple


def has_print_statements(content: str) -> bool:
    """Check if file contains print statements."""
    return bool(re.search(r'\bprint\s*\(', content))


def has_logging_import(content: str) -> bool:
    """Check if file already imports logging."""
    return bool(re.search(r'import logging', content))


def add_logging_import(content: str) -> str:
    """Add logging import after other imports."""
    lines = content.split('\n')
    
    # Find the last import line
    last_import_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('import ') or line.strip().startswith('from '):
            last_import_idx = i
    
    # Insert logging import after last import
    if last_import_idx > 0:
        lines.insert(last_import_idx + 1, 'import logging')
        lines.insert(last_import_idx + 2, '')
        lines.insert(last_import_idx + 3, 'logger = logging.getLogger(__name__)')
    
    return '\n'.join(lines)


def replace_print_with_logging(content: str, verbose: bool = False) -> str:
    """
    Replace print() statements with logger calls.
    
    Pattern detection:
    - Training progress -> logger.info()
    - Error messages -> logger.error()
    - Debug info -> logger.debug()
    - Warnings -> logger.warning()
    """
    lines = content.split('\n')
    modified = False
    
    for i, line in enumerate(lines):
        # Skip comments
        if line.strip().startswith('#'):
            continue
        
        # Find print statements
        print_match = re.search(r'print\s*\((.*)\)', line)
        if print_match:
            message = print_match.group(1)
            indent = len(line) - len(line.lstrip())
            indent_str = ' ' * indent
            
            # Determine log level based on content
            message_lower = message.lower()
            
            if any(word in message_lower for word in ['error', 'fail', 'exception']):
                log_level = 'error'
            elif any(word in message_lower for word in ['warn', 'warning']):
                log_level = 'warning'
            elif any(word in message_lower for word in ['debug', 'trace']):
                log_level = 'debug'
            else:
                log_level = 'info'
            
            # Replace print with logger
            # Check if it's conditional on verbose
            if 'if self.verbose:' in lines[i-1] if i > 0 else False:
                # Already conditional, just replace print
                new_line = f"{indent_str}logger.{log_level}({message})"
            else:
                # Add verbose check
                new_line = f"{indent_str}if self.verbose:\n{indent_str}    logger.{log_level}({message})"
            
            lines[i] = new_line
            modified = True
            
            if verbose:
                print(f"  Replaced: print({message}) -> logger.{log_level}()")
    
    if modified and not has_logging_import(content):
        content = '\n'.join(lines)
        content = add_logging_import(content)
        return content
    
    return '\n'.join(lines)


def process_file(file_path: Path, dry_run: bool = False, verbose: bool = False) -> bool:
    """Process a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not has_print_statements(content):
            return False
        
        new_content = replace_print_with_logging(content, verbose=verbose)
        
        if new_content != content:
            if not dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"✓ Fixed: {file_path}")
            else:
                print(f"Would fix: {file_path}")
            return True
        
        return False
    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")
        return False


def find_engine_files(base_dir: Path) -> List[Path]:
    """Find all Python files in engines directory."""
    engine_files = []
    
    for root, dirs, files in os.walk(base_dir):
        # Skip test files and __pycache__
        if '__pycache__' in root or 'test' in root.lower():
            continue
        
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                engine_files.append(Path(root) / file)
    
    return engine_files


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Add logging to engine files')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without modifying files')
    parser.add_argument('--verbose', action='store_true', help='Show detailed changes')
    parser.add_argument('--path', type=str, default=None, help='Specific file or directory to process')
    
    args = parser.parse_args()
    
    # Determine base directory
    base_dir = Path(__file__).parent.parent / 'corerec' / 'engines'
    if args.path:
        target_path = Path(args.path)
        if target_path.is_file():
            files_to_process = [target_path]
        else:
            files_to_process = find_engine_files(target_path)
    else:
        files_to_process = find_engine_files(base_dir)
    
    print(f"{'DRY RUN - ' if args.dry_run else ''}Processing {len(files_to_process)} engine files...")
    print(f"Base directory: {base_dir}\n")
    
    modified_count = 0
    
    for file_path in files_to_process:
        if process_file(file_path, dry_run=args.dry_run, verbose=args.verbose):
            modified_count += 1
    
    print(f"\n{'Would modify' if args.dry_run else 'Modified'}: {modified_count}/{len(files_to_process)} files")
    
    if args.dry_run:
        print("\nRun without --dry-run to apply changes")


if __name__ == '__main__':
    main()

