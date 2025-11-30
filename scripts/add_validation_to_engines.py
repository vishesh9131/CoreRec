"""
Script to add input validation to all engine fit() and recommend() methods.

This script:
1. Finds fit() methods in engine files
2. Adds validate_fit_inputs() at the beginning
3. Finds recommend() methods
4. Adds validate_model_fitted(), validate_user_id(), validate_top_k()
5. Adds necessary imports

Run: python scripts/add_validation_to_engines.py
"""

import os
import re
from pathlib import Path
from typing import List


VALIDATION_IMPORTS = """from corerec.utils.validation import (
    validate_fit_inputs,
    validate_user_id,
    validate_item_id,
    validate_top_k,
    validate_model_fitted,
    ValidationError
)"""


def has_validation_import(content: str) -> bool:
    """Check if file already imports validation utilities."""
    return "from corerec.utils.validation import" in content or "validate_fit_inputs" in content


def add_validation_import(content: str) -> str:
    """Add validation imports after other imports."""
    lines = content.split("\n")

    # Find the last import line
    last_import_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("import ") or line.strip().startswith("from "):
            last_import_idx = i

    # Insert validation imports
    if last_import_idx > 0:
        lines.insert(last_import_idx + 1, "")
        lines.insert(last_import_idx + 2, VALIDATION_IMPORTS)

    return "\n".join(lines)


def add_validation_to_fit(content: str) -> str:
    """Add validation to fit() methods."""
    lines = content.split("\n")
    modified = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # Find fit() method definition
        if re.match(r"\s*def fit\s*\(", line):
            # Find the end of method signature and docstring
            j = i + 1
            in_docstring = False
            docstring_char = None

            while j < len(lines):
                if '"""' in lines[j] or "'''" in lines[j]:
                    if not in_docstring:
                        in_docstring = True
                        docstring_char = '"""' if '"""' in lines[j] else "'''"
                    elif docstring_char in lines[j]:
                        in_docstring = False
                        j += 1
                        break
                elif not in_docstring and lines[j].strip() and not lines[j].strip().startswith("#"):
                    break
                j += 1

            # Check if validation already exists
            validation_exists = False
            for k in range(j, min(j + 10, len(lines))):
                if "validate_fit_inputs" in lines[k]:
                    validation_exists = True
                    break

            if not validation_exists:
                # Add validation after docstring/signature
                indent = len(lines[i]) - len(lines[i].lstrip())
                indent_str = " " * (indent + 4)

                # Insert validation
                validation_lines = [
                    f"{indent_str}# Validate inputs",
                    f"{indent_str}validate_fit_inputs(user_ids, item_ids, ratings)",
                    f"{indent_str}",
                ]

                for idx, val_line in enumerate(validation_lines):
                    lines.insert(j + idx, val_line)

                modified = True
                i = j + len(validation_lines)
            else:
                i = j

        i += 1

    return "\n".join(lines), modified


def add_validation_to_recommend(content: str) -> str:
    """Add validation to recommend() methods."""
    lines = content.split("\n")
    modified = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # Find recommend() method definition
        if re.match(r"\s*def recommend\s*\(", line):
            # Find the end of method signature and docstring
            j = i + 1
            in_docstring = False
            docstring_char = None

            while j < len(lines):
                if '"""' in lines[j] or "'''" in lines[j]:
                    if not in_docstring:
                        in_docstring = True
                        docstring_char = '"""' if '"""' in lines[j] else "'''"
                    elif docstring_char in lines[j]:
                        in_docstring = False
                        j += 1
                        break
                elif not in_docstring and lines[j].strip() and not lines[j].strip().startswith("#"):
                    break
                j += 1

            # Check if validation already exists
            validation_exists = False
            for k in range(j, min(j + 10, len(lines))):
                if "validate_model_fitted" in lines[k] or "validate_user_id" in lines[k]:
                    validation_exists = True
                    break

            if not validation_exists:
                # Add validation after docstring/signature
                indent = len(lines[i]) - len(lines[i].lstrip())
                indent_str = " " * (indent + 4)

                # Insert validation
                validation_lines = [
                    f"{indent_str}# Validate inputs",
                    f"{indent_str}validate_model_fitted(self.is_fitted, self.name)",
                    f"{indent_str}validate_user_id(user_id, self.user_map if hasattr(self, 'user_map') else {{}})",
                    f"{indent_str}validate_top_k(top_k if 'top_k' in locals() else 10)",
                    f"{indent_str}",
                ]

                for idx, val_line in enumerate(validation_lines):
                    lines.insert(j + idx, val_line)

                modified = True
                i = j + len(validation_lines)
            else:
                i = j

        i += 1

    return "\n".join(lines), modified


def process_file(file_path: Path, dry_run: bool = False) -> bool:
    """Process a single file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check if file has fit() or recommend() methods
        has_fit = re.search(r"def fit\s*\(", content)
        has_recommend = re.search(r"def recommend\s*\(", content)

        if not (has_fit or has_recommend):
            return False

        modified = False
        new_content = content

        # Add validation to fit() if exists
        if has_fit:
            new_content, fit_modified = add_validation_to_fit(new_content)
            modified = modified or fit_modified

        # Add validation to recommend() if exists
        if has_recommend:
            new_content, rec_modified = add_validation_to_recommend(new_content)
            modified = modified or rec_modified

        # Add imports if modified and not already present
        if modified and not has_validation_import(new_content):
            new_content = add_validation_import(new_content)

        if modified:
            if not dry_run:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                print(f"✓ Added validation to: {file_path}")
            else:
                print(f"Would add validation to: {file_path}")
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
        if "__pycache__" in root or "test" in root.lower():
            continue

        for file in files:
            if file.endswith(".py") and not file.startswith("__"):
                engine_files.append(Path(root) / file)

    return engine_files


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Add validation to engine files")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be changed without modifying files"
    )
    parser.add_argument(
        "--path", type=str, default=None, help="Specific file or directory to process"
    )

    args = parser.parse_args()

    # Determine base directory
    base_dir = Path(__file__).parent.parent / "corerec" / "engines"
    if args.path:
        target_path = Path(args.path)
        if target_path.is_file():
            files_to_process = [target_path]
        else:
            files_to_process = find_engine_files(target_path)
    else:
        files_to_process = find_engine_files(base_dir)

    print(
        f"{'DRY RUN - ' if args.dry_run else ''}Processing {len(files_to_process)} engine files..."
    )
    print(f"Base directory: {base_dir}\n")

    modified_count = 0

    for file_path in files_to_process:
        if process_file(file_path, dry_run=args.dry_run):
            modified_count += 1

    print(
        f"\n{'Would modify' if args.dry_run else 'Modified'}: {modified_count}/{len(files_to_process)} files"
    )

    if args.dry_run:
        print("\nRun without --dry-run to apply changes")


if __name__ == "__main__":
    main()
