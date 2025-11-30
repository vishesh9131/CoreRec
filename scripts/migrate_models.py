#!/usr/bin/env python3
"""
Automated Migration Script for CoreRec Models

This script automates the migration of models from BaseCorerec to BaseRecommender.
It handles:
- Import statement updates
- Class inheritance changes
- Adding save/load methods
- Updating exception handling

Usage:
    python scripts/migrate_model.py <model_file_path>
    python scripts/migrate_model.py --all  # Migrate all models

Author: Vishesh Yadav
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple


# Template for save/load methods
SAVE_LOAD_TEMPLATE = '''
    def save(self, path: Union[str, Path], **kwargs) -> None:
        """Save model to disk."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path_obj, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        if self.verbose:
            logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> '{class_name}':
        """Load model from disk."""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        if hasattr(model, 'verbose') and model.verbose:
            logger.info(f"Model loaded from {path}")
        
        return model
'''


def migrate_imports(content: str) -> Tuple[str, bool]:
    """Migrate import statements."""
    modified = False

    # Replace BaseCorerec import
    if "from corerec.base_recommender import BaseCorerec" in content:
        content = content.replace(
            "from corerec.base_recommender import BaseCorerec",
            "from corerec.api.base_recommender import BaseRecommender",
        )
        modified = True

    # Add necessary imports if not present
    imports_to_add = []

    if "import pickle" not in content and "from pickle import" not in content:
        imports_to_add.append("import pickle")

    if "from pathlib import Path" not in content and "import pathlib" not in content:
        imports_to_add.append("from pathlib import Path")

    if "Union[str, Path]" in content and "from typing import" in content:
        # Check if Union is already imported
        if not re.search(r"from typing import.*Union", content):
            # Find the typing import line and add Union
            content = re.sub(r"(from typing import[^)]+)", r"\1, Union", content)
            modified = True

    # Add exception imports
    if "ValidationError" in content and "from corerec.api.exceptions" not in content:
        imports_to_add.append(
            "from corerec.api.exceptions import ModelNotFittedError, InvalidParameterError"
        )

    # Insert imports after existing imports
    if imports_to_add:
        # Find the last import statement
        import_lines = [
            i
            for i, line in enumerate(content.split("\n"))
            if line.strip().startswith(("import ", "from "))
        ]
        if import_lines:
            lines = content.split("\n")
            last_import_idx = import_lines[-1]
            for imp in reversed(imports_to_add):
                lines.insert(last_import_idx + 1, imp)
            content = "\n".join(lines)
            modified = True

    return content, modified


def migrate_class_definition(content: str) -> Tuple[str, bool]:
    """Migrate class definitions from BaseCorerec to BaseRecommender."""
    modified = False

    # Replace class inheritance
    pattern = r"class\s+(\w+)\(BaseCorerec\)"
    if re.search(pattern, content):
        content = re.sub(pattern, r"class \1(BaseRecommender)", content)
        modified = True

    return content, modified


def migrate_exceptions(content: str) -> Tuple[str, bool]:
    """Migrate exception handling to use custom exceptions."""
    modified = False

    # Replace ValueError for parameter validation
    if "raise ValueError" in content:
        # This is a heuristic - might need manual review
        content = re.sub(
            r"raise ValueError\((.*?must.*?)\)", r"raise InvalidParameterError(\1)", content
        )
        modified = True

    return content, modified


def add_save_load_methods(content: str, class_name: str) -> Tuple[str, bool]:
    """Add save/load methods if not present."""
    modified = False

    # Check if save method exists
    if "def save(self" not in content:
        # Find the end of the class (last method)
        # Insert save/load methods before the last line of the class
        save_load = SAVE_LOAD_TEMPLATE.format(class_name=class_name)

        # Find the class definition
        class_match = re.search(rf"class {class_name}\(.*?\):", content)
        if class_match:
            # Find the last method in the class (heuristic: last def before next class or EOF)
            # This is a simple approach - insert at end of file if it's the only class
            content = content.rstrip() + "\n" + save_load + "\n"
            modified = True

    return content, modified


def migrate_fit_return(content: str, class_name: str) -> Tuple[str, bool]:
    """Ensure fit method returns self."""
    modified = False

    # Find fit method
    fit_pattern = r"def fit\(self[^)]*\)[^:]*:"
    if re.search(fit_pattern, content):
        # Check if there's already a return self
        # This is a heuristic check
        lines = content.split("\n")
        in_fit = False
        fit_indent = 0

        for i, line in enumerate(lines):
            if re.match(r"\s+def fit\(", line):
                in_fit = True
                fit_indent = len(line) - len(line.lstrip())
            elif in_fit:
                # Check for end of fit method (next method or same indent level)
                if line.strip() and not line.startswith(" " * (fit_indent + 4)):
                    # End of fit method
                    if "return self" not in "\n".join(lines[max(0, i - 10) : i]):
                        # Add return self before this line
                        indent = " " * (fit_indent + 4)
                        lines.insert(i, f"{indent}return self")
                        modified = True
                    in_fit = False

        if modified:
            content = "\n".join(lines)

    return content, modified


def migrate_file(file_path: Path, dry_run: bool = False) -> bool:
    """
    Migrate a single model file.

    Args:
        file_path: Path to the model file
        dry_run: If True, only print what would be changed

    Returns:
        True if file was modified
    """
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Processing: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content
        overall_modified = False

        # Extract class name
        class_match = re.search(r"class\s+(\w+)\(BaseCorerec\)", content)
        if not class_match:
            print(f"  ⚠️  No BaseCorerec class found, skipping")
            return False

        class_name = class_match.group(1)
        print(f"  Found class: {class_name}")

        # Apply migrations
        content, mod = migrate_imports(content)
        overall_modified = overall_modified or mod
        if mod:
            print("  ✓ Updated imports")

        content, mod = migrate_class_definition(content)
        overall_modified = overall_modified or mod
        if mod:
            print(f"  ✓ Changed {class_name}(BaseCorerec) → {class_name}(BaseRecommender)")

        content, mod = migrate_exceptions(content)
        overall_modified = overall_modified or mod
        if mod:
            print("  ✓ Updated exception handling")

        content, mod = add_save_load_methods(content, class_name)
        overall_modified = overall_modified or mod
        if mod:
            print("  ✓ Added save/load methods")

        content, mod = migrate_fit_return(content, class_name)
        overall_modified = overall_modified or mod
        if mod:
            print("  ✓ Updated fit() to return self")

        # Write back if modified
        if overall_modified and not dry_run:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  ✅ File migrated successfully!")
        elif overall_modified:
            print(f"  📝 Would modify file (dry run)")
        else:
            print(f"  ℹ️  No changes needed")

        return overall_modified

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def find_all_model_files(base_dir: Path) -> List[Path]:
    """Find all Python files that might be models."""
    model_files = []

    # Search in engines directory
    engines_dir = base_dir / "corerec" / "engines"
    if engines_dir.exists():
        for file in engines_dir.rglob("*.py"):
            if file.stem not in ["__init__", "__pycache__"]:
                model_files.append(file)

    return model_files


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Migrate CoreRec models to BaseRecommender")
    parser.add_argument("files", nargs="*", help="Specific files to migrate")
    parser.add_argument("--all", action="store_true", help="Migrate all model files")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be changed without modifying files"
    )
    parser.add_argument("--base-dir", type=str, default=".", help="Base directory of CoreRec")

    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()

    if args.all:
        print("🔍 Finding all model files...")
        files_to_migrate = find_all_model_files(base_dir)
        print(f"Found {len(files_to_migrate)} model files")
    elif args.files:
        files_to_migrate = [Path(f) for f in args.files]
    else:
        parser.print_help()
        return

    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No files will be modified\n")

    print(f"\n{'='*60}")
    print(f"CoreRec Model Migration Script")
    print(f"{'='*60}")

    migrated = 0
    skipped = 0

    for file_path in files_to_migrate:
        if migrate_file(file_path, dry_run=args.dry_run):
            migrated += 1
        else:
            skipped += 1

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Migrated: {migrated}")
    print(f"  Skipped: {skipped}")
    print(f"  Total: {len(files_to_migrate)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
