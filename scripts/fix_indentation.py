#!/usr/bin/env python3
"""
Fix indentation issues in injected methods.

The method injection script added methods at wrong indentation level.
This script fixes them to be properly indented as class methods.
"""

import re
from pathlib import Path
from typing import List

# Files that had methods injected
FILES_TO_FIX = [
    "corerec/engines/contentFilterEngine/graph_based_algorithms/gnn.py",
    "corerec/engines/contentFilterEngine/tfidf_recommender.py",
    "corerec/engines/deepfm.py",
    "corerec/engines/gnnrec.py",
    "corerec/engines/mind.py",
    "corerec/engines/nasrec.py",
    "corerec/engines/sasrec.py",
    "corerec/engines/unionizedFilterEngine/bayesian_method_base/bpr_base.py",
    "corerec/engines/unionizedFilterEngine/bayesian_method_base/bprmf_base.py",
    "corerec/engines/unionizedFilterEngine/bayesian_method_base/vmf_base.py",
    "corerec/engines/unionizedFilterEngine/graph_based_base/geoimc_base.py",
    "corerec/engines/unionizedFilterEngine/graph_based_base/lightgcn.py",
    "corerec/engines/unionizedFilterEngine/graph_based_base/lightgcn_base.py",
    "corerec/engines/unionizedFilterEngine/mf_base/A2SVD_base.py",
    "corerec/engines/unionizedFilterEngine/mf_base/ALS_base.py",
    "corerec/engines/unionizedFilterEngine/mf_base/factorization_machine_base.py",
    "corerec/engines/unionizedFilterEngine/mf_base/matrix_factorization.py",
    "corerec/engines/unionizedFilterEngine/mf_base/matrix_factorization_base.py",
    "corerec/engines/unionizedFilterEngine/mf_base/svd_base.py",
    "corerec/engines/unionizedFilterEngine/mf_base/user_based_uf.py",
    "corerec/engines/unionizedFilterEngine/nn_base/AFM_base.py",
    "corerec/engines/unionizedFilterEngine/nn_base/AutoFI_base.py",
    "corerec/engines/unionizedFilterEngine/nn_base/AutoInt_base.py",
    "corerec/engines/unionizedFilterEngine/nn_base/BST_base.py",
    "corerec/engines/unionizedFilterEngine/nn_base/DLRM_base.py",
    "corerec/engines/unionizedFilterEngine/nn_base/DeepRec_base.py",
    "corerec/engines/unionizedFilterEngine/nn_base/bivae_base.py",
    "corerec/engines/unionizedFilterEngine/nn_base/gru_cf.py",
    "corerec/engines/unionizedFilterEngine/nn_base/nextitnet.py",
    "corerec/engines/unionizedFilterEngine/rbm.py",
    "corerec/engines/unionizedFilterEngine/rlrmc.py",
    "corerec/engines/unionizedFilterEngine/sar.py",
    "corerec/engines/unionizedFilterEngine/sli.py",
    "corerec/engines/unionizedFilterEngine/sum.py",
]


def fix_indentation(file_path: Path) -> bool:
    """Fix indentation of wrongly indented methods."""
    try:
        with open(file_path, "r") as f:
            content = f.read()

        # Pattern to find module-level def that should be class method
        # Look for lines that start with "    def " at the beginning
        pattern = r"^    def (predict|save|load)\("

        if not re.search(pattern, content, re.MULTILINE):
            # Methods are already properly indented or don't exist
            return True

        # Fix by adding proper indentation (4 more spaces)
        lines = content.split("\n")
        fixed_lines = []
        in_wrongly_indented_method = False

        for line in lines:
            # Check if this is a wrongly indented method definition
            if re.match(r"^    def (predict|save|load|recommend)\(", line):
                in_wrongly_indented_method = True
                # Add 4 spaces of indentation
                fixed_lines.append("    " + line)
            elif in_wrongly_indented_method:
                # Continue fixing indentation until we hit a line that's not indented
                if line and not line[0].isspace():
                    # End of method
                    in_wrongly_indented_method = False
                    fixed_lines.append(line)
                elif line.strip() == "":
                    # Empty line
                    fixed_lines.append(line)
                else:
                    # Add 4 spaces to maintain relative indentation
                    fixed_lines.append("    " + line)
            else:
                fixed_lines.append(line)

        # Write back
        with open(file_path, "w") as f:
            f.write("\n".join(fixed_lines))

        return True
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False


def main():
    base_dir = Path("/Users/visheshyadav/Documents/GitHub/CoreRec")

    print("Fixing indentation in injected methods...")
    print("=" * 60)

    success = 0
    failed = 0

    for rel_path in FILES_TO_FIX:
        file_path = base_dir / rel_path
        if not file_path.exists():
            continue

        print(f"Fixing: {rel_path}")
        if fix_indentation(file_path):
            print("  ✓ Fixed")
            success += 1
        else:
            print("  ✗ Failed")
            failed += 1

    print("=" * 60)
    print(f"Fixed: {success}, Failed: {failed}")


if __name__ == "__main__":
    main()
