"""
Script to fix all stub engine implementations by adding NotImplementedError.

This script identifies engines that are just stubs (pass statements) and 
replaces them with helpful NotImplementedError messages guiding users to
working alternatives.

Run: python scripts/fix_stub_engines.py
"""

import os
import re
from pathlib import Path


# Stub files to fix (relative to corerec/engines/)
STUB_FILES = {
    # Matrix Factorization stubs
    "unionizedFilterEngine/mf_base/weighted_matrix_factorization_base.py": "MatrixFactorizationBase or ALSRecommender",
    "unionizedFilterEngine/mf_base/Implicit_feedback_mf_base.py": "ALSRecommender with implicit=True",
    "unionizedFilterEngine/mf_base/sgd_matrix_factorization_base.py": "MatrixFactorizationBase",
    "unionizedFilterEngine/mf_base/neural_matrix_factorization_base.py": "NCF (Neural Collaborative Filtering)",
    "unionizedFilterEngine/mf_base/svd_base.py": "sklearn.decomposition.TruncatedSVD",
    "unionizedFilterEngine/mf_base/hierarchical_poisson_factorization_base.py": "ALSRecommender",
    "unionizedFilterEngine/mf_base/contextual_matrix_factorization_base.py": "DeepFM or DCN",
    "unionizedFilterEngine/mf_base/hybrid_matrix_factorization_base.py": "DeepFM",
    "unionizedFilterEngine/mf_base/incremental_matrix_factorization_base.py": "MatrixFactorizationBase with warm_start",
    "unionizedFilterEngine/mf_base/kernelized_matrix_factorization_base.py": "sklearn.kernel_approximation",
    "unionizedFilterEngine/mf_base/nmf_base.py": "sklearn.decomposition.NMF",
    "unionizedFilterEngine/mf_base/pmf_base.py": "MatrixFactorizationBase",
    "unionizedFilterEngine/mf_base/temporal_matrix_factorization_base.py": "SASRec or DIEN",
    "unionizedFilterEngine/mf_base/svdpp_base.py": "MatrixFactorizationBase",
    # Graph-based stubs
    "unionizedFilterEngine/graph_based_base/graph_based_ufilter_base.py": "LightGCN or GNNRec",
    "unionizedFilterEngine/graph_based_base/heterogeneous_network_ufilter_base.py": "GNNRec",
    "unionizedFilterEngine/graph_based_base/geoimc_base.py": "LightGCN",
    "unionizedFilterEngine/graph_based_base/multi_view_ufilter_base.py": "GNNRec",
    "unionizedFilterEngine/graph_based_base/multi_relational_ufilter_base.py": "GNNRec",
    "unionizedFilterEngine/graph_based_base/edge_aware_ufilter_base.py": "LightGCN",
    "unionizedFilterEngine/graph_based_base/gnn_ufilter_base.py": "GNNRec",
    # Attention/Transformer stubs
    "unionizedFilterEngine/attention_mechanism_base/Transformer_based_uf_base.py": "SASRec",
    "unionizedFilterEngine/attention_mechanism_base/Attention_based_uf_base.py": "DIN or DIEN",
    # Bayesian stubs
    "unionizedFilterEngine/bayesian_method_base/Bayesian_mf_base.py": "ALSRecommender",
    "unionizedFilterEngine/bayesian_method_base/Bayesian_Personalized_Ranking_Extensions_base.py": "NCF",
    "unionizedFilterEngine/bayesian_method_base/PGM_uf.py": "ALSRecommender",
    "unionizedFilterEngine/bayesian_method_base/multinomial_vae.py": "VAE-based models",
}


TEMPLATE = '''"""
{algorithm_name} - NOT YET IMPLEMENTED

WARNING: This feature is currently under development and not ready for production use.

For similar functionality, please use: {alternative}

Expected implementation: CoreRec v0.6.0 or later
Track progress: https://github.com/vishesh9131/CoreRec/issues
"""


class {class_name}:
    """
    {algorithm_name} - Placeholder for future implementation.
    
    This class will raise NotImplementedError when instantiated.
    Please use the recommended alternatives listed in the module docstring.
    
    Raises:
        NotImplementedError: This feature is not yet implemented
    """
    
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            f"\\n\\n{class_name} is not yet implemented.\\n\\n"
            f"This feature is planned for CoreRec v0.6.0 or later.\\n\\n"
            f"For similar functionality, please use: {alternative}\\n\\n"
            f"Track implementation progress:\\n"
            f"https://github.com/vishesh9131/CoreRec/issues"
        )
'''


def extract_class_name(file_path: Path) -> str:
    """Extract class name from file content."""
    try:
        with open(file_path, "r") as f:
            content = f.read()
            match = re.search(r"class\s+(\w+)", content)
            if match:
                return match.group(1)
    except Exception:
        pass

    # Fallback: derive from filename
    return "".join(word.capitalize() for word in file_path.stem.split("_"))


def fix_stub_file(file_path: Path, alternative: str):
    """Fix a single stub file."""
    class_name = extract_class_name(file_path)
    algorithm_name = " ".join(class_name.replace("Base", "").split("_")).title()

    new_content = TEMPLATE.format(
        algorithm_name=algorithm_name, class_name=class_name, alternative=alternative
    )

    with open(file_path, "w") as f:
        f.write(new_content)

    print(f"Fixed: {file_path}")


def main():
    """Fix all stub engine files."""
    base_dir = Path(__file__).parent.parent / "corerec" / "engines"

    print("Fixing stub engine implementations...")
    print(f"Base directory: {base_dir}")
    print(f"Total stubs to fix: {len(STUB_FILES)}\n")

    fixed_count = 0
    errors = []

    for rel_path, alternative in STUB_FILES.items():
        file_path = base_dir / rel_path

        if not file_path.exists():
            errors.append(f"File not found: {file_path}")
            continue

        try:
            fix_stub_file(file_path, alternative)
            fixed_count += 1
        except Exception as e:
            errors.append(f"Error fixing {file_path}: {str(e)}")

    print(f"\n\nSummary:")
    print(f"Successfully fixed: {fixed_count}/{len(STUB_FILES)} files")

    if errors:
        print(f"\nErrors encountered:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("All stub files fixed successfully!")


if __name__ == "__main__":
    main()
