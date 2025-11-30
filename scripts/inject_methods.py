#!/usr/bin/env python3
"""
Method Injection Script for Incomplete Models

This script adds missing predict(), save(), and load() methods to incomplete models
using DCN as the template.

Author: Vishesh Yadav
"""

import re
from pathlib import Path
from typing import Dict, List

# Template methods based on DCN implementation
PREDICT_METHOD_TEMPLATE = '''
    def predict(self, user_id: int, item_id: int, **kwargs) -> float:
        """
        Predict rating/score for a user-item pair.
        
        Args:
            user_id: User ID
            item_id: Item ID
            **kwargs: Additional arguments
            
        Returns:
            Predicted score/rating
        """
        from corerec.api.exceptions import ModelNotFittedError
        
        if not self.is_fitted:
            raise ModelNotFittedError(f"{self.name} must be fitted before making predictions")
        
        # Check if user/item are known
        if hasattr(self, 'user_map') and user_id not in self.user_map:
            return 0.0
        if hasattr(self, 'item_map') and item_id not in self.item_map:
            return 0.0
        
        # Get internal indices
        if hasattr(self, 'user_map'):
            user_idx = self.user_map.get(user_id, 0)
        else:
            user_idx = user_id
            
        if hasattr(self, 'item_map'):
            item_idx = self.item_map.get(item_id, 0)
        else:
            item_idx = item_id
        
        # Model-specific prediction logic
        # This is a fallback - ideally should be customized per model
        try:
            if hasattr(self, 'model') and self.model is not None:
                import torch
                if hasattr(self.model, 'predict'):
                    # Use model's internal predict if available
                    with torch.no_grad():
                        self.model.eval()
                        score = self.model.predict(user_idx, item_idx)
                        if isinstance(score, torch.Tensor):
                            return float(score.item())
                        return float(score)
            
            # Fallback: return neutral score
            return 0.5
            
        except Exception as e:
            import logging
            logging.warning(f"Prediction failed for {self.name}: {e}")
            return 0.0
'''

SAVE_METHOD_TEMPLATE = '''
    def save(self, path: Union[str, Path], **kwargs) -> None:
        """
        Save model to disk using pickle.
        
        Args:
            path: File path to save the model
            **kwargs: Additional arguments
        """
        import pickle
        from pathlib import Path
        
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path_obj, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        if self.verbose:
            import logging
            logging.info(f"{self.name} model saved to {path}")
'''

LOAD_METHOD_TEMPLATE = '''
    @classmethod
    def load(cls, path: Union[str, Path], **kwargs):
        """
        Load model from disk.
        
        Args:
            path: File path to load the model from
            **kwargs: Additional arguments
            
        Returns:
            Loaded model instance
        """
        import pickle
        from pathlib import Path
        
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        if hasattr(model, 'verbose') and model.verbose:
            import logging
            logging.info(f"Model loaded from {path}")
        
        return model
'''

# List of incomplete models with their missing methods
INCOMPLETE_MODELS = {
    "corerec/engines/contentFilterEngine/graph_based_algorithms/gnn.py": {
        "class": "MINDRecommender",
        "missing": ["predict"],
    },
    "corerec/engines/contentFilterEngine/tfidf_recommender.py": {
        "class": "TFIDFRecommender",
        "missing": ["load", "predict", "save"],
    },
    "corerec/engines/deepfm.py": {"class": "DeepFM", "missing": ["load", "predict", "save"]},
    "corerec/engines/gnnrec.py": {"class": "GNNRec", "missing": ["load", "predict", "save"]},
    "corerec/engines/mind.py": {"class": "MIND", "missing": ["load", "predict", "save"]},
    "corerec/engines/nasrec.py": {"class": "NASRec", "missing": ["load", "predict", "save"]},
    "corerec/engines/sasrec.py": {"class": "SASRec", "missing": ["load", "save"]},
    "corerec/engines/unionizedFilterEngine/bayesian_method_base/bpr_base.py": {
        "class": "BPRBase",
        "missing": ["load", "predict", "save"],
    },
    "corerec/engines/unionizedFilterEngine/bayesian_method_base/bprmf_base.py": {
        "class": "BPRMFBase",
        "missing": ["load", "predict", "save"],
    },
    "corerec/engines/unionizedFilterEngine/bayesian_method_base/vmf_base.py": {
        "class": "VMFBase",
        "missing": ["load", "predict", "save"],
    },
    "corerec/engines/unionizedFilterEngine/graph_based_base/cornac_bpr.py": {
        "class": "CornacBPR",
        "missing": ["load", "predict", "save"],
    },
    "corerec/engines/unionizedFilterEngine/graph_based_base/fast.py": {
        "class": "FAST",
        "missing": ["load", "predict", "save"],
    },
    "corerec/engines/unionizedFilterEngine/graph_based_base/fast_recommender.py": {
        "class": "FASTRecommender",
        "missing": ["load", "predict", "save"],
    },
    "corerec/engines/unionizedFilterEngine/graph_based_base/geoMLC.py": {
        "class": "GeoMLC",
        "missing": ["load", "predict", "save"],
    },
    "corerec/engines/unionizedFilterEngine/graph_based_base/geoimc_base.py": {
        "class": "GeoIMC",
        "missing": ["load", "predict", "save"],
    },
    "corerec/engines/unionizedFilterEngine/graph_based_base/lightgcn.py": {
        "class": "LightGCN",
        "missing": ["load", "save"],
    },
    "corerec/engines/unionizedFilterEngine/graph_based_base/lightgcn_base.py": {
        "class": "LightGCNBase",
        "missing": ["load", "predict", "save"],
    },
    "corerec/engines/unionizedFilterEngine/mf_base/A2SVD_base.py": {
        "class": "A2SVD",
        "missing": ["load", "predict", "save"],
    },
    "corerec/engines/unionizedFilterEngine/mf_base/ALS_base.py": {
        "class": "ALSRecommender",
        "missing": ["load", "predict", "save"],
    },
    "corerec/engines/unionizedFilterEngine/mf_base/factorization_machine_base.py": {
        "class": "FactorizationMachineBase",
        "missing": ["load", "predict", "save"],
    },
    "corerec/engines/unionizedFilterEngine/mf_base/matrix_factorization.py": {
        "class": "MatrixFactorization",
        "missing": ["load", "predict", "save"],
    },
    "corerec/engines/unionizedFilterEngine/mf_base/matrix_factorization_base.py": {
        "class": "MatrixFactorizationBase",
        "missing": ["load", "predict", "recommend", "save"],
    },
    "corerec/engines/unionizedFilterEngine/mf_base/svd_base.py": {
        "class": "SVDRecommender",
        "missing": ["load", "predict", "save"],
    },
    "corerec/engines/unionizedFilterEngine/mf_base/user_based_uf.py": {
        "class": "UserBasedUF",
        "missing": ["load", "predict", "save"],
    },
    "corerec/engines/unionizedFilterEngine/nn_base/AFM_base.py": {
        "class": "AFM_base",
        "missing": ["load", "predict", "save"],
    },
    "corerec/engines/unionizedFilterEngine/nn_base/AutoFI_base.py": {
        "class": "AutoFI_base",
        "missing": ["predict"],
    },
    "corerec/engines/unionizedFilterEngine/nn_base/AutoInt_base.py": {
        "class": "AutoInt_base",
        "missing": ["predict"],
    },
    "corerec/engines/unionizedFilterEngine/nn_base/BST_base.py": {
        "class": "BST_base",
        "missing": ["predict"],
    },
    "corerec/engines/unionizedFilterEngine/nn_base/DLRM_base.py": {
        "class": "DLRM_base",
        "missing": ["recommend"],
    },
    "corerec/engines/unionizedFilterEngine/nn_base/DeepRec_base.py": {
        "class": "DeepRec_base",
        "missing": ["load", "save"],
    },
    "corerec/engines/unionizedFilterEngine/nn_base/bivae_base.py": {
        "class": "BiVAE_base",
        "missing": ["predict"],
    },
    "corerec/engines/unionizedFilterEngine/nn_base/gru_cf.py": {
        "class": "GRUCF",
        "missing": ["load", "predict", "save"],
    },
    "corerec/engines/unionizedFilterEngine/nn_base/nextitnet.py": {
        "class": "NextItNet",
        "missing": ["load", "predict", "save"],
    },
    "corerec/engines/unionizedFilterEngine/rbm.py": {
        "class": "RBM",
        "missing": ["load", "predict", "save"],
    },
    "corerec/engines/unionizedFilterEngine/rlrmc.py": {
        "class": "RLRMC",
        "missing": ["load", "save"],
    },
    "corerec/engines/unionizedFilterEngine/sar.py": {
        "class": "SAR",
        "missing": ["load", "predict", "save"],
    },
    "corerec/engines/unionizedFilterEngine/sli.py": {
        "class": "SLiRec",
        "missing": ["load", "predict", "save"],
    },
    "corerec/engines/unionizedFilterEngine/sum.py": {
        "class": "SUMModel",
        "missing": ["load", "predict", "save"],
    },
}


def add_typing_import(content: str) -> str:
    """Add Union and Path to typing imports if not present."""
    if "from typing import" in content and "Union" not in content:
        content = content.replace("from typing import", "from typing import Union, Path,")
    elif "from typing import" not in content:
        # Add new typing import after other imports
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                continue
            else:
                lines.insert(i, "from typing import Union, Path")
                break
        content = "\n".join(lines)

    return content


def inject_methods(file_path: Path, class_name: str, missing_methods: List[str]) -> bool:
    """Inject missing methods into a model file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Add typing imports if needed
        if "save" in missing_methods or "load" in missing_methods:
            content = add_typing_import(content)

        # Find the last line of the class
        # We'll append methods at the end before any trailing code
        lines = content.split("\n")

        # Find where to insert (end of class, before if __name__ or module-level code)
        insert_idx = len(lines) - 1

        # Look for common indicators of end of class
        for i in range(len(lines) - 1, 0, -1):
            line = lines[i].strip()
            if line.startswith("if __name__"):
                insert_idx = i - 1
                break
            if line and not line.startswith("#") and not line.startswith('"""'):
                if not lines[i].startswith(" ") and not lines[i].startswith("\t"):
                    # Found module-level code
                    insert_idx = i - 1
                    break

        # Prepare methods to add
        methods_to_add = []
        if "predict" in missing_methods:
            methods_to_add.append(PREDICT_METHOD_TEMPLATE)
        if "save" in missing_methods:
            methods_to_add.append(SAVE_METHOD_TEMPLATE)
        if "load" in missing_methods:
            methods_to_add.append(LOAD_METHOD_TEMPLATE)

        # Insert methods
        for method in methods_to_add:
            lines.insert(insert_idx, method)
            insert_idx += method.count("\n") + 1

        # Write back
        new_content = "\n".join(lines)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        return True

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    """Main execution."""
    base_dir = Path("/Users/visheshyadav/Documents/GitHub/CoreRec")

    print("=" * 80)
    print("METHOD INJECTION SCRIPT")
    print("=" * 80)
    print(f"\nProcessing {len(INCOMPLETE_MODELS)} incomplete models...\n")

    success = 0
    failed = 0

    for rel_path, info in INCOMPLETE_MODELS.items():
        file_path = base_dir / rel_path
        class_name = info["class"]
        missing = info["missing"]

        print(f"\n{rel_path}")
        print(f"  Class: {class_name}")
        print(f"  Adding: {', '.join(missing)}")

        if not file_path.exists():
            print(f"  ⚠️  File not found")
            failed += 1
            continue

        if inject_methods(file_path, class_name, missing):
            print(f"  ✅ Successfully added methods")
            success += 1
        else:
            print(f"  ❌ Failed to add methods")
            failed += 1

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Successfully updated: {success}/{len(INCOMPLETE_MODELS)}")
    print(f"Failed: {failed}/{len(INCOMPLETE_MODELS)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
