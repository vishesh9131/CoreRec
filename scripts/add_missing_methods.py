"""
Utility script to add missing abstract methods to migrated models.

This script adds predict(), save(), and load() methods to models that need them.

Author: Vishesh Yadav
"""

import os
from pathlib import Path

# Template for predict method
PREDICT_TEMPLATE = '''
    def predict(self, user_id: int, item_id: int, **kwargs) -> float:
        """Predict score for a user-item pair."""
        if not self.is_fitted:
            raise ModelNotFittedError()
        
        if user_id not in self.user_map or item_id not in self.item_map:
            return 0.0
        
        # Get indices
        user_idx = self.user_map[user_id]
        item_idx = self.item_map[item_id]
        
        # Create feature tensors
        user_tensor = torch.LongTensor([user_idx]).to(self.device)
        item_tensor = torch.LongTensor([item_idx]).to(self.device)
        
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            score = self.model(user_tensor, item_tensor).item()
        
        return score
'''

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


def add_methods_to_file(file_path: Path, class_name: str):
    """Add missing methods to a model file."""
    with open(file_path, "r") as f:
        content = f.read()

    # Check if predict method exists
    if "def predict(self" not in content:
        print(f"  Adding predict() method to {class_name}")
        # Insert before the last line (assuming end of class)
        content = content.rstrip() + "\n" + PREDICT_TEMPLATE

    # Check if save method exists
    if "def save(self" not in content:
        print(f"  Adding save()/load() methods to {class_name}")
        save_load = SAVE_LOAD_TEMPLATE.format(class_name=class_name)
        content = content.rstrip() + "\n" + save_load + "\n"

    # Write back
    with open(file_path, "w") as f:
        f.write(content)


# Files to process
models = [
    ("corerec/engines/deepfm.py", "DeepFM"),
    ("corerec/engines/gnnrec.py", "GNNRec"),
    ("corerec/engines/nasrec.py", "NASRec"),
]

base_dir = Path("/Users/visheshyadav/Documents/GitHub/CoreRec")

for file_rel, class_name in models:
    file_path = base_dir / file_rel
    print(f"\nProcessing {file_rel}...")
    add_methods_to_file(file_path, class_name)
    print(f"  ✓ Completed")

print("\n✅ All methods added successfully!")
