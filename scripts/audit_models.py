import os
import re
from pathlib import Path
from scripts.model_database import MODEL_DATABASE


def find_potential_models(root_dir):
    models = set()

    # Walk through the directory
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py") and not file.startswith("__"):
                # Check file content for class definitions inheriting from BaseRecommender or similar
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Look for class definitions
                    # Heuristic: classes in engines/ usually represent models
                    if "corerec/engines" in file_path:
                        # Extract class names
                        classes = re.findall(r"class\s+(\w+)", content)
                        for cls in classes:
                            if "Recommender" in cls or "Model" in cls or "Engine" in cls:
                                # Filter out base classes if possible
                                if cls not in ["BaseRecommender", "BaseEngine", "Model"]:
                                    models.add(cls)

                        # Also use filename as a fallback for model name identification
                        base_name = (
                            file.replace(".py", "")
                            .replace("_recommender", "")
                            .replace("_model", "")
                        )
                        if len(base_name) > 2:  # Ignore short names like 'ui'
                            models.add(base_name.upper())

                except Exception:
                    pass

    return models


# Known documented models (normalized)
documented = set(MODEL_DATABASE.keys())
documented_normalized = {k.upper().replace("-", "").replace("_", "") for k in documented}

# Scan
root_dir = "/Users/visheshyadav/Documents/GitHub/CoreRec/corerec/engines"
potential_models = find_potential_models(root_dir)

# Filter
remaining = []
for m in potential_models:
    # Normalize for comparison
    m_norm = m.upper().replace("-", "").replace("_", "")

    # Check if already documented
    is_documented = False
    for d in documented_normalized:
        if d in m_norm or m_norm in d:
            is_documented = True
            break

    if not is_documented:
        remaining.append(m)

# Manual cleanup/filtering of the list to remove noise
# This is a heuristic list, might need manual refinement
clean_remaining = sorted(list(set(remaining)))

print(f"Documented models: {len(documented)}")
print(f"Potential remaining models found: {len(clean_remaining)}")
print("Potential remaining models:")
for m in clean_remaining:
    print(f" - {m}")
