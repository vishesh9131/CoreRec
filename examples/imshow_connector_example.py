#!/usr/bin/env python3
# minimal example of plug-and-play frontend with imshow. it's tiny and a bit rough

import os
import sys
import random
from typing import List, Dict, Any, Optional

# make sure project root is on path when running directly
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from corerec.imshow import quick_demo, list_frontends
except Exception as e:
    print("Failed to import corerec.imshow. Please check installation.", e)
    sys.exit(1)

# a super simple item catalog to demonstrate
CATALOG = [
    {
        "id": f"vid_{i}",
        "title": f"Demo video {i}",
        "thumbnail": f"https://picsum.photos/seed/im{i}/200/120",
        "url": f"https://example.com/watch/{i}",
        "genres": ["demo", "synthetic"],
    }
    for i in range(1, 51)
]


def predict(user_id: Any, k: int = 10, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Minimal predict function format for imshow Connector.
    Returns a ranked list of item dicts with at least: id, title, thumbnail, url.
    This is obv. not a real model – just random scoring for demo.
    """
    rng = random.Random(hash(user_id) % (2**32))
    scored = [(rng.random(), item) for item in CATALOG]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [it for _, it in scored[:k]]


if __name__ == "__main__":
    # choose a frontend; try youtube first, fallback to first available
    available = list_frontends()
    frontend = "youtube" if "youtube" in available else (available[0] if available else None)
    if not frontend:
        print("No frontends available from corerec.imshow. aborting")
        sys.exit(1)

    # start a tiny local server to preview the UI
    # tip: open http://127.0.0.1:8000 in browser
    print(f"Starting quick demo with frontend='{frontend}'. Available frontends: {available}")
    quick_demo(predict, frontend=frontend, host="127.0.0.1", port=8000) 