# CoreRec Branch Structure

## Overview

The codebase is organized into 6 feature branches + main for clean development.

```
main (production-ready)
 |
 +-- feature/cleanup-engines       (Top 5 methods, sandbox organization)
 +-- feature/modern-pipeline       (Pipeline architecture)
 +-- feature/two-tower-retrieval   (Two-Tower model)
 +-- feature/sequential-models     (BERT4Rec, transformer models)
 +-- feature/vector-store          (FAISS/Annoy integration)
 +-- feature/multimodal-fusion     (Multi-modal fusion strategies)
```

## Branch Details

### 1. `feature/cleanup-engines`
**Purpose:** Clean main engines, move 75+ methods to sandbox

**Contains:**
- Sandbox backups (`sandbox/collaborative_full/`, `sandbox/content_based_full/`)
- Refactored `__init__.py` files (Top 5 only)
- Cleanup scripts and documentation
- Engine migration guides

**Status:** Ready for review

---

### 2. `feature/modern-pipeline`
**Purpose:** Multi-stage recommendation pipeline

**Contains:**
- `corerec/pipelines/recommendation_pipeline.py`
- Three-stage architecture: Retrieval -> Ranking -> Reranking
- Business rule support (diversity, freshness)
- Pipeline orchestrator

**Status:** Ready for review

---

### 3. `feature/two-tower-retrieval`
**Purpose:** Industry-standard retrieval architecture

**Contains:**
- `corerec/engines/two_tower.py`
- User/Item tower separation
- Contrastive learning (InfoNCE)
- Pre-computed embeddings

**Status:** Ready for review

---

### 4. `feature/sequential-models`
**Purpose:** Transformer-based sequential recommendation

**Contains:**
- `corerec/engines/bert4rec.py`
- Bidirectional attention
- Mask-and-predict training
- Positional encodings

**Status:** Ready for review

---

### 5. `feature/vector-store`
**Purpose:** Fast similarity search for retrieval

**Contains:**
- `corerec/retrieval/vector_store.py`
- Numpy backend (simple)
- FAISS backend (production)
- Annoy backend (medium scale)

**Status:** Ready for review

---

### 6. `feature/multimodal-fusion`
**Purpose:** Combine text, images, metadata

**Contains:**
- `corerec/multimodal/fusion_strategies.py`
- Attention fusion
- Gated fusion
- Bilinear pooling

**Status:** Ready for review

---

## Workflow

### Development
```bash
# Work on a specific feature
git checkout feature/two-tower-retrieval
# ... make changes ...
git add -A && git commit -m "feat: ..."
```

### Integration
```bash
# Merge feature into main when ready
git checkout main
git merge feature/two-tower-retrieval
```

### Full Integration (all features)
```bash
git checkout main
git merge feature/cleanup-engines
git merge feature/modern-pipeline
git merge feature/two-tower-retrieval
git merge feature/sequential-models
git merge feature/vector-store
git merge feature/multimodal-fusion
```

## Current Status

| Branch | Status | Lines Changed |
|--------|--------|---------------|
| `feature/cleanup-engines` | Ready | ~2000+ |
| `feature/modern-pipeline` | Ready | ~300 |
| `feature/two-tower-retrieval` | Ready | ~370 |
| `feature/sequential-models` | Ready | ~390 |
| `feature/vector-store` | Ready | ~350 |
| `feature/multimodal-fusion` | Ready | ~330 |

## Quick Commands

```bash
# See all branches
git branch -v

# Switch to a branch
git checkout feature/cleanup-engines

# See changes in a branch
git log main..feature/two-tower-retrieval --oneline

# Compare branches
git diff main..feature/two-tower-retrieval --stat
```

## Testing

Each branch can be tested independently:

```bash
# Test Two-Tower
git checkout feature/two-tower-retrieval
python -c "from corerec.engines.two_tower import TwoTower; print('OK')"

# Test Pipeline
git checkout feature/modern-pipeline
python -c "from corerec.pipelines.recommendation_pipeline import RecommendationPipeline; print('OK')"

# Test Vector Store
git checkout feature/vector-store
python -c "from corerec.retrieval.vector_store import create_index; print('OK')"
```

## Notes

- `main` branch is production-ready
- Feature branches contain isolated changes
- Merge order doesn't matter (no conflicts between features)
- Each feature can be released independently
- Full integration gives complete modern RecSys framework
