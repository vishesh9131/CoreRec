# Engine Refactoring Guide

## What Changed?

CoreRec's collaborative and content-based engines have been refactored to focus on **quality over quantity**.

### Before: 80+ Methods All Exposed

```
corerec.engines.collaborative  (50+ methods, mixed quality)
corerec.engines.content_based  (40+ methods, various maturity levels)
```

Problems:
- Overwhelming for new users
- Mixed production-ready and experimental code
- Hard to maintain
- Unclear which methods to use

### After: Top 5 + Sandbox

```
corerec.engines.collaborative  (Top 5 battle-tested)
corerec.engines.content_based  (Top 5 production-ready)
corerec.sandbox/*              (75+ methods under development)
```

Benefits:
- Clear "go-to" methods for production
- Experimental methods isolated
- Easier to maintain and test
- Gradual quality improvement

---

## Top 5 Methods

### Collaborative Filtering

| Method | Use Case | Why Top 5? |
|--------|----------|------------|
| **TwoTower** | Modern retrieval, large-scale | Industry standard (YouTube, Netflix) |
| **SAR** | Quick baseline, simple | Fast, proven, no deep learning needed |
| **LightGCN** | Graph-based, social | Modern, scales well, network effects |
| **NCF** | Neural collaborative | Foundational, interpretable embeddings |
| **FastRecommender** | Rapid prototyping | Quick to train, good baseline |

### Content-Based

| Method | Use Case | Why Top 5? |
|--------|----------|------------|
| **TFIDFRecommender** | Text-based items | Classic, reliable, no dependencies |
| **YoutubeDNN** | Large-scale deployment | Industry-proven multi-stage |
| **DSSM** | Semantic matching | Microsoft's standard, deep features |
| **BERT4Rec** | Sequential behavior | Transformer-based, state-of-art |
| **Word2VecRecommender** | Embedding-based | Versatile, good item-item |

---

## Migration Guide

### For Existing Code (No Breaking Changes!)

All your existing code still works. Methods are forwarded to sandbox:

```python
# This still works
from corerec.engines.collaborative import DeepFM
from corerec.engines.content_based import CNN

# But you'll get a deprecation hint in logs
```

### Recommended Updates

#### If you're using a Top 5 method:

```python
# Old (still works)
from corerec.engines import unionized
model = unionized.SAR()

# New (same thing, cleaner)
from corerec.engines import unionized
model = unionized.SAR()

# No change needed! You're already using best practices.
```

#### If you're using a sandbox method:

```python
# Old (still works)
from corerec.engines.collaborative import DeepFM

# New (explicit about experimental status)
from corerec.sandbox.collaborative import DeepFM

# Or via submodule
from corerec.sandbox.collaborative.nn import DeepFM
```

### For New Projects

**Start with Top 5:**

```python
from corerec.engines import unionized, content

# Collaborative
model = unionized.TwoTower(embedding_dim=256)  # Modern standard
# or
model = unionized.SAR()  # Simple baseline

# Content-based
model = content.TFIDFRecommender()  # Classic
# or
model = content.YoutubeDNN()  # Deep learning
```

**Only use sandbox if:**
- You need a specific algorithm not in Top 5
- You're doing research/experimentation
- You understand the tradeoffs

---

## Sandbox Structure

### What's in Sandbox?

**Collaborative (45+ methods):**
- Matrix Factorization: SVD, ALS, NMF, PMF, BPR
- Neural Networks: DeepFM, DCN, AutoInt, PNN, xDeepFM, DLRM
- Sequential: GRU, LSTM, Caser, SASRec, NextItNet
- Attention: DIEN, DIN, BST, Transformer variants
- Graph: DeepWalk, various GNN models
- Variational: VAE, BiVAE, CVAE
- Bayesian: Bayesian MF, MCMC methods
- Multi-Task: MMOE, PLE, ESMM
- Others: RLRMC, SLI, SUM, RBM, GeoMLC

**Content-Based (35+ methods):**
- Traditional ML: SVM, LightGBM, Decision Trees
- Neural: CNN, RNN, Transformer, VAE, Autoencoder
- Graph: GNN, semantic models
- Hybrid: Ensemble, attention mechanisms
- Context: User/item profiling
- Fairness: Fair ranking, explainable AI
- Learning: Transfer, meta, few-shot, zero-shot
- Multi-Modal: Fusion strategies
- Others: Doc2Vec, TDM, MIND, AITM

### Accessing Sandbox Methods

```python
# Option 1: Direct import
from corerec.sandbox.collaborative import DeepFM, SASRec
from corerec.sandbox.content_based import CNN, Transformer

# Option 2: Via submodule
from corerec.sandbox.collaborative.nn import DeepFM
from corerec.sandbox.content_based.nn import CNN

# Option 3: Browse what's available
from corerec.engines import unionized, content
print(unionized.sandbox.list_available())
print(content.sandbox.list_available())
```

### Sandbox Status Levels

Methods are tagged by maturity:
- **Alpha**: Early development, API may change
- **Beta**: Feature complete, needs testing
- **Stable**: Ready for graduation

Check docstrings for status.

---

## Why This Approach?

### 1. User Experience

**Before:** "Which of these 50 methods should I use?"
**After:** "Start with these 5, explore sandbox if needed."

### 2. Quality Control

**Before:** All methods mixed together
**After:** Production-ready vs experimental clearly separated

### 3. Development Velocity

**Before:** Hard to improve methods without breaking things
**After:** Sandbox allows rapid iteration

### 4. Graduation Path

Methods improve in sandbox → graduate to Top 5 → replace weaker methods

Example pipeline:
```
SASRec (sandbox, beta) → extensive testing → 
→ graduation to main engine → 
→ potentially replaces a weaker Top 5 method
```

---

## Decision Tree: Which Method to Use?

### Collaborative Filtering

```
Need real-time serving at scale?
  YES → TwoTower
  NO ↓

Have graph structure (social, connections)?
  YES → LightGCN
  NO ↓

Need quick baseline (no ML infrastructure)?
  YES → SAR
  NO ↓

Want neural collaborative filtering?
  YES → NCF or FastRecommender
  NO → Check sandbox
```

### Content-Based

```
Text-based items (articles, products)?
  YES → TFIDFRecommender
  NO ↓

Large-scale production deployment?
  YES → YoutubeDNN
  NO ↓

Need semantic understanding?
  YES → DSSM
  NO ↓

Have sequential user behavior?
  YES → BERT4Rec
  NO ↓

Want item-item embeddings?
  YES → Word2VecRecommender
  NO → Check sandbox
```

---

## FAQ

### Q: Will sandbox methods ever be removed?

A: No. They're accessible indefinitely. They may:
- Graduate to main engine (promoted)
- Stay in sandbox (experimental)
- Be marked deprecated (if better alternatives exist)

### Q: Can I use sandbox methods in production?

A: Yes, but:
- Check the status (Alpha/Beta/Stable)
- Test thoroughly
- Be prepared for API changes (Alpha methods)
- Consider Top 5 first

### Q: How do methods graduate from sandbox?

Requirements:
1. Comprehensive tests (>80% coverage)
2. Documentation (theory + examples)
3. Performance benchmarks
4. Real-world validation
5. API stability

### Q: What if my preferred method is in sandbox?

Keep using it! Import from `corerec.sandbox.*` to be explicit about status.
Help it graduate by:
- Reporting bugs
- Contributing tests
- Sharing results
- Improving documentation

### Q: Are all 80+ methods still maintained?

Yes, but with different priority:
- Top 5: Actively maintained, bug fixes ASAP
- Sandbox (Stable): Maintained, bugs fixed regularly
- Sandbox (Beta): Active development
- Sandbox (Alpha): Best effort

---

## Examples

### Example 1: Simple Collaborative Filtering

```python
from corerec.engines import unionized
import numpy as np

# Your data
user_ids = ['u1', 'u2', 'u3']
item_ids = ['i1', 'i2', 'i3']
interactions = np.array([
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 1]
])

# Use SAR (fast, simple)
model = unionized.SAR()
model.fit(user_ids, item_ids, interactions)
recs = model.recommend('u1', top_k=5)
```

### Example 2: Modern Large-Scale

```python
from corerec.engines import unionized
from corerec.retrieval.vector_store import create_index

# Use TwoTower (modern standard)
model = unionized.TwoTower(embedding_dim=256)
model.fit(user_ids, item_ids, interactions)

# Build fast index
item_embs = model.get_item_embeddings()
index = create_index("faiss", dim=256)
index.add(item_embs, item_ids)

# Lightning-fast retrieval
candidates = model.recommend('u1', top_k=10)
```

### Example 3: Experimental Method

```python
from corerec.sandbox.collaborative import DeepFM

# Explicitly using sandbox method
model = DeepFM(embedding_dim=64)
# ... rest of your code

# Check status
print(DeepFM.__doc__)  # Status should be documented
```

---

## Contributing

Want to help sandbox methods graduate?

1. **Test**: Run with your data, report results
2. **Benchmark**: Compare against Top 5
3. **Document**: Improve docstrings and examples
4. **Debug**: Find and fix edge cases
5. **Optimize**: Improve performance

See `CONTRIBUTING.md` for details.

---

## Roadmap

### Phase 1: Setup (Complete)
- ✅ Identify Top 5 for each engine
- ✅ Create sandbox structure
- ✅ Forward imports (no breaking changes)
- ✅ Documentation

### Phase 2: Organization (In Progress)
- 🔄 Categorize sandbox methods by maturity
- 🔄 Add status tags to all methods
- 🔄 Comprehensive testing
- 🔄 Performance benchmarks

### Phase 3: Graduation (Ongoing)
- ⏳ DeepFM testing complete → graduate
- ⏳ SASRec polish → graduate
- ⏳ CNN multi-modal integration → graduate

### Phase 4: Maintenance (Continuous)
- Regular quality reviews
- Promotion/demotion based on performance
- Top 5 list may change over time

---

## Summary

**For most users:** Use the Top 5. They're battle-tested and production-ready.

**For researchers:** Explore sandbox. Help methods graduate.

**For everyone:** No breaking changes. Your code still works.

**Questions?** 
- GitHub Issues: https://github.com/vishesh9131/CoreRec/issues
- Email: sciencely98@gmail.com

---

Last updated: 2025-01-07

