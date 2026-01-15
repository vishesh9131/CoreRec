# ⚠️ BEFORE RUNNING CLEANUP - READ THIS

## What Will Happen

The cleanup script will **physically delete** files from main engines, keeping only Top 5 methods.

## Safety

✅ **Full backup created:**
- `corerec/sandbox/collaborative_full/` - Complete copy of collaborative engine
- `corerec/sandbox/content_based/` - Complete copy of content_based engine

All 80+ methods are preserved and accessible via sandbox.

## What Gets Deleted

### From `corerec/engines/collaborative/`:
**DELETE These Files:**
- rbm.py
- rlrmc.py
- sli.py
- sum.py
- geomlc.py
- cornac_bpr.py
- base_recommender.py
- device_manager.py
- initializer.py
- cr_unionizedFactory.py

**DELETE These Directories:**
- mf_base/ (all matrix factorization methods)
- sequential_model_base/ (GRU, LSTM, Caser, etc.)
- bayesian_method_base/ (Bayesian methods)
- attention_mechanism_base/ (attention mechanisms)
- variational_encoder_base/ (VAE, CVAE, etc.)
- regularization_based_base/ (regularization methods)

**TRIM These Directories (keep specific files only):**
- nn_base/ → Keep only NCF-related files
- graph_based_base/ → Keep only LightGCN-related files

**KEEP These Files:**
- __init__.py (refactored)
- sar.py
- fast_recommender.py or fast.py
- nn_base/ncf.py (or ncf_base.py)
- graph_based_base/lightgcn_base.py

### From `corerec/engines/content_based/`:
**DELETE These Files:**
- cr_contentFilterFactory.py

**DELETE These Directories:**
- traditional_ml_algorithms/
- graph_based_algorithms/
- hybrid_ensemble_methods/
- context_personalization/
- special_techniques/
- probabilistic_statistical_methods/
- performance_scalability/
- other_approaches/
- miscellaneous_techniques/
- fairness_explainability/
- learning_paradigms/
- multi_modal_cross_domain_methods/

**TRIM These Directories:**
- nn_based_algorithms/ → Keep only Youtube_dnn.py, DSSM.py
- embedding_representation_learning/ → Keep only word2vec.py

**KEEP These Files:**
- __init__.py (refactored)
- tfidf_recommender.py
- nn_based_algorithms/Youtube_dnn.py
- nn_based_algorithms/DSSM.py
- embedding_representation_learning/word2vec.py

## Final Structure

### Collaborative Engine (After Cleanup)
```
corerec/engines/collaborative/
├── __init__.py                    # Top 5 only
├── sar.py                         # #2
├── fast_recommender.py            # #5
├── nn_base/
│   ├── __init__.py
│   └── ncf.py                     # #4
└── graph_based_base/
    ├── __init__.py
    └── lightgcn_base.py           # #3
```
(+TwoTower #1 is in engines/two_tower.py)

### Content-Based Engine (After Cleanup)
```
corerec/engines/content_based/
├── __init__.py                    # Top 5 only
├── tfidf_recommender.py           # #1
├── nn_based_algorithms/
│   ├── __init__.py
│   ├── Youtube_dnn.py             # #2
│   └── DSSM.py                    # #3
└── embedding_representation_learning/
    ├── __init__.py
    └── word2vec.py                # #5
```
(+BERT4Rec #4 is in engines/bert4rec.py)

### All Other Methods in Sandbox
```
corerec/sandbox/
├── collaborative_full/            # Complete backup (45+ methods)
└── content_based_full/            # Complete backup (35+ methods)
```

## How to Run Cleanup

```bash
cd /Users/visheshyadav/Documents/GitHub/CoreRec
./cleanup_engines.sh
```

The script will:
1. Ask for confirmation
2. Delete non-Top-5 files
3. Show summary

## After Cleanup

### Imports Still Work

**Old code (still works):**
```python
from corerec.engines.collaborative import DeepFM  # Auto-redirects to sandbox
from corerec.engines.content_based import CNN      # Auto-redirects to sandbox
```

**New recommended way:**
```python
from corerec.sandbox.collaborative import DeepFM
from corerec.sandbox.content_based import CNN
```

### Top 5 Usage
```python
from corerec.engines import unionized, content

# Collaborative
model = unionized.TwoTower()        # #1 Modern
model = unionized.SAR()             # #2 Simple
model = unionized.LightGCN()        # #3 Graph
model = unionized.NCF()             # #4 Neural
model = unionized.FastRecommender() # #5 Proto

# Content-Based
model = content.TFIDFRecommender()  # #1 Classic
model = content.YoutubeDNN()        # #2 Industry
model = content.DSSM()              # #3 Semantic
model = content.BERT4Rec()          # #4 Sequential
model = content.Word2VecRecommender() # #5 Embedding
```

## Testing After Cleanup

```python
# Test Top 5 imports
from corerec.engines import unionized, content

# Should work
model1 = unionized.SAR()
model2 = content.TFIDFRecommender()

# Test sandbox imports
from corerec.sandbox.collaborative import DeepFM, RBM
from corerec.sandbox.content_based import CNN, Transformer

# Should work
model3 = DeepFM()
model4 = CNN()

print("All imports successful!")
```

## Rollback Plan

If something goes wrong:

1. **Full backup exists:**
   ```bash
   # Restore from backup
   cp -r corerec/sandbox/collaborative_full/* corerec/engines/collaborative/
   cp -r corerec/sandbox/content_based_full/* corerec/engines/content_based/
   ```

2. **Or use git:**
   ```bash
   git checkout corerec/engines/collaborative/
   git checkout corerec/engines/content_based/
   ```

## Benefits After Cleanup

1. **Cleaner Codebase:**
   - Main engines: 10 files (Top 5 each)
   - Sandbox: 80+ files (organized)

2. **Faster Imports:**
   - Less to scan on import
   - Clearer structure

3. **Better Maintenance:**
   - Focus quality on Top 5
   - Experiment freely in sandbox

4. **Clear Separation:**
   - Production-ready vs Experimental
   - No confusion about which to use

## Ready?

Before running cleanup:
- ✅ Backup created (sandbox/*_full/)
- ✅ Script tested and ready
- ✅ Imports will still work (via sandbox)
- ✅ Rollback plan available
- ✅ Documentation updated

Run when ready:
```bash
./cleanup_engines.sh
```

## Questions?

- Check: `ENGINE_REFACTORING_GUIDE.md` for full details
- Check: `TOP5_QUICK_REFERENCE.md` for method guide
- Check: `CLEANUP_PLAN.md` for technical details

---

**Remember:** This is reversible. Full backup in sandbox/*_full/

