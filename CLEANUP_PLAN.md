# Engine Cleanup Plan

## Goal
Move 75+ methods to sandbox, keep only Top 5 in main engines.

## Backup Status
✅ Full backup created:
- `corerec/sandbox/collaborative_full/` (all 45+ methods)
- `corerec/sandbox/content_based_full/` (all 35+ methods)

## Files to Keep in Main Engines

### Collaborative Engine (Keep These Only)
```
corerec/engines/collaborative/
├── __init__.py (refactored - Top 5 only)
├── sar.py (Top 5 #2)
├── fast_recommender.py or fast.py (Top 5 #5)
└── graph_based_base/
    └── lightgcn_base.py (Top 5 #3)
└── nn_base/
    └── ncf.py or ncf_base.py (Top 5 #4)
```

Note: TwoTower (#1) is in `corerec/engines/two_tower.py` (new file)

### Content-Based Engine (Keep These Only)
```
corerec/engines/content_based/
├── __init__.py (refactored - Top 5 only)
├── tfidf_recommender.py (Top 5 #1)
└── nn_based_algorithms/
    ├── Youtube_dnn.py (Top 5 #2)
    └── DSSM.py (Top 5 #3)
└── embedding_representation_learning/
    └── word2vec.py (Top 5 #5)
```

Note: BERT4Rec (#4) is in `corerec/engines/bert4rec.py` (new file)

## Files to Delete from Main Engines

### Collaborative - DELETE
- rbm.py
- rlrmc.py
- sli.py
- sum.py
- geomlc.py
- cornac_bpr.py
- base_recommender.py (move to sandbox)
- device_manager.py (move to utils)
- initializer.py (move to utils)
- cr_unionizedFactory.py (deprecated)
- All subdirectories except needed files:
  - mf_base/ (DELETE - move to sandbox)
  - nn_base/ (TRIM - keep only NCF)
  - graph_based_base/ (TRIM - keep only LightGCN)
  - sequential_model_base/ (DELETE - move to sandbox)
  - bayesian_method_base/ (DELETE - move to sandbox)
  - attention_mechanism_base/ (DELETE - move to sandbox)
  - variational_encoder_base/ (DELETE - move to sandbox)
  - regularization_based_base/ (DELETE - move to sandbox)

### Content-Based - DELETE
- cr_contentFilterFactory.py (deprecated)
- All subdirectories except needed files:
  - traditional_ml_algorithms/ (DELETE - move to sandbox)
  - nn_based_algorithms/ (TRIM - keep only Youtube_dnn.py, DSSM.py)
  - graph_based_algorithms/ (DELETE - move to sandbox)
  - hybrid_ensemble_methods/ (DELETE - move to sandbox)
  - context_personalization/ (DELETE - move to sandbox)
  - special_techniques/ (DELETE - move to sandbox)
  - probabilistic_statistical_methods/ (DELETE - move to sandbox)
  - performance_scalability/ (DELETE - move to sandbox)
  - other_approaches/ (DELETE - move to sandbox)
  - miscellaneous_techniques/ (DELETE - move to sandbox)
  - fairness_explainability/ (DELETE - move to sandbox)
  - embedding_representation_learning/ (TRIM - keep only word2vec.py)
  - learning_paradigms/ (DELETE - move to sandbox)
  - multi_modal_cross_domain_methods/ (DELETE - move to sandbox)

## Execution Steps

1. ✅ Backup created (sandbox/collaborative_full, sandbox/content_based_full)
2. Update sandbox __init__.py to import from _full directories
3. Clean collaborative engine (delete non-Top-5 files)
4. Clean content_based engine (delete non-Top-5 files)
5. Update main __init__.py files
6. Test imports
7. Create migration script for users

## Result
Clean codebase with:
- 5 collaborative methods in main engine
- 5 content methods in main engine
- 75+ methods safely in sandbox
- All imports still work (via sandbox)

