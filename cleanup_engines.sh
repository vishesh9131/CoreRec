#!/bin/bash
# Script to clean up main engines, keeping only Top 5 methods
# Backup already created in sandbox/collaborative_full and sandbox/content_based_full

set -e  # Exit on error

COREREC_ROOT="/Users/visheshyadav/Documents/GitHub/CoreRec/corerec"

echo "=========================================="
echo "CoreRec Engine Cleanup Script"
echo "=========================================="
echo ""
echo "This will remove non-Top-5 files from main engines."
echo "Full backup exists in sandbox/*_full directories."
echo ""
read -p "Continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Cleanup cancelled."
    exit 0
fi

echo ""
echo "Starting cleanup..."
echo ""

# ============================================================================
# COLLABORATIVE ENGINE CLEANUP
# ============================================================================

echo "Cleaning collaborative engine..."
cd "$COREREC_ROOT/engines/collaborative"

# Remove non-Top-5 files
echo "  - Removing rbm.py..."
rm -f rbm.py

echo "  - Removing rlrmc.py..."
rm -f rlrmc.py

echo "  - Removing sli.py..."
rm -f sli.py

echo "  - Removing sum.py..."
rm -f sum.py

echo "  - Removing geomlc.py..."
rm -f geomlc.py

echo "  - Removing cornac_bpr.py..."
rm -f cornac_bpr.py

echo "  - Removing base_recommender.py..."
rm -f base_recommender.py

echo "  - Removing device_manager.py..."
rm -f device_manager.py

echo "  - Removing initializer.py..."
rm -f initializer.py

echo "  - Removing cr_unionizedFactory.py..."
rm -f cr_unionizedFactory.py

# Remove entire subdirectories that aren't needed
echo "  - Removing mf_base/ (all matrix factorization to sandbox)..."
rm -rf mf_base/

echo "  - Removing sequential_model_base/ (to sandbox)..."
rm -rf sequential_model_base/

echo "  - Removing bayesian_method_base/ (to sandbox)..."
rm -rf bayesian_method_base/

echo "  - Removing attention_mechanism_base/ (to sandbox)..."
rm -rf attention_mechanism_base/

echo "  - Removing variational_encoder_base/ (to sandbox)..."
rm -rf variational_encoder_base/

echo "  - Removing regularization_based_base/ (to sandbox)..."
rm -rf regularization_based_base/

# Keep nn_base but clean it (keep only NCF)
echo "  - Cleaning nn_base/ (keeping only NCF)..."
cd nn_base/
# Keep only NCF-related files
find . -type f ! -name '__init__.py' ! -name 'ncf.py' ! -name 'ncf_base.py' ! -name '*ncf*' -delete
cd ..

# Keep graph_based_base but clean it (keep only LightGCN)
echo "  - Cleaning graph_based_base/ (keeping only LightGCN)..."
cd graph_based_base/ 2>/dev/null || echo "    graph_based_base/ not found, skipping..."
if [ -d "." ]; then
    find . -type f ! -name '__init__.py' ! -name '*lightgcn*' ! -name '*LightGCN*' -delete 2>/dev/null || true
    cd ..
fi

echo "Collaborative engine cleaned!"
echo ""

# ============================================================================
# CONTENT-BASED ENGINE CLEANUP
# ============================================================================

echo "Cleaning content-based engine..."
cd "$COREREC_ROOT/engines/content_based"

# Remove factory
echo "  - Removing cr_contentFilterFactory.py..."
rm -f cr_contentFilterFactory.py

# Remove entire subdirectories except needed ones
echo "  - Removing traditional_ml_algorithms/ (to sandbox)..."
rm -rf traditional_ml_algorithms/

echo "  - Removing graph_based_algorithms/ (to sandbox)..."
rm -rf graph_based_algorithms/

echo "  - Removing hybrid_ensemble_methods/ (to sandbox)..."
rm -rf hybrid_ensemble_methods/

echo "  - Removing context_personalization/ (to sandbox)..."
rm -rf context_personalization/

echo "  - Removing special_techniques/ (to sandbox)..."
rm -rf special_techniques/

echo "  - Removing probabilistic_statistical_methods/ (to sandbox)..."
rm -rf probabilistic_statistical_methods/

echo "  - Removing performance_scalability/ (to sandbox)..."
rm -rf performance_scalability/

echo "  - Removing other_approaches/ (to sandbox)..."
rm -rf other_approaches/

echo "  - Removing miscellaneous_techniques/ (to sandbox)..."
rm -rf miscellaneous_techniques/

echo "  - Removing fairness_explainability/ (to sandbox)..."
rm -rf fairness_explainability/

echo "  - Removing learning_paradigms/ (to sandbox)..."
rm -rf learning_paradigms/

echo "  - Removing multi_modal_cross_domain_methods/ (to sandbox)..."
rm -rf multi_modal_cross_domain_methods/

# Keep nn_based_algorithms but clean it (keep only Youtube_dnn.py and DSSM.py)
echo "  - Cleaning nn_based_algorithms/ (keeping only Youtube_dnn.py, DSSM.py)..."
cd nn_based_algorithms/
find . -type f ! -name '__init__.py' ! -name 'Youtube_dnn.py' ! -name 'DSSM.py' -delete
cd ..

# Keep embedding_representation_learning but clean it (keep only word2vec.py)
echo "  - Cleaning embedding_representation_learning/ (keeping only word2vec.py)..."
cd embedding_representation_learning/ 2>/dev/null || echo "    embedding dir not found, skipping..."
if [ -d "." ]; then
    find . -type f ! -name '__init__.py' ! -name 'word2vec.py' ! -name '*word2vec*' -delete 2>/dev/null || true
    cd ..
fi

echo "Content-based engine cleaned!"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================

echo "=========================================="
echo "Cleanup Complete!"
echo "=========================================="
echo ""
echo "Main engines now contain only Top 5 methods:"
echo ""
echo "Collaborative:"
echo "  1. TwoTower (in engines/two_tower.py)"
echo "  2. SAR (sar.py)"
echo "  3. LightGCN (graph_based_base/lightgcn_base.py)"
echo "  4. NCF (nn_base/ncf.py)"
echo "  5. FastRecommender (fast_recommender.py)"
echo ""
echo "Content-Based:"
echo "  1. TFIDFRecommender (tfidf_recommender.py)"
echo "  2. YoutubeDNN (nn_based_algorithms/Youtube_dnn.py)"
echo "  3. DSSM (nn_based_algorithms/DSSM.py)"
echo "  4. BERT4Rec (in engines/bert4rec.py)"
echo "  5. Word2VecRecommender (embedding_representation_learning/word2vec.py)"
echo ""
echo "All other methods preserved in:"
echo "  - sandbox/collaborative_full/"
echo "  - sandbox/content_based_full/"
echo ""
echo "Accessible via:"
echo "  from corerec.sandbox.collaborative import <Method>"
echo "  from corerec.sandbox.content_based import <Method>"
echo ""
echo "=========================================="

