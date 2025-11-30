#!/bin/bash
# Bulk migration script for all remaining BaseCorerec models
# This script migrates class definitions and imports

echo "===== CoreRec Model Migration Script ====="
echo ""

# List of files found to migrate
FILES=(
  "corerec/engines/mind.py"
  "corerec/engines/sasrec.py"
  "corerec/engines/unionizedFilterEngine/nn_base/FLEN_base.py"
  "corerec/engines/unionizedFilterEngine/nn_base/FM_base.py"
  "corerec/engines/unionizedFilterEngine/nn_base/AutoFI_base.py"
  "corerec/engines/unionizedFilterEngine/nn_base/GNN_base.py"
  "corerec/engines/unionizedFilterEngine/nn_base/AutoInt_base.py"
  "corerec/engines/unionizedFilterEngine/nn_base/bivae_base.py"
  "corerec/engines/unionizedFilterEngine/nn_base/Fibinet_base.py"
  "corerec/engines/unionizedFilterEngine/nn_base/NFM_base.py"
  "corerec/engines/unionizedFilterEngine/nn_base/ESCMM_base.py"
  "corerec/engines/unionizedFilterEngine/nn_base/ENSFM_base.py"
  "corerec/engines/unionizedFilterEngine/nn_base/DeepFM_base.py"
  "corerec/engines/unionizedFilterEngine/nn_base/ESMM_base.py"
  "corerec/engines/unionizedFilterEngine/nn_base/gan_ufilter_base.py"
  "corerec/engines/unionizedFilterEngine/nn_base/DCN_base.py"
  "corerec/engines/unionizedFilterEngine/nn_base/caser_base.py"
  "corerec/engines/unionizedFilterEngine/nn_base/Bert4Rec_base.py"
  "corerec/engines/unionizedFilterEngine/nn_base/DLRM_base.py"
  "corerec/engines/unionizedFilterEngine/nn_base/FFM_base.py"
  "corerec/engines/unionizedFilterEngine/nn_base/autoencoder_cf_base.py"
  "corerec/engines/unionizedFilterEngine/nn_base/DIEN_base.py"
  "corerec/engines/unionizedFilterEngine/nn_base/FGCNN_base.py"
  "corerec/engines/unionizedFilterEngine/nn_base/DeepRec_base.py"
  "corerec/engines/unionizedFilterEngine/nn_base/GateNet_base.py"
  "corerec/engines/unionizedFilterEngine/nn_base/BST_base.py"
  "corerec/engines/unionizedFilterEngine/nn_base/DeepCrossing_base.py"
  "corerec/engines/unionizedFilterEngine/nn_base/ncf.py"
  "corerec/engines/unionizedFilterEngine/nn_base/AFM_base.py"
  "corerec/engines/unionizedFilterEngine/mf_base/matrix_factorization.py"
  "corerec/engines/unionizedFilterEngine/mf_base/deep_matrix_factorization_base.py"
  "corerec/engines/unionizedFilterEngine/mf_base/user_based_uf.py"
)

migrated=0
failed=0

for file in "${FILES[@]}"; do
  echo "Processing: $file"
  
  # Check if file exists
  if [ ! -f "$file" ]; then
    echo "  ⚠ File not found, skipping"
    continue
  fi
  
  # Step 1: Update import statement
  sed -i '' 's/from corerec\.base_recommender import BaseCorerec/from corerec.api.base_recommender import BaseRecommender/' "$file"
  
  #Step 2: Update class definitions
  sed -i '' 's/class \([A-Za-z0-9_]*\)(BaseCorerec):/class \1(BaseRecommender):/' "$file"
  
  # Step 3: Check if validate functions are used and add imports if needed
  if grep -q "validate_fit_inputs\|validate_user_id\|validate_top_k\|validate_model_fitted" "$file"; then
    # Check if validation imports already exist
    if ! grep -q "from corerec.utils.validation import" "$file"; then
      # Add import after the base_recommender import
      sed -i '' '/from corerec.api.base_recommender import BaseRecommender/a\
from corerec.utils.validation import validate_fit_inputs, validate_user_id, validate_top_k, validate_model_fitted
' "$file"
    fi
  fi
  
  if [ $? -eq 0 ]; then
    echo "  ✓ Migrated successfully"
    ((migrated++))
  else
    echo "  ✗ Migration failed"
    ((failed++))
  fi
done

echo ""
echo "======================================"
echo "Migration Summary:"
echo "  Successfully migrated: $migrated"
echo "  Failed: $failed"
echo "  Total files: ${#FILES[@]}"
echo "======================================"
