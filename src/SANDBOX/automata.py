# import os
# # This script is used to generate the directory structure and files for the content_based
# # Define the directory structure and naming conventions
# directory_structure = {
#     "nn_based_algorithms": [
#         "dkn.py",
#         "lstur.py",
#         "naml.py",
#         "npa.py",
#         "nrms.py",
#         "cnn.py",
#         "rnn.py",
#         "transformer.py",
#         "autoencoder.py",
#         "vae.py"
#     ],
#     "traditional_ml_algorithms": [
#         "lightgbm.py",
#         "decision_tree.py",
#         "svm.py",
#         "tfidf.py",
#         "vw.py"
#     ],
#     "probabilistic_statistical_methods": [
#         "lsa.py",
#         "lda.py",
#         "bayesian.py",
#         "fuzzy_logic.py"
#     ],
#     "embedding_representation_learning": [
#         "word2vec.py",
#         "doc2vec.py",
#         "personalized_embeddings.py"
#     ],
#     "graph_based_algorithms": [
#         "graph_filtering.py",
#         "gnn.py",
#         "semantic_models.py"
#     ],
#     "context_personalization": [
#         "context_aware.py",
#         "user_profiling.py",
#         "item_profiling.py"
#     ],
#     "hybrid_ensemble_methods": [
#         "hybrid_collaborative.py",
#         "attention_mechanisms.py",
#         "ensemble_methods.py"
#     ],
#     "multi_modal_cross_domain_methods": [
#         "multi_modal.py",
#         "cross_domain.py",
#         "cross_lingual.py"
#     ],
#     "learning_paradigms": [
#         "few_shot.py",
#         "zero_shot.py",
#         "transfer_learning.py",
#         "meta_learning.py"
#     ],
#     "performance_scalability": [
#         "scalable_algorithms.py",
#         "feature_extraction.py",
#         "load_balancing.py"
#     ],
#     "special_techniques": [
#         "dynamic_filtering.py",
#         "temporal_filtering.py",
#         "interactive_filtering.py"
#     ],
#     "fairness_explainability": [
#         "explainable.py",
#         "fairness_aware.py",
#         "privacy_preserving.py"
#     ],
#     "miscellaneous_techniques": [
#         "feature_selection.py",
#         "noise_handling.py",
#         "cold_start.py"
#     ],
#     "other_approaches": [
#         "rule_based.py",
#         "ontology_based.py",
#         "sentiment_analysis.py"
#     ]
# }

# # Base directory for your project
# base_dir = "corerec/engines/content_based"

# # Create the directory structure
# for category, files in directory_structure.items():
#     category_path = os.path.join(base_dir, category)
#     os.makedirs(category_path, exist_ok=True)

#     # Create the Python files in each category
#     for file_name in files:
#         file_path = os.path.join(category_path, file_name)
#         with open(file_path, 'w') as f:
#             f.write(f"# {file_name.replace('.py', '')} implementation\n")
#             f.write("pass\n")

#     # Create __init__.py for each category
#     init_file_path = os.path.join(category_path, "__init__.py")
#     with open(init_file_path, 'w') as init_file:
#         # Generate import statements based on file names
#         for file_name in files:
#             algorithm_name = file_name.replace('.py', '').upper()
#             alias_name = f"{category[0:3].upper()}_{algorithm_name}"
#             init_file.write(f"from .{file_name[:-3]} import {algorithm_name} as {alias_name}\n")

# print("Directory structure and files created successfully.")




# This script is used to generate the directory structure and files for the collaborative

import os

# Define the directory structure and naming conventions for collaborative
directory_structure = {
    "graph_based_algorithms": [
        "graph_based_cf_base.py",
        "multi_view_cf_base.py",
        "edge_aware_cf_base.py",
        "heterogeneous_network_cf_base.py",
        "multi_relational_cf_base.py",
        "lightgcn_base.py",
        "geoimc_base.py",
        "gnn_cf_base.py"
    ],
    "matrix_factorization_algorithms": [
        "svdpp_base.py",
        "svd_base.py",
        "A2SVD_base.py",
        "ALS_base.py",
        "bayesian_matrix_factorization_base.py",
        "hierarchical_poisson_factorization_base.py",
        "incremental_matrix_factorization_base.py",
        "kernelized_matrix_factorization_base.py",
        "deep_matrix_factorization_base.py",
        "neural_matrix_factorization_base.py",
        "pmf_base.py",
        "nmf_base.py",
        "temporal_matrix_factorization_base.py",
        "contextual_matrix_factorization_base.py",
        "hybrid_regularization_matrix_factorization_base.py",
        "hybrid_matrix_factorization_base.py",
        "weighted_matrix_factorization_base.py",
        "sgd_matrix_factorization_base.py",
        "Implicit_feedback_mf_base.py"
    ],
    "neural_network_based_algorithms": [
        "xdeepfm_base.py",
        "self_supervised_learning_cf_base.py",
        "nextitnet_base.py",
        "ssept_base.py",
        "sasrec_base.py",
        "ncf_base.py",
        "bivae_base.py",
        "autoencoder_cf_base.py",
        "slirec_base.py",
        "rnn_sequential_recommendation_base.py",
        "caser_base.py",
        "hybrid_deep_learning_base.py"
    ],
    "variational_encoder_based_algorithms": [
        "Multinomial_VAE_base.py",
        "BPRE_base.py",
        "Standard_VAE_base.py",
        "PGM_uf_base.py"
    ],
    "bayesian_method_based_algorithms": [
        "Bayesian[mf]_base.py",
        "Bayesian[Personalized_Ranking][Extensions]_base.py",
        "PGM_uf.py"
    ],
    "attention_mechanism_based_algorithms": [
        "Transformer_based_uf_base.py",
        "SASRec_base.py"
    ],
    "regularization_based_algorithms": [
        "Hybrid_Regularization[mf].py"
    ],
    "factory": [
        "cr_unionizedFactory.py",
        "als_recommender.py",
        "svd_recommender.py",
        "base_recommender.py"
    ]
}

# Base directory for your project
base_dir = "corerec/engines/test_struct_UF"

# Create the directory structure
for category, files in directory_structure.items():
    category_path = os.path.join(base_dir, category)
    os.makedirs(category_path, exist_ok=True)

    # Create the Python files in each category
    for file_name in files:
        file_path = os.path.join(category_path, file_name)
        with open(file_path, 'w') as f:
            f.write(f"# {file_name.replace('.py', '')} implementation\n")
            f.write("pass\n")

    # Create __init__.py for each category
    init_file_path = os.path.join(category_path, "__init__.py")
    with open(init_file_path, 'w') as init_file:
        # Generate import statements based on file names
        for file_name in files:
            algorithm_name = file_name.replace('.py', '').upper()
            alias_name = f"{category[0:3].upper()}_{algorithm_name}"
            init_file.write(f"from .{file_name[:-3]} import {algorithm_name} as {alias_name}\n")

print("Directory structure and files created successfully.")