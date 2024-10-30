CONFIG = {
    "collaborative_filtering": {
        "method": "matrix_factorization",  # Options: matrix_factorization, user_based_cf, etc.
        "params": {
            "num_factors": 20,
            "learning_rate": 0.01,
            "regularization": 0.02,
            "epochs": 20
        }
    }
}