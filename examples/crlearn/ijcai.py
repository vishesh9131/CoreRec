from cr_learn.ijcai import load
import pandas as pd

data = load()

if __name__ == "__main__":
    print(f"Number of users: {len(data['users'])}")
    print(f"Number of merchants in training: {len(data['merchant_train']['merchant_id'].unique())}")
    print(f"Number of merchants in testing: {len(data['merchant_test']['merchant_id'].unique())}")

    print("\nSample of user data:")
    print(data["users"].head())

    print("\nSample of merchant training data:")
    print(data["merchant_train"].head())

    print("\nSample of merchant features:")
    sample_merchants = list(data["merchant_features"].keys())[:3]
    for merchant_id in sample_merchants:
        print(f"Merchant {merchant_id}:", data["merchant_features"][merchant_id])

    print("\nSample of user-merchant interactions:")
    sample_users = list(data["user_merchant_interaction"].keys())[:3]
    for user_id in sample_users:
        print(
            f"User {user_id} interacted with merchants:",
            data["user_merchant_interaction"][user_id][:5],
        )
