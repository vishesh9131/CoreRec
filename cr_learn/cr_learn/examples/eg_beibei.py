from cr_learn import beibei as bei

datasets = bei.load()

for name, dataset in datasets.items():
    print(f"\nProcessing dataset: {name}")
    
    print(f"Number of users: {dataset.get_user_count()}")
    print(f"Number of items: {dataset.get_item_count()}")
    print(f"Number of interactions: {dataset.get_interaction_count()}")
    
    sparsity = dataset.get_sparsity()
    print(f"Sparsity of the dataset: {sparsity:.2%}")
    
    popular_items = dataset.get_popular_items(top_n=5)
    print("Top 5 popular items:")
    print(popular_items)
    
    active_users = dataset.get_active_users(top_n=5)
    print("Top 5 active users:")
    print(active_users)
    
    negative_samples = dataset.get_negative_samples(sample_size=5)
    print("Sampled negative interactions:")
    print(negative_samples)
    
    print("\n" + "-"*40 + "\n")
