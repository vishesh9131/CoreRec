# Correct cr_learn API Usage

## What Actually Works

```python
# ✅ CORRECT - Use specific dataset modules
from cr_learn import ml_1m  # MovieLens 1M
from cr_learn import ml      # MovieLens 100K  
from cr_learn import beibei  # Beibei dataset
from sklearn.model_selection import train_test_split

# Load data
data = ml_1m.load()  # Returns dict
ratings_df = data['ratings']  # DataFrame

# Split
train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

# Extract arrays for model
train_users = train_df['user_id'].values
train_items = train_df['movie_id'].values  
train_ratings = train_df['rating'].values
```

## What DOESN'T Work (My Mistake)

```python
# ❌ WRONG - This API doesn't exist
data = cr_learn.load_dataset('movielens-100k')
train_data, test_data = data.train_test_split()
```

## Available Datasets in cr_learn

- `cr_learn.ml` - MovieLens 100K
- `cr_learn.ml_1m` - MovieLens 1M
- `cr_learn.beibei` - Beibei e-commerce
- `cr_learn.steam_games` - Steam games
- `cr_learn.tmall` - Tmall purchases

All return a dict with keys:
- `'ratings'`: DataFrame with user_id, item_id/movie_id, rating, timestamp
- `'users'`: User metadata
- `'movies'` or `'items'`: Item metadata

## Fix for All Tutorials

Update Step 1 in all tutorials to use the correct API shown above.
