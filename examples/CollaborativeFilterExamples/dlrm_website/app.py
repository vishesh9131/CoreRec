import os
import json
import random
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from pathlib import Path
import sys

# Add the parent directory to the path so we can import the DLRM model
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from examples.CollaborativeFilterExamples.dlrm_eg import DLRMRecommender

app = Flask(__name__)
app.secret_key = 'dlrm_recommender_secret_key'

# Global variables
DATA_DIR = str(Path(__file__).parent.parent.parent.parent / "src/SANDBOX/dataset/IJCAI-15")
MODEL_PATH = str(Path(__file__).parent.parent / "dlrm_ijcai_model.pt")
DEFAULT_USER_ID = 163968  # Use a default user ID from the dataset

# Load merchant data (simplified for demo)
def load_merchant_data():
    # In a real application, you would load actual merchant data
    # For this demo, we'll create synthetic merchant data
    merchants = []
    # Try to load from train data to get real merchant IDs
    try:
        train_data = pd.read_csv(f"{DATA_DIR}/train_format1.csv")
        merchant_ids = train_data['merchant_id'].unique().tolist()
        
        # Create synthetic data for these merchants
        categories = ['Electronics', 'Fashion', 'Home & Kitchen', 'Books', 'Sports', 'Beauty', 'Toys']
        for merchant_id in merchant_ids[:100]:  # Limit to 100 merchants for demo
            merchant = {
                'id': int(merchant_id),
                'name': f"Merchant {merchant_id}",
                'category': random.choice(categories),
                'rating': round(random.uniform(3.0, 5.0), 1),
                'price': round(random.uniform(9.99, 299.99), 2),
                'image': f"product_{random.randint(1, 20)}.jpg",  # We'll create these sample images
                'description': f"This is a great product from Merchant {merchant_id}. High quality and good value."
            }
            merchants.append(merchant)
    except Exception as e:
        print(f"Error loading merchant data: {e}")
        # Fallback to completely synthetic data
        for i in range(1, 101):
            merchant = {
                'id': i,
                'name': f"Merchant {i}",
                'category': random.choice(categories),
                'rating': round(random.uniform(3.0, 5.0), 1),
                'price': round(random.uniform(9.99, 299.99), 2),
                'image': f"product_{random.randint(1, 20)}.jpg",
                'description': f"This is a great product from Merchant {i}. High quality and good value."
            }
            merchants.append(merchant)
    
    return merchants

# Load user data
def load_user_data():
    users = []
    try:
        user_info = pd.read_csv(f"{DATA_DIR}/user_info_format1.csv")
        for _, row in user_info.head(20).iterrows():  # Limit to 20 users for demo
            user = {
                'id': int(row['user_id']),
                'age_range': int(row['age_range']) if not np.isnan(row['age_range']) else 0,
                'gender': int(row['gender']) if not np.isnan(row['gender']) else 0,
                'name': f"User {row['user_id']}"
            }
            users.append(user)
    except Exception as e:
        print(f"Error loading user data: {e}")
        # Fallback to synthetic users
        for i in range(1, 21):
            user = {
                'id': i,
                'age_range': random.randint(1, 6),
                'gender': random.randint(0, 1),
                'name': f"User {i}"
            }
            users.append(user)
    
    return users

# Load or create the DLRM recommender model
def get_recommender():
    try:
        # Try to load the pretrained model
        print(f"Loading model from {MODEL_PATH}")
        recommender = DLRMRecommender.load(MODEL_PATH)
        
        # We need to set up the user_info, user_activity, and merchant_ids for recommendation generation
        train_data = pd.read_csv(f"{DATA_DIR}/train_format1.csv")
        user_info = pd.read_csv(f"{DATA_DIR}/user_info_format1.csv")
        user_logs = pd.read_csv(f"{DATA_DIR}/user_log_format1.csv")
        
        # Calculate user activity features from logs
        user_activity = user_logs.groupby('user_id').agg({
            'item_id': 'count',  # Number of items viewed
            'cat_id': 'nunique',  # Number of unique categories
            'brand_id': 'nunique',  # Number of unique brands
            'action_type': lambda x: (x == 0).sum()  # Number of clicks
        }).reset_index()
        
        user_activity.columns = ['user_id', 'total_items', 'unique_categories', 'unique_brands', 'total_clicks']
        
        recommender.user_info = user_info
        recommender.user_activity = user_activity
        recommender.merchant_ids = train_data['merchant_id'].unique()
        
        return recommender
    except Exception as e:
        print(f"Error loading recommender model: {e}")
        print("Using random recommendations as fallback")
        return None

# Initialize data
merchants = load_merchant_data()
users = load_user_data()
recommender = get_recommender()

# Routes
@app.route('/')
def index():
    # Get the current user from session or use default
    user_id = session.get('user_id', DEFAULT_USER_ID)
    user = next((u for u in users if u['id'] == user_id), users[0])
    
    # Get featured merchants (random selection for demo)
    featured = random.sample(merchants, min(8, len(merchants)))
    
    # Get personalized recommendations
    recommended = get_recommendations(user_id, 8)
    
    # Get trending merchants (random for demo)
    trending = random.sample(merchants, min(4, len(merchants)))
    
    return render_template('index.html', 
                          user=user,
                          users=users,
                          featured=featured,
                          recommended=recommended,
                          trending=trending)

@app.route('/product/<int:merchant_id>')
def product_detail(merchant_id):
    # Find the merchant
    merchant = next((m for m in merchants if m['id'] == merchant_id), None)
    if not merchant:
        return redirect(url_for('index'))
    
    # Get the current user
    user_id = session.get('user_id', DEFAULT_USER_ID)
    user = next((u for u in users if u['id'] == user_id), users[0])
    
    # Get related products (random for demo)
    related = random.sample([m for m in merchants if m['id'] != merchant_id], 
                           min(4, len(merchants)-1))
    
    return render_template('product.html', 
                          user=user,
                          users=users,
                          product=merchant,
                          related=related)

@app.route('/category/<category>')
def category(category):
    # Filter merchants by category
    category_merchants = [m for m in merchants if m['category'] == category]
    if not category_merchants:
        return redirect(url_for('index'))
    
    # Get the current user
    user_id = session.get('user_id', DEFAULT_USER_ID)
    user = next((u for u in users if u['id'] == user_id), users[0])
    
    return render_template('category.html', 
                          user=user,
                          users=users,
                          category=category,
                          products=category_merchants)

@app.route('/switch_user/<int:user_id>')
def switch_user(user_id):
    session['user_id'] = user_id
    return redirect(request.referrer or url_for('index'))

@app.route('/api/recommendations/<int:user_id>')
def api_recommendations(user_id):
    count = request.args.get('count', 8, type=int)
    recommendations = get_recommendations(user_id, count)
    return jsonify(recommendations)

# Helper function to get recommendations
def get_recommendations(user_id, count=8):
    if recommender:
        try:
            # Get DLRM recommendations
            recs = recommender.recommend(user_id, n_recommendations=count)
            # Match recommendations with merchant data
            recommended = []
            for rec in recs:
                merchant = next((m for m in merchants if m['id'] == rec['item_id']), None)
                if merchant:
                    # Add the score to the merchant data
                    merchant_copy = merchant.copy()
                    merchant_copy['score'] = rec['score']
                    recommended.append(merchant_copy)
            
            # If we don't have enough recommendations, pad with random ones
            if len(recommended) < count:
                remaining = count - len(recommended)
                # Get merchants not already in recommendations
                remaining_merchants = [m for m in merchants if m['id'] not in [r['id'] for r in recommended]]
                if remaining_merchants:
                    random_recs = random.sample(remaining_merchants, min(remaining, len(remaining_merchants)))
                    for m in random_recs:
                        m_copy = m.copy()
                        m_copy['score'] = 0.1  # Low score for random recommendations
                        recommended.append(m_copy)
            
            return recommended
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            # Fall back to random recommendations
            return random.sample(merchants, min(count, len(merchants)))
    else:
        # If no recommender, return random merchants
        return random.sample(merchants, min(count, len(merchants)))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 