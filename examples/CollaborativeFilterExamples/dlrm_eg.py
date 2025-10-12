import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from corerec.engines.unionizedFilterEngine.nn_base.DLRM_base import DLRM_base
from corerec.base_recommender import BaseCorerec




class DLRMRecommender(DLRM_base):
    """
    DLRM Recommender implementation that extends DLRM_base.
    Implements the required recommend method for recommendation generation.
    """
    
    def __init__(
        self,
        name: str = "DLRM",
        embed_dim: int = 16,
        bottom_mlp_dims: List[int] = [64, 32],
        top_mlp_dims: List[int] = [64, 32, 1],
        dropout: float = 0.1,
        batchnorm: bool = True,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        num_epochs: int = 20,
        patience: int = 5,
        shuffle: bool = True,
        device: str = None,
        seed: int = 42,
        verbose: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """Pass name correctly to the BaseCorerec parent class."""
        # Initialize BaseCorerec with the name parameter
        BaseCorerec.__init__(self, name=name)
        
        # Continue with DLRM_base initialization
        self.name = name
        self.seed = seed
        self.verbose = verbose
        self.is_fitted = False
        
        # Set random seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Process config if provided
        if config is not None:
            self.embed_dim = config.get("embed_dim", embed_dim)
            self.bottom_mlp_dims = config.get("bottom_mlp_dims", bottom_mlp_dims)
            self.top_mlp_dims = config.get("top_mlp_dims", top_mlp_dims)
            self.dropout = config.get("dropout", dropout)
            self.batchnorm = config.get("batchnorm", batchnorm)
            self.learning_rate = config.get("learning_rate", learning_rate)
            self.batch_size = config.get("batch_size", batch_size)
            self.num_epochs = config.get("num_epochs", num_epochs)
            self.patience = config.get("patience", patience)
            self.shuffle = config.get("shuffle", shuffle)
        else:
            self.embed_dim = embed_dim
            self.bottom_mlp_dims = bottom_mlp_dims
            self.top_mlp_dims = top_mlp_dims
            self.dropout = dropout
            self.batchnorm = batchnorm
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.num_epochs = num_epochs
            self.patience = patience
            self.shuffle = shuffle
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Setup logger
        self._setup_logger()
        
        # Initialize model
        self.model = None
        self.optimizer = None
        self.loss_fn = torch.nn.BCELoss()
        
        # Initialize data structures for users, items, and features
        self.categorical_map = {}
        self.categorical_names = []
        self.field_dims = []
        self.dense_features = []
        self.dense_dim = 0
        
        # Initialize hook manager for model introspection
        self.hook_manager = None
        
        if self.verbose:
            self.logger.info(f"Initialized {self.name} model with {self.embed_dim} embedding dimensions")
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: ID of the user to generate recommendations for
            n_recommendations: Number of recommendations to generate
            
        Returns:
            List of recommendation dictionaries containing item_id and score
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making recommendations")
            
        # Get user features
        user_features = self._get_user_features(user_id)
        
        # Get all possible items
        all_items = self._get_all_items()
        
        # Generate predictions for all items
        predictions = []
        for item_id in all_items:
            # Create feature dictionary for this user-item pair
            features = {**user_features, 'merchant_id': item_id}
            
            # Get prediction
            score = self.predict(features)
            
            predictions.append({
                'item_id': item_id,
                'score': float(score)
            })
        
        # Sort by score and return top N
        predictions.sort(key=lambda x: x['score'], reverse=True)
        return predictions[:n_recommendations]
    
    def _get_user_features(self, user_id: int) -> Dict[str, Any]:
        """Get features for a specific user."""
        # Get user info
        user_info = self.user_info[self.user_info['user_id'] == user_id].iloc[0]
        
        # Get user activity
        user_activity = self.user_activity[self.user_activity['user_id'] == user_id]
        
        features = {
            'user_id': user_id,
            'age_range': user_info['age_range'],
            'gender': user_info['gender']
        }
        
        if not user_activity.empty:
            activity = user_activity.iloc[0]
            features.update({
                'total_items': activity['total_items'],
                'unique_categories': activity['unique_categories'],
                'unique_brands': activity['unique_brands'],
                'total_clicks': activity['total_clicks']
            })
        else:
            features.update({
                'total_items': 0,
                'unique_categories': 0,
                'unique_brands': 0,
                'total_clicks': 0
            })
            
        return features
    
    def _get_all_items(self) -> List[int]:
        """Get list of all unique merchant IDs."""
        return self.merchant_ids

def load_ijcai_data(data_dir: str):
    """
    Load and preprocess IJCAI-15 dataset.
    
    Args:
        data_dir: Directory containing the dataset files
        
    Returns:
        Tuple of (train_data, test_data, user_info, user_logs)
    """
    print("Loading IJCAI-15 dataset...")
    
    # Load user information
    user_info = pd.read_csv(f"{data_dir}/user_info_format1.csv")
    
    # Load training and test data
    train_data = pd.read_csv(f"{data_dir}/train_format1.csv")
    test_data = pd.read_csv(f"{data_dir}/test_format1.csv")
    
    # Load user logs (we'll use this for feature engineering)
    user_logs = pd.read_csv(f"{data_dir}/user_log_format1.csv")
    
    return train_data, test_data, user_info, user_logs

def create_features(train_data, test_data, user_info, user_logs):
    """
    Create features for DLRM model from the dataset.
    
    Args:
        train_data: Training data DataFrame
        test_data: Test data DataFrame
        user_info: User information DataFrame
        user_logs: User logs DataFrame
        
    Returns:
        Tuple of (train_features, test_features)
    """
    print("Creating features...")
    
    # Merge user information with train/test data
    train_features = train_data.merge(user_info, on='user_id', how='left')
    test_features = test_data.merge(user_info, on='user_id', how='left')
    
    # Calculate user activity features from logs
    user_activity = user_logs.groupby('user_id').agg({
        'item_id': 'count',  # Number of items viewed
        'cat_id': 'nunique',  # Number of unique categories
        'brand_id': 'nunique',  # Number of unique brands
        'action_type': lambda x: (x == 0).sum()  # Number of clicks
    }).reset_index()
    
    user_activity.columns = ['user_id', 'total_items', 'unique_categories', 'unique_brands', 'total_clicks']
    
    # Merge activity features
    train_features = train_features.merge(user_activity, on='user_id', how='left')
    test_features = test_features.merge(user_activity, on='user_id', how='left')
    
    # Fill missing values
    train_features = train_features.fillna(0)
    test_features = test_features.fillna(0)
    
    # Convert to list of dictionaries format required by DLRM
    train_features_list = train_features.to_dict('records')
    test_features_list = test_features.to_dict('records')
    
    return train_features_list, test_features_list, user_info, user_activity, train_data['merchant_id'].unique()

def main():
    # Set data directory
    data_dir = "/Users/visheshyadav/Documents/GitHub/CoreRec/src/SANDBOX/dataset/IJCAI-15"
    
    print("""
    IMPORTANT: This example requires NumPy < 2.0 
    If you're getting NumPy compatibility errors, please run:
    
    pip install "numpy<2.0" pandas torch
    
    Then try running this example again.
    """)
    
    # Load data
    train_data, test_data, user_info, user_logs = load_ijcai_data(data_dir)
    
    # Create features
    train_features, test_features, user_info, user_activity, merchant_ids = create_features(
        train_data, test_data, user_info, user_logs
    )
    
    # Initialize DLRM model
    print("\nInitializing DLRM model...")
    model = DLRMRecommender(
        name="DLRM_IJCAI",
        embed_dim=16,
        bottom_mlp_dims=[128, 64, 32],
        top_mlp_dims=[64, 32, 1],
        dropout=0.2,
        batchnorm=True,
        learning_rate=0.001,
        batch_size=1024,
        num_epochs=20,
        patience=5,
        verbose=True
    )
    
    # Store necessary data for recommendations
    model.user_info = user_info
    model.user_activity = user_activity
    model.merchant_ids = merchant_ids
    
    # Train the model
    print("\nTraining DLRM model...")
    model.fit(train_features)
    
    # Make predictions on test data
    print("\nMaking predictions on test data...")
    predictions = []
    for sample in test_features[:5]:  # Show predictions for first 5 samples
        # Remove label for prediction
        features = {k: v for k, v in sample.items() if k != 'label'}
        pred = model.predict(features)
        predictions.append(pred)
        print(f"Sample: {features}")
        print(f"Predicted probability: {pred:.4f}")
        print(f"Actual label: {sample.get('label', 'N/A')}")
        print("-" * 50)
    
    # Generate recommendations for a sample user
    print("\nGenerating recommendations for a sample user...")
    sample_user_id = test_data['user_id'].iloc[0]
    recommendations = model.recommend(sample_user_id, n_recommendations=5)
    print(f"\nTop 5 recommendations for user {sample_user_id}:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. Merchant ID: {rec['item_id']}, Score: {rec['score']:.4f}")
    
    # Save the model
    print("\nSaving model...")
    model.save("dlrm_ijcai_model.pt")
    
    # Load the model
    print("\nLoading model...")
    loaded_model = DLRMRecommender.load("dlrm_ijcai_model.pt")
    
    # Verify loaded model predictions
    print("\nVerifying loaded model predictions...")
    for sample in test_features[:2]:  # Show predictions for first 2 samples
        features = {k: v for k, v in sample.items() if k != 'label'}
        pred = loaded_model.predict(features)
        print(f"Sample: {features}")
        print(f"Loaded model prediction: {pred:.4f}")
        print(f"Original model prediction: {model.predict(features):.4f}")
        print(f"Actual label: {sample.get('label', 'N/A')}")
        print("-" * 50)

if __name__ == "__main__":
    main()
