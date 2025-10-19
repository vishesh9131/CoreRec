"""
Model Evaluator

Tools for evaluating and comparing recommendation models.

Author: Vishesh Yadav (mail: sciencely98@gmail.com)
"""

from typing import Dict, List, Any, Optional
import numpy as np
from corerec.evaluation.metrics import RankingMetrics


class Evaluator:
    """
    Model evaluator for recommendation systems.
    
    Evaluates models on test data using multiple metrics.
    
    Example:
        from corerec.evaluation import Evaluator
        
        evaluator = Evaluator(metrics=['ndcg@10', 'map@10', 'recall@20'])
        
        # Single model evaluation
        results = evaluator.evaluate(model, test_data)
        
        # Model comparison
        comparison = evaluator.compare_models({
            'NCF': ncf_model,
            'DeepFM': deepfm_model
        }, test_data)
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(self, metrics: List[str] = None):
        """
        Initialize evaluator.
        
        Args:
            metrics: List of metrics to compute (e.g., ['ndcg@10', 'map@10'])
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.metrics = metrics or ['ndcg@10', 'map@10', 'precision@10', 'recall@10']
        self.ranking_metrics = RankingMetrics()
    
    def evaluate(self, model, test_data: Dict[Any, List]) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            model: Recommendation model with recommend() method
            test_data: Dict mapping user_id to list of relevant items
            
        Returns:
            Dictionary of metric_name -> score
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        results = {metric: [] for metric in self.metrics}
        
        for user_id, ground_truth in test_data.items():
            try:
                # Get recommendations
                predictions = model.recommend(user_id, top_k=20)
                
                # Compute each metric
                for metric_name in self.metrics:
                    # Parse metric name (e.g., 'ndcg@10')
                    if '@' in metric_name:
                        metric, k = metric_name.split('@')
                        k = int(k)
                    else:
                        metric = metric_name
                        k = 10
                    
                    # Compute metric
                    if metric == 'ndcg':
                        score = self.ranking_metrics.ndcg_at_k(predictions, ground_truth, k)
                    elif metric == 'map':
                        score = self.ranking_metrics.map_at_k(predictions, ground_truth, k)
                    elif metric == 'mrr':
                        score = self.ranking_metrics.mrr_at_k(predictions, ground_truth, k)
                    elif metric == 'precision':
                        score = self.ranking_metrics.precision_at_k(predictions, ground_truth, k)
                    elif metric == 'recall':
                        score = self.ranking_metrics.recall_at_k(predictions, ground_truth, k)
                    elif metric == 'hit_rate':
                        score = self.ranking_metrics.hit_rate_at_k(predictions, ground_truth, k)
                    else:
                        continue
                    
                    results[metric_name].append(score)
            
            except Exception as e:
                print(f"Error evaluating user {user_id}: {e}")
                continue
        
        # Average results
        return {k: np.mean(v) if v else 0.0 for k, v in results.items()}
    
    def compare_models(self, models: Dict[str, Any], 
                      test_data: Dict[Any, List]) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models on same test data.
        
        Args:
            models: Dict mapping model_name to model instance
            test_data: Test data (user_id -> relevant items)
            
        Returns:
            Dict mapping model_name to evaluation results
            
        Example:
            results = evaluator.compare_models({
                'NCF': ncf_model,
                'DeepFM': deepfm_model
            }, test_data)
            
            # Results:
            # {
            #   'NCF': {'ndcg@10': 0.45, 'map@10': 0.38, ...},
            #   'DeepFM': {'ndcg@10': 0.48, 'map@10': 0.41, ...}
            # }
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        comparison = {}
        
        for model_name, model in models.items():
            print(f"Evaluating {model_name}...")
            results = self.evaluate(model, test_data)
            comparison[model_name] = results
        
        return comparison
    
    def generate_report(self, results: Dict[str, Dict[str, float]]) -> str:
        """
        Generate human-readable evaluation report.
        
        Args:
            results: Evaluation results from compare_models()
            
        Returns:
            Formatted report string
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        report = "=" * 60 + "\n"
        report += "Model Evaluation Report\n"
        report += "=" * 60 + "\n\n"
        
        # Get all metrics
        all_metrics = set()
        for model_results in results.values():
            all_metrics.update(model_results.keys())
        all_metrics = sorted(all_metrics)
        
        # Create table
        header = f"{'Model':<20}"
        for metric in all_metrics:
            header += f" {metric:>12}"
        report += header + "\n"
        report += "-" * len(header) + "\n"
        
        for model_name, model_results in results.items():
            row = f"{model_name:<20}"
            for metric in all_metrics:
                value = model_results.get(metric, 0.0)
                row += f" {value:>12.4f}"
            report += row + "\n"
        
        return report


class CrossValidator:
    """
    Cross-validation utilities.
    
    Example:
        cv = CrossValidator(n_folds=5)
        avg_score = cv.cross_validate(model, data, metric='ndcg@10')
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(self, n_folds: int = 5, random_state: int = 42):
        """
        Initialize cross-validator.
        
        Args:
            n_folds: Number of folds
            random_state: Random seed
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.n_folds = n_folds
        self.random_state = random_state
    
    def split(self, data: Any, n_folds: Optional[int] = None) -> List[tuple]:
        """
        Split data into folds.
        
        Args:
            data: Data to split
            n_folds: Number of folds (uses self.n_folds if None)
            
        Returns:
            List of (train, test) tuples
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        n_folds = n_folds or self.n_folds
        
        # Simple implementation - can be enhanced
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            data = data.sample(frac=1, random_state=self.random_state)  # Shuffle
            fold_size = len(data) // n_folds
            
            folds = []
            for i in range(n_folds):
                test_start = i * fold_size
                test_end = (i + 1) * fold_size if i < n_folds - 1 else len(data)
                
                test_data = data.iloc[test_start:test_end]
                train_data = pd.concat([
                    data.iloc[:test_start],
                    data.iloc[test_end:]
                ])
                
                folds.append((train_data, test_data))
            
            return folds
        else:
            raise NotImplementedError("Only DataFrame supported currently")

