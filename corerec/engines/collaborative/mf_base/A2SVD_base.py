from typing import Union, Optional, List, Tuple, Dict, Callable
from pathlib import Path
import pickle
import numpy as np
from scipy.linalg import svd, cholesky
from corerec.api.exceptions import ModelNotFittedError
import logging

logger = logging.getLogger(__name__)

        
class A2SVDBase:
    """
    Activation-aware Singular Value Decomposition for compressing weight matrices.
    
    Implementation of ASVD approach for training-free compression of weight
    matrices by accounting for activation distributions. Handles outliers in
    activations and layer-specific sensitivity differences.
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        transform_method: str = "magnitude",
        use_cholesky: bool = False,
        absorb_singular_values: bool = True,
        verbose: bool = False
    ):
        """
        Initialize ASVD base class.
        
        Parameters:
        -----------
        alpha : float, default=0.5
            Control factor for magnitude-based transform scaling
        transform_method : str, default="magnitude"
            Method for computing transform matrix: "magnitude" or "cholesky"
        use_cholesky : bool, default=False
            If True, use Cholesky decomposition method (ASVD+ variant)
        absorb_singular_values : bool, default=True
            Whether to fuse singular values into U and V matrices
        verbose : bool, default=False
            Enable verbose logging
        """
        self.alpha = alpha
        self.transform_method = transform_method
        self.use_cholesky = use_cholesky
        self.absorb_singular_values = absorb_singular_values
        self.verbose = verbose
        
        self.decomposed_matrices = {}
        self.transform_matrices = {}
        self.layer_ranks = {}
        self.is_fitted = False
        
    def _compute_magnitude_transform(self, activations: np.ndarray) -> np.ndarray:
        """
        Compute diagonal transform matrix based on activation magnitudes.
        
        Uses absolute mean value of activations per channel.
        """
        n_channels = activations.shape[1]
        diagonal_values = np.zeros(n_channels)
        
        for i in range(n_channels):
            channel_data = activations[:, i]
            abs_mean = np.mean(np.abs(channel_data))
            # avoid div by zero, set minimum threshold
            if abs_mean > 1e-8:
                diagonal_values[i] = abs_mean ** self.alpha
            else:
                diagonal_values[i] = 1.0
        
        S = np.diag(diagonal_values)
        return S
    
    def _compute_cholesky_transform(self, activations: np.ndarray) -> np.ndarray:
        """
        Compute transform matrix using Cholesky decomposition of XX^T.
        
        This minimizes output error directly.
        """
        X = activations.T
        XXT = X @ X.T
        
        try:
            L = cholesky(XXT, lower=True)
            return L
        except (np.linalg.LinAlgError, ValueError) as e:
            # fallback if cholesky fails (e.g. matrix not positive definite)
            if self.verbose:
                logger.warning(f"Cholesky failed, using magnitude method: {e}")
            return self._compute_magnitude_transform(activations)
    
    def _compute_transform_matrix(self, activations: np.ndarray) -> np.ndarray:
        """
        Compute transformation matrix S based on activation patterns.
        """
        if self.use_cholesky or self.transform_method == "cholesky":
            return self._compute_cholesky_transform(activations)
        else:
            return self._compute_magnitude_transform(activations)
    
    def decompose_weight_matrix(
        self,
        W: np.ndarray,
        activations: Optional[np.ndarray] = None,
        rank: Optional[int] = None,
        param_ratio: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Decompose weight matrix using activation-aware SVD.
        
        Parameters:
        -----------
        W : np.ndarray, shape (m, n)
            Weight matrix to decompose
        activations : np.ndarray, optional, shape (L, n)
            Input activations for computing transform matrix
        rank : int, optional
            Target rank for decomposition. If None, computed from param_ratio
        param_ratio : float, optional
            Target parameter ratio (0-1). Used to compute rank if rank is None
            
        Returns:
        --------
        U : np.ndarray
            Left singular vectors
        V : np.ndarray  
            Right singular vectors (transformed)
        Sigma : np.ndarray, optional
            Singular values (None if absorbed)
        S : np.ndarray
            Transform matrix used
        """
        m, n = W.shape
        
        if activations is None:
            S = np.eye(n)
            S_inv = np.eye(n)
        else:
            S = self._compute_transform_matrix(activations)
            S_inv = np.linalg.inv(S)
        
        WS = W @ S
        
        if rank is None and param_ratio is not None:
            k = int(np.sqrt(param_ratio * m * n / (m + n)))
            k = max(1, min(k, min(m, n)))
        elif rank is None:
            k = min(m, n)
        else:
            k = max(1, min(rank, min(m, n)))
        
        U, s, Vt = svd(WS, full_matrices=False)
        
        U_k = U[:, :k]
        s_k = s[:k]
        Vt_k = Vt[:k, :]
        
        Vt_k_transformed = Vt_k @ S_inv
        
        if self.absorb_singular_values:
            sqrt_s = np.sqrt(s_k)
            U_final = U_k * sqrt_s[np.newaxis, :]
            V_final = (sqrt_s[:, np.newaxis] * Vt_k_transformed).T
            return U_final, V_final, None, S
        else:
            return U_k, Vt_k_transformed, np.diag(s_k), S
    
    def reconstruct_weight_matrix(
        self,
        U: np.ndarray,
        V: np.ndarray,
        Sigma: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Reconstruct weight matrix from decomposed components.
        
        Parameters:
        -----------
        U : np.ndarray
            Left singular vectors
        V : np.ndarray
            Right singular vectors
        Sigma : np.ndarray, optional
            Singular values (if not absorbed)
            
        Returns:
        --------
        W_reconstructed : np.ndarray
            Reconstructed weight matrix
        """
        if Sigma is None:
            return U @ V.T
        else:
            if Sigma.ndim == 1:
                return U @ np.diag(Sigma) @ V.T
            else:
                return U @ Sigma @ V.T
    
    def _compute_output_error(
        self,
        W_original: np.ndarray,
        W_approx: np.ndarray,
        activations: np.ndarray
    ) -> float:
        """
        Compute Frobenius norm of output error.
        
        Error = ||(W_k - W)X||_F
        """
        delta_W = W_approx - W_original
        delta_Y = delta_W @ activations.T
        error = np.linalg.norm(delta_Y, ord='fro')
        return float(error)
    
    def _evaluate_rank_candidates(
        self,
        W: np.ndarray,
        activations: np.ndarray,
        candidate_ratios: List[float]
    ) -> List[Tuple[float, float, int]]:
        """
        Evaluate different rank candidates and return sensitivities.
        
        Returns list of (ratio, error, rank) tuples sorted by error.
        """
        results = []
        m, n = W.shape
        max_rank = min(m, n)
        
        for ratio in candidate_ratios:
            k = int(np.sqrt(ratio * m * n / (m + n)))
            k = max(1, min(k, max_rank))
            
            U, V, Sigma, S = self.decompose_weight_matrix(
                W, activations, rank=k
            )
            
            W_approx = self.reconstruct_weight_matrix(U, V, Sigma)
            error = self._compute_output_error(W, W_approx, activations)
            
            results.append((ratio, error, k))
        
        results.sort(key=lambda x: x[1])
        return results
    
    def binary_search_rank(
        self,
        W: np.ndarray,
        activations: np.ndarray,
        target_param_ratio: float,
        tolerance: float = 0.01
    ) -> int:
        """
        Binary search for optimal rank given target parameter ratio.
        
        Parameters:
        -----------
        W : np.ndarray
            Weight matrix
        activations : np.ndarray
            Input activations
        target_param_ratio : float
            Target parameter ratio (0-1)
        tolerance : float
            Tolerance for binary search convergence
            
        Returns:
        --------
        optimal_rank : int
            Optimal rank found
        """
        m, n = W.shape
        max_rank = min(m, n)
        min_rank = 1
        
        def get_param_ratio(rank):
            return (rank * (m + n)) / (m * n)
        
        while max_rank - min_rank > 1:
            mid_rank = (min_rank + max_rank) // 2
            current_ratio = get_param_ratio(mid_rank)
            
            if current_ratio <= target_param_ratio:
                min_rank = mid_rank
            else:
                max_rank = mid_rank
        
        U, V, _, _ = self.decompose_weight_matrix(W, activations, rank=min_rank)
        W_approx_min = self.reconstruct_weight_matrix(U, V, None)
        error_min = self._compute_output_error(W, W_approx_min, activations)
        
        U, V, _, _ = self.decompose_weight_matrix(W, activations, rank=max_rank)
        W_approx_max = self.reconstruct_weight_matrix(U, V, None)
        error_max = self._compute_output_error(W, W_approx_max, activations)
        
        if error_min <= error_max:
            return min_rank
        else:
            return max_rank
    
    def compress_layer(
        self,
        layer_name: str,
        W: np.ndarray,
        activations: Optional[np.ndarray] = None,
        rank: Optional[int] = None,
        param_ratio: Optional[float] = None
    ) -> Dict:
        """
        Compress a single layer's weight matrix.
        
        Parameters:
        -----------
        layer_name : str
            Name identifier for the layer
        W : np.ndarray
            Weight matrix to compress
        activations : np.ndarray, optional
            Input activations for this layer
        rank : int, optional
            Target rank
        param_ratio : float, optional
            Target parameter ratio
            
        Returns:
        --------
        compression_info : dict
            Dictionary with compression details
        """
        if rank is None and param_ratio is not None:
            rank = self.binary_search_rank(W, activations, param_ratio)
        
        U, V, Sigma, S = self.decompose_weight_matrix(W, activations, rank=rank)
        W_approx = self.reconstruct_weight_matrix(U, V, Sigma)
        
        original_params = W.size
        compressed_params = U.size + V.size
        if Sigma is not None:
            compressed_params += Sigma.size
        
        compression_ratio = compressed_params / original_params
        
        error = self._compute_output_error(W, W_approx, activations) if activations is not None else None
        
        compression_info = {
            'U': U,
            'V': V,
            'Sigma': Sigma,
            'S': S,
            'rank': rank,
            'compression_ratio': compression_ratio,
            'original_params': original_params,
            'compressed_params': compressed_params,
            'error': error
        }
        
        self.decomposed_matrices[layer_name] = compression_info
        self.transform_matrices[layer_name] = S
        self.layer_ranks[layer_name] = rank
        
        if self.verbose:
            logger.info(f"Layer {layer_name}: rank={rank}, ratio={compression_ratio:.3f}")
        
        return compression_info
    
    def get_compressed_weight(self, layer_name: str) -> np.ndarray:
        """
        Get reconstructed weight matrix for a layer.
        
        Parameters:
        -----------
        layer_name : str
            Layer identifier
            
        Returns:
        --------
        W_reconstructed : np.ndarray
            Reconstructed weight matrix
        """
        if layer_name not in self.decomposed_matrices:
            raise ModelNotFittedError(f"Layer {layer_name} not found. Call compress_layer first.")
        
        info = self.decomposed_matrices[layer_name]
        return self.reconstruct_weight_matrix(info['U'], info['V'], info['Sigma'])

    def sensitivity_based_rank_search(
        self,
        layers: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]],
        target_param_ratio: float,
        candidate_ratios: Optional[List[float]] = None,
        eval_function: Optional[Callable] = None
    ) -> Dict[str, int]:
        """
        Sensitivity-based truncation rank searching for multiple layers.
        
        Parameters:
        -----------
        layers : Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]
            Dictionary mapping layer names to (weight_matrix, activations) tuples
        target_param_ratio : float
            Overall target parameter ratio across all layers
        candidate_ratios : List[float], optional
            Candidate parameter ratios to evaluate. Default: [0.1, 0.2, ..., 0.9]
        eval_function : Callable, optional
            Custom evaluation function. If None, uses output error
            
        Returns:
        --------
        layer_ranks : Dict[str, int]
            Dictionary mapping layer names to optimal ranks
        """
        if candidate_ratios is None:
            candidate_ratios = [i / 10.0 for i in range(1, 10)]
        
        layer_sensitivities = {}
        
        for layer_name, (W, activations) in layers.items():
            if activations is None:
                activations = np.random.randn(100, W.shape[1])
            
            results = self._evaluate_rank_candidates(W, activations, candidate_ratios)
            layer_sensitivities[layer_name] = results
        
        # binary search for optimal rank configuration
        all_sensitivities = []
        for layer_name, results in layer_sensitivities.items():
            for ratio, error, rank in results:
                all_sensitivities.append((layer_name, ratio, error, rank))
        
        all_sensitivities.sort(key=lambda x: x[2])
        
        pL, pH = 0, len(all_sensitivities) - 1
        optimal_ranks = {}
        
        while pL <= pH:
            pM = (pL + pH) // 2
            
            # set ranks based on position pM
            current_ranks = {}
            for layer_name, _, _, rank in all_sensitivities[pM:]:
                if layer_name not in current_ranks:
                    current_ranks[layer_name] = rank
                else:
                    current_ranks[layer_name] = min(current_ranks[layer_name], rank)
            
            # compute total param ratio
            total_original = sum(layers[name][0].size for name in layers.keys())
            total_compressed = 0
            for name, (W, _) in layers.items():
                rank = current_ranks.get(name, min(W.shape))
                m, n = W.shape
                total_compressed += rank * (m + n)
            
            current_ratio = total_compressed / total_original if total_original > 0 else 1.0
            
            if current_ratio <= target_param_ratio:
                optimal_ranks = current_ranks.copy()
                pH = pM - 1
            else:
                pL = pM + 1
        
        return optimal_ranks
    
    def compress_kv_cache(
        self,
        k_proj_weights: np.ndarray,
        v_proj_weights: np.ndarray,
        k_activations: Optional[np.ndarray] = None,
        v_activations: Optional[np.ndarray] = None,
        rank: Optional[int] = None,
        cache_ratio: Optional[float] = None
    ) -> Dict[str, Dict]:
        """
        Compress Key/Value projection matrices for KV cache compression.
        
        By decomposing K/V projection matrices, intermediate activations can
        be stored in low-rank form, reducing KV cache memory requirements.
        
        Parameters:
        -----------
        k_proj_weights : np.ndarray
            Key projection weight matrix
        v_proj_weights : np.ndarray
            Value projection weight matrix
        k_activations : np.ndarray, optional
            Key input activations
        v_activations : np.ndarray, optional
            Value input activations
        rank : int, optional
            Target rank for decomposition
        cache_ratio : float, optional
            Target KV cache compression ratio
            
        Returns:
        --------
        compression_info : Dict[str, Dict]
            Compression info for both K and V projections
        """
        if rank is None and cache_ratio is not None:
            # estimate rank from cache ratio
            N = k_proj_weights.shape[0]
            target_dim = int(N * cache_ratio)
            rank = max(1, target_dim)
        
        k_info = self.compress_layer("k_projection", k_proj_weights, k_activations, rank=rank)
        v_info = self.compress_layer("v_projection", v_proj_weights, v_activations, rank=rank)
        
        return {
            'k_projection': k_info,
            'v_projection': v_info
        }

    def save(self, path: Union[str, Path], **kwargs) -> None:
        """
        Save model to disk using pickle.
        
        Parameters:
        -----------
        path : Union[str, Path]
            File path to save the model
        **kwargs : dict
            Additional arguments (unused)
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path_obj, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        if self.verbose:
            logger.info(f"A2SVDBase model saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> 'A2SVDBase':
        """
            Load model from disk.
        
        Parameters:
        -----------
        path : Union[str, Path]
            File path to load the model from
        **kwargs : dict
            Additional arguments (unused)
            
            Returns:
        --------
                Loaded model instance
        """
        path_obj = Path(path)
        
        with open(path_obj, 'rb') as f:
            model = pickle.load(f)
        
        if not isinstance(model, cls):
            raise ValueError(f"Loaded object is not instance of {cls.__name__}")
        
        if model.verbose:
            logger.info(f"A2SVDBase model loaded from {path}")
        
        return model
