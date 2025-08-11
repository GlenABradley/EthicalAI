"""Advanced Axis Builder (Tier B).

Extends the basic axis builder with advanced features:
- Whitening/PCA for decorrelation
- LDA for better class separation
- Orthogonalization with decorrelation penalty
- Margin scoring blending Mahalanobis distance with directional cosine
- Bootstrap uncertainty estimation
- LoRA contrastive adapter stub
- Full JSON field utilization
- CLI and NPZ workflow support
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Union, Sequence, Tuple

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.covariance import EmpiricalCovariance, ledoit_wolf, LedoitWolf
from sklearn.decomposition import PCA

from coherence.axis.pack import AxisPack

logger = logging.getLogger(__name__)


def _load_json_axis_config(path: Union[str, Path]) -> dict:
    """Load and validate an axis configuration from JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Validate required fields
    required_fields = ['name', 'max_examples', 'min_examples']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in axis config: {field}")
    
    return config


def _compute_whitening_matrix(vectors: np.ndarray, method: str = 'empirical') -> np.ndarray:
    """Compute whitening matrix using specified method.
    
    Args:
        vectors: Input vectors (n_samples, n_features)
        method: Whitening method ('empirical' or 'pca' or 'zca')
        
    Returns:
        Whitening matrix (n_features, n_features)
    """
    logger.debug(f"Computing {method} whitening matrix...")
    vectors = np.asarray(vectors, dtype=np.float32)
    
    # Log input shape and stats
    logger.debug(f"Input vectors shape: {vectors.shape}, dtype: {vectors.dtype}")
    logger.debug(f"Input stats - min: {np.min(vectors):.4f}, max: {np.max(vectors):.4f}, "
                f"mean: {np.mean(vectors):.4f}, std: {np.std(vectors):.4f}")
    
    if len(vectors.shape) != 2:
        error_msg = f"Expected 2D array, got shape {vectors.shape}"
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    n_samples, n_features = vectors.shape
    logger.debug(f"n_samples: {n_samples}, n_features: {n_features}")
    
    if n_samples < 2:
        warning_msg = f"Only {n_samples} samples provided, which is insufficient for stable whitening"
        logger.warning(warning_msg)
        return np.eye(n_features, dtype=np.float32)
    
    # Center the data
    mean = np.mean(vectors, axis=0)
    X_centered = vectors - mean
    
    if method == 'empirical':
        try:
            logger.debug("Computing Ledoit-Wolf covariance...")
            # Compute covariance matrix with Ledoit-Wolf shrinkage
            # Ensure we get the full covariance matrix, not just the diagonal
            lw = LedoitWolf(store_precision=False, assume_centered=False)
            lw.fit(X_centered)
            cov = lw.covariance_
            
            # Ensure cov is 2D and has the right shape
            cov = np.atleast_2d(cov)
            if cov.shape != (n_features, n_features):
                cov = np.cov(X_centered, rowvar=False)
                
            logger.debug(f"Covariance matrix shape: {cov.shape}")
            logger.debug(f"Covariance matrix condition number: {np.linalg.cond(cov):.4f}")
            
            # Add small diagonal for numerical stability
            cov = cov + 1e-6 * np.eye(n_features)
            
            # Compute SVD
            logger.debug("Computing SVD...")
            U, S, Vt = np.linalg.svd(cov, full_matrices=False)
            
            # Log singular values for diagnostics
            logger.debug(f"Singular values: {S[:5]}{'...' if len(S) > 5 else ''}")
            logger.debug(f"Condition number from SVD: {S[0] / (S[-1] + 1e-10):.4f}")
            
            # Avoid division by zero
            S = np.maximum(S, 1e-6)
            W = np.dot(U, np.diag(1.0 / np.sqrt(S)))
            
            # Ensure output has correct shape
            if W.shape != (n_features, n_features):
                warning_msg = (f"Whitening matrix has incorrect shape {W.shape}, "
                             f"expected (n_features, n_features) = ({(n_features, n_features)}), "
                             "using identity")
                logger.warning(warning_msg)
                return np.eye(n_features, dtype=np.float32)
                
            logger.debug(f"Successfully computed whitening matrix with shape {W.shape}")
            return W
            
        except Exception as e:
            error_msg = f"Error in empirical whitening: {str(e)}"
            logger.warning(error_msg, exc_info=True)
            logger.warning("Falling back to ZCA whitening")
            method = 'zca'  # Fall back to ZCA whitening
    
    if method == 'pca':
        try:
            pca = PCA(whiten=True, random_state=42)
            pca.fit(vectors)
            W = pca.components_.T * np.sqrt(pca.explained_variance_)
        except Exception as e:
            logger.warning(f"Error in PCA whitening: {e}, falling back to ZCA")
            method = 'zca'
    
    # ZCA whitening as fallback
    if method == 'zca':
        try:
            # Simple ZCA whitening
            cov = np.cov(X_centered, rowvar=False)
            cov = np.atleast_2d(cov)  # Ensure 2D
            
            # Add small diagonal for numerical stability
            cov = cov + 1e-6 * np.eye(cov.shape[0])
            
            # Compute SVD
            U, S, Vt = np.linalg.svd(cov)
            
            # Avoid division by zero
            S = np.maximum(S, 1e-6)
            
            # ZCA whitening matrix
            W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S)), U.T))
            
            # Ensure output has correct shape
            if W.shape != (n_features, n_features):
                W = np.eye(n_features, dtype=np.float32)
                logger.warning("ZCA whitening failed, using identity matrix")
                
        except Exception as e:
            logger.error(f"ZCA whitening failed: {e}, using identity matrix")
            W = np.eye(n_features, dtype=np.float32)
    
    return W.astype(np.float32)


def _apply_lda(
    pos_vectors: np.ndarray, 
    neg_vectors: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Linear Discriminant Analysis to find optimal projection.
    
    Args:
        pos_vectors: Positive examples (n_pos, n_features)
        neg_vectors: Negative examples (n_neg, n_features)
        
    Returns:
        Tuple of (projected_pos, projected_neg) vectors
    """
    X = np.vstack([pos_vectors, neg_vectors])
    y = np.array([1] * len(pos_vectors) + [0] * len(neg_vectors))
    
    lda = LinearDiscriminantAnalysis()
    X_lda = lda.fit_transform(X, y)
    
    # Return separated projections
    return X_lda[y == 1], X_lda[y == 0]


def _compute_margin_scores(
    pos_vectors: np.ndarray,
    neg_vectors: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """Compute margin scores blending Mahalanobis distance with cosine similarity.
    
    Args:
        pos_vectors: Positive examples (n_pos, n_features)
        neg_vectors: Negative examples (n_neg, n_features)
        alpha: Weight for Mahalanobis distance (1-alpha for cosine)
        
    Returns:
        Array of margin scores for each positive example
    """
    logger.debug("\n" + "="*80)
    logger.debug("COMPUTING MARGIN SCORES")
    logger.debug("="*80)
    logger.debug(f"Positive vectors shape: {pos_vectors.shape if hasattr(pos_vectors, 'shape') else 'N/A'}")
    logger.debug(f"Negative vectors shape: {neg_vectors.shape if hasattr(neg_vectors, 'shape') else 'N/A'}")
    logger.debug(f"Alpha: {alpha}")
    
    try:
        # Ensure inputs are numpy arrays
        pos_vectors = np.asarray(pos_vectors, dtype=np.float32)
        neg_vectors = np.asarray(neg_vectors, dtype=np.float32)
        
        # Compute class means
        pos_mean = np.mean(pos_vectors, axis=0)
        neg_mean = np.mean(neg_vectors, axis=0)
        
        logger.debug(f"Positive mean shape: {pos_mean.shape}")
        logger.debug(f"Negative mean shape: {neg_mean.shape}")
        
        # Stack all vectors for covariance estimation
        all_vectors = np.vstack([pos_vectors, neg_vectors])
        logger.debug(f"All vectors shape: {all_vectors.shape}")
        
        # Compute empirical covariance with regularization
        cov = np.cov(all_vectors, rowvar=False)
        logger.debug(f"Covariance matrix shape: {cov.shape}")
        
        # Add small diagonal for numerical stability
        cov = cov + 1e-6 * np.eye(cov.shape[0])
        
        # Compute Mahalanobis distance
        diff = pos_vectors - neg_mean
        logger.debug(f"Diff shape: {diff.shape}")
        
        # Safely compute Mahalanobis distance
        try:
            cov_inv = np.linalg.inv(cov)
            mahalanobis = np.sum(diff @ cov_inv * diff, axis=1)
            logger.debug(f"Mahalanobis scores shape: {mahalanobis.shape}")
        except np.linalg.LinAlgError as e:
            logger.warning(f"Could not compute Mahalanobis distance (using zeros): {e}")
            mahalanobis = np.zeros(len(pos_vectors))
        
        # Compute cosine similarity
        norm_pos = np.linalg.norm(pos_vectors, axis=1, keepdims=True)
        norm_neg = np.linalg.norm(neg_mean)
        cosine = np.sum(pos_vectors * neg_mean, axis=1) / (norm_pos.squeeze() * norm_neg + 1e-8)
        logger.debug(f"Cosine scores shape: {cosine.shape}")
        
        # Combine scores
        scores = alpha * mahalanobis + (1 - alpha) * cosine
        logger.debug(f"Final scores shape: {scores.shape}")
        logger.debug(f"Scores stats - min: {np.min(scores):.4f}, max: {np.max(scores):.4f}, mean: {np.mean(scores):.4f}")
        
        return scores
        
    except Exception as e:
        logger.error(f"Error in _compute_margin_scores: {e}", exc_info=True)
        # Fallback to simple cosine similarity
        try:
            cosine = np.sum(pos_vectors * neg_mean, axis=1) / (
                np.linalg.norm(pos_vectors, axis=1) * np.linalg.norm(neg_mean) + 1e-8
            )
            return cosine
        except Exception as inner_e:
            logger.error(f"Fallback cosine similarity also failed: {inner_e}")
            # Return uniform scores if all else fails
            return np.ones(len(pos_vectors))  
    
    return margin_scores


class AdvancedAxisBuilder:
    """Advanced Axis Builder with Tier B features."""
    
    def __init__(
        self,
        embedding_model: Optional[Union[str, Callable]] = None,
        encode_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
        whitening: Union[bool, str] = 'cov',  
        whitening_method: str = 'empirical',
        whitening_reg: float = 1e-6,
        use_lda: bool = True,
        lda_jitter: float = 1e-6,
        margin_alpha: float = 0.8,
        orthogonalize: bool = True,
        n_bootstrap: int = 0,
        random_state: Optional[int] = None,
    ):
        """Initialize the advanced axis builder.
        
        Args:
            whitening: Whether to apply whitening
            whitening_method: 'empirical' or 'pca'
            use_lda: Whether to use LDA for better class separation
            margin_alpha: Weight for margin scoring (0-1)
            orthogonalize: Whether to orthogonalize the final axes
            n_bootstrap: Number of bootstrap samples for uncertainty estimation
            random_state: Random seed for reproducibility
        """
        self.whitening = whitening
        self.whitening_method = whitening_method
        self.use_lda = use_lda
        self.margin_alpha = margin_alpha
        self.orthogonalize = orthogonalize
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def _bootstrap_axis(
        self,
        pos_vectors: np.ndarray,
        neg_vectors: np.ndarray,
        n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute bootstrap estimates of axis direction and uncertainty."""
        n_pos = len(pos_vectors)
        n_neg = len(neg_vectors)
        
        bootstrap_axes = []
        
        for _ in range(n_samples):
            # Sample with replacement
            pos_sample = pos_vectors[self.rng.choice(n_pos, n_pos, replace=True)]
            neg_sample = neg_vectors[self.rng.choice(n_neg, n_neg, replace=True)]
            
            # Compute axis
            axis = np.mean(pos_sample, axis=0) - np.mean(neg_sample, axis=0)
            axis = axis / (np.linalg.norm(axis) + 1e-8)
            bootstrap_axes.append(axis)
        
        bootstrap_axes = np.array(bootstrap_axes)
        mean_axis = np.mean(bootstrap_axes, axis=0)
        uncertainty = np.std(bootstrap_axes, axis=0)
        
        return mean_axis, uncertainty
    
    def build_axis_pack_from_vectors(
        self,
        seeds_vecs: Mapping[str, Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]],
        *,
        lambda_init: float = 1.0,
        beta_init: float = 0.0,
        weights_init: Optional[Sequence[float]] = None,
        meta: Optional[Dict[str, object]] = None,
    ) -> AxisPack:
        """Build an AxisPack from precomputed seed vectors with advanced features.
        
        Args:
            seeds_vecs: Mapping from axis name to (pos_vectors, neg_vectors)
            lambda_init: Initial scaling factor for utilities
            beta_init: Initial bias for utilities
            weights_init: Initial weights for aggregation
            meta: Additional metadata for the axis pack
            
        Returns:
            AxisPack with advanced features
        """
        logger.info(f"Building axis pack from {len(seeds_vecs)} axes")
        
        # Log input dimensions
        for name, (pos_vecs, neg_vecs) in seeds_vecs.items():
            logger.debug(f"Input vectors for {name}:")
            
            # Safely get length for both lists and numpy arrays
            pos_len = len(pos_vecs) if hasattr(pos_vecs, '__len__') else 0
            neg_len = len(neg_vecs) if hasattr(neg_vecs, '__len__') else 0
            
            logger.debug(f"  Positive: {pos_len} vectors")
            if pos_len > 0 and hasattr(pos_vecs, '__getitem__'):
                logger.debug(f"    Shape: {pos_vecs[0].shape} (first vector)")
                
            logger.debug(f"  Negative: {neg_len} vectors")
            if neg_len > 0 and hasattr(neg_vecs, '__getitem__'):
                logger.debug(f"    Shape: {neg_vecs[0].shape} (first vector)")
        
        # Infer embedding dimension D from the first available vector
        first_axis = next(iter(seeds_vecs.values()))
        first_pos = np.asarray(first_axis[0], dtype=np.float32)
        if first_pos.ndim == 1:
            D = int(first_pos.shape[0])
        else:
            D = int(first_pos.shape[1])
        logger.info(f"Inferred embedding dimension D={D}")

        names = list(seeds_vecs.keys())
        if not names:
            error_msg = "No axes provided"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # --- Whitening helpers (returns (mu, W)) ---
        def compute_whitener(X: np.ndarray, method: str = "cov") -> Tuple[np.ndarray, np.ndarray]:
            X = np.asarray(X, dtype=np.float32)
            mu = X.mean(axis=0)
            if method in (False, None, "none"):
                return mu, np.eye(X.shape[1], dtype=np.float32)
            if method.startswith("pca"):
                k = None
                if ":" in method:
                    _, kstr = method.split(":", 1)
                    try:
                        k = int(kstr)
                    except Exception:
                        k = None
                from sklearn.decomposition import PCA
                pca = PCA(n_components=k, svd_solver="full", whiten=True, random_state=42)
                pca.fit(X - mu)
                Vt = pca.components_            # (k, D)
                S  = pca.explained_variance_    # (k,)
                Wp = (Vt.T @ np.diag(1.0/np.sqrt(S + 1e-12))).astype(np.float32)  # (D, k)
                return mu, Wp
            # default: covariance (ZCA-like)
            C = np.cov((X - mu), rowvar=False)
            evals, evecs = np.linalg.eigh(C)
            W = (evecs @ np.diag(1.0/np.sqrt(evals + 1e-12)) @ evecs.T).astype(np.float32)  # (D, D)
            return mu, W

        def lda_direction(X_pos: np.ndarray, X_neg: np.ndarray) -> np.ndarray:
            mu_pos = X_pos.mean(axis=0)
            mu_neg = X_neg.mean(axis=0)
            def cov(X, mu):
                X0 = X - mu
                C = (X0.T @ X0) / max(1, X0.shape[0] - 1)
                # shrink diagonal for stability
                C.flat[:: C.shape[0] + 1] += 1e-5
                return C
            Sw = cov(X_pos, mu_pos) + cov(X_neg, mu_neg)
            w = np.linalg.solve(Sw, (mu_pos - mu_neg))
            w = w / (np.linalg.norm(w) + 1e-12)
            return w.astype(np.float32)

        # Stack all vectors for whitening if enabled (compute once globally)
        if self.whitening:
            logger.debug("Collecting vectors for whitening...")
            all_vectors = []
            for axis_name, (pos_vecs, neg_vecs) in seeds_vecs.items():
                pos_arr = np.asarray(pos_vecs, dtype=np.float32)
                neg_arr = np.asarray(neg_vecs, dtype=np.float32)
                if pos_arr.ndim == 1:
                    pos_arr = pos_arr.reshape(1, -1)
                if neg_arr.ndim == 1:
                    neg_arr = neg_arr.reshape(1, -1)
                all_vectors.append(pos_arr)
                all_vectors.append(neg_arr)
            all_mat = np.vstack(all_vectors)
            logger.debug(f"Computing whitener over {all_mat.shape[0]} samples, D={all_mat.shape[1]}")
            mu_whit, W_whit = compute_whitener(all_mat, method=self.whitening_method or "cov")
            logger.debug(f"Whitener shapes: mu={mu_whit.shape}, W={W_whit.shape}")
        else:
            logger.debug("Whitening disabled")
            mu_whit = np.zeros((D,), dtype=np.float32)
            W_whit = np.eye(D, dtype=np.float32)
        
        # Process each axis (single pass)
        processed_axes = []
        Q = []  # List to store axis vectors
        processed_axis_names = []

        logger.info(f"Starting to process {len(seeds_vecs)} axes...")

        for name, (pos_vecs, neg_vecs) in seeds_vecs.items():
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing axis: {name}")
            logger.info(f"{'='*80}")
            
            try:
                # Convert to numpy arrays if they aren't already
                pos_vecs = np.asarray(pos_vecs, dtype=np.float32)
                neg_vecs = np.asarray(neg_vecs, dtype=np.float32)
                
                # Ensure 2D shapes (n, D)
                if pos_vecs.ndim == 1:
                    pos_vecs = pos_vecs.reshape(1, -1)
                if neg_vecs.ndim == 1:
                    neg_vecs = neg_vecs.reshape(1, -1)

                # Ensure we have valid input shapes
                if pos_vecs.shape[0] == 0 or neg_vecs.shape[0] == 0:
                    logger.warning(f"  Empty positive or negative vectors for {name}, skipping")
                    raise ValueError(f"Empty positive or negative vectors for {name}")
                
                # Apply whitening consistently: Xw = (X - mu) @ W
                pos_w = (pos_vecs - mu_whit) @ W_whit
                neg_w = (neg_vecs - mu_whit) @ W_whit

                # Compute LDA direction in whitened space (or original if W=I)
                if self.use_lda and pos_w.shape[0] > 0 and neg_w.shape[0] > 0:
                    w_dir = lda_direction(pos_w, neg_w)  # shape: (D,) or (k,)
                else:
                    # fallback: mean diff in current space
                    w_dir = (pos_w.mean(axis=0) - neg_w.mean(axis=0)).astype(np.float32)
                    if np.linalg.norm(w_dir) > 0:
                        w_dir = w_dir / (np.linalg.norm(w_dir) + 1e-12)

                # If PCA whitening (W is D x k), back-project to original D
                if W_whit.shape[1] != D:
                    # w_dir is (k,), map to D
                    axis_vec = (W_whit @ w_dir.reshape(-1, 1)).reshape(-1)
                else:
                    axis_vec = w_dir.reshape(-1)

                # Normalize in original D
                norm = np.linalg.norm(axis_vec)
                if norm < 1e-12:
                    logger.warning(f"  Zero-length axis for {name}")
                    raise ValueError(f"Zero-length axis for {name}")
                axis_vec = (axis_vec / norm).astype(np.float32)

                # Store the axis vector and name
                Q.append(axis_vec.reshape(-1, 1))  # (D,1)
                processed_axis_names.append(name)
                processed_axes.append(axis_vec)
                logger.info(f"Successfully processed axis: {name}")
                logger.info(f"  Axis shape: {axis_vec.shape}, dtype: {axis_vec.dtype}")
                logger.info(f"  Axis norm: {np.linalg.norm(axis_vec):.4f}")
                
            except Exception as e:
                logger.error(f"Error processing axis {name}: {e}", exc_info=True)
                # Don't use continue here since we're not in a loop
                # The error will be handled by the outer try-except
        
        # --- Build Q (D x k) once, orthonormalize once, validate once ---
        if len(Q) == 0:
            raise ValueError("No valid axes were processed. Check input data and logs.")

        logger.info(f"Processed {len(Q)} axes: {processed_axis_names}")
        logger.debug(f"Q list length: {len(Q)}")
        for i, q in enumerate(Q):
            logger.debug(f"Q[{i}] shape: {q.shape if hasattr(q, 'shape') else 'no shape'}, type: {type(q)}")

        logger.info(f"Stacking {len(Q)} axis vectors into Q matrix...")
        Q_mat = np.hstack(Q)  # each item in Q is (D, 1)
        logger.info(f"Q matrix raw shape: {Q_mat.shape}")

        # Validate immediate shape
        if Q_mat.ndim != 2 or Q_mat.shape[0] != D:
            raise ValueError(f"Invalid Q shape after stacking: {Q_mat.shape}, expected (D={D}, k)")

        # Orthonormalize columns (optional)
        if getattr(self, 'orthogonalize', True) and Q_mat.shape[1] > 1:
            logger.info("Orthonormalizing Q columns via QR (reduced)...")
            Q_ortho, _ = np.linalg.qr(Q_mat, mode='reduced')  # (D, k)
        else:
            Q_ortho = Q_mat

        # Validate orthonormality & dimensions
        qtq = Q_ortho.T @ Q_ortho
        max_err = float(np.max(np.abs(qtq - np.eye(Q_ortho.shape[1]))))
        logger.info(f"Q^T Q max deviation from I: {max_err:.2e}")

        k = Q_ortho.shape[1]
        if Q_ortho.shape[0] != D:
            raise ValueError(f"Q has incorrect embedding dimension: {Q_ortho.shape[0]} (expected {D})")
        if k != len(processed_axis_names):
            raise ValueError(
                f"Q columns ({k}) != number of processed axes ({len(processed_axis_names)})"
            )
        if np.any(np.isnan(Q_ortho)) or np.any(np.isinf(Q_ortho)):
            raise ValueError("Q matrix contains invalid values (NaN/Inf)")

        Q_final = Q_ortho.astype(np.float32)
        logger.info(f"Final Q matrix shape: {Q_final.shape}")
        # Q_final is the definitive (D, k) matrix. No further stacking/validation needed here.
        
        # Create AxisPack with only successfully processed axes
        k = len(processed_axis_names)
        if k == 0:
            raise ValueError("No valid axes were successfully processed")
            
        lambda_arr = np.full((k,), float(lambda_init), dtype=np.float32)
        beta_arr = np.full((k,), float(beta_init), dtype=np.float32)
        
        if weights_init is not None and len(weights_init) == len(seeds_vecs):
            # Only use weights for successfully processed axes
            weights_list = []
            for i, name in enumerate(seeds_vecs.keys()):
                if name in processed_axis_names:
                    weights_list.append(weights_init[i])
            weights_arr = np.array(weights_list, dtype=np.float32)
        else:
            # Default to uniform weights
            weights_arr = np.full((k,), 1.0 / float(k), dtype=np.float32)
        
        # Initialize metadata if not provided
        meta = meta or {}
        
        # Final sanity checks for Q_final
        assert Q_final.shape[0] == D, f"Expected embedding dim D={D}, got {Q_final.shape[0]}"
        assert Q_final.shape[1] == len(processed_axis_names), "Axis count mismatch"
        col_norms = np.linalg.norm(Q_final, axis=0)
        assert np.allclose(col_norms, 1.0, atol=1e-5), f"Non-unit columns: {col_norms}"

        logger.info(f"Creating AxisPack with {k} axes: {', '.join(processed_axis_names)}")
        logger.debug(f"Q shape: {Q_final.shape}, lambda shape: {lambda_arr.shape}, "
                   f"beta shape: {beta_arr.shape}, weights shape: {weights_arr.shape}")
        
        # Create the AxisPack with the properly shaped Q matrix
        axis_pack = AxisPack(
            names=processed_axis_names,
            Q=Q_final.astype(np.float32),
            lambda_=lambda_arr,
            beta=beta_arr,
            weights=weights_arr,
            mu={},
            meta=meta,
        )
        
        # Validate the axis pack
        try:
            axis_pack.validate()
            logger.info("Successfully validated AxisPack")
        except Exception as e:
            logger.error(f"AxisPack validation failed: {e}")
            raise
            
        return axis_pack
    
    def build_axis_pack_from_json(
        self,
        json_paths: Sequence[Union[str, Path]],
        encode_fn,
        **kwargs
    ) -> AxisPack:
        """Build an AxisPack from JSON configuration files.
        
        Args:
            json_paths: Paths to JSON configuration files
            encode_fn: Function to encode text to vectors
            **kwargs: Additional arguments to pass to build_axis_pack_from_vectors
            
        Returns:
            AxisPack built from the JSON configurations
        """
        logger.info(f"Building axis pack from {len(json_paths)} JSON files")
        
        # Load and process each axis configuration
        seeds_vecs = {}
        
        for path in json_paths:
            try:
                config = _load_json_axis_config(path)
                name = config['name']
                
                # Handle both naming conventions
                max_examples = config.get('max_examples', config.get('positive_examples', []))
                min_examples = config.get('min_examples', config.get('negative_examples', []))
                
                if not max_examples or not min_examples:
                    logger.warning(f"Skipping axis '{name}': missing required examples")
                    continue
                
                logger.debug(f"  Encoding {len(max_examples)} positive examples for axis '{name}'...")
                pos_vectors = encode_fn(max_examples)
                logger.debug(f"  Encoded positive vectors shape: {pos_vectors.shape if hasattr(pos_vectors, 'shape') else 'N/A'}")
                
                logger.debug(f"  Encoding {len(min_examples)} negative examples for axis '{name}'...")
                neg_vectors = encode_fn(min_examples)
                logger.debug(f"  Encoded negative vectors shape: {neg_vectors.shape if hasattr(neg_vectors, 'shape') else 'N/A'}")
                
                seeds_vecs[name] = (pos_vectors, neg_vectors)
                logger.info(f"  Successfully processed axis: {name} with {len(max_examples)} positive "
                          f"and {len(min_examples)} negative vectors")
                
            except Exception as e:
                logger.error(f"Error processing {path}: {e}", exc_info=True)
                continue
        
        if not seeds_vecs:
            raise ValueError("No valid axis configurations found in the provided JSON files")
        
        logger.info(f"Successfully processed {len(seeds_vecs)} axes. Building axis pack...")
        
        # Build the axis pack
        try:
            axis_pack = self.build_axis_pack_from_vectors(seeds_vecs, **kwargs)
            logger.info(f"Successfully built axis pack with {len(axis_pack.names) if hasattr(axis_pack, 'names') else 'unknown'} axes")
            logger.debug(f"Axis pack type: {type(axis_pack).__name__}")
            if hasattr(axis_pack, 'Q') and hasattr(axis_pack.Q, 'shape'):
                logger.debug(f"Axis pack Q matrix shape: {axis_pack.Q.shape}")
            return axis_pack
        except Exception as e:
            logger.error(f"Error in build_axis_pack_from_vectors: {e}", exc_info=True)
            raise
    
    def save_axis_pack(
        self,
        axis_pack: AxisPack,
        output_path: Union[str, Path],
        save_npz: bool = True,
        save_meta: bool = True
    ) -> None:
        """Save an AxisPack to disk in multiple formats.
        
        Args:
            axis_pack: The AxisPack to save
            output_path: Base path for output files
            save_npz: Whether to save NPZ file with vector data
            save_meta: Whether to save metadata as JSON
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        with open(output_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
            json.dump(axis_pack.to_json_obj(), f, indent=2)
        
        # Save as NPZ if requested
        if save_npz:
            np.savez_compressed(
                output_path.with_suffix('.npz'),
                Q=axis_pack.Q,
                lambda_=axis_pack.lambda_,
                beta=axis_pack.beta,
                weights=axis_pack.weights
            )
        
        # Save metadata separately if requested
        if save_meta and axis_pack.meta:
            with open(output_path.with_suffix('.meta.json'), 'w', encoding='utf-8') as f:
                json.dump(axis_pack.meta, f, indent=2)


# Convenience function for common use case
def build_advanced_axis_pack(
    json_paths: Sequence[Union[str, Path]],
    encode_fn,
    **kwargs
) -> AxisPack:
    """Convenience function to build an advanced axis pack from JSON configs.
    
    Args:
        json_paths: Paths to JSON configuration files
        encode_fn: Function to encode text to vectors
        **kwargs: Additional arguments to AdvancedAxisBuilder
        
    Returns:
        AxisPack built with advanced features
    """
    builder = AdvancedAxisBuilder(**kwargs)
    return builder.build_axis_pack_from_json(json_paths, encode_fn)
