"""
Hyperparameter Search for Maiti 2021 SVM Replication

This module implements grid search over:
- Gaussian smoothing filter width (sigma)
- SVM kernel type
- SVM C parameter
- SVM gamma parameter (for RBF kernel)

Two feature extraction paths:
A) Whole SR: Use all spectral points in the window
B) PCA: Extract first 4 principal components

=============================
HYPERPARAMETER SEARCH MODULE 
=============================

This module implements the grid search strategy from Maiti 2021 paper.

KEY FEATURES:
-------------
1. Two feature extraction paths:
   - Path A: Whole SR (all ~30 spectral points)
   - Path B: PCA (first 4 principal components)

2. Optimizes multiple hyperparameters jointly:
   - Gaussian smoothing width (σ)
   - SVM kernel (RBF, linear, poly)
   - SVM C parameter (regularization)
   - SVM gamma (for RBF/poly kernels)
   - Polynomial degree (for poly kernel)

3. Fixed parameters (based on paper context):
   - class_weight='balanced' (handles class imbalance) -REMOVED
   - probability=True (enables threshold optimization) -REMOVED
   - random_state=42 (reproducibility)

USAGE EXAMPLE:
--------------

from hyperparameter_search import SVMHyperparameterSearch

# Load preprocessed SR data
# X_train: shape (n_samples, n_spectral_points)
# y_train: shape (n_samples,) - labels (0=healthy, 1=cancer)

# Example: SR_1005 for 32 samples, 125 spectral points
X_train = preprocessed_srs['SR_1005']  # (32, 30)
y_train = labels  # (32,)

# ===== Path A: Whole SR =====
searcher_whole = SVMHyperparameterSearch(feature_type='whole_sr')
best_params_whole = searcher_whole.search(X_train, y_train, cv_folds=5)

print(f"Best accuracy (whole SR): {searcher_whole.best_score_:.4f}")
print(f"Best config: {best_params_whole}")

# ===== Path B: PCA =====
searcher_pca = SVMHyperparameterSearch(feature_type='pca', n_pca_components=4)
best_params_pca = searcher_pca.search(X_train, y_train, cv_folds=5)

print(f"Best accuracy (PCA): {searcher_pca.best_score_:.4f}")
print(f"Best config: {best_params_pca}")

# Get top 5 configurations
top_5 = searcher_whole.get_top_k_configs(k=5)
for i, config in enumerate(top_5, 1):
    print(f"{i}. Acc={config['mean_accuracy']:.4f}, σ={config['sigma']}, "
          f"kernel={config['kernel']}, C={config['C']}")

PARAMETER GRIDS:
----------------
Default grids (can be modified):

gaussian_sigma: [0, 5, 10, 15, 20, 25, 30]
  - 0 = no smoothing
  - Higher values = more smoothing
  
kernel: ['rbf', 'linear', 'poly']
  - RBF: Radial Basis Function (most common)
  - Linear: Simple linear separator
  - Poly: Polynomial kernel
  
C: [0.01, 0.1, 1.0, 10.0, 100.0]
  - Lower C = more regularization
  - Higher C = less regularization
  
gamma: ['scale', 'auto']
  - Only for RBF/poly kernels
  - 'scale': 1 / (n_features * X.var())
  - 'auto': 1 / n_features
  
degree: [2, 3, 4]
  - Only for poly kernel
  - Polynomial degree

MODIFYING THE GRID:
-------------------

# Create searcher
searcher = SVMHyperparameterSearch(feature_type='whole_sr')

# Reduce grid for faster search
searcher.param_grid['gaussian_sigma'] = [0, 10, 20]
searcher.param_grid['kernel'] = ['rbf', 'linear']
searcher.param_grid['C'] = [1.0, 10.0]

# Run search
best_params = searcher.search(X_train, y_train, cv_folds=5, verbose=1)

VERBOSITY LEVELS:
-----------------
verbose=0: Silent (no output)
verbose=1: Progress updates + best configs
verbose=2: Detailed (every configuration tested)

ATTRIBUTES AFTER SEARCH:
------------------------
searcher.best_params_     : Best hyperparameter configuration (dict)
searcher.best_score_      : Best CV accuracy (float)
searcher.search_results_  : All tested configurations (list of dicts)

NEXT STEPS:
-----------
After finding best hyperparameters:

1. Train final model with best params
2. Apply to validation strategies (k-fold, LOOCV, blind)
3. Calculate performance metrics
4. Compare whole SR vs PCA results
5. Replicate Table 1 from paper

IMPLEMENTATION NOTES:
---------------------

1. Gaussian smoothing applied BEFORE feature extraction
   - For whole SR: smooth, then use all points
   - For PCA: smooth, then extract PCs

2. Stratified K-Fold CV preserves class distribution
   - Important for imbalanced datasets

3. PCA fitted on training fold only
   - Prevents data leakage during CV

4. Grid search can be slow
   - ~18 configs for minimal grid
   - ~1000+ configs for full grid
   - Use verbose=1 to monitor progress

5. Returns all results for analysis
   - Can inspect suboptimal configs
   - Useful for understanding parameter effects
"""


"""
FIXED Hyperparameter Search - No Data Leakage
==============================================

Fixes:
1. ✅ PCA fitted INSIDE CV folds (sklearn Pipeline)
2. ✅ Gaussian smoothing applied PER fold
3. ✅ Nested CV option for unbiased evaluation
4. ✅ Feature selection for high-dimensional data
5. ✅ Proper stratification for small samples
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import gaussian_filter1d
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances


# ============================================================================
# GAMMA ESTIMATION (kernlab-style)
# ============================================================================

def sigest(X, frac=0.5, quantiles=(0.1, 0.9), random_state=42):
    """
    Estimate gamma for RBF kernel, replicating kernlab::sigest.

    Computes pairwise squared distances on a subsample and returns
    three data-driven gamma values:

      'median'    : 1 / (2 * median(sq_dists))
                    — plain median heuristic

      'kpar_auto' : exp(mean(log([low, high]))) = geometric mean of sigest range
                    — replicates kernlab kpar="automatic"

      'low'/'high': bounds of the sigest range for reference

    Parameters
    ----------
    X : array (n_samples, n_features)
    frac : float
        Fraction of samples used for pairwise distances (kernlab default 0.5).
        With small n (e.g. ~32), set frac=1.0 to use all pairs.
    quantiles : tuple
        (lower_q, upper_q) quantiles, kernlab uses (0.1, 0.9).
    random_state : int

    Returns
    -------
    dict with keys: 'low', 'median', 'kpar_auto', 'high'
    """
    n = X.shape[0]
    n_sub = max(2, int(n * frac))

    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=n_sub, replace=False)
    X_sub = X[idx]

    sq_dists = pairwise_distances(X_sub, metric='sqeuclidean')
    upper = sq_dists[np.triu_indices_from(sq_dists, k=1)]
    upper = upper[upper > 0]  # exclude self-distances

    q_low_dist, q_high_dist = np.quantile(upper, quantiles)

    low  = 1.0 / (2.0 * q_high_dist)   # conservative
    high = 1.0 / (2.0 * q_low_dist)    # aggressive
    median_gamma  = 1.0 / (2.0 * np.median(upper))
    kpar_auto = np.exp(np.mean(np.log([low, high])))  # geometric mean, matches kpar="automatic"

    return {
        'low':       low,
        'median':    median_gamma,
        'kpar_auto': kpar_auto,
        'high':      high,
    }


# ============================================================================
# CUSTOM TRANSFORMERS (to put preprocessing in sklearn Pipeline)
# ============================================================================

class GaussianSmoother(BaseEstimator, TransformerMixin):
    """
    Applies Gaussian smoothing to each sample independently.
    Safe to use in CV - no fitting required.
    """
    def __init__(self, sigma=0):
        self.sigma = sigma
    
    def fit(self, X, y=None):
        return self  # No fitting needed
    
    def transform(self, X):
        if self.sigma == 0:
            return X.copy()
        
        X_smoothed = np.zeros_like(X)
        for i in range(X.shape[0]): #for each individual in df (row)
            X_smoothed[i, :] = gaussian_filter1d(X[i, :], sigma=self.sigma)
        return X_smoothed


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Select top K features by variance (for high-dimensional data).
    """
    def __init__(self, n_features='all'):
        self.n_features = n_features
    
    def fit(self, X, y=None):
        if self.n_features == 'all':
            self.selected_indices_ = np.arange(X.shape[1])
        else:
            # Select features with highest variance
            variances = np.var(X, axis=0)
            self.selected_indices_ = np.argsort(variances)[-self.n_features:]
        return self
    
    def transform(self, X):
        return X[:, self.selected_indices_]


# ============================================================================
# MAIN SEARCH CLASS (FIXED VERSION)
# ============================================================================

class SVMHyperparameterSearchFixed:
    """
    Fixed hyperparameter search with no data leakage.
    
    Key improvements:
    - Uses sklearn Pipeline (ensures all preprocessing happens inside CV)
    - Optional nested CV for unbiased evaluation
    - Feature selection for high-dimensional data
    - Proper handling of small sample sizes
    """
    
    def __init__(self, 
                 feature_type: str = 'whole_sr',
                 n_pca_components: int = 4,
                 n_feature_select: str = 'all',
                 use_nested_cv: bool = False):
        """
        Parameters:
        -----------
        feature_type : str
            'whole_sr' or 'pca'
        n_pca_components : int
            Number of PCs (only for feature_type='pca')
        n_feature_select : str or int
            'all' or number of features to select by variance
            Recommended: min(100, n_features) for high-dim data
        use_nested_cv : bool
            If True, uses nested CV (outer for eval, inner for tuning)
            Slower but gives unbiased accuracy estimate
        """
        assert feature_type in ['whole_sr', 'pca']
        
        self.feature_type = feature_type
        self.n_pca_components = n_pca_components
        self.n_feature_select = n_feature_select
        self.use_nested_cv = use_nested_cv
        
        # Parameter grid
        self.param_grid = {
            'gaussian_sigma': [0, 5, 10, 15, 20],
            'kernel': ['rbf', 'linear'],
            'C': [0.01, 0.2575, 0.505, 0.7525, 1.0],
            'gamma': ['scale', 'auto'],
        }
        
        self.best_params_ = None
        self.best_score_ = 0.0
        self.search_results_ = []
    
    
    def _build_pipeline(self, sigma, kernel, C, gamma, degree):
        """
        Build sklearn Pipeline with all preprocessing steps.
        
        Pipeline ensures:
        - All transformers fit on training fold only
        - No data leakage across CV folds
        """
        steps = []
        
        # Step 1: Gaussian smoothing
        steps.append(('smoother', GaussianSmoother(sigma=sigma)))
        
        # Step 2: Feature selection (optional, for high-dim data)
        if self.n_feature_select != 'all':
            steps.append(('feature_select', FeatureSelector(n_features=self.n_feature_select)))
        
        # Step 3: PCA (optional)
        if self.feature_type == 'pca':
            steps.append(('pca', PCA(n_components=self.n_pca_components)))
        
        # Step 4: SVM
        svm_params = {
            'kernel': kernel,
            'C': C,
            'class_weight': 'balanced',
            # 'probability': True,
            'random_state': 42,
            'max_iter': -1
        }
        
        if kernel in ['rbf', 'poly']:
            svm_params['gamma'] = gamma
        if kernel == 'poly':
            svm_params['degree'] = degree
        
        steps.append(('svm', SVC(**svm_params)))
        
        return Pipeline(steps)
    
    
    def _evaluate_single_config(self, 
                                X, y, 
                                sigma, kernel, C, gamma, degree,
                                cv_folds=5):
        """
        Evaluate one configuration using CV.
        Now uses Pipeline - ensuring NO DATA LEAKAGE!
        """
        # Build pipeline
        pipeline = self._build_pipeline(sigma, kernel, C, gamma, degree)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        try:
            # Pipeline ensures all preprocessing happens INSIDE each fold
            cv_scores = cross_val_score(pipeline, X, y, 
                                       cv=cv, 
                                       scoring='accuracy',
                                       n_jobs=-1)
            
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()
            
        except Exception as e:
            print(f"Warning: Config failed - {e}")
            mean_score = 0.0
            std_score = 0.0
        
        return {
            'sigma': sigma,
            'kernel': kernel,
            'C': C,
            'gamma': gamma if kernel in ['rbf', 'poly'] else None,
            'degree': degree if kernel == 'poly' else None,
            'mean_accuracy': mean_score,
            'std_accuracy': std_score,
            'feature_type': self.feature_type
        }
    
    
    def search(self, X, y, cv_folds=5, verbose=1):
        """
        Grid search with optional nested CV.
        
        If use_nested_cv=False:
            Simple grid search (faster, but optimistic)
        
        If use_nested_cv=True:
            Nested CV (slower, but unbiased accuracy estimate)
            - Outer loop: Evaluation
            - Inner loop: Hyperparameter selection
        """
        if not self.use_nested_cv:
            return self._simple_grid_search(X, y, cv_folds, verbose)
        else:
            return self._nested_cv_search(X, y, cv_folds, verbose)
    
    
    def _simple_grid_search(self, X, y, cv_folds, verbose):
        """
        Standard grid search (may be optimistic).
        """
        self.search_results_ = []
        self.best_score_ = 0.0
        
        if verbose >= 1:
            print(f"\n{'='*70}")
            print(f"Simple Grid Search - Feature Type: {self.feature_type}")
            print(f"Data shape: {X.shape}")
            print(f"CV folds: {cv_folds}")
            print(f"{'='*70}\n")
        
        for sigma in self.param_grid['gaussian_sigma']:
            for kernel in self.param_grid['kernel']:
                for C in self.param_grid['C']:
                    
                    gamma_values = (self.param_grid['gamma'] 
                                   if kernel in ['rbf', 'poly'] 
                                   else ['scale'])
                    degree_values = (self.param_grid['degree'] 
                                    if kernel == 'poly' 
                                    else [3]) # can be any value if kernel is not poly; dummy value
                    
                    for gamma in gamma_values:
                        for degree in degree_values:
                            
                            result = self._evaluate_single_config(
                                X, y, sigma, kernel, C, gamma, degree, cv_folds
                            )
                            
                            self.search_results_.append(result)
                            
                            if result['mean_accuracy'] > self.best_score_:
                                self.best_score_ = result['mean_accuracy']
                                self.best_params_ = result
                                
                                if verbose >= 1:
                                    print(f"  ✓ New best: {self.best_score_:.4f} "
                                          f"(σ={sigma}, {kernel}, C={C})")
        
        if verbose >= 1:
            print(f"\n{'='*70}")
            print(f"Best CV accuracy: {self.best_score_:.4f}")
            print(f"{'='*70}\n")
        
        return self.best_params_
    
    
    def _nested_cv_search(self, X, y, cv_folds, verbose):
        """
        Nested CV for unbiased evaluation.
        
        Outer CV: Splits data into train/test
        Inner CV: Finds best hyperparameters on train fold
        """
        if verbose >= 1:
            print(f"\n{'='*70}")
            print(f"NESTED CV - Feature Type: {self.feature_type}")
            print(f"Data shape: {X.shape}")
            print(f"Outer CV folds: {cv_folds}")
            print(f"Inner CV folds: {cv_folds}")
            print(f"{'='*70}\n")
        
        outer_cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        outer_scores = []
        fold_best_params = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            if verbose >= 1:
                print(f"Outer fold {fold_idx}/{cv_folds}:")
                print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")
            
            # Inner CV: Find best hyperparameters on THIS training fold
            best_score_inner = 0.0
            best_params_inner = None
            
            for sigma in self.param_grid['gaussian_sigma']:
                for kernel in self.param_grid['kernel']:
                    for C in self.param_grid['C']:
                        
                        gamma_values = (self.param_grid['gamma'] 
                                       if kernel in ['rbf', 'poly'] 
                                       else ['scale'])
                        degree_values = (self.param_grid['degree'] 
                                        if kernel == 'poly' 
                                        else [3])
                        
                        for gamma in gamma_values:
                            for degree in degree_values:
                                
                                # Evaluate on TRAINING fold only (inner CV)
                                result = self._evaluate_single_config(
                                    X_train, y_train, sigma, kernel, C, gamma, degree, cv_folds
                                )
                                
                                if result['mean_accuracy'] > best_score_inner:
                                    best_score_inner = result['mean_accuracy']
                                    best_params_inner = result
            
            # Train final model on entire training fold with best params
            pipeline = self._build_pipeline(
                best_params_inner['sigma'],
                best_params_inner['kernel'],
                best_params_inner['C'],
                best_params_inner['gamma'] or 'scale',
                best_params_inner['degree'] or 3
            )
            
            pipeline.fit(X_train, y_train)
            test_score = pipeline.score(X_test, y_test)
            
            outer_scores.append(test_score)
            fold_best_params.append(best_params_inner)
            
            if verbose >= 1:
                print(f"  Inner best: {best_score_inner:.4f}")
                print(f"  Outer test: {test_score:.4f}\n")
        
        # Summary
        mean_outer_score = np.mean(outer_scores)
        std_outer_score = np.std(outer_scores)
        
        if verbose >= 1:
            print(f"{'='*70}")
            print(f"NESTED CV Results:")
            print(f"  Mean accuracy: {mean_outer_score:.4f} ± {std_outer_score:.4f}")
            print(f"  Fold scores: {[f'{s:.3f}' for s in outer_scores]}")
            print(f"{'='*70}\n")
        
        # Nested CV score = unbiased accuracy estimate (what you report)
        self.best_score_ = mean_outer_score
        self.nested_cv_scores_ = outer_scores

        # Select final params by majority vote across folds
        # (most frequent kernel+C+sigma combo wins)
        from collections import Counter
        param_keys = [(p['kernel'], p['C'], p['sigma'], p['gamma'], p['degree'])
                      for p in fold_best_params]
        most_common = Counter(param_keys).most_common(1)[0][0]
        self.best_params_ = next(p for p in fold_best_params
                                 if (p['kernel'], p['C'], p['sigma'],
                                     p['gamma'], p['degree']) == most_common)

        if verbose >= 1:
            print(f"  Final params (majority vote across {cv_folds} folds):")
            print(f"  kernel={self.best_params_['kernel']}, "
                  f"C={self.best_params_['C']}, "
                  f"sigma={self.best_params_['sigma']}")
            print(f"  NOTE: nested CV accuracy ({mean_outer_score:.4f}) is your "
                  f"reportable estimate. Re-run simple grid search on full data "
                  f"to confirm final params.\n")

        return self.best_params_



