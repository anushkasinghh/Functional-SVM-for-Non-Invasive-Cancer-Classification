"""
Functional Support Vector Classification (FSVC)
================================================

Implementation of Algorithm 1 from Xie & Ogden (2024, Biostatistics, 25(4):1178-1194).

Dependencies:
    - numpy, scipy, scikit-learn
    - rpy2 + R with refund package (required for FACE via fpca.face)

Algorithm overview (paper Algorithm 1 / Section 2.1):
    1. Perform FPCA via FACE (Xiao et al. 2016) on the training spectra to
       estimate FPC scores Â_i = (Â_i1, ..., Â_iK)^T.
    2. Solve the SVM classification problem (paper eq. 2.5/2.6) using Â_i
       as input features, with Gaussian or linear kernel.
    3. Simultaneously select optimal tuning parameters by 5-fold CV:
         - τ   : smoothing parameter for the FACE covariance estimator
         - K   : number of FPC scores retained
         - C   : SVM regularisation parameter
         - γ   : Gaussian kernel bandwidth (if rbf kernel), set by the
                 median heuristic γ = 1/median(||Â_i − Â_l||²) as stated
                 in Sections 3 and 4 of the paper.

fpca.face scaling convention (Xiao et al. 2016):
    FPCA is a continuous theory. Eigenfunctions φ_k(t) satisfy ∫φ_k(t)²dt = 1.
    On a discrete grid of J points, this means Σⱼ φ_k(tⱼ)² = J (not 1).
    fpca.face returns outputs on this continuous-normalization scale:

  ┌────────────┬─────────────────────────────────────────────┬────────────────────────────────────────┐
  │   Output   │                 What it is                  │             Why scaled                 │
  ├────────────┼─────────────────────────────────────────────┼────────────────────────────────────────┤
  │ efunctions │ φ_k(tⱼ) evaluated at grid points            │ ‖φ_k‖₂² = J (continuous norm)          │
  ├────────────┼─────────────────────────────────────────────┼────────────────────────────────────────┤
  │ evalues    │ λ_k × J                                     │ Matches eigenfunction norm convention  │
  ├────────────┼─────────────────────────────────────────────┼────────────────────────────────────────┤
  │ scores     │ ξ_ik × √J  (BLUP estimates E[A_ik | x_i])  │ Var(scores) = evalues                  │
  └────────────┴─────────────────────────────────────────────┴────────────────────────────────────────┘

Kernel convention (sklearn vs paper):
    Paper eq. 2.5: k(Â_i, Â_l) = exp(−γ‖Â_i - Â_l‖²)
    sklearn SVC:   k(x, x')    = exp(−gamma‖x - x'‖²)
    So: sklearn gamma = paper γ.  They are the same parameter.
"""

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.svm import SVC
from sklearn.model_selection import KFold, StratifiedKFold


# ============================================================================
# Part 1: FPCA via FACE (Xiao et al. 2016, refund::fpca.face)
# ============================================================================

@dataclass
class FPCAResult:
    """Container for fpca.face output."""
    mu: np.ndarray           # (J,)   mean function
    efunctions: np.ndarray   # (J, npc)  eigenfunctions evaluated on grid
    evalues: np.ndarray      # (npc,) eigenvalues
    scores: np.ndarray       # (n, npc) FPC scores
    sigma2: float            # estimated noise variance
    npc: int                 # number of components retained


def fpca_face_via_r(
    # type annotations/ hints:
    Y: np.ndarray,          # (n, J) data matrix — each row is one functional observation
    npc: int = 10,          # max number of Functional Principal Components to extract
    lam: float | None = None, # smoothing parameter (lambda in R); None = auto-select via GCV
    knots: int = 35,        # number of knots for the B-spline basis
    p: int = 3,             # degree of B-splines (3 = cubic)
    m: int = 2,             # order of difference penalty
    pve: float = 0.99,      # proportion of variance explained threshold: the smoother is penalized for having high curvature, which is standard for spectral data — you want smooth curves, not ones that chase every noise spike.
    center: bool = True,    # whether to subtract column means before fittingb spline is like a black box to me, even fpca face, 
) -> FPCAResult:
    """
    Call R's refund::fpca.face via rpy2.

    This is a thin wrapper: it converts numpy → R matrix, calls fpca.face
    with the exact same parameter names as the R function, and converts
    the results back to numpy.

    Parameters
    ----------
    Y : ndarray of shape (n, J)
        Data matrix. Each row is one functional observation on a regular grid.
    npc : int
        Maximum number of FPCs to extract.
    lam : float or None
        Smoothing parameter (called 'lambda' in R; renamed here to avoid
        Python keyword conflict). If None, fpca.face selects via GCV (generalized cross val) GCV(λ) = (1/n) ||y - ŷ||² / (1 - trace(H)/n)².
        where H is the "hat matrix" (the matrix that maps observed data to fitted values). Minimise this over λ without ever actually refitting the model.  
    knots : int
        Number of knots for the B-spline basis in the sandwich smoother.
    p : int
        Degree of B-splines (default 3 = cubic).
    m : int
        Order of difference penalty (default 2).
    pve : float
        Proportion of variance explained threshold (only used if npc is None
        in the R function — here we always pass npc explicitly).
    center : bool
        Whether to center the data (subtract column means).

    Returns
    -------
    FPCAResult
        Contains mu, efunctions, evalues, scores, sigma2, npc.

    Raises
    ------
    ImportError
        If rpy2 or R's refund package is not available.

    fpca.face does the following
    ------
    1. take n spectra (n x J matrix Y)
    2. Optionally center: subtract mean spectrum 
    3. Compute covariance between every pair of wavenumbers -> J x J covariance matrix
    4. SMOOTH covariance matrix using B-splines (face "trick" - raw covariance is noisy, smoothening => better eigenfunctions) (knots, p, m , lambda are useful here)
    5. Find eigenfucntions of smoothed covariance 
    6. Project each spectrum onto top npc eigenfunctions 
    7. gives n x npc score matrix 

    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri, default_converter
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.packages import importr
    except ImportError:
        raise ImportError(
            "rpy2 is required for fpca_face_via_r. Install with:\n"
            "  pip install rpy2\n"
            "and ensure R is installed with the 'refund' package:\n"
            "  install.packages('refund')"
        )

    try:
        refund = importr("refund")
    except Exception as e:
        raise ImportError(
            f"Could not load R package 'refund': {e}\n"
            "Install in R with: install.packages('refund')"
        )

    np_cv = default_converter + numpy2ri.converter

    # Use localconverter for the entire call.
    # Inside the context rpy2 auto-converts the return value to a Python
    # NamedList (each element already a numpy array), so we access fields
    # with result["name"] rather than result.rx2("name").
    with localconverter(np_cv):
        r_Y = ro.r["matrix"](Y, nrow=Y.shape[0], ncol=Y.shape[1])

        kwargs = {
            "Y": r_Y,
            "var": True,
            "simul": False,
            "npc": npc,
            "knots": knots,
            "p": p,
            "m": m,
            "center": center,
            "pve": pve,
        }
        if lam is not None:
            kwargs["lambda"] = lam  # passed via **dict so "lambda" is a valid key

        result = refund.fpca_face(**kwargs)  # NamedList — integer-indexed only

        # NamedList does not support string indexing; look up position via .names
        names = list(result.names())
        def _get(key):
            return result[names.index(key)]

        mu         = np.array(_get("mu")).flatten()
        efunctions = np.array(_get("efunctions"))
        evalues    = np.array(_get("evalues")).flatten()
        scores     = np.array(_get("scores"))
        try:
            sigma2 = float(np.array(_get("sigma2")).flatten()[0])
        except ValueError:  # key not in names
            sigma2 = float(np.array(_get("error_var")).flatten()[0])
        npc_out = int(np.array(_get("npc")).flatten()[0])

    return FPCAResult(
        mu=mu,
        efunctions=efunctions,
        evalues=evalues,
        scores=scores,
        sigma2=sigma2, # estimated noise variance, useful to asess data quality
        npc=npc_out,
    )




# ============================================================================
# Part 2: BLUP Score Estimation for New Data
# ============================================================================

def estimate_pc_scores(
    Y_new: np.ndarray,
    mu: np.ndarray,
    sigma2: float,
    evalues: np.ndarray,
    efunctions: np.ndarray,
    return_shrinkage: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Best Linear Unbiased Prediction (BLUP) of FPC scores for new observations.

    This is a line-by-line translation of the R function `estPCscore` from
    FSVC.R (lines 4-36).

    Mathematical formula for each subject i:
        Y_tilde_i = Y_new[i, :] - mu              (centered observation)
        Z = efunctions                              (J x npc matrix)
        D_inv = diag(1 / evalues)                   (npc x npc)
        scores_i = (Z'Z + sigma2 * D_inv)^{-1} Z' Y_tilde_i

    This is the BLUP/shrinkage estimator that accounts for measurement noise.
    When sigma2 is large, scores are shrunk toward zero (regularization).
    When sigma2 → 0, this reduces to the standard projection.

    The R code also handles missing data (NA values) per subject by
    subsetting Z to observed points. We preserve this capability.

    Exact shrinkage weights (when return_shrinkage=True)
    ----------------------------------------------------
    The per-component shrinkage factor is defined as:
        w_k = (Z'Z)_kk / (Z'Z + σ² D⁻¹)_kk
            = (Z'Z)_kk / ((Z'Z)_kk + σ²/λ_k)

    This uses the actual diagonal of Z'Z (not the J·I approximation), so it
    is exact given the estimated eigenfunctions. w_k ∈ (0, 1]: values near 1
    mean BLUP ≈ naive projection (low noise); values near 0 mean heavy
    shrinkage toward zero (component dominated by noise).

    Parameters
    ----------
    Y_new : ndarray (n_new, J)
        New spectra to project.
    mu : ndarray (J,)
        Mean function from FPCA.
    sigma2 : float
        Noise variance from FPCA.
    evalues : ndarray (npc,)
        Eigenvalues from FPCA.
    efunctions : ndarray (J, npc)
        Eigenfunctions evaluated on grid from FPCA.
    return_shrinkage : bool
        If True, also return exact per-component shrinkage weights (npc,).

    Returns
    -------
    scores : ndarray (n_new, npc)
    shrinkage_weights : ndarray (npc,)  — only if return_shrinkage=True
    """
    n_new, J = Y_new.shape
    npc = efunctions.shape[1]

    D_inv = np.diag(1.0 / evalues)   # diag(1/λ_k)
    Z = efunctions                    # (J, npc)

    scores = np.full((n_new, npc), np.nan)
    Y_tilde = Y_new - mu[np.newaxis, :]  # centre each observation

    # Precompute Z'Z once (same for all samples on a regular grid)
    ZtZ = Z.T @ Z        # (npc, npc)
    A   = ZtZ + sigma2 * D_inv  # (npc, npc)

    for i in range(n_new):
        # Handle irregular / missing observations per subject
        obs_points = ~np.isnan(Y_new[i, :])

        if obs_points.all():
            # No missing data — fast path (common case for regular grids)
            Z_cur     = Z
            A_i       = A            # precomputed from full Z
            y_tilde_i = Y_tilde[i, :]
        else:
            Z_cur     = Z[obs_points, :]
            A_i       = Z_cur.T @ Z_cur + sigma2 * D_inv
            y_tilde_i = Y_tilde[i, obs_points]

        # BLUP: score_i = (Z'Z + σ²D⁻¹)⁻¹ Z'ỹ_i  [paper Section 2.1]
        b = Z_cur.T @ y_tilde_i  # (npc,)
        scores[i, :] = np.linalg.solve(A_i, b)

    if return_shrinkage:
        # Exact shrinkage weights from the actual Z'Z diagonal
        ZtZ_diag = np.diag(ZtZ)                      # (npc,)  = ||φ_k||² on grid
        A_diag   = ZtZ_diag + sigma2 / evalues        # (npc,)
        shrinkage_weights = ZtZ_diag / A_diag         # (npc,)  ∈ (0, 1]
        return scores, shrinkage_weights

    return scores


# ============================================================================
# Part 3: Gamma (bandwidth) Estimation for Gaussian kernel
#
# The paper (Sections 3 and 4) states γ is determined by the median heuristic:
#   γ = 1 / median(‖Â_i − Â_l‖²)   over all pairs i ≠ l
#
# The authors' R code uses kernlab's kpar="automatic", which is actually
# mean(1/q90, 1/q10) of pairwise squared distances — a different value.
# Both options are provided below. compute_gamma_automatic (sigest-style)
# is the default to match the R code's actual behaviour; use
# compute_gamma_median_heuristic to match the paper's stated method.
# ============================================================================

def sigest_like_kernlab(X: np.ndarray, frac: float = 0.5, rng: np.random.RandomState | None = None) -> np.ndarray:
    """
    Replicate kernlab::sigest: subsample, compute pairwise ||x-x'||², return
    [1/q90, 1/median, 1/q10].

    kernlab defines k(x,x') = exp(-σ‖x-x'‖²); sklearn uses exp(-γ‖x-x'‖²).
    So σ (kernlab) = γ (sklearn).

    Parameters
    ----------
    X : ndarray (n, d)
    frac : float
        Fraction of rows to subsample (0.5 matches kernlab::sigest default).

    Returns
    -------
    ndarray (3,) : [1/q90, 1/median, 1/q10] of pairwise squared distances.
    """
    n = X.shape[0]
    n_sub = max(2, int(n * frac))

    if n_sub < n:
        if rng is None:
            rng = np.random.RandomState()
        idx = rng.choice(n, size=n_sub, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X

    dists_sq = pdist(X_sub, metric="sqeuclidean")
    dists_sq = dists_sq[dists_sq > 0]

    if len(dists_sq) == 0:
        warnings.warn("All pairwise distances are zero; returning gamma=1.0")
        return np.array([1.0, 1.0, 1.0])

    return np.array([
        1.0 / np.quantile(dists_sq, 0.9),
        1.0 / np.quantile(dists_sq, 0.5),
        1.0 / np.quantile(dists_sq, 0.1),
    ])


def compute_gamma_automatic(X: np.ndarray, rng: np.random.RandomState | None = None) -> float:
    """
    Compute γ via the sigest heuristic: mean(1/q90, 1/q10) of pairwise ‖Â_i−Â_l‖².

    This matches the authors' R implementation (kernlab kpar="automatic") but
    differs from the paper's stated median heuristic. Use as default to
    reproduce the published numerical results.

    Parameters
    ----------
    X : ndarray (n, d)  — FPCA scores, unscaled.

    Returns
    -------
    float : γ for sklearn SVC(kernel='rbf', gamma=...).
    """
    srange = sigest_like_kernlab(X, frac=0.5, rng=rng)
    return float(np.mean([srange[0], srange[2]]))


def compute_gamma_median_heuristic(X: np.ndarray) -> float:
    """
    Compute γ via the median heuristic as stated in Xie & Ogden (2024) Sections 3–4:
        γ = 1 / median(‖Â_i − Â_l‖²)   over all pairs i ≠ l.

    Uses all pairwise distances (no subsampling). Appropriate for the small
    sample sizes (n ≈ 50–150) typical of this application.

    Parameters
    ----------
    X : ndarray (n, d)  — FPCA scores, unscaled.

    Returns
    -------
    float : γ for sklearn SVC(kernel='rbf', gamma=...).
    """
    dists_sq = pdist(X, metric="sqeuclidean")
    dists_sq = dists_sq[dists_sq > 0]
    if len(dists_sq) == 0:
        warnings.warn("All pairwise distances are zero; returning gamma=1.0")
        return 1.0
    return float(1.0 / np.median(dists_sq))


# ============================================================================
# Part 4: FPCA via FACE
# ============================================================================

def run_fpca(
    Y: np.ndarray,
    npc: int = 10,
    lam: float | None = None,
    knots: int = 35,
) -> FPCAResult:
    """
    Estimate FPC scores using FACE (Xiao et al. 2016) via refund::fpca.face.

    This is Step 1 of Algorithm 1 in Xie & Ogden (2024). FACE is required;
    rpy2 and R's refund package must be installed.

    Parameters
    ----------
    Y : ndarray (n, J)
    npc : int
        Maximum FPCs to extract.
    lam : float or None
        Smoothing parameter τ. None triggers GCV selection inside fpca.face.
    knots : int
        B-spline knots for the sandwich smoother. Paper uses 10 (J=40) or
        35 (J≥50); default 35 covers the breath spectra grid.

    Returns
    -------
    FPCAResult
    """
    return fpca_face_via_r(Y, npc=npc, lam=lam, knots=knots)


# ============================================================================
# Part 5: FSVC — Functional Support Vector Classification
# ============================================================================

@dataclass
class FSVCResult:
    """Container for fitted FSVC model."""
    # Optimal tuning parameters
    opt_tau: float
    opt_C: float
    opt_K: int
    opt_gamma: float | None  # None for linear kernel

    # FPCA model (fitted on full training data with opt_tau)
    fpca_result: FPCAResult

    # Fitted SVM
    svm_model: SVC

    # CV results for diagnostics
    cv_accuracy_matrix: np.ndarray  # (len(smoothers), len(Cs))
    cv_best_K_per_sc: np.ndarray    # best K for each (s,c) combo

    # Training performance
    train_accuracy: float | None = None
    train_predictions: np.ndarray | None = None


def fsvc(
    X: np.ndarray,
    y: np.ndarray,
    kernel: Literal["rbf", "linear"] = "rbf",
    Ks: list[int] | None = None,
    smoothers: list[float] | None = None,
    Cs: list[float] | None = None,
    npc: int = 10,
    knots: int = 35,
    n_folds: int = 5,
    fit: bool = True,
    random_state: int | None = None,
    stratified_folds: bool = True,
) -> FSVCResult:
    """
    Functional Support Vector Classification — Algorithm 1 of Xie & Ogden (2024).

    Joint 5-fold CV selects the optimal combination of (τ, K, C) — and γ
    when using the Gaussian kernel — by minimising cross-validated
    misclassification error (paper Algorithm 1 Step 3).

    Parameters
    ----------
    X : ndarray (n, J)
        Spectral data matrix. Rows are samples, columns are wavenumbers.
    y : ndarray (n,)
        Class labels (any two distinct values; passed to sklearn as-is).
    kernel : "rbf" or "linear"
        SVM kernel. "rbf" = Gaussian (paper eq. 2.5), "linear" (paper eq. 2.4).
    Ks : list of int
        Search grid for number of FPCs K. Paper real-data applications use
        K ∈ {1, …, 10}; simulations use {1, …, 5}.
    smoothers : list of float
        Search grid for FACE smoothing parameter τ.
        Paper grids: {0.5, 1, 5, 10} (Sections 3 and 4).
    Cs : list of float
        Search grid for SVM regularisation C.
        Paper grid: {0.01, 0.2575, 0.5050, 0.7525, 1} (Sections 3 and 4).
    npc : int
        Maximum FPCs extracted by FACE. Must be ≥ max(Ks).
    knots : int
        B-spline knots for fpca.face (paper: 10 when J=40, 35 when J≥50).
    n_folds : int
        CV folds (paper: 5).
    fit : bool
        If True, compute resubstitution training accuracy after fitting.
    random_state : int or None
        Seed for CV fold splits and gamma estimation reproducibility.
    stratified_folds : bool
        If True (default), use StratifiedKFold to preserve class balance in
        each fold. Recommended for small n with class imbalance. The paper
        does not specify stratification; set False to use plain random splits.

    Returns
    -------
    FSVCResult
    """
    if Ks is None:
        Ks = list(range(1, 11))  # [1, 2, ..., 10]
    if smoothers is None:
        smoothers = [0.5, 1.0, 5.0, 10.0]
    if Cs is None:
        Cs = [0.01, 0.2575, 0.505, 0.7525, 1.0]

    assert npc >= max(Ks), f"npc ({npc}) must be >= max(Ks) ({max(Ks)})"

    n = len(y)

    # Seeded RNG for reproducible gamma estimation across CV folds
    _gamma_rng = np.random.RandomState(random_state)

    # ---- Create CV folds ----
    # Paper specifies 5-fold CV (Sections 3 and 4) but does not specify
    # stratification. Stratified splits are used by default to prevent
    # degenerate folds when n is small and classes are imbalanced.
    if stratified_folds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        folds = list(skf.split(X, y))
    else:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        folds = list(kf.split(X))

    # ---- Joint CV grid search over (τ, C, K) ----
    # Implements Algorithm 1 Step 3: simultaneously minimise cross-validated
    # misclassification error over all tuning parameters.
    accuracy_grid = np.zeros((len(smoothers), len(Cs)))  # best-over-K accuracy
    best_K_for_sc = np.zeros((len(smoothers), len(Cs)), dtype=int)

    for s_idx, tau in enumerate(smoothers):
        for c_idx, C_val in enumerate(Cs):
            # Accuracy accumulated over folds for each K
            accuracy_per_K = np.zeros(len(Ks))

            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                X_train_fold = X[train_idx]
                X_test_fold = X[test_idx]
                y_train_fold = y[train_idx]
                y_test_fold = y[test_idx]

                # ---- Step 1: FACE on training fold (Algorithm 1 Step 1) ----
                fpca_cv = run_fpca(X_train_fold, npc=npc, lam=tau, knots=knots)

                # ---- Step 2: BLUP scores for test fold (Algorithm 1 Step 1) ----
                # fpca.face returns BLUP scores E[A_ik|x_i] for the training fold
                # directly. Test-fold scores are estimated via the explicit BLUP
                # formula (paper Section 2.1) using the training eigenfunctions.
                scores_train = fpca_cv.scores  # (n_train, npc)
                scores_test = estimate_pc_scores(
                    X_test_fold,
                    mu=fpca_cv.mu,
                    sigma2=fpca_cv.sigma2,
                    evalues=fpca_cv.evalues,
                    efunctions=fpca_cv.efunctions,
                )

                # ---- Step 3: For each K, train SVM and predict ----
                for k_idx, K in enumerate(Ks):
                    S_train = scores_train[:, :K]
                    S_test = scores_test[:, :K]

                    # γ by sigest heuristic (default); swap to
                    # compute_gamma_median_heuristic to match paper Section 3.
                    if kernel == "rbf":
                        gamma_val = compute_gamma_automatic(S_train, rng=_gamma_rng)
                    else:
                        gamma_val = "scale"

                    # probability=False in CV: predict() only, no Platt scaling.
                    svm = SVC(
                        kernel=kernel,
                        C=C_val,
                        gamma=gamma_val if kernel == "rbf" else "scale",
                        probability=False,
                        # class_weight="balanced",
                    )
                    svm.fit(S_train, y_train_fold)
                    pred = svm.predict(S_test)

                    # Accumulate average fold accuracy (Algorithm 1 Step 3)
                    fold_acc = np.mean(pred == y_test_fold)
                    accuracy_per_K[k_idx] += fold_acc / n_folds

            # Best K for this (τ, C) pair; best CV accuracy
            accuracy_grid[s_idx, c_idx] = np.max(accuracy_per_K)
            best_K_for_sc[s_idx, c_idx] = Ks[np.argmax(accuracy_per_K)]

    # ---- Find optimal parameters (Algorithm 1 Step 3) ----
    best_idx = np.unravel_index(np.argmax(accuracy_grid), accuracy_grid.shape)
    opt_s_idx, opt_c_idx = best_idx
    opt_tau = smoothers[opt_s_idx]
    opt_C = Cs[opt_c_idx]
    opt_K = best_K_for_sc[opt_s_idx, opt_c_idx]

    print(f"Joint CV complete. Best: tau={opt_tau}, K={opt_K}, C={opt_C}, "
          f"CV accuracy={accuracy_grid[opt_s_idx, opt_c_idx]:.4f}")

    # ---- Refit on full training data with optimal parameters ----
    # Algorithm 1 Steps 1–2 applied to all n samples.
    fpca_full = run_fpca(X, npc=npc, lam=opt_tau, knots=knots)

    # fpca.face scores are BLUP estimates; use directly as SVM training features.
    S_full = fpca_full.scores[:, :opt_K]

    if kernel == "rbf":
        opt_gamma = compute_gamma_automatic(S_full, rng=_gamma_rng)
    else:
        opt_gamma = None

    # Final SVM fit
    final_svm = SVC(
        kernel=kernel,
        C=opt_C,
        gamma=opt_gamma if kernel == "rbf" else "scale",
        probability=True,
        # class_weight="balanced",
    )
    final_svm.fit(S_full, y)

    # ---- Optionally compute training predictions ----
    train_acc = None
    train_pred = None
    if fit:
        train_pred = fsvc_predict(X, fpca_full, final_svm, opt_K)
        train_acc = np.mean(train_pred == y)
        print(f"Training accuracy (resubstitution): {train_acc:.4f}")

    return FSVCResult(
        opt_tau=opt_tau,
        opt_C=opt_C,
        opt_K=opt_K,
        opt_gamma=opt_gamma,
        fpca_result=fpca_full,
        svm_model=final_svm,
        cv_accuracy_matrix=accuracy_grid,
        cv_best_K_per_sc=best_K_for_sc,
        train_accuracy=train_acc,
        train_predictions=train_pred,
    )


# ============================================================================
# Part 6: Prediction
# ============================================================================

def fsvc_predict(
    X_new: np.ndarray,
    fpca_result: FPCAResult,
    svm_model: SVC,
    opt_K: int,
    return_proba: bool = False,
) -> np.ndarray:
    """
    Predict class labels for new spectral data (Algorithm 1, prediction step).

    1. Estimate FPC scores via BLUP (paper Section 2.1)
    2. Truncate to first opt_K scores
    3. Predict with the trained SVM

    Parameters
    ----------
    X_new : ndarray (n_new, J)
        New spectra (same preprocessing as training data).
    fpca_result : FPCAResult
        FPCA model fitted on training data.
    svm_model : SVC
        Trained SVM model.
    opt_K : int
        Number of FPC scores to use.
    return_proba : bool
        If True, return class probabilities instead of labels.

    Returns
    -------
    ndarray (n_new,) or (n_new, 2)
        Predicted labels or probabilities.
    """
    # BLUP scores for new observations (paper Section 2.1)
    scores_new = estimate_pc_scores(
        X_new,
        mu=fpca_result.mu,
        sigma2=fpca_result.sigma2,
        evalues=fpca_result.evalues,
        efunctions=fpca_result.efunctions,
    )

    S_new = scores_new[:, :opt_K]
    if return_proba:
        return svm_model.predict_proba(S_new)
    else:
        return svm_model.predict(S_new)


# ============================================================================
# Part 7: Convenience function for blind test evaluation
# ============================================================================

def evaluate_blind_test(
    model: FSVCResult,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluate a fitted FSVC model on the blind test set.

    Parameters
    ----------
    model : FSVCResult
        Fitted model from fsvc().
    X_test : ndarray (n_test, J)
        Blind test spectra.
    y_test : ndarray (n_test,)
        True labels.

    Returns
    -------
    dict with keys: predictions, probabilities, accuracy, confusion_matrix,
         sensitivity, specificity, auc, mcc, and exact confidence intervals.
    """
    from sklearn.metrics import (
        confusion_matrix, matthews_corrcoef, roc_auc_score,
        classification_report,
    )
    from scipy.stats import beta as beta_dist

    pred = fsvc_predict(X_test, model.fpca_result, model.svm_model, model.opt_K)
    proba = fsvc_predict(
        X_test, model.fpca_result, model.svm_model, model.opt_K,
        return_proba=True,
    )

    acc = np.mean(pred == y_test)
    cm = confusion_matrix(y_test, pred)
    mcc = matthews_corrcoef(y_test, pred)

    # Sensitivity and specificity (assumes positive class is the second label)
    labels = np.unique(y_test)
    if len(labels) == 2:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Clopper-Pearson exact 95% CI:
        # lo = Beta(alpha/2; k, n-k+1), hi = Beta(1-alpha/2; k+1, n-k)
        def clopper_pearson(k, n, alpha=0.05):
            lo = beta_dist.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
            hi = beta_dist.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
            return (float(lo), float(hi))

        sens_ci = clopper_pearson(tp, tp + fn)
        spec_ci = clopper_pearson(tn, tn + fp)

        # AUC — use probabilities for the positive class
        try:
            # proba[:, 1] = P(positive class)
            auc = roc_auc_score(y_test, proba[:, 1])
        except Exception:
            auc = None
    else:
        sensitivity = specificity = auc = None
        sens_ci = spec_ci = None

    return {
        "predictions": pred,
        "probabilities": proba,
        "accuracy": acc,
        "confusion_matrix": cm,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "sensitivity_95ci": sens_ci,
        "specificity_95ci": spec_ci,
        "auc": auc,
        "mcc": mcc,
        "classification_report": classification_report(y_test, pred),
        "n_support_vectors": model.svm_model.n_support_,
    }

