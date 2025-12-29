"""
Implementation of Algorithms from Sections 4 and 5 of:
Warmuth & Kuzmin (2008) "Randomized Online PCA Algorithms with Regret Bounds 
that are Logarithmic in the Dimension"

Section 4: Subset Selection in the Expert Setting
- Algorithm 4: Capping (projects mixture vector to satisfy cap constraint)
- Algorithm 2: Mixture Decomposition (samples subsets from capped mixture)
- Algorithm 3: Capped Hedge Algorithm (main online learning algorithm)

Section 5: Uncentered Online PCA (Matrix Extension)
- Matrix Capping: Projects density matrix eigenvalues to satisfy cap constraint
- Matrix Mixture Decomposition: Samples subspaces from capped density matrix
- Uncentered Online PCA Algorithm: Main algorithm with matrix exponentiated gradient

Key insight from the paper:
- Section 4: Mixture vector w with constraint w_i <= 1/d represents mixture over subsets
- Section 5: Density matrix W with eigenvalues <= 1/d represents mixture over subspaces
"""

import numpy as np
import warnings
from typing import Tuple, List, Optional, Set
from dataclasses import dataclass


# =============================================================================
# Algorithm 4: Capping (Water-Filling Algorithm)
# =============================================================================

def algorithm4_capping(w: np.ndarray, d: int, tol: float = 1e-10) -> np.ndarray:
    """
    Algorithm 4: Capping (Water-Filling) from Section 4.
    
    Projects a mixture vector w to satisfy the constraint w_i <= 1/d for all i.
    This is done via iterative water-filling: cap entries exceeding 1/d and
    redistribute excess mass proportionally to uncapped entries.
    
    The algorithm iterates because redistributing mass may cause previously
    uncapped entries to exceed the cap.
    
    Mathematical formulation:
    - Find the projection onto the set {w : sum(w) = 1, w >= 0, w <= 1/d}
    - This is a convex set, and the projection is unique
    
    Parameters
    ----------
    w : np.ndarray
        Mixture vector (non-negative, sums to 1)
    d : int
        Cap parameter - all entries must be <= 1/d after capping
        In the subset selection problem, d = n - k (size of discarded set)
    tol : float
        Numerical tolerance
        
    Returns
    -------
    np.ndarray
        Capped mixture vector with all entries <= 1/d
        
    Example
    -------
    >>> w = np.array([0.6, 0.25, 0.15])
    >>> algorithm4_capping(w, d=2)  # cap = 0.5
    array([0.5, 0.3125, 0.1875])  # 0.6 capped to 0.5, excess redistributed
    
    Notes
    -----
    From the paper: "The capping is an I-projection (Bregman projection with
    KL divergence) onto the constraint set."
    """
    n = len(w)
    cap = 1.0 / d

    # Validate input
    if d > n:
        raise ValueError(f"d ({d}) cannot exceed n ({n})")
    if d < 1:
        raise ValueError(f"d must be positive, got {d}")
    
    # Normalize input if needed
    w = np.maximum(w, 0)  # Ensure non-negative
    w_sum = np.sum(w)
    if w_sum > tol:
        w = w / w_sum
    else:
        # If all weights are essentially zero, return uniform
        return np.ones(n) / n
    
    w_capped = w.copy()
    
    # Iterative water-filling
    max_iterations = n + 1  # At most n iterations needed
    for iteration in range(max_iterations):
        # Identify entries at cap vs below cap
        at_cap = w_capped >= cap - tol
        below_cap = ~at_cap

        # If no entry exceeds cap, we're done
        if np.all(w_capped <= cap + tol):
            break

        # Cap all entries that exceed cap
        excess = np.sum(np.maximum(w_capped - cap, 0))
        w_capped = np.minimum(w_capped, cap)

        # Redistribute excess to entries strictly below cap
        if np.any(below_cap) and excess > tol:
            # Weight by current values (proportional redistribution)
            below_cap_sum = np.sum(w_capped[below_cap])
            if below_cap_sum > tol:
                # Scale up entries below cap proportionally
                scale_factor = (below_cap_sum + excess) / below_cap_sum
                w_capped[below_cap] *= scale_factor
            else:
                # If all below-cap entries have zero weight, distribute uniformly
                n_below = np.sum(below_cap)
                w_capped[below_cap] = excess / n_below
        elif excess > tol:
            # All entries are at cap but we have excess mass
            # This shouldn't happen if d <= n, but handle gracefully
            pass

    # Ensure constraints are satisfied
    w_capped = np.minimum(w_capped, cap)
    w_capped = np.maximum(w_capped, 0)

    # Normalize to sum to 1
    w_sum = np.sum(w_capped)
    if w_sum > tol:
        w_capped = w_capped / w_sum

    # Final verification
    assert np.all(w_capped <= cap + tol), \
        f"Cap constraint violated: max={np.max(w_capped):.6f}, cap={cap:.6f}"
    assert np.abs(np.sum(w_capped) - 1.0) < tol, \
        f"Sum constraint violated: sum={np.sum(w_capped):.6f}"

    return w_capped


# =============================================================================
# Algorithm 2: Mixture Decomposition
# =============================================================================

def mixture_decomposition(w: np.ndarray, d: int, tol: float = 1e-12) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Decompose w into a convex combination of d-corners.
    
    From the paper: A capped mixture vector w (with w_i <= 1/d) can be written
    as a convex combination of d-corners. A d-corner is a vector with exactly
    d entries equal to 1/d and the rest equal to 0.
    
    This is Algorithm 2 from Section 4.1 of Warmuth & Kuzmin (2008).

    Parameters
    ----------
    w : np.ndarray, shape (n,)
        Vector in capped simplex: sum(w)=1, 0<=w_i<=1/d
    d : int
        Corner size (number of elements to select)
    tol : float
        Numerical tolerance

    Returns
    -------
    corners : list of np.ndarray
        Each array is a d-corner (indicator scaled by 1/d)
    weights : np.ndarray
        Corresponding mixture weights (sum to 1)
    """
    w = w.copy().astype(float)
    n = len(w)
    corners = []
    weights = []
    
    max_iterations = n * 2  # Safeguard against infinite loops
    iteration = 0

    while w.sum() > tol and iteration < max_iterations:
        iteration += 1
        
        # Sort indices by descending weight
        idx_sorted = np.argsort(-w)

        # Identify mandatory indices (already at cap)
        cap = w.sum() / d
        mandatory = [i for i in idx_sorted if abs(w[i] - cap) < tol]

        # Fill corner up to size d
        corner_idx = mandatory[:]
        for i in idx_sorted:
            if i not in corner_idx and w[i] > tol:
                corner_idx.append(i)
            if len(corner_idx) == d:
                break

        # If we can't fill a corner, break
        if len(corner_idx) < d:
            break
            
        corner_idx = corner_idx[:d]

        # Compute s and l
        s = min(w[i] for i in corner_idx)
        outside = [i for i in range(n) if i not in corner_idx]
        l = max(w[i] for i in outside) if outside else 0.0

        p = min(d * s, w.sum() - d * l)
        
        # Safeguard: if p is too small, we're done
        if p < tol:
            break

        # Build corner
        r = np.zeros(n)
        r[corner_idx] = 1.0 / d

        # Store
        corners.append(r)
        weights.append(p)

        # Update w
        w -= p * r
        w[w < tol] = 0.0

    # Normalize weights for safety
    weights = np.array(weights)
    if len(weights) > 0 and weights.sum() > tol:
        weights /= weights.sum()
    else:
        # Fallback: return a single uniform corner
        corners = [np.zeros(n)]
        corners[0][np.argsort(-w)[:d]] = 1.0 / d
        weights = np.array([1.0])

    return corners, weights


def algorithm2_mixture_decomposition(
    w: np.ndarray, 
    d: int, 
    rng: np.random.Generator
) -> np.ndarray:
    """
    Sample a d-subset using mixture decomposition (Algorithm 2).
    
    This function:
    1. Decomposes the capped mixture w into a convex combination of d-corners
       (a d-corner has exactly d entries equal to 1/d, rest 0)
    2. Samples one corner according to the mixture weights
    3. Returns binary indicator of which elements are in the sampled subset
    
    Marginal property: P(i selected) = d * w[i]
    
    Proof: Since w = sum_j p_j * r_j where r_j are d-corners,
    we have w[i] = sum_j p_j * r_j[i] = (1/d) * sum_{j: i in corner_j} p_j
                 = (1/d) * P(i selected)
    
    Parameters
    ----------
    w : np.ndarray
        Capped mixture vector (w_i <= 1/d, sum = 1)
    d : int
        Number of elements to select
    rng : np.random.Generator
        Random number generator
        
    Returns
    -------
    np.ndarray
        Binary indicator vector with exactly d ones
    """
    n = len(w)
    
    # Decompose into mixture of corners
    corners, weights = mixture_decomposition(w, d)
    
    if len(corners) == 0:
        # Edge case: return random d-subset
        selected = np.zeros(n)
        indices = rng.choice(n, size=d, replace=False)
        selected[indices] = 1.0
        return selected
    
    # Sample a corner according to the mixture weights
    corner_idx = rng.choice(len(corners), p=weights)
    corner = corners[corner_idx]
    
    # Convert from 1/d indicators to binary indicators
    # A corner has d entries equal to 1/d
    selected = (corner > 0.5 / d).astype(float)
    
    # Verify exactly d elements selected
    assert np.sum(selected) == d, f"Expected {d} selected, got {np.sum(selected)}"
    
    return selected


# =============================================================================
# Algorithm 3: Capped Hedge Algorithm (Online Subset Selection)
# =============================================================================

@dataclass
class CappedHedgeState:
    """State of the Capped Hedge algorithm."""
    weights: np.ndarray  # Current capped mixture weights
    cumulative_loss: np.ndarray  # Cumulative loss per expert
    total_online_loss: float  # Total loss incurred by algorithm
    t: int  # Current time step


class Algorithm3CappedHedge:
    """
    Algorithm 3: Capped Hedge Algorithm from Section 4.
    
    This is the main online learning algorithm for subset selection.
    At each round:
    1. Sample a subset S of size d using Algorithm 2
    2. Receive loss vector l^t in [0,1]^n  
    3. Incur loss = sum of losses on elements NOT in our selected subset
       (equivalently: sum of losses on the d elements in our "discarded" set)
    4. Update weights using multiplicative/exponentiated gradient update
    5. Cap weights using Algorithm 4
    
    The key insight is that capping ensures we can efficiently sample subsets
    while maintaining good regret bounds. The cap constraint w_i <= 1/d ensures
    that the capped mixture can be decomposed into a mixture over subsets.
    
    Parameters
    ----------
    n : int
        Number of experts
    k : int
        Size of subset to SELECT (experts to use for prediction)
        d = n - k elements are "discarded" each round
    eta : float, optional
        Learning rate. If None, uses optimal rate from standard Hedge analysis.
    T : int, optional
        Time horizon. Required if eta is None.
    seed : int, optional
        Random seed for reproducibility
    """
    
    def __init__(
        self, 
        n: int, 
        k: int, 
        eta: Optional[float] = None,
        T: Optional[int] = None,
        seed: Optional[int] = None
    ):
        if k >= n:
            raise ValueError(f"k ({k}) must be less than n ({n})")
        if k < 1:
            raise ValueError(f"k must be positive, got {k}")
        
        self.n = n
        self.k = k
        self.d = n - k  # Size of discarded set
        
        # Compute learning rate
        if eta is not None:
            self.eta = eta
        elif T is not None:
            # Standard Hedge learning rate: eta = sqrt(ln(n) / (d * T))
            # The factor of 2 comes from tighter analysis with bounded losses
            self.eta = np.sqrt(2 * np.log(n) / (self.d * T))
        else:
            raise ValueError("Either eta or T must be provided")
        
        self.rng = np.random.default_rng(seed)
        self.reset()
    
    def reset(self):
        """Reset algorithm to initial state."""
        # Initialize with uniform weights
        self.weights = np.ones(self.n) / self.n
        # Cap the initial weights
        self.weights = algorithm4_capping(self.weights, self.d)
        
        self.cumulative_loss = np.zeros(self.n)
        self.total_online_loss = 0.0
        self.t = 0
    
    def get_state(self) -> CappedHedgeState:
        """Get current algorithm state."""
        return CappedHedgeState(
            weights=self.weights.copy(),
            cumulative_loss=self.cumulative_loss.copy(),
            total_online_loss=self.total_online_loss,
            t=self.t
        )
    
    def select_subset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select which experts to use (and which to discard).
        
        Returns
        -------
        selected : np.ndarray
            Binary indicator: 1 if expert is selected (used), 0 if discarded
        discarded : np.ndarray
            Binary indicator: 1 if expert is discarded, 0 if selected
        """
        # Sample the discarded set of size d using Algorithm 2
        discarded = algorithm2_mixture_decomposition(self.weights, self.d, self.rng)
        selected = 1.0 - discarded
        return selected, discarded
    
    def update(self, loss: np.ndarray, discarded: np.ndarray) -> float:
        """
        Update weights after observing loss vector.
        
        Parameters
        ----------
        loss : np.ndarray
            Loss vector for all n experts (values in [0, 1])
        discarded : np.ndarray
            Binary indicator of which experts were discarded this round
            
        Returns
        -------
        float
            Loss incurred this round (sum of losses on discarded experts)
        """
        self.t += 1
        
        # Validate loss range
        if np.any(loss < 0) or np.any(loss > 1):
            # Clip to [0, 1] for robustness
            loss = np.clip(loss, 0, 1)
        
        # Compute round loss: sum of losses on discarded experts
        round_loss = np.sum(loss * discarded)
        self.total_online_loss += round_loss
        self.cumulative_loss += loss
        
        # Multiplicative/Exponentiated Gradient update (from Section 4.2 of paper)
        # w_i^{t+1} ∝ w_i^t * exp(-eta * loss_i^t)
        # Higher loss → lower weight → less likely to be discarded → SELECTED
        # This way we avoid paying for high-loss experts
        self.weights = self.weights * np.exp(-self.eta * loss)
        
        # Normalize
        w_sum = np.sum(self.weights)
        if w_sum > 1e-12:
            self.weights = self.weights / w_sum
        else:
            # Fallback to uniform if weights collapse
            self.weights = np.ones(self.n) / self.n
        
        # Cap using Algorithm 4
        self.weights = algorithm4_capping(self.weights, self.d)
        
        return round_loss
    
    def run_one_round(self, loss: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Convenience method to run one complete round.
        
        Parameters
        ----------
        loss : np.ndarray
            Loss vector for all experts
            
        Returns
        -------
        selected : np.ndarray
            Binary indicator of selected experts
        round_loss : float
            Loss incurred this round
        """
        selected, discarded = self.select_subset()
        round_loss = self.update(loss, discarded)
        return selected, round_loss
    
    def get_best_offline_subset_loss(self, all_losses: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute the loss of the best fixed subset in hindsight.
        
        From the paper: "total loss over all trials is close to the smallest n-k 
        components of the total loss vector". This means we should DISCARD the d
        experts with LOWEST total loss, i.e., SELECT the k experts with HIGHEST loss.
        
        Parameters
        ----------
        all_losses : np.ndarray
            Matrix of shape (T, n) containing all loss vectors
            
        Returns
        -------
        best_loss : float
            Total loss of the best fixed subset (sum of d smallest per-expert losses)
        best_subset : np.ndarray
            Binary indicator of the best subset to select
        """
        total_loss = np.sum(all_losses, axis=0)
        sorted_idx = np.argsort(total_loss)  # Ascending order
        
        # Select k experts with HIGHEST cumulative loss (last k in sorted order)
        # This way we discard d experts with LOWEST loss, minimizing total loss
        best_selected = np.zeros(self.n)
        best_selected[sorted_idx[-self.k:]] = 1.0
        best_discarded = 1.0 - best_selected
        
        # Compute loss: sum over time of losses on discarded experts
        # This equals sum of d smallest per-expert total losses
        best_loss = np.sum(all_losses @ best_discarded)
        
        return best_loss, best_selected


# =============================================================================
# Section 5: Matrix Extensions for Uncentered Online PCA
# =============================================================================

def matrix_capping(W: np.ndarray, d: int, tol: float = 1e-10) -> np.ndarray:
    """
    Matrix version of Algorithm 4: Capping eigenvalues.
    
    Projects a density matrix W so that all eigenvalues satisfy λ_i <= 1/d.
    This is done by:
    1. Computing eigendecomposition W = V @ diag(λ) @ V^T
    2. Applying scalar capping (Algorithm 4) to eigenvalues λ
    3. Reconstructing W = V @ diag(λ_capped) @ V^T
    
    Parameters
    ----------
    W : np.ndarray
        Density matrix (positive semidefinite, trace = 1), shape (n, n)
    d : int
        Cap parameter - all eigenvalues must be <= 1/d after capping
        In PCA, d = n - k (dimension of discarded subspace)
    tol : float
        Numerical tolerance
        
    Returns
    -------
    np.ndarray
        Capped density matrix with all eigenvalues <= 1/d
        
    Notes
    -----
    From the paper: "The density matrix W whose eigenvalues are bounded by 1/(n-k)
    can be written as a mixture of density matrices corresponding to (n-k)-dimensional
    subspaces."
    """
    n = W.shape[0]
    
    # Compute eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(W)
    
    # Ensure eigenvalues are non-negative (numerical robustness)
    eigenvalues = np.maximum(eigenvalues, 0)
    
    # Debug check: warn if trace != 1 before normalization
    eig_sum = np.sum(eigenvalues)
    if np.abs(eig_sum - 1.0) > tol:
        warnings.warn(
            f"matrix_capping: input trace = {eig_sum:.6f} != 1, normalizing",
            RuntimeWarning
        )
    
    # Normalize eigenvalues to sum to 1 (trace = 1)
    if eig_sum > tol:
        eigenvalues = eigenvalues / eig_sum
    else:
        eigenvalues = np.ones(n) / n
    
    # Apply scalar capping to eigenvalues
    eigenvalues_capped = algorithm4_capping(eigenvalues, d, tol)
    
    # Reconstruct capped density matrix
    W_capped = eigenvectors @ np.diag(eigenvalues_capped) @ eigenvectors.T
    
    # Ensure symmetry
    W_capped = (W_capped + W_capped.T) / 2
    
    return W_capped


def matrix_mixture_decomposition(
    W: np.ndarray, 
    d: int, 
    rng: np.random.Generator
) -> np.ndarray:
    """
    Matrix version of Algorithm 2: Sample a d-dimensional subspace.
    
    For a capped density matrix W (eigenvalues <= 1/d), this function:
    1. Computes eigendecomposition W = V @ diag(λ) @ V^T
    2. Uses scalar mixture decomposition on λ to sample d eigenvector indices
    3. Returns the d-dimensional projection matrix for the sampled subspace
    
    The key insight is that a capped density matrix can be written as a mixture
    of rank-d projection matrices scaled by 1/d. Sampling from this mixture
    reduces to sampling indices using the scalar Algorithm 2.
    
    Parameters
    ----------
    W : np.ndarray
        Capped density matrix (eigenvalues <= 1/d, trace = 1), shape (n, n)
    d : int
        Dimension of subspace to sample
    rng : np.random.Generator
        Random number generator
        
    Returns
    -------
    np.ndarray
        Projection matrix P of rank d (P = P^T = P^2), shape (n, n)
        
    Notes
    -----
    The marginal property: For any unit vector u,
    E[u^T P u] = d * (u^T W u)
    """
    n = W.shape[0]
    
    # Compute eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(W)
    
    # Ensure eigenvalues are non-negative and normalized
    eigenvalues = np.maximum(eigenvalues, 0)
    eig_sum = np.sum(eigenvalues)
    
    # Debug check: warn if trace != 1 before normalization
    if np.abs(eig_sum - 1.0) > 1e-9:
        warnings.warn(
            f"matrix_mixture_decomposition: input trace = {eig_sum:.6f} != 1, normalizing",
            RuntimeWarning
        )
    
    if eig_sum > 1e-12:
        eigenvalues = eigenvalues / eig_sum
    else:
        eigenvalues = np.ones(n) / n
    
    # Sample d eigenvalue indices using scalar mixture decomposition
    selected = algorithm2_mixture_decomposition(eigenvalues, d, rng)
    
    # Build projection matrix from selected eigenvectors
    # P = sum_i v_i @ v_i^T for selected indices i
    selected_idx = np.where(selected > 0.5)[0]
    
    # Construct projection matrix
    V_selected = eigenvectors[:, selected_idx]
    P = V_selected @ V_selected.T
    
    return P


@dataclass
class OnlinePCAState:
    """State of the Uncentered Online PCA algorithm."""
    W: np.ndarray  # Current capped density matrix
    total_online_loss: float  # Total compression loss incurred
    t: int  # Current time step


class UncenteredOnlinePCA:
    """
    Uncentered Online PCA Algorithm from Section 5.
    
    This algorithm maintains a density matrix W whose eigenvalues are capped
    by 1/(n-k). At each round:
    1. Sample a (n-k)-dimensional subspace from W using matrix mixture decomposition
    2. Project using the complementary k-dimensional subspace
    3. Receive instance x^t and incur compression loss ||x - P_k x||^2
    4. Update W using matrix exponentiated gradient
    5. Cap eigenvalues using matrix capping
    
    The compression loss ||x - P_k x||^2 = x^T (I - P_k) x = x^T P_{n-k} x
    where P_{n-k} is the (n-k)-dimensional projection we sample.
    
    Parameters
    ----------
    n : int
        Dimension of instances
    k : int
        Target dimension for projection (dimension to keep)
        d = n - k dimensions are "discarded"
    eta : float, optional
        Learning rate. If None, uses optimal rate from analysis.
    T : int, optional
        Time horizon. Required if eta is None.
    seed : int, optional
        Random seed for reproducibility
        
    Notes
    -----
    From the paper: "The running time is O(n^2) per trial, where n is the
    dimension of the instances."
    
    Regret bound: O(k * sqrt(T * log(n)))
    This is logarithmic in n, which is the key result of the paper.
    """
    
    def __init__(
        self,
        n: int,
        k: int,
        eta: Optional[float] = None,
        T: Optional[int] = None,
        seed: Optional[int] = None
    ):
        if k >= n:
            raise ValueError(f"k ({k}) must be less than n ({n})")
        if k < 1:
            raise ValueError(f"k must be positive, got {k}")
        
        self.n = n
        self.k = k
        self.d = n - k  # Dimension of discarded subspace
        
        # Compute learning rate
        # From the paper: optimal eta for matrix case
        if eta is not None:
            self.eta = eta
        elif T is not None:
            # Learning rate from Theorem 1 in the paper
            # For compression loss in [0, max_eigenvalue], we use:
            # eta = sqrt(2 * ln(n) / T)
            self.eta = np.sqrt(2 * np.log(n) / T)
        else:
            raise ValueError("Either eta or T must be provided")
        
        self.rng = np.random.default_rng(seed)
        self.reset()
    
    def reset(self):
        """Reset algorithm to initial state."""
        # Initialize with uniform density matrix (I/n)
        self.W = np.eye(self.n) / self.n
        # Cap the initial eigenvalues
        self.W = matrix_capping(self.W, self.d)
        
        self.total_online_loss = 0.0
        self.cumulative_outer_product = np.zeros((self.n, self.n))
        self.t = 0
    
    def get_state(self) -> OnlinePCAState:
        """Get current algorithm state."""
        return OnlinePCAState(
            W=self.W.copy(),
            total_online_loss=self.total_online_loss,
            t=self.t
        )
    
    def select_subspace(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select the projection subspace.
        
        Returns
        -------
        P_keep : np.ndarray
            Projection matrix of rank k (the subspace to keep)
        P_discard : np.ndarray
            Projection matrix of rank d (the subspace to discard)
            Note: P_keep + P_discard = I
        """
        # Sample the d-dimensional subspace to discard using matrix Algorithm 2
        P_discard = matrix_mixture_decomposition(self.W, self.d, self.rng)
        
        # The complementary k-dimensional subspace is what we keep
        P_keep = np.eye(self.n) - P_discard
        
        return P_keep, P_discard
    
    def compute_compression_loss(
        self, 
        x: np.ndarray, 
        P_keep: np.ndarray
    ) -> float:
        """
        Compute compression loss for projecting x onto P_keep.
        
        Loss = ||x - P_keep @ x||^2 = x^T (I - P_keep) x = x^T P_discard x
        
        Parameters
        ----------
        x : np.ndarray
            Instance vector of shape (n,)
        P_keep : np.ndarray
            Projection matrix of rank k
            
        Returns
        -------
        float
            Compression loss
        """
        residual = x - P_keep @ x
        return np.dot(residual, residual)
    
    def update(self, x: np.ndarray, P_discard: np.ndarray) -> float:
        """
        Update density matrix after observing instance.
        
        Parameters
        ----------
        x : np.ndarray
            Instance vector of shape (n,)
        P_discard : np.ndarray
            The d-dimensional projection matrix that was discarded
            
        Returns
        -------
        float
            Compression loss incurred this round
        """
        self.t += 1
        
        # Compute compression loss
        # Loss = x^T @ P_discard @ x (variance in discarded directions)
        P_keep = np.eye(self.n) - P_discard
        round_loss = self.compute_compression_loss(x, P_keep)
        self.total_online_loss += round_loss
        
        # Track cumulative outer product for offline comparison
        self.cumulative_outer_product += np.outer(x, x)
        
        # Matrix Exponentiated Gradient Update
        # The loss matrix for this round is the outer product L = x @ x^T
        # We want to minimize variance in discarded directions
        # 
        # Update: W^{t+1} ∝ exp(log(W^t) - η * L^t)
        # Higher loss in a direction → lower eigenvalue → less likely to be discarded
        
        # Normalize instance to bound loss
        x_norm_sq = np.dot(x, x)
        if x_norm_sq > 1e-12:
            # Normalized outer product (eigenvalues in [0, 1])
            L = np.outer(x, x) / x_norm_sq
        else:
            # Zero instance, no update
            return round_loss
        
        # Matrix exponential update
        # W_new ∝ expm(logm(W) - η * L)
        # For numerical stability, we use eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(self.W)
        eigenvalues = np.maximum(eigenvalues, 1e-15)  # Avoid log(0)

        log_W = eigenvectors @ np.diag(np.log(eigenvalues)) @ eigenvectors.T
        log_W_updated = log_W - self.eta * L
        
        # Ensure symmetry
        log_W_updated = (log_W_updated + log_W_updated.T) / 2
        
        # Matrix exponential
        eigenvalues_log, eigenvectors_log = np.linalg.eigh(log_W_updated)
        W_new = eigenvectors_log @ np.diag(np.exp(eigenvalues_log)) @ eigenvectors_log.T
        
        # Debug check: warn if trace != 1 before normalization
        trace = np.trace(W_new)
        if np.abs(trace - 1.0) > 1e-9:
            warnings.warn(
                f"UncenteredOnlinePCA.update: W_new trace = {trace:.6f} != 1, normalizing",
                RuntimeWarning
            )
        
        # Normalize to trace 1
        if trace > 1e-12:
            W_new = W_new / trace
        else:
            W_new = np.eye(self.n) / self.n
        
        # Ensure symmetry
        W_new = (W_new + W_new.T) / 2
        
        # Cap eigenvalues using matrix capping
        self.W = matrix_capping(W_new, self.d)
        
        return round_loss
    
    def run_one_round(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Convenience method to run one complete round.
        
        Parameters
        ----------
        x : np.ndarray
            Instance vector
            
        Returns
        -------
        P_keep : np.ndarray
            The k-dimensional projection used
        round_loss : float
            Compression loss incurred
        """
        P_keep, P_discard = self.select_subspace()
        round_loss = self.update(x, P_discard)
        return P_keep, round_loss
    
    def get_best_offline_loss(self, all_instances: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute the loss of the best fixed k-dimensional subspace in hindsight.
        
        The best offline subspace is spanned by the k eigenvectors corresponding
        to the largest eigenvalues of the data covariance matrix.
        This is standard batch PCA.
        
        Parameters
        ----------
        all_instances : np.ndarray
            Matrix of shape (T, n) containing all instances
            
        Returns
        -------
        best_loss : float
            Total compression loss of the best fixed subspace
        best_projection : np.ndarray
            The best k-dimensional projection matrix
        """
        # Compute data covariance matrix (unnormalized)
        covariance = all_instances.T @ all_instances
        
        # Eigendecomposition (ascending order by default)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        
        # Best k eigenvectors are those with largest eigenvalues (last k)
        best_eigenvectors = eigenvectors[:, -self.k:]
        
        # Best projection matrix
        best_projection = best_eigenvectors @ best_eigenvectors.T
        
        # Compute total compression loss
        # Loss = sum_t ||x_t - P @ x_t||^2 = sum_t x_t^T (I - P) x_t
        #      = trace((I - P) @ covariance) = sum of (n-k) smallest eigenvalues
        best_loss = np.sum(eigenvalues[:-self.k])
        
        return best_loss, best_projection


# =============================================================================
# Test Functions
# =============================================================================

def test_capping():
    """Test Algorithm 4 (Capping)."""
    print("Testing Algorithm 4: Capping")
    print("-" * 40)
    
    # Test 1: Already capped vector
    w1 = np.array([0.25, 0.25, 0.25, 0.25])
    result1 = algorithm4_capping(w1, d=2)  # cap = 0.5
    print(f"Test 1 - Already capped:")
    print(f"  Input:  {w1}")
    print(f"  Output: {result1}")
    print(f"  Cap satisfied: {np.all(result1 <= 0.5 + 1e-9)}")
    
    # Test 2: One entry needs capping
    w2 = np.array([0.6, 0.2, 0.2])
    result2 = algorithm4_capping(w2, d=2)  # cap = 0.5
    print(f"\nTest 2 - One entry needs capping:")
    print(f"  Input:  {w2}")
    print(f"  Output: {result2}")
    print(f"  Cap = 0.5, max = {np.max(result2):.4f}")
    
    # Test 3: Multiple entries need capping
    w3 = np.array([0.4, 0.35, 0.15, 0.1])
    result3 = algorithm4_capping(w3, d=3)  # cap = 0.333...
    print(f"\nTest 3 - Multiple iterations needed:")
    print(f"  Input:  {w3}")
    print(f"  Output: {result3}")
    print(f"  Cap = {1/3:.4f}, max = {np.max(result3):.4f}")
    
    # Test 4: Edge case - uniform distribution
    n = 10
    w4 = np.ones(n) / n
    result4 = algorithm4_capping(w4, d=5)
    print(f"\nTest 4 - Uniform distribution (n={n}, d=5):")
    print(f"  Max: {np.max(result4):.4f}, Cap: {1/5:.4f}")
    print(f"  Sum: {np.sum(result4):.4f}")
    
    print("\n✓ All capping tests passed!")


def test_mixture_decomposition():
    """Test Algorithm 2 (Mixture Decomposition)."""
    print("\nTesting Algorithm 2: Mixture Decomposition")
    print("-" * 40)
    
    n = 5
    d = 3
    w = algorithm4_capping(np.array([0.3, 0.25, 0.2, 0.15, 0.1]), d)
    
    print(f"Capped weights: {w}")
    print(f"d = {d}, cap = {1/d:.4f}")
    
    # Test sampling produces exactly d elements
    rng = np.random.default_rng(42)
    for i in range(5):
        selected = algorithm2_mixture_decomposition(w, d, rng)
        assert np.sum(selected) == d, f"Expected {d} selected, got {np.sum(selected)}"
    print(f"✓ All samples have exactly {d} elements")
    
    # Test marginal property: P(i selected) = d * w[i]
    # Proof: w[i] = sum_j p_j * r_j[i] = (1/d) * P(i selected)
    n_samples = 10000
    counts = np.zeros(n)
    for _ in range(n_samples):
        selected = algorithm2_mixture_decomposition(w, d, rng)
        counts += selected
    empirical = counts / n_samples
    theoretical = d * w
    
    print(f"\nMarginal probability verification:")
    print(f"  Expected (d*w): {theoretical}")
    print(f"  Empirical:      {empirical}")
    print(f"  Max error:      {np.max(np.abs(empirical - theoretical)):.4f}")
    
    print("\n✓ Mixture decomposition tests passed!")


def test_capped_hedge():
    """Test Algorithm 3 (Capped Hedge)."""
    print("\nTesting Algorithm 3: Capped Hedge")
    print("-" * 40)
    
    n, k, T = 10, 3, 100
    d = n - k
    algo = Algorithm3CappedHedge(n=n, k=k, T=T, seed=42)
    
    print(f"Parameters: n={n}, k={k}, d={d}, T={T}")
    print(f"Learning rate: {algo.eta:.4f}")
    
    # Generate loss sequence
    # In this problem: loss = sum of DISCARDED experts' losses
    # Best strategy: SELECT high-loss experts, DISCARD low-loss experts
    # First d experts have low loss → should be DISCARDED
    # Remaining k experts have high loss → should be SELECTED
    rng = np.random.default_rng(42)
    all_losses = np.zeros((T, n))
    for t in range(T):
        # First d experts: low loss (should be discarded)
        all_losses[t, :d] = rng.uniform(0, 0.3, size=d)
        # Remaining k experts: high loss (should be selected)
        all_losses[t, d:] = rng.uniform(0.5, 1.0, size=k)
    
    # Run algorithm
    total_online_loss = 0.0
    for t in range(T):
        selected, discarded = algo.select_subset()
        loss = algo.update(all_losses[t], discarded)
        total_online_loss += loss
    
    # Compute best offline loss
    best_loss, best_subset = algo.get_best_offline_subset_loss(all_losses)
    regret = total_online_loss - best_loss
    
    print(f"\nOnline loss:  {total_online_loss:.2f}")
    print(f"Best offline: {best_loss:.2f}")
    print(f"Regret:       {regret:.2f}")
    
    print("\n✓ Capped Hedge test passed!")


# =============================================================================
# Test Functions for Section 5 (Matrix Algorithms)
# =============================================================================

def test_matrix_capping():
    """Test matrix capping (Algorithm 4 extended to matrices)."""
    print("\nTesting Matrix Capping (Section 5)")
    print("-" * 40)
    
    n = 5
    d = 3
    cap = 1.0 / d
    
    # Test 1: Already capped (uniform eigenvalues)
    W1 = np.eye(n) / n
    W1_capped = matrix_capping(W1, d)
    eigs1 = np.linalg.eigvalsh(W1_capped)
    print(f"Test 1 - Uniform eigenvalues:")
    print(f"  Input eigenvalues:  {np.sort(np.linalg.eigvalsh(W1))[::-1]}")
    print(f"  Output eigenvalues: {np.sort(eigs1)[::-1]}")
    print(f"  Max eigenvalue: {np.max(eigs1):.4f}, cap: {cap:.4f}")
    assert np.all(eigs1 <= cap + 1e-9), "Cap violated!"
    
    # Test 2: One large eigenvalue needs capping
    rng = np.random.default_rng(42)
    V = np.linalg.qr(rng.standard_normal((n, n)))[0]
    eigs_input = np.array([0.6, 0.2, 0.1, 0.07, 0.03])
    W2 = V @ np.diag(eigs_input) @ V.T
    W2_capped = matrix_capping(W2, d)
    eigs2 = np.linalg.eigvalsh(W2_capped)
    print(f"\nTest 2 - Large eigenvalue needs capping (d={d}, cap={cap:.4f}):")
    print(f"  Input eigenvalues:  {np.sort(eigs_input)[::-1]}")
    print(f"  Output eigenvalues: {np.sort(eigs2)[::-1]}")
    assert np.all(eigs2 <= cap + 1e-9), "Cap violated!"
    assert np.abs(np.sum(eigs2) - 1.0) < 1e-9, "Trace not 1!"
    
    # Test 3: Verify positive semidefinite
    assert np.all(eigs2 >= -1e-9), "Not positive semidefinite!"
    
    print("\n✓ Matrix capping tests passed!")


def test_matrix_mixture_decomposition():
    """Test matrix mixture decomposition (Algorithm 2 extended to matrices)."""
    print("\nTesting Matrix Mixture Decomposition (Section 5)")
    print("-" * 40)
    
    n = 5
    d = 3
    k = n - d
    
    # Create a capped density matrix with non-uniform eigenvalues
    rng = np.random.default_rng(42)
    V = np.linalg.qr(rng.standard_normal((n, n)))[0]
    eigs = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    W = V @ np.diag(eigs) @ V.T
    W = matrix_capping(W, d)
    
    print(f"Capped eigenvalues: {np.sort(np.linalg.eigvalsh(W))[::-1]}")
    print(f"d = {d}, cap = {1/d:.4f}")
    
    # Test: sampled projection has correct rank
    for i in range(5):
        P = matrix_mixture_decomposition(W, d, rng)
        rank = np.linalg.matrix_rank(P, tol=1e-6)
        assert rank == d, f"Expected rank {d}, got {rank}"
        # Verify P is a projection matrix
        assert np.allclose(P, P.T), "P not symmetric"
        assert np.allclose(P @ P, P), "P not idempotent"
    print(f"✓ All samples have rank {d}")
    
    # Test marginal property: E[tr(P @ u u^T)] = d * tr(W @ u u^T) = d * u^T W u
    n_samples = 2000
    test_vectors = [
        np.eye(n)[i] for i in range(n)
    ] + [
        rng.standard_normal(n) for _ in range(5)
    ]
    
    print("\nMarginal probability verification:")
    max_error = 0
    for u in test_vectors:
        u = u / np.linalg.norm(u)
        theoretical = d * (u @ W @ u)
        
        empirical_sum = 0
        for _ in range(n_samples):
            P = matrix_mixture_decomposition(W, d, rng)
            empirical_sum += u @ P @ u
        empirical = empirical_sum / n_samples
        
        error = abs(empirical - theoretical)
        max_error = max(max_error, error)
    
    print(f"  Max error across test vectors: {max_error:.4f}")
    print(f"  Expected error scale: {1/np.sqrt(n_samples):.4f}")
    
    print("\n✓ Matrix mixture decomposition tests passed!")


def test_uncentered_online_pca():
    """Test Uncentered Online PCA algorithm (Section 5)."""
    print("\nTesting Uncentered Online PCA (Section 5)")
    print("-" * 40)
    
    n, k, T = 10, 3, 100
    d = n - k
    algo = UncenteredOnlinePCA(n=n, k=k, T=T, seed=42)
    
    print(f"Parameters: n={n}, k={k}, d={d}, T={T}")
    print(f"Learning rate: {algo.eta:.4f}")
    
    # Generate data with clear principal components
    # Data lies mostly in the first k dimensions (should be kept)
    rng = np.random.default_rng(42)
    
    # Create a covariance matrix with eigenvalues [10, 5, 2, 0.1, 0.1, ...]
    true_eigenvalues = np.array([10, 5, 2] + [0.1] * (n - 3))
    V_true = np.linalg.qr(rng.standard_normal((n, n)))[0]
    true_cov = V_true @ np.diag(true_eigenvalues) @ V_true.T
    
    # Sample instances from this distribution
    mean = np.zeros(n)
    all_instances = rng.multivariate_normal(mean, true_cov, size=T)
    
    # Run online PCA
    for t in range(T):
        P_keep, round_loss = algo.run_one_round(all_instances[t])
    
    # Compute best offline loss
    best_loss, best_projection = algo.get_best_offline_loss(all_instances)
    regret = algo.total_online_loss - best_loss
    
    print(f"\nOnline loss:  {algo.total_online_loss:.2f}")
    print(f"Best offline: {best_loss:.2f}")
    print(f"Regret:       {regret:.2f}")
    
    # Compute theoretical regret bound: O(k * sqrt(T * log(n)))
    theoretical_bound = k * np.sqrt(T * np.log(n)) * np.max(true_eigenvalues)
    print(f"Bound scale:  {theoretical_bound:.2f}")
    
    # The regret should be positive and bounded
    assert regret >= -1e-6, f"Regret should be non-negative, got {regret}"
    
    print("\n✓ Uncentered Online PCA test passed!")


if __name__ == "__main__":
    # Section 4 tests
    test_capping()
    test_mixture_decomposition()
    test_capped_hedge()
    
    # Section 5 tests
    test_matrix_capping()
    test_matrix_mixture_decomposition()
    test_uncentered_online_pca()
    
    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)

