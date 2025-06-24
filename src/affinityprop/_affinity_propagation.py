"""Affinity Propagation clustering algorithm."""

import warnings

import numpy as np

from sklearn.exceptions import ConvergenceWarning


def _affinity_propagation(
    S: np.ndarray,
    preference: float | None = None,
    damping: float = 0.9,
    max_iter: int = 200,
    convergence_iter: int = 15,
    verbose: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Affinity Propagation clustering from scratch based on the original paper by Frey and Dueck (2007).

    Parameters
    ----------
    S : ndarray of shape (n_samples, n_samples)
        Similarity matrix where S[i, k] indicates how well point k is suited to be the exemplar for point i.
        Diagonal entries S[k, k] represent preferences—how suitable each point is to be its own exemplar.

    preference : float or None, default=None
        The common value to set on the diagonal of S (i.e., S[k,k]). If None, the median of S is used.
        Larger values lead to more clusters; smaller values produce fewer.

    damping : float, default=0.9
        Damping factor between 0.5 and 1 to control the update rate. Higher means slower convergence but greater stability.

    max_iter : int, default=200
        Maximum number of iterations to perform.

    convergence_iter : int, default=15
        Number of iterations where cluster assignments must remain unchanged to declare convergence.

    verbose : bool, default=False
        If True, prints progress information.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster labels assigned to each point.

    exemplars : ndarray of int
        Indices of exemplar points (cluster centers).
    """
    n = S.shape[0]

    # Step 1: Initialize preference if not provided
    # ----------------------------------------------------------------
    # If no preference is specified, the diagonal entries of the similarity matrix are assigned the median value of S. 
    # Larger values result in more clusters, while smaller values reduce the number of exemplars.
    if preference is None:
        preference = np.median(S)
    S = S.copy()
    np.fill_diagonal(S, preference)

    # Step 2a: Initialize responsibility and availability matrices
    # ----------------------------------------------------------------
    # Responsibility matrix R encodes the degree to which point k is suited to be the exemplar for point i. 
    # Availability matrix A reflects the accumulated evidence for how appropriate it is for point i to choose k.
    R = np.zeros((n, n), dtype=np.float64)
    A = np.zeros((n, n), dtype=np.float64)

    # Step 2b: Initialize variables to monitor exemplar stability
    # ----------------------------------------------------------------
    # Stores the previous exemplar set and counts how many consecutive iterations the exemplars remain unchanged.
    exemplars_old = np.array([], dtype=int)
    stable_iter = 0

    for it in range(max_iter):
        # Step 3: Compute combined messages (A + S)
        # ----------------------------------------------------------------
        # Computes A + S to identify candidate exemplars. For each point i,
        # the highest and second-highest values are extracted to update the
        # responsibility matrix accordingly.
        AS = A + S
        max1 = np.max(AS, axis=1)
        idx1 = np.argmax(AS, axis=1)

        # Temporarily mask the top candidate to extract the second-best score
        AS[np.arange(n), idx1] = -np.inf
        max2 = np.max(AS, axis=1)
        AS[np.arange(n), idx1] = max1 

        # Step 4: Update responsibility matrix
        # ----------------------------------------------------------------
        # Each responsibility R[i, k] is updated to reflect the relative
        # strength of similarity between i and candidate exemplar k.
        # Values are smoothed using the damping parameter.
        R_new = S - max1[:, None]
        R_new[np.arange(n), idx1] = S[np.arange(n), idx1] - max2
        R = damping * R + (1 - damping) * R_new

        # Step 5: Update availability matrix
        # ----------------------------------------------------------------
        # Aggregates incoming responsibilities to determine how supported
        # each candidate k is. The diagonal reflects total support; off-diagonal
        # values represent the degree to which i should consider k as exemplar.
        Rp = np.maximum(R, 0)
        np.fill_diagonal(Rp, R.diagonal())  # Keep diagonal unaltered
        A_new = np.sum(Rp, axis=0)[None, :] - Rp
        A_new = np.minimum(0, A_new)
        diag = np.sum(Rp, axis=0) - Rp.diagonal()
        np.fill_diagonal(A_new, diag)
        A = damping * A + (1 - damping) * A_new

        # Step 6a: Check for convergence based on stability of exemplars
        # ----------------------------------------------------------------
        # Implements the convergence criterion described in the original paper:
        # local exemplar decisions must remain unchanged for 'convergence_iter' consecutive steps.
        # If no exemplars are selected, convergence check is skipped for this iteration.
        E = np.diag(A + R) > 0
        exemplars = np.where(E)[0]

        if exemplars.size == 0:
            continue  # No exemplars this round: skip convergence check

        if np.array_equal(exemplars, exemplars_old):
            stable_iter += 1
            if verbose:
                print(f"Iter {it+1:03d} | stable_iter = {stable_iter}")
            if stable_iter >= convergence_iter:
                if verbose:
                    print(f"Converged (exemplars stable) at iteration {it+1}")
                break
        else:
            stable_iter = 0

        exemplars_old = exemplars.copy()

    else:
        # Step 6b: Raise warning if convergence criterion not met
        # ----------------------------------------------------------------
        # If maximum iterations are reached without detecting convergence,
        # a warning is issued to indicate potential instability in the result.
        warnings.warn(
            "Affinity Propagation did not converge within max_iter. "
            "Cluster assignments may be unreliable.",
            ConvergenceWarning
        )

    # Step 7: Extract final exemplars from message matrix
    # ----------------------------------------------------------------
    # Points corresponding to positive diagonal entries of A + R are selected as exemplars. These represent cluster centers.
    exemplars = np.where(np.diag(A + R) > 0)[0]
    if exemplars.size == 0:
        raise ValueError("No exemplars found. Try lowering `preference`.")

    # Step 8: Assign labels according to maximal similarity
    # ----------------------------------------------------------------
    # Each point is assigned to the exemplar with which it shares the highest similarity. This defines the final clustering output.
    labels = np.argmax(S[:, exemplars], axis=1)
    return labels, exemplars



class AffinityPropagation:
    """
    Object-oriented wrapper for a procedural affinity propagation implementation.

    Provides a structured interface for configuring, fitting, and retrieving
    clustering results from the algorithm. After fitting, parameters
    and computed outputs such as labels and exemplar indices are stored as attributes.


    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each input sample, assigned after fit.

    cluster_centers_indices_ : ndarray of int
        Indices of exemplar points (cluster centers), assigned after fit.
    """

    def __init__(
        self,
        preference: float | None = None,
        damping: float = 0.9,
        max_iter: int = 200,
        convergence_iter: int = 15,
        verbose: bool = False
    ) -> None:
        """
        Initialize the AffinityPropagation model with optional parameters.

        Parameters
        ----------
        preference : float or None
            Preference value for self-exemplars. If None, will use median of similarities.
        
        damping : float, default=0.9
            Damping factor to stabilize updates. Must be in (0.5, 1).
        
        max_iter : int, default=200
            Maximum number of iterations allowed.
        
        convergence_iter : int, default=15
            Number of stable iterations required for convergence.
        
        verbose : bool, default=False
            If True, displays convergence messages during training.
        """
        if not 0.5 < damping < 1:
            raise ValueError("damping must be in the interval (0.5, 1)")
        if max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        if convergence_iter <= 0:
            raise ValueError("convergence_iter must be a positive integer")

        self.preference = preference
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.verbose = verbose

    def fit(self, S: np.ndarray) -> "AffinityPropagation":
        """
        Fit the Affinity Propagation model to a similarity matrix.

        Parameters
        ----------
        S : ndarray of shape (n_samples, n_samples)
            Precomputed similarity matrix. S[i, j] represents how well j is suited to be the exemplar for i.

        Returns
        -------
        self : object
            Fitted instance with attributes `labels_` and `cluster_centers_indices_`.
        """
        if not isinstance(S, np.ndarray):
            raise TypeError("Input similarity matrix S must be a NumPy array.")
        if S.ndim != 2 or S.shape[0] != S.shape[1]:
            raise ValueError("S must be a square similarity matrix (n_samples, n_samples)")

        labels, exemplars = _affinity_propagation(
            S,
            preference=self.preference,
            damping=self.damping,
            max_iter=self.max_iter,
            convergence_iter=self.convergence_iter,
            verbose=self.verbose
        )
        self.labels_ = labels
        self.cluster_centers_indices_ = exemplars
        return self

    def fit_predict(self, S: np.ndarray) -> np.ndarray:
        """
        Fit the model and return cluster labels.

        Parameters
        ----------
        S : ndarray of shape (n_samples, n_samples)
            Similarity matrix.

        Returns
        -------
        labels_ : ndarray
            Cluster labels for each sample.
        """
        self.fit(S)
        return self.labels_