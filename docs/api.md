# API Reference

This document provides a concise overview of the main public interface exposed by the `affinityprop` package.

The package provides a single class-based entry point: `AffinityPropagation`.

```python
AffinityPropagation(
    preference=None,
    damping=0.9,
    max_iter=200,
    convergence_iter=15,
    verbose=False
)
```

#### Parameters

- `preference` (`float` or `None`, default: `None`)  
  Preference value for self-exemplars. Controls the number of clusters selected.  
  If `None`, the median of the similarity matrix is used.

- `damping` (`float`, default: `0.9`)  
  Damping factor to stabilize updates. Must lie in the interval `(0.5, 1)`.

- `max_iter` (`int`, default: `200`)  
  Maximum number of iterations allowed for message passing.

- `convergence_iter` (`int`, default: `15`)  
  Number of consecutive stable iterations required to signal convergence.

- `verbose` (`bool`, default: `False`)  
  If enabled, prints convergence progress during training.

#### Attributes

- `labels_` (`ndarray` of shape `(n_samples,)`)  
  Cluster assignment for each input sample, assigned after fitting the model.

- `cluster_centers_indices_` (`ndarray` of `int`)  
  Indices of the identified exemplars, computed after fitting the model.


#### Methods

- `fit(S)` → `AffinityPropagation`  
  Fits the model to a similarity matrix `S` using the Affinity Propagation algorithm.  
  **Returns**: the fitted instance itself, with attributes `labels_` and `cluster_centers_indices_`.

- `fit_predict(S)` → `np.ndarray`  
  Fits the model and directly returns the cluster labels.  
  **Returns**: a 1D NumPy array of integer cluster assignments of shape `(n_samples,)`.
