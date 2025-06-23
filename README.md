# Affinity Propagation Clustering via Custom Python Package Implementation

## Introduction  

This repository is designed for a modular and customizable Python implementation of the Affinity Propagation clustering algorithm, originally introduced by FREY B.J., DUECK D. (2007) in their paper *"Clustering by Passing Messages Between Data Points"* (Science, Vol. 315, Issue 5814, pp. 972–976, [10.1126/science.1136800](https://doi.org/10.1126/science.1136800)).


Affinity Propagation works by transmitting messages between data points based on pairwise similarities, allowing the algorithm to identify a set of representative exemplars without needing to pre-specify the number of clusters. 



## Getting Started

To set up the repository properly, follow these steps:

**1.** **Install the Package** 

Clone the repository and install the package using the configuration defined in the `pyproject.toml`:

```bash
pip install .
```

**2.** **Run Unit Tests**  

To validate the core logic of the algorithm, run the unit tests located in the `tests/` directory using `pytest`. Make sure it is installed in your environment:

```bash
pip install pytest
pytest tests/
```



## License  

This project is licensed under the **MIT License**, which allows for open-source use, modification, and distribution with minimal restrictions. For more details, refer to the file included in this repository.  
