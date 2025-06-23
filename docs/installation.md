# Installation Guide

This guide walks you through installing the `affinityprop` package using Conda and pip. It includes instructions for both editable and non-editable installation modes.

**1.** **Create a Conda Environment** 

It is recommended to create a dedicated Conda environment to isolate dependencies and match the required Python version:

```bash
conda create -n myenv python=3.11.13
conda activate myenv
```

**2.1.** **Install the Package in Editable Mode**

If the package is under active development, it is recommended to install it in editable mode to enable immediate reflection of local source code modifications:

```bash
pip install -e .
```

**2.2.** **Install the Package in Non-Editable Mode**

If no modifications to the source code are required and the package is intended for standard usage, install it in non-editable mode as follows:

```bash
pip install .
