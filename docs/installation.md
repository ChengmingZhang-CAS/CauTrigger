# Installation

You need to have Python 3.10 or newer installed on your system. This process will be completed in just a few minutes under normal network conditions.

<!--
1) Install the latest release of `CauTrigger` from [PyPI][]:

```bash
pip install CauTrigger
```
-->

```bash
pip install git+https://github.com/ChengmingZhang-CAS/CauTrigger.git@main
```

You can see the basic dependency packages from 'pyproject.toml':
```
dependencies = [
  "numpy>=1.22",
  "pandas>=1.5",
  "anndata",
  "scanpy>=1.9",
  "scvi-tools>=1.1",
  "scikit-learn>=1.3",
  "matplotlib>=3.7",
  "seaborn>=0.12",
  "tqdm",
  "shap",
  "captum",
  "pyarrow",
  "session-info2",
]
```
