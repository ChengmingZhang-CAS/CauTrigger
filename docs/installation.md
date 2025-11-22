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

You can see the basic dependency packages in the [environment.yml](https://github.com/ChengmingZhang-CAS/CauTrigger/blob/main/environment.yml) file.
```
name: cautrigger-env
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults

dependencies:
  - python=3.10
  - pytorch=2.3.0
  - pytorch-cuda=12.1
  - numpy
  - pandas
  - anndata
  - scanpy>=1.9
  - scvi-tools>=1.1
  - scikit-learn
  - matplotlib
  - seaborn
  - tqdm
  - shap
  - pyarrow
  # notebook
  - jupyter
  - ipykernel

  - pip:
      - captum
      - velocyto
```
