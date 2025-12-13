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

The velocyto package is installed here to utilize its colDeltaCorpartial function for calculating local partial correlations. Since this function depends on Cython extensions, prerequisites like Cython and NumPy must be pre-installed. For the full installation guide, please refer to the official documentation: https://velocyto.org/velocyto.py/install/index.html#
