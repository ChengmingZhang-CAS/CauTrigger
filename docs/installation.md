# Installation

CauTrigger requires **Python 3.10 or newer**.

```bash
pip install git+https://github.com/ChengmingZhang-CAS/CauTrigger.git@main
````

A reference list of dependency packages is provided in
[`environment.yml`](https://github.com/ChengmingZhang-CAS/CauTrigger/blob/main/environment.yml).

CauTrigger uses `velocyto` to access `colDeltaCorpartial` for local partial correlation analysis.
Since `velocyto` relies on Cython extensions, please ensure that prerequisites such as NumPy and
Cython are available in your environment.
For details, see the official installation guide:
[https://velocyto.org/velocyto.py/install/index.html#](https://velocyto.org/velocyto.py/install/index.html#)
