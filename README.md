# <img src="https://raw.githubusercontent.com/ChengmingZhang-CAS/CauTrigger/main/docs/_static/logo.png" width="60"> CauTrigger

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]


[badge-tests]: https://img.shields.io/github/actions/workflow/status/ChengmingZhang-CAS/CauTrigger/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/CauTrigger

<p align="center">
  <strong>Deciphering biological system state transitions by hierarchical causal decomposition</strong>
</p>

<div align="center">
  <img src="https://raw.githubusercontent.com/ChengmingZhang-CAS/CauTrigger/main/docs/_static/CauTrigger_overview.png" width="640" />
</div>

> <small>
Causal decoupling model constructed on a dual-flow variational autoencoder (DFVAE) framework to identify causal triggers influencing state transition. Triggers ($x^n$) are processed through a feature selection layer to separate causal triggers ($x^{c_n}$) and others ($\tilde{x}^{c_n}$ ), and then encoded them into latent space $z$ consists of causal ($z^{c_n}$) and spurious ($z^{s_n}$) components. This latent space is decoded to generate downstream conductors ($x^{c_{n-1}},...,x^{c_1}$) and to predict the final cell state ($y$). The model strives to maximize the causal information flow, $I(z^{c_n}→y)$, from $z^{c_n}$ to $y$, thus delineating the causal path from $x^{c_n}$ to $y$ via $z^{c_n}$.
> </small>
---

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

## Installation

You need to have Python 3.10 or newer installed on your system. This process will be completed in just a few minutes under normal network conditions.


```bash
pip install git+https://github.com/ChengmingZhang-CAS/CauTrigger.git@main
```
The velocyto package is installed here to utilize its colDeltaCorpartial function for calculating local partial correlations. Since this function depends on Cython extensions, prerequisites like Cython and NumPy must be pre-installed. For the full installation guide, please refer to the official documentation [velocyto-install][].

## Analyze Your Own Datasets
### Step 1: Prepare Your Dataset
- Load preprocessed data in adata format
- `var` has a *unique* index (e.g. gene symbol)
- `obs` has a *unique* index and one column indicating the states (e.g. 'cell_type', 'state')
- If you want to use CauTrigger-2L or 3L, `obsm` must have corresponding feature matrix called 'X_down' or ('X_down1', 'X_down2)
- Any 2D visualizations/embeddings (e.g., UMAP, t-SNE) that should be available and need to adhere to these rules:
  - stored in `.obsm` with name `X_{name}`
  - type: `np.ndarray` (NOT `pd.DataFrame`), dtype: float/int/uint
  - shape: `(n_obs, 2)`
  - all values finite or NaN (NO +Inf or -Inf)

### Step 2: Run CauTrigger
```bash
model = CauTrigger1L(adata)
model.train()
```

### Step 3: Analysis
Select potential causal triggers. We use top-k=10 here as example, you can select by other way
```bash
topk = 10
weight_df_weight1 = model.get_up_feature_weights(normalize=True, method="Grad", sort_by_weight=False)[0]['weight']
causal_factors_layer1_indices = np.argsort(weight_df_weight1)[-topk:][::-1]
```

Visualize the causal latent space and then you can do in silico perturbation on it or others(e.g. UMAP)
```bash
adata.obsm['X_ct'] = model.get_model_output()['latent'][:, :2]
```

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/ChengmingZhang-CAS/CauTrigger/issues
[tests]: https://github.com/ChengmingZhang-CAS/CauTrigger/actions/workflows/test.yaml
[documentation]: https://cautrigger.readthedocs.io
[changelog]: https://cautrigger.readthedocs.io/en/latest/changelog.html
[api documentation]: https://cautrigger.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/CauTrigger
[velocyto-install]: https://velocyto.org/velocyto.py/install/index.html#
