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
CauTrigger is a causal decoupling model constructed on a dual-flow variational autoencoder (DFVAE) framework to identify causal triggers that influence biological state transitions in a hierarchical manner. Triggers ($x^n$) are processed through a feature selection layer to separate causal triggers ($x^{c_n}$) and others ($\tilde{x}^{c_n}$), and are then encoded into a latent space $z$ that consists of causal ($z^{c_n}$) and spurious ($z^{s_n}$) components. This latent space is decoded to generate downstream conductors ($x^{c_{n-1}}, \dots, x^{c_1}$) and to predict the final cell state ($y$). The model strives to maximize the causal information flow $I(z^{c_n} \rightarrow y)$ from $z^{c_n}$ to $y$, thereby delineating hierarchical causal paths from $x^{c_n}$ to $y$ via $z^{c_n}$.
> </small>

---

## Getting started

Please refer to the [documentation][] for installation instructions, API reference, and end-to-end tutorials.

## Tutorials at a glance

The documentation includes four end-to-end tutorials (see the [tutorials section](https://cautrigger.readthedocs.io/en/latest/notebooks/index.html)):

- **Two-layer synthetic causal simulation** – demonstrates how CauTrigger recovers known causal vs. spurious features in a controlled two-layer system.
- **hESC differentiation causal analysis** – applies a two-layer TF/TG model to human embryonic stem cell differentiation (0 h vs 96 h).
- **T2D multi-omics causal analysis** – uses a three-layer TF–CRE–TG hierarchy to analyze pancreatic islet multi-omics in control, pre-T2D, and T2D donors.
- **Gut–brain axis hierarchical analysis in ASD** – models microbiota → host proteome → ASD states to illustrate hierarchical triggers and in silico perturbation.

For new applications, we recommend starting from the tutorial whose data structure is closest to your own dataset and adapting the data-preparation steps accordingly.

## Installation

You need to have Python 3.10 or newer installed on your system. Installation usually finishes within a few minutes.

```bash
pip install git+https://github.com/ChengmingZhang-CAS/CauTrigger.git@main
```

The `velocyto` package is used to access the `colDeltaCorpartial` function for calculating local partial correlations. Since this function depends on Cython extensions, prerequisites like Cython and NumPy must be pre-installed. For the full installation guide, please refer to the official documentation [velocyto-install][].

## Analyze Your Own Datasets
### Step 1: Prepare Your Dataset
**Core AnnData structure**
- `adata` is a preprocessed `AnnData` object (cells in `obs`, features in `var`).
- `adata.var` has a *unique* index (e.g. gene symbols).
- `adata.obs` has a *unique* index for all cells.

**State labels**
- The system state $y$ is stored in a numeric column `"labels"` in `adata.obs`.
  - For binary tasks (`n_state = 2`), `"labels"` should lie in [0, 1] (e.g. 0/1 or continuous values).
  - For multi-class tasks (`n_state > 2`), `"labels"` contains integer class indices 0, 1, ..., `n_state - 1`.

**Hierarchical inputs (2-layer / 3-layer)**
- For CauTrigger-2L or 3L, `adata.obsm` contains the downstream feature matrices, e.g. `'X_down'` or (`'X_down1'`, `'X_down2'`). See the tutorials for complete two-layer and three-layer examples.

### Step 2: Run CauTrigger
```python
model = CauTrigger1L(adata)
model.train()
```

### Step 3: Analysis
Select potential causal triggers. By default, SHAP-based attribution is used; you may optionally switch to gradient-based attribution by setting `method="Grad"`.

```python
topk = 10
weight_df_weight1 = model.get_up_feature_weights(
    normalize=True,
    method="SHAP",  # default; use "Grad" for gradient-based attribution
    sort_by_weight=False
)[0]["weight"]
causal_factors_layer1_indices = np.argsort(weight_df_weight1)[-topk:][::-1]
```

Visualize the causal latent space and then you can do in silico perturbation on it or other embeddings (e.g. UMAP):

```python
adata.obsm["X_ct"] = model.get_model_output()["latent"][:, :2]
```

## Contact
For questions, discussions, or bug reports, please use the GitHub issue tracker.
- **GitHub Issues**: https://github.com/ChengmingZhang-CAS/CauTrigger/issues

## Citation

> t.b.a

[issue tracker]: https://github.com/ChengmingZhang-CAS/CauTrigger/issues
[documentation]: https://cautrigger.readthedocs.io
[changelog]: https://cautrigger.readthedocs.io/en/latest/changelog.html
[api documentation]: https://cautrigger.readthedocs.io/en/latest/api.html
[velocyto-install]: https://velocyto.org/velocyto.py/install/index.html#
