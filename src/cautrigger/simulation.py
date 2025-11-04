"""
Simulation utilities for generating synthetic hierarchical causal data.

This module provides functions for constructing multi-layer causal
structures, sampling expression profiles using NB/ZINB distributions,
injecting spurious signals, and controlling sparsity/noise levels.

Used in:
- CauTrigger simulation benchmarking
- Layer-known / layer-unknown experiments
- Two-layer and three-layer hierarchical simulations
"""
# NOTE:
# These run_ct_* functions are designed ONLY for synthetic benchmark
# experiments used in the CauTrigger paper. They require ground-truth
# layer and causal annotation in adata.var, which real datasets do not
# provide.

import os
import pandas as pd
import sys
import math
import gc
import logging
import warnings
import scanpy as sc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from anndata import AnnData
from itertools import combinations
from scipy.stats import pearsonr, spearmanr, f_oneway, norm
from scipy.stats import ttest_ind
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.preprocessing import minmax_scale, MinMaxScaler
from scipy.stats import zscore

from cautrigger.utils import set_seed, select_features
from cautrigger.model import CauTrigger3L, CauTrigger2L, CauTrigger1L

# sys.path.append('../')  # add root path
warnings.filterwarnings("ignore")
logging.getLogger('matplotlib.font_manager').disabled = True
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

def apply_activation(x, activation=None):
    if activation == "tanh":
        return np.tanh(x)
    elif activation == "relu":
        return np.maximum(0, x)
    elif activation == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    elif activation is None or activation == 'linear':
        return x
    else:
        raise ValueError(f"Unsupported activation: {activation}")

def zscore_normalization(x):
    return (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + 1e-6)

def softplus(x):
    return np.log1p(np.exp(x))

def linear_act(x, in_dim, out_dim, activation="linear", weight_clip=3, seed=None, sparsity=0.5):
    """
    Linear layer with optional sparsity and activation.

    Args:
        x: input array
        in_dim: input dimension
        out_dim: output dimension
        activation: activation function to apply
        weight_clip: clip weights to this range
        seed: random seed
        sparsity: proportion of smallest weights (by magnitude) to zero out (e.g., 0.9 keeps top 10%)
    """
    if seed is not None:
        np.random.seed(seed)
    w = np.random.randn(in_dim, out_dim)

    if weight_clip is not None:
        w = np.clip(w, -weight_clip, weight_clip)

    if sparsity > 0:
        threshold = np.percentile(np.abs(w), sparsity * 100)
        w[np.abs(w) < threshold] = 0

    return apply_activation(x @ w, activation)


def generate_two_layer_synthetic_data(
        n_samples=200,
        oversampling_factor=5,
        n_causal_features_1=20,
        n_causal_features_2=10,
        n_spurious_features_1=80,
        n_spurious_features_2=40,
        n_hidden=10,
        n_latent=5,
        noise_scale=0.1,
        causal_strength=0.2,
        spurious_mode='hrc',  # Options: 'flat', 'hrc'
        simulate_single_cell=True,
        dist='zinb',
        nb_shape=10.0,
        p_zero=0.1,
        activation='linear',
        normalization='zscore',
        mu_transform='softplus',
        seed=42,
        weight_clip=3,
):
    """Generate synthetic data with two-layer causal and spurious variables.
    """
    np.random.seed(seed)
    total_samples = n_samples * oversampling_factor
    pos_samples = n_samples // 2
    neg_samples = n_samples - pos_samples

    # Input latent sources
    c = np.clip(np.random.randn(total_samples, n_latent), -2, 2)
    s = np.clip(np.random.randn(total_samples, n_latent), -2, 2)

    # ====== Causal path ======
    xc2 = linear_act(c, n_latent, n_hidden, activation, weight_clip)
    xc2 = linear_act(xc2, n_hidden, n_causal_features_2, activation, weight_clip)
    xc1 = linear_act(xc2, n_causal_features_2, n_hidden, activation, weight_clip)
    xc1 = linear_act(xc1, n_hidden, n_causal_features_1, activation, weight_clip)
    yc = linear_act(xc1, n_causal_features_1, n_hidden, activation, weight_clip)
    yc = yc @ np.random.randn(n_hidden, 1)

    # ====== Spurious path ======
    if spurious_mode == 'flat':
        xs2 = linear_act(s, n_latent, n_hidden, activation, weight_clip)
        xs2 = linear_act(xs2, n_hidden, n_spurious_features_2, activation, weight_clip)
        xs1 = linear_act(s, n_latent, n_hidden, activation, weight_clip)
        xs1 = linear_act(xs1, n_hidden, n_spurious_features_1, activation, weight_clip)

        ys = linear_act(s, n_latent, n_hidden, activation, weight_clip)
        ys = ys @ np.random.randn(n_hidden, 1)
    elif spurious_mode == 'hrc':
        s2 = linear_act(s, n_latent, n_hidden, activation, weight_clip)
        s1 = linear_act(s2, n_hidden, n_hidden, activation, weight_clip)

        ys = linear_act(s1, n_hidden, n_hidden, activation, weight_clip)
        ys = ys @ np.random.randn(n_hidden, 1)

        xs2 = linear_act(s2, n_hidden, n_spurious_features_2, activation, weight_clip)
        xs1 = linear_act(s1, n_hidden, n_spurious_features_1, activation, weight_clip)

    elif spurious_mode == 'semi_hrc':
        xs2 = linear_act(s, n_latent, n_spurious_features_2, activation, weight_clip)

        s1 = linear_act(s, n_latent, n_hidden, activation, weight_clip)

        xs1 = linear_act(s1, n_hidden, n_spurious_features_1, activation, weight_clip)
        ys = linear_act(s1, n_hidden, n_hidden, activation, weight_clip)
        ys = ys @ np.random.randn(n_hidden, 1)

    else:
        raise ValueError(f"Unsupported spurious_mode: {spurious_mode}")

    # Combine yc and ys into final output
    yc = (yc - yc.mean()) / yc.std()
    ys = (ys - ys.mean()) / ys.std()
    # yc = yc / np.abs(yc).max()
    # ys = ys / np.abs(ys).max()

    Y = causal_strength * yc + (1 - causal_strength) * ys

    # ====== Assemble feature matrix ======
    layer1 = np.hstack([xc1, xs1])
    layer2 = np.hstack([xc2, xs2])
    full_data = np.hstack([layer1, layer2])

    # ====== Simulate single-cell data ======
    if simulate_single_cell:
        if mu_transform == 'softplus':
            mu = np.log1p(np.exp(full_data))
        elif mu_transform == 'exp':
            mu = np.exp(np.clip(full_data, a_min=-10, a_max=10))
        else:
            raise ValueError(f"Unsupported mu_transform: {mu_transform}")

        mu = np.clip(mu, 1e-3, 1e3)
        r = nb_shape
        if dist == 'zinb':
            prob_zero = np.random.uniform(0, 1, size=mu.shape) < p_zero
            nb_data = np.random.negative_binomial(r, r / (mu + r))
            full_data = np.where(prob_zero, 0, nb_data)
        elif dist == 'nb':
            full_data = np.random.negative_binomial(r, r / (mu + r))
    else:
        noise = np.random.laplace(scale=noise_scale, size=full_data.shape)
        full_data += noise
        if normalization == 'zscore':
            full_data = zscore(full_data, axis=0)
        elif normalization == 'minmax':
            full_data = MinMaxScaler().fit_transform(full_data)
        elif normalization == 'zscore_minmax':
            full_data = MinMaxScaler().fit_transform(zscore(full_data, axis=0))

    # ====== Label assignment ======
    sorted_indices = np.argsort(Y.ravel())
    positive_idx = sorted_indices[-pos_samples:]
    negative_idx = sorted_indices[:neg_samples]
    selected_idx = np.concatenate([positive_idx, negative_idx])
    labels = np.full(total_samples, -1, dtype=int)
    labels[positive_idx] = 1
    labels[negative_idx] = 0

    # ====== Create AnnData ======
    adata = AnnData(full_data[selected_idx])
    adata.obs["labels"] = labels[selected_idx]

    layer1_names = [f"layer1_f{i + 1}" for i in range(layer1.shape[1])]
    layer2_names = [f"layer2_f{i + 1}" for i in range(layer2.shape[1])]
    feature_names = layer1_names + layer2_names
    feature_types = (
            ['causal1'] * n_causal_features_1 +
            ['spurious1'] * n_spurious_features_1 +
            ['causal2'] * n_causal_features_2 +
            ['spurious2'] * n_spurious_features_2
    )

    adata.var.index = feature_names
    adata.var["feat_type"] = feature_types
    adata.var["layer"] = ["layer1"] * len(layer1_names) + ["layer2"] * len(layer2_names)
    adata.var["is_causal"] = [1 if "causal" in ft else 0 for ft in feature_types]
    adata.obsm["layer1"] = adata.X[:, :layer1.shape[1]]
    adata.obsm["layer2"] = adata.X[:, layer1.shape[1]:]

    print("adata max, mean, min: ", adata.X.max(), adata.X.mean(), adata.X.min())
    return adata


def generate_three_layer_synthetic_data(
        n_samples=200,
        oversampling_factor=5,
        n_causal_features_1=20,
        n_causal_features_2=20,
        n_causal_features_3=10,
        n_spurious_features_1=80,
        n_spurious_features_2=80,
        n_spurious_features_3=40,
        n_hidden=10,
        n_latent=5,
        noise_scale=0.1,
        causal_strength=0.2,
        spurious_mode='hrc',  # Options: 'flat', 'hrc', 'semi_hrc'
        simulate_single_cell=True,
        dist='zinb',
        nb_shape=10.0,
        p_zero=0.1,
        activation='linear',
        normalization='zscore',
        mu_transform='softplus',
        seed=42,
        weight_clip=1,
):
    np.random.seed(seed)
    total_samples = n_samples * oversampling_factor
    pos_samples = n_samples // 2
    neg_samples = n_samples - pos_samples

    # Input latent sources
    c = np.clip(np.random.randn(total_samples, n_latent), -2, 2)
    s = np.clip(np.random.randn(total_samples, n_latent), -2, 2)

    # ====== Causal path ======
    xc3 = linear_act(c, n_latent, n_hidden, activation, weight_clip)
    xc3 = linear_act(xc3, n_hidden, n_causal_features_3, activation, weight_clip)

    xc2 = linear_act(xc3, n_causal_features_3, n_hidden, activation, weight_clip)
    xc2 = linear_act(xc2, n_hidden, n_causal_features_2, activation, weight_clip)

    xc1 = linear_act(xc2, n_causal_features_2, n_hidden, activation, weight_clip)
    xc1 = linear_act(xc1, n_hidden, n_causal_features_1, activation, weight_clip)

    yc = linear_act(xc1, n_causal_features_1, n_hidden, activation, weight_clip)
    yc = yc @ np.random.randn(n_hidden, 1)

    # ====== Spurious path ======
    if spurious_mode == 'flat':
        xs3 = linear_act(s, n_latent, n_hidden, activation, weight_clip)
        xs3 = linear_act(xs3, n_hidden, n_spurious_features_3, activation, weight_clip)

        xs2 = linear_act(s, n_latent, n_hidden, activation, weight_clip)
        xs2 = linear_act(xs2, n_hidden, n_spurious_features_2, activation, weight_clip)

        xs1 = linear_act(s, n_latent, n_hidden, activation, weight_clip)
        xs1 = linear_act(xs1, n_hidden, n_spurious_features_1, activation, weight_clip)

        ys = linear_act(s, n_latent, n_hidden, activation, weight_clip)
        ys = ys @ np.random.randn(n_hidden, 1)

    elif spurious_mode == 'hrc':
        s3 = linear_act(s, n_latent, n_hidden, activation, weight_clip)
        s2 = linear_act(s3, n_hidden, n_hidden, activation, weight_clip)
        s1 = linear_act(s2, n_hidden, n_hidden, activation, weight_clip)

        xs3 = linear_act(s3, n_hidden, n_spurious_features_3, activation, weight_clip)
        xs2 = linear_act(s2, n_hidden, n_spurious_features_2, activation, weight_clip)
        xs1 = linear_act(s1, n_hidden, n_spurious_features_1, activation, weight_clip)

        ys = linear_act(s1, n_hidden, n_hidden, activation, weight_clip)
        ys = ys @ np.random.randn(n_hidden, 1)

    elif spurious_mode == 'semi_hrc':
        xs3 = linear_act(s, n_latent, n_spurious_features_3, activation, weight_clip)
        xs2 = linear_act(s, n_latent, n_spurious_features_2, activation, weight_clip)

        s1 = linear_act(s, n_latent, n_hidden, activation, weight_clip)
        xs1 = linear_act(s1, n_hidden, n_spurious_features_1, activation, weight_clip)

        ys = linear_act(s1, n_hidden, n_hidden, activation, weight_clip)
        ys = ys @ np.random.randn(n_hidden, 1)

    else:
        raise ValueError(f"Unsupported spurious_mode: {spurious_mode}")

    # Combine yc and ys into final output
    yc = (yc - yc.mean()) / yc.std()
    ys = (ys - ys.mean()) / ys.std()
    Y = causal_strength * yc + (1 - causal_strength) * ys

    # Assemble feature matrix
    layer1 = np.hstack([xc1, xs1])
    layer2 = np.hstack([xc2, xs2])
    layer3 = np.hstack([xc3, xs3])
    full_data = np.hstack([layer1, layer2, layer3])

    # Simulate single-cell data
    if simulate_single_cell:
        if mu_transform == 'softplus':
            mu = np.log1p(np.exp(full_data))
        elif mu_transform == 'exp':
            mu = np.exp(np.clip(full_data, a_min=-10, a_max=10))
        else:
            raise ValueError(f"Unsupported mu_transform: {mu_transform}")
        mu = np.clip(mu, 1e-3, 1e3)
        r = nb_shape
        if dist == 'zinb':
            prob_zero = np.random.uniform(0, 1, size=mu.shape) < p_zero
            nb_data = np.random.negative_binomial(r, r / (mu + r))
            full_data = np.where(prob_zero, 0, nb_data)
        elif dist == 'nb':
            full_data = np.random.negative_binomial(r, r / (mu + r))
    else:
        noise = np.random.laplace(scale=noise_scale, size=full_data.shape)
        full_data += noise
        if normalization == 'zscore':
            full_data = zscore(full_data, axis=0)
        elif normalization == 'minmax':
            full_data = MinMaxScaler().fit_transform(full_data)
        elif normalization == 'zscore_minmax':
            full_data = MinMaxScaler().fit_transform(zscore(full_data, axis=0))

    # Label assignment
    sorted_indices = np.argsort(Y.ravel())
    positive_idx = sorted_indices[-pos_samples:]
    negative_idx = sorted_indices[:neg_samples]
    selected_idx = np.concatenate([positive_idx, negative_idx])
    labels = np.full(total_samples, -1, dtype=int)
    labels[positive_idx] = 1
    labels[negative_idx] = 0

    # Create AnnData
    adata = AnnData(full_data[selected_idx])
    adata.obs["labels"] = labels[selected_idx]

    layer1_names = [f"layer1_f{i + 1}" for i in range(layer1.shape[1])]
    layer2_names = [f"layer2_f{i + 1}" for i in range(layer2.shape[1])]
    layer3_names = [f"layer3_f{i + 1}" for i in range(layer3.shape[1])]
    feature_names = layer1_names + layer2_names + layer3_names

    feature_types = (
        ['causal1'] * n_causal_features_1 +
        ['spurious1'] * n_spurious_features_1 +
        ['causal2'] * n_causal_features_2 +
        ['spurious2'] * n_spurious_features_2 +
        ['causal3'] * n_causal_features_3 +
        ['spurious3'] * n_spurious_features_3
    )

    adata.var.index = feature_names
    adata.var["feat_type"] = feature_types
    adata.var["layer"] = (
        ["layer1"] * len(layer1_names) +
        ["layer2"] * len(layer2_names) +
        ["layer3"] * len(layer3_names)
    )
    adata.var["is_causal"] = [1 if "causal" in ft else 0 for ft in feature_types]

    adata.obsm["layer1"] = adata.X[:, :layer1.shape[1]]
    adata.obsm["layer2"] = adata.X[:, layer1.shape[1]:layer1.shape[1]+layer2.shape[1]]
    adata.obsm["layer3"] = adata.X[:, -layer3.shape[1]:]

    print("adata max, mean, min: ", adata.X.max(), adata.X.mean(), adata.X.min())
    return adata


def run_ct_2l_known(adata, output_dir=None, mode="SHAP", topk=20, topk_ratio=None, is_log1p=False, full_input=False):
    """
    Run CauTrigger in two steps:
    Step 1: Use CauTrigger1L on layer1 (closer to Y) to get top-k causal genes.
    Step 2: Use those genes as X_down in CauTrigger2L to score layer2 (upstream).
    Return:
        dict {
            "layer1": DataFrame with scores (index = layer1 gene names),
            "layer2": DataFrame with scores (index = layer2 gene names)
        }
    """
    np.random.seed(42)
    if is_log1p:
        print("CauTrigger log1p normalization applied to adata.X, adata.obsm['layer1'], and adata.obsm['layer2']")
        adata.X = np.log1p(adata.X)
        adata.obsm["layer1"] = np.log1p(adata.obsm["layer1"])
        adata.obsm["layer2"] = np.log1p(adata.obsm["layer2"])
        adata._log1p_applied = True
        print("[INFO] log1p transformation applied to adata.obsm['layer1'] and ['layer2'].")

    # === Step 1: layer1 → downstream (closer to Y) ===
    layer1_vars = adata.var_names[adata.var["layer"] == "layer1"]
    adata_layer1 = AnnData(
        X=adata.obsm["layer1"],
        obs=adata.obs.copy(),
        var=adata.var.loc[layer1_vars].copy()
    )

    model_1L = CauTrigger1L(
        adata_layer1,
        n_latent=10,
        n_hidden=128,
        n_layers_encoder=0,
        n_layers_decoder=0,
        n_layers_dpd=0,
        dropout_rate_encoder=0.0,
        dropout_rate_decoder=0.0,
        dropout_rate_dpd=0.0,
        use_batch_norm='none',
        use_batch_norm_dpd=True,
        decoder_linear=False,
        dpd_linear=True,
        init_weight=None,
        init_thresh=0.4,
        attention=False,
        att_mean=False,
        # zs_to_dpd=True,  # Use zs to predict dpd
    )
    model_1L.train(max_epochs=200, stage_training=True, weight_scheme="sim")
    df_layer1, _ = model_1L.get_up_feature_weights(method=mode, normalize=False, sort_by_weight=True)

    print("df_layer1", df_layer1.head(20))

    # === Step 2: layer2 → upstream, use layer1 top-k as X_down ===
    layer2_vars = adata.var_names[adata.var["layer"] == "layer2"]
    df_layer1 = df_layer1.loc[layer1_vars]  # ensure order matches obsm["layer1"]

    if topk_ratio is not None:
        sorted_weights = df_layer1["weight"].sort_values(ascending=False)
        cumulative = sorted_weights.cumsum() / sorted_weights.sum()
        selected = cumulative[cumulative < topk_ratio].index
        print(f"[INFO] Selected {len(selected)} features by cumulative ratio {topk_ratio}")
        X_down = adata.obsm["layer1"] if full_input else adata.obsm["layer1"][:,
                                                         [df_layer1.index.get_loc(i) for i in selected]]
    else:
        topk_indices = df_layer1["weight"].values.argsort()[-topk:]
        X_down = adata.obsm["layer1"] if full_input else adata.obsm["layer1"][:, topk_indices]

    adata_layer2 = AnnData(
        X=adata.obsm["layer2"],
        obs=adata.obs.copy(),
        var=adata.var.loc[layer2_vars].copy(),
        obsm={"X_down": X_down}
    )

    model_2L = CauTrigger2L(
        adata_layer2,
        n_latent=10,
        n_hidden=128,
        n_layers_encoder=0,
        n_layers_decoder=0,
        n_layers_dpd=0,
        dropout_rate_encoder=0.0,
        dropout_rate_decoder=0.0,
        dropout_rate_dpd=0.0,
        use_batch_norm='none',
        use_batch_norm_dpd=True,
        decoder_linear=False,
        dpd_linear=True,
        init_weight=None,
        init_thresh=0.4,
        attention=False,
        att_mean=False,
    )
    model_2L.train(max_epochs=200, stage_training=True, weight_scheme="sim")
    df_layer2, _ = model_2L.get_up_feature_weights(method=mode, normalize=False, sort_by_weight=True)
    print("df_layer2", df_layer2.head(10))
    df_layer2 = df_layer2.loc[layer2_vars]

    # Set correct index for both outputs
    assert df_layer1.index.equals(layer1_vars)
    assert df_layer2.index.equals(layer2_vars)

    return {
        "layer1": df_layer1,
        "layer2": df_layer2,
    }


def run_ct_2l_unknown(adata, output_dir=None, mode="SHAP", topk=30, topk_ratio=None, is_log1p=False, full_input=False):
    """
    Run CauTrigger in two steps:
    Step 1: Use CauTrigger1L on layer1 (closer to Y) to get top-k causal genes.
    Step 2: Use those genes as X_down in CauTrigger2L to score layer2 (upstream).
    Return:
        dict {
            "layer1": DataFrame with scores (index = layer1 gene names),
            "layer2": DataFrame with scores (index = layer2 gene names)
        }
    """
    np.random.seed(42)
    if is_log1p:
        print("CauTrigger log1p normalization applied to adata.X, adata.obsm['layer1'], and adata.obsm['layer2']")
        adata.X = np.log1p(adata.X)
        adata.obsm["layer1"] = np.log1p(adata.obsm["layer1"])
        adata.obsm["layer2"] = np.log1p(adata.obsm["layer2"])
        adata._log1p_applied = True
        print("[INFO] log1p transformation applied to adata.obsm['layer1'] and ['layer2'].")

    # === Step 1: Score all genes ===
    adata_step1 = adata.copy()

    model_1L = CauTrigger1L(
        adata_step1,
        n_latent=10,
        n_hidden=128,
        n_layers_encoder=0,
        n_layers_decoder=0,
        n_layers_dpd=0,
        dropout_rate_encoder=0.0,
        dropout_rate_decoder=0.0,
        dropout_rate_dpd=0.0,
        use_batch_norm='none',
        use_batch_norm_dpd=True,
        decoder_linear=False,
        dpd_linear=True,
        init_weight=None,
        init_thresh=0.4,
        attention=False,
        att_mean=False,
    )
    model_1L.train(max_epochs=200, stage_training=True, weight_scheme="sim")
    df_step1, _ = model_1L.get_up_feature_weights(method=mode, normalize=False, sort_by_weight=True)
    print("df_step1", df_step1.head(20))

    # Step 2: select top-k genes from step1 as X_down
    if topk_ratio is not None:
        sorted_weights = df_step1["weight"].sort_values(ascending=False)
        cumulative = sorted_weights.cumsum() / sorted_weights.sum()
        selected = cumulative[cumulative <= topk_ratio].index.tolist()
        topk_genes = selected
        print(f"[INFO] Selected {len(topk_genes)} genes by cumulative ratio {topk_ratio}")
    else:
        topk_genes = df_step1["weight"].nlargest(topk).index.tolist()
    all_genes = adata.var_names.tolist()
    remaining_genes = [g for g in all_genes if g not in topk_genes]

    # construct X_down and X_upstream
    X_down = adata[:, topk_genes].X
    X_upstream = adata[:, remaining_genes].X

    # construct step2 adata
    adata_step2 = AnnData(
        X=X_upstream,
        obs=adata.obs.copy(),
        var=adata.var.loc[remaining_genes].copy(),
        obsm={"X_down": X_down if not full_input else adata.X}
    )

    model_2L = CauTrigger2L(
        adata_step2,
        n_latent=10,
        n_hidden=128,
        n_layers_encoder=0,
        n_layers_decoder=0,
        n_layers_dpd=0,
        dropout_rate_encoder=0.0,
        dropout_rate_decoder=0.0,
        dropout_rate_dpd=0.0,
        use_batch_norm='none',
        use_batch_norm_dpd=True,
        decoder_linear=False,
        dpd_linear=True,
        init_weight=None,
        init_thresh=0.1,
        attention=False,
        att_mean=False,
    )
    model_2L.train(max_epochs=200, stage_training=True, weight_scheme="sim")
    df_step2, _ = model_2L.get_up_feature_weights(method=mode, normalize=False, sort_by_weight=True)
    print("df_step2", df_step2.head(10))

    # Mark the estimation step
    df_step1["step"] = "step1"
    df_step2["step"] = "step2"

    # Amplify only top-k weights from step1
    amplify_factor = 2.0
    df_step1_topk = df_step1.loc[topk_genes].copy()
    df_step1_topk["weight"] *= amplify_factor

    # Concatenate step1_topk and all step2 genes (unaltered)
    df_all = pd.concat([df_step1_topk, df_step2])

    # Add ground truth causal label
    df_all["is_causal"] = adata.var.loc[df_all.index, "is_causal"]

    # Assign step labels
    df_all["step"] = ["step1" if g in topk_genes else "step2" for g in df_all.index]

    # Keep selected columns and align with original gene order
    df_all = df_all[["weight", "is_causal", "step"]]
    df_all = df_all.reindex(adata.var_names)  # Ensure consistent order

    # === Construct step1 and step2 full tables for separate layer evaluation ===
    df_step1_full = pd.DataFrame(index=adata.var_names)
    df_step1_full["weight"] = 0.0
    df_step1_full.loc[df_step1.index, "weight"] = df_step1["weight"]
    df_step1_full["step"] = "step1"
    df_step1_full["is_causal"] = 0
    layer1_genes = adata.var_names[adata.var["layer"] == "layer1"]
    df_step1_full.loc[layer1_genes, "is_causal"] = adata.var.loc[layer1_genes, "is_causal"]

    df_step2_full = pd.DataFrame(index=adata.var_names)
    df_step2_full["weight"] = 0.0
    df_step2_full.loc[df_step2.index, "weight"] = df_step2["weight"]
    df_step2_full["step"] = "step2"
    df_step2_full["is_causal"] = 0
    layer2_genes = adata.var_names[adata.var["layer"] == "layer2"]
    df_step2_full.loc[layer2_genes, "is_causal"] = adata.var.loc[layer2_genes, "is_causal"]

    # Return all outputs
    return {
        "step1": df_step1_full,
        "step2": df_step2_full,
        "all": df_all
    }


def run_ct_3l_known(adata, output_dir=None, mode="SHAP", topk=30, topk_ratio=None, is_log1p=False, full_input=False):
    """
    Run CauTrigger in two steps:
    Step 1: Use CauTrigger1L on layer1 (closer to Y) to get top-k causal genes.
    Step 2: Use those genes as X_down in CauTrigger2L to score layer2 (upstream).
    Step 3: Use top-k from layer2 as X_down in CauTrigger3L to score layer3 (upstream).
    Return:
        dict {
            "layer1": DataFrame with scores (index = layer1 gene names),
            "layer2": DataFrame with scores (index = layer2 gene names)
        }
    """
    np.random.seed(42)
    if is_log1p:
        print("CauTrigger log1p normalization applied to adata.X, adata.obsm['layer1'], and adata.obsm['layer2']")
        adata.X = np.log1p(adata.X)
        adata.obsm["layer1"] = np.log1p(adata.obsm["layer1"])
        adata.obsm["layer2"] = np.log1p(adata.obsm["layer2"])
        adata.obsm["layer3"] = np.log1p(adata.obsm["layer3"])
        adata._log1p_applied = True
        print("[INFO] log1p transformation applied to adata.obsm['layer1'] ['layer2'] and ['layer3'].")

    # === Step 1: layer1 → downstream (closer to Y) ===
    layer1_vars = adata.var_names[adata.var["layer"] == "layer1"]
    adata_layer1 = AnnData(
        X=adata.obsm["layer1"],
        obs=adata.obs.copy(),
        var=adata.var.loc[layer1_vars].copy()
    )

    model_1L = CauTrigger1L(
        adata_layer1,
        n_latent=10,
        n_hidden=128,
        n_layers_encoder=0,
        n_layers_decoder=0,
        n_layers_dpd=0,
        dropout_rate_encoder=0.0,
        dropout_rate_decoder=0.0,
        dropout_rate_dpd=0.0,
        use_batch_norm='none',
        use_batch_norm_dpd=True,
        decoder_linear=False,
        dpd_linear=True,
        init_weight=None,
        init_thresh=0.4,
        attention=False,
        att_mean=False,
    )
    model_1L.train(max_epochs=200, stage_training=True, weight_scheme="sim")
    df_layer1, _ = model_1L.get_up_feature_weights(method=mode, normalize=False, sort_by_weight=True)
    print("df_layer1", df_layer1.head(20))

    # === Step 2: layer2 → upstream, use layer1 top-k as X_down ===
    df_layer1 = df_layer1.loc[layer1_vars]
    if topk_ratio is not None:
        sorted_weights = df_layer1["weight"].sort_values(ascending=False)
        cumulative = sorted_weights.cumsum() / sorted_weights.sum()
        selected1 = cumulative[cumulative < topk_ratio].index
        print(f"[INFO] Selected {len(selected1)} layer1 features by cumulative ratio {topk_ratio}")
        X_down = adata.obsm["layer1"] if full_input else adata.obsm["layer1"][
            :, [df_layer1.index.get_loc(i) for i in selected1]]
    else:
        topk_genes_layer1 = df_layer1.sort_values("weight", ascending=False).index[:topk]
        topk_indices_layer1 = pd.Index(layer1_vars).get_indexer(topk_genes_layer1)
        X_down = adata.obsm["layer1"] if full_input else adata.obsm["layer1"][:, topk_indices_layer1]

    layer2_vars = adata.var_names[adata.var["layer"] == "layer2"]

    adata_layer2 = AnnData(
        X=adata.obsm["layer2"],
        obs=adata.obs.copy(),
        var=adata.var.loc[layer2_vars].copy(),
        obsm={"X_down": X_down}
    )

    model_2L = CauTrigger2L(
        adata_layer2,
        n_latent=10,
        n_hidden=128,
        n_layers_encoder=0,
        n_layers_decoder=0,
        n_layers_dpd=0,
        dropout_rate_encoder=0.0,
        dropout_rate_decoder=0.0,
        dropout_rate_dpd=0.0,
        use_batch_norm='none',
        use_batch_norm_dpd=True,
        decoder_linear=False,
        dpd_linear=True,
        init_weight=None,
        init_thresh=0.4,
        attention=False,
        att_mean=False,
    )
    model_2L.train(max_epochs=200, stage_training=True, weight_scheme="sim")
    df_layer2, _ = model_2L.get_up_feature_weights(method=mode, normalize=False, sort_by_weight=True)
    print("df_layer2", df_layer2.head(20))

    # === Step 3: layer3 → top-k from layer2 ===
    df_layer2 = df_layer2.loc[layer2_vars]
    if topk_ratio is not None:
        sorted_weights2 = df_layer2["weight"].sort_values(ascending=False)
        cumulative2 = sorted_weights2.cumsum() / sorted_weights2.sum()
        selected2 = cumulative2[cumulative2 < topk_ratio].index
        print(f"[INFO] Selected {len(selected2)} layer2 features by cumulative ratio {topk_ratio}")
        topk_indices_layer2 = [df_layer2.index.get_loc(i) for i in selected2]
    else:
        topk_genes_layer2 = df_layer2.sort_values("weight", ascending=False).index[:topk]
        topk_indices_layer2 = pd.Index(layer2_vars).get_indexer(topk_genes_layer2)

    topk_genes_layer1 = df_layer1.sort_values("weight", ascending=False).index[:topk]
    topk_indices_layer1 = pd.Index(layer1_vars).get_indexer(topk_genes_layer1)

    if topk_ratio is not None:
        X_down1 = adata.obsm["layer1"] if full_input else adata.obsm["layer1"][
            :, [df_layer1.index.get_loc(i) for i in selected1]]
    else:
        X_down1 = adata.obsm["layer1"] if full_input else adata.obsm["layer1"][:, topk_indices_layer1]

    X_down2 = adata.obsm["layer2"] if full_input else adata.obsm["layer2"][:, topk_indices_layer2]

    layer3_vars = adata.var_names[adata.var["layer"] == "layer3"]

    adata_layer3 = AnnData(
        X=adata.obsm["layer3"],
        obs=adata.obs.copy(),
        var=adata.var.loc[layer3_vars].copy(),
        obsm={
            "X_down1": X_down1,
            "X_down2": X_down2
        }
    )

    model_3L = CauTrigger3L(
        adata_layer3,
        n_latent=10,
        n_hidden=128,
        n_layers_encoder=0,
        n_layers_decoder=0,
        n_layers_dpd=0,
        dropout_rate_encoder=0.0,
        dropout_rate_decoder=0.0,
        dropout_rate_dpd=0.0,
        use_batch_norm='none',
        use_batch_norm_dpd=True,
        decoder_linear=False,
        dpd_linear=True,
        init_weight=None,
        init_thresh=0.4,
        attention=False,
        att_mean=False,
    )
    model_3L.train(max_epochs=200, stage_training=True, weight_scheme="sim")
    df_layer3, _ = model_3L.get_up_feature_weights(method=mode, normalize=False, sort_by_weight=True)
    print("df_layer3", df_layer3.head(10))
    df_layer3 = df_layer3.loc[layer3_vars]

    return {
        "layer1": df_layer1,
        "layer2": df_layer2,
        "layer3": df_layer3
    }


def run_ct_3l_unknown(adata, output_dir=None, mode="SHAP", topk=30, topk_ratio=None, is_log1p=False, full_input=False):
    """
    Run CauTrigger in two steps:
    Step 1: Use CauTrigger1L on layer1 (closer to Y) to get top-k causal genes.
    Step 2: Use those genes as X_down in CauTrigger2L to score layer2 (upstream).
    Return:
        dict {
            "layer1": DataFrame with scores (index = layer1 gene names),
            "layer2": DataFrame with scores (index = layer2 gene names)
        }
    """
    np.random.seed(42)
    if is_log1p:
        print("CauTrigger log1p normalization applied to adata.X, adata.obsm['layer1'], and adata.obsm['layer2']")
        adata.X = np.log1p(adata.X)
        adata.obsm["layer1"] = np.log1p(adata.obsm["layer1"])
        adata.obsm["layer2"] = np.log1p(adata.obsm["layer2"])
        adata._log1p_applied = True
        print("[INFO] log1p transformation applied to adata.obsm['layer1'] and ['layer2'].")

    # === Step 1: Score all genes ===
    adata_step1 = adata.copy()

    model_1L = CauTrigger1L(
        adata_step1,
        n_latent=10,
        n_hidden=128,
        n_layers_encoder=0,
        n_layers_decoder=0,
        n_layers_dpd=0,
        dropout_rate_encoder=0.0,
        dropout_rate_decoder=0.0,
        dropout_rate_dpd=0.0,
        use_batch_norm='none',
        use_batch_norm_dpd=True,
        decoder_linear=False,
        dpd_linear=True,
        init_weight=None,
        init_thresh=0.4,
        attention=False,
        att_mean=False,
    )
    model_1L.train(max_epochs=200, stage_training=True, weight_scheme="sim")
    df_step1, _ = model_1L.get_up_feature_weights(method=mode, normalize=False, sort_by_weight=True)
    print("df_step1", df_step1.head(20))

    # Step 2: select top-k genes from step1 as X_down
    if topk_ratio is not None:
        sorted_weights = df_step1["weight"].sort_values(ascending=False)
        cumulative = sorted_weights.cumsum() / sorted_weights.sum()
        down_genes_2L = cumulative[cumulative <= topk_ratio].index.tolist()
    else:
        down_genes_2L = df_step1["weight"].nlargest(topk).index.tolist()

    up_genes_2L = [g for g in adata.var_names if g not in down_genes_2L]

    # 构建 X_down 和 X_upstream
    X_down_2L = adata[:, down_genes_2L].X
    X_up_2L = adata[:, up_genes_2L].X

    # 构建 step2 的 AnnData
    adata_step2 = AnnData(
        X=X_up_2L,
        obs=adata.obs.copy(),
        var=adata.var.loc[up_genes_2L].copy(),
        obsm={"X_down": X_down_2L if not full_input else adata.X}
    )

    model_2L = CauTrigger2L(
        adata_step2,
        n_latent=10,
        n_hidden=128,
        n_layers_encoder=0,
        n_layers_decoder=0,
        n_layers_dpd=0,
        dropout_rate_encoder=0.0,
        dropout_rate_decoder=0.0,
        dropout_rate_dpd=0.0,
        use_batch_norm='none',
        use_batch_norm_dpd=True,
        decoder_linear=False,
        dpd_linear=True,
        init_weight=None,
        init_thresh=0.1,
        attention=False,
        att_mean=False,
    )
    model_2L.train(max_epochs=200, stage_training=True, weight_scheme="sim")
    df_step2, _ = model_2L.get_up_feature_weights(method=mode, normalize=False, sort_by_weight=True)
    print("df_step2", df_step2.head(20))

    # === Step 3: Use cumulative ratio if provided ===
    if topk_ratio is not None:
        sorted_weights1 = df_step1["weight"].sort_values(ascending=False)
        cumulative1 = sorted_weights1.cumsum() / sorted_weights1.sum()
        down1_genes_3L = cumulative1[cumulative1 <= topk_ratio].index.tolist()

        sorted_weights2 = df_step2["weight"].sort_values(ascending=False)
        cumulative2 = sorted_weights2.cumsum() / sorted_weights2.sum()
        down2_genes_3L = cumulative2[cumulative2 <= topk_ratio].index.tolist()
    else:
        down1_genes_3L = df_step1["weight"].nlargest(topk).index.tolist()
        down2_genes_3L = df_step2["weight"].nlargest(topk).index.tolist()

    down_genes_3L = down1_genes_3L + down2_genes_3L
    up_genes_3L = [g for g in adata.var_names if g not in down_genes_3L]

    X_down1_3L = adata[:, down1_genes_3L].X
    X_down2_3L = adata[:, down2_genes_3L].X
    X_up_3L = adata[:, up_genes_3L].X

    adata_step3 = AnnData(
        X=X_up_3L,
        obs=adata.obs.copy(),
        var=adata.var.loc[up_genes_3L].copy(),
        obsm={
            "X_down1": X_down1_3L if not full_input else adata.X,
            "X_down2": X_down2_3L if not full_input else adata.X
        }
    )

    model_3L = CauTrigger3L(
        adata_step3,
        n_latent=10,
        n_hidden=128,
        n_layers_encoder=0,
        n_layers_decoder=0,
        n_layers_dpd=0,
        dropout_rate_encoder=0.0,
        dropout_rate_decoder=0.0,
        dropout_rate_dpd=0.0,
        use_batch_norm='none',
        use_batch_norm_dpd=True,
        decoder_linear=False,
        dpd_linear=True,
        init_weight=None,
        init_thresh=0.4,
        attention=False,
        att_mean=False,
    )
    model_3L.train(max_epochs=200, stage_training=True, weight_scheme="sim")
    df_step3, _ = model_3L.get_up_feature_weights(method=mode, normalize=False, sort_by_weight=True)
    print("df_step3", df_step3.head(30))

    # === Merge and assign weights ===
    df_step1_topk = df_step1.loc[down1_genes_3L].copy()
    df_step2_topk = df_step2.loc[down2_genes_3L].copy()
    df_step1_topk["weight"] *= 2.0
    df_step2_topk["weight"] *= 1.5

    df_all = pd.concat([df_step1_topk, df_step2_topk, df_step3])
    df_all["is_causal"] = adata.var.loc[df_all.index, "is_causal"]
    df_all["step"] = ["step1" if g in down1_genes_3L else
                      "step2" if g in down2_genes_3L else
                      "step3" for g in df_all.index]
    df_all = df_all[["weight", "is_causal", "step"]]
    df_all = df_all.reindex(adata.var_names)

    # === Create stepX full table ===
    df_step1_full = pd.DataFrame(index=adata.var_names)
    df_step1_full["weight"] = 0.0
    df_step1_full.loc[df_step1.index, "weight"] = df_step1["weight"]
    df_step1_full["step"] = "step1"
    df_step1_full["is_causal"] = 0
    layer1_genes = adata.var_names[adata.var["layer"] == "layer1"]
    df_step1_full.loc[layer1_genes, "is_causal"] = adata.var.loc[layer1_genes, "is_causal"]
    # df_step1_full = df_step1_full.loc[layer1_genes, :]

    df_step2_full = pd.DataFrame(index=adata.var_names)
    df_step2_full["weight"] = 0.0
    df_step2_full.loc[df_step2.index, "weight"] = df_step2["weight"]
    df_step2_full["step"] = "step2"
    df_step2_full["is_causal"] = 0
    layer2_genes = adata.var_names[adata.var["layer"] == "layer2"]
    df_step2_full.loc[layer2_genes, "is_causal"] = adata.var.loc[layer2_genes, "is_causal"]
    # df_step2_full = df_step2_full.loc[layer2_genes, :]

    df_step3_full = pd.DataFrame(index=adata.var_names)
    df_step3_full["weight"] = 0.0
    df_step3_full.loc[df_step3.index, "weight"] = df_step3["weight"]
    df_step3_full["step"] = "step3"
    df_step3_full["is_causal"] = 0
    layer3_genes = adata.var_names[adata.var["layer"] == "layer3"]
    df_step3_full.loc[layer3_genes, "is_causal"] = adata.var.loc[layer3_genes, "is_causal"]
    # df_step3_full = df_step3_full.loc[layer3_genes, :]

    return {
        "step1": df_step1_full,
        "step2": df_step2_full,
        "step3": df_step3_full,
        "all": df_all
    }


def run_PC(adata, output_dir):
    def gauss_ci_test(suff_stat, i, j, K):
        corr_matrix = suff_stat["C"]
        n_samples = suff_stat["n"]

        if len(K) == 0:
            r = corr_matrix[i, j]
        elif len(K) == 1:
            k = K[0]
            r = (corr_matrix[i, j] - corr_matrix[i, k] * corr_matrix[j, k]) / math.sqrt(
                (1 - corr_matrix[i, k] ** 2) * (1 - corr_matrix[j, k] ** 2)
            )
        else:
            sub_corr = corr_matrix[np.ix_([i, j] + K, [i, j] + K)]
            precision_matrix = np.linalg.pinv(sub_corr)
            r = (-1 * precision_matrix[0, 1]) / math.sqrt(
                abs(precision_matrix[0, 0] * precision_matrix[1, 1])
            )

        r = max(min(r, 0.99999), -0.99999)
        z = 0.5 * math.log1p((2 * r) / (1 - r))
        z_standard = z * math.sqrt(n_samples - len(K) - 3)
        p_value = 2 * (1 - norm.cdf(abs(z_standard)))

        return p_value

    def get_neighbors(G, x, exclude_y):
        return [i for i, connected in enumerate(G[x]) if connected and i != exclude_y]

    def skeleton(suff_stat, alpha):
        p_value_mat = np.zeros_like(suff_stat["C"])
        n_nodes = suff_stat["C"].shape[0]
        O = [[[] for _ in range(n_nodes)] for _ in range(n_nodes)]
        G = [[i != j for i in range(n_nodes)] for j in range(n_nodes)]
        pairs = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]

        done = False
        l = 0

        while not done and any(any(row) for row in G):
            done = True

            for x, y in pairs:
                if G[x][y]:
                    neighbors = get_neighbors(G, x, y)
                    if len(neighbors) >= l:
                        done = False
                        for K in combinations(neighbors, l):
                            p_value = gauss_ci_test(suff_stat, x, y, list(K))
                            if p_value > p_value_mat[x][y]:
                                p_value_mat[x][y] = p_value_mat[y][x] = p_value
                            if p_value >= alpha:
                                G[x][y] = G[y][x] = False
                                O[x][y] = O[y][x] = list(K)
                                break
            l += 1

        return np.asarray(G, dtype=int), O, p_value_mat

    def extend_cpdag(G, O):
        n_nodes = G.shape[0]

        def rule1(g):
            pairs = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if g[i][j] == 1 and g[j][i] == 0]
            for i, j in pairs:
                all_k = [k for k in range(n_nodes) if
                         (g[j][k] == 1 and g[k][j] == 1) and (g[i][k] == 0 and g[k][i] == 0)]
                for k in all_k:
                    g[j][k] = 1
                    g[k][j] = 0
            return g

        def rule2(g):
            pairs = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if g[i][j] == 1 and g[j][i] == 1]
            for i, j in pairs:
                all_k = [k for k in range(n_nodes) if
                         (g[i][k] == 1 and g[k][i] == 0) and (g[k][j] == 1 and g[j][k] == 0)]
                if len(all_k) > 0:
                    g[i][j] = 1
                    g[j][i] = 0
            return g

        def rule3(g):
            pairs = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if g[i][j] == 1 and g[j][i] == 1]
            for i, j in pairs:
                all_k = [k for k in range(n_nodes) if
                         (g[i][k] == 1 and g[k][i] == 1) and (g[k][j] == 1 and g[j][k] == 0)]
                if len(all_k) >= 2:
                    for k1, k2 in combinations(all_k, 2):
                        if g[k1][k2] == 0 and g[k2][k1] == 0:
                            g[i][j] = 1
                            g[j][i] = 0
                            break
            return g

        pairs = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if G[i][j] == 1]
        for x, y in sorted(pairs, key=lambda x: (x[1], x[0])):
            all_z = [z for z in range(n_nodes) if G[y][z] == 1 and z != x]
            for z in all_z:
                if G[x][z] == 0 and y not in O[x][z]:
                    G[x][y] = G[z][y] = 1
                    G[y][x] = G[y][z] = 0

        old_G = np.zeros((n_nodes, n_nodes))
        while not np.array_equal(old_G, G):
            old_G = G.copy()
            G = rule1(G)
            G = rule2(G)
            G = rule3(G)

        return np.array(G)

    def pc(suff_stat, alpha=0.5, verbose=False):
        G, O, pvm = skeleton(suff_stat, alpha)
        cpdag = extend_cpdag(G, O)
        if verbose:
            print(cpdag)
        return cpdag, pvm

    alpha = 0.05
    X = adata.X
    if np.issubdtype(X.dtype, np.integer) or X.max() > 100:  # Rough check for count data
        X = np.log1p(X)  # Apply log1p for count data
    y = adata.obs['labels'].values
    data = pd.DataFrame(np.column_stack((X, y)))
    cpdag, pvm = pc(
        suff_stat={"C": data.corr().values, "n": data.shape[0]},
        alpha=alpha
    )
    pv = pvm[:-1, -1]
    arr = np.array(1 - pv).reshape(1, -1)
    normalized_arr = sk_normalize(arr, norm='l1', axis=1)
    return (normalized_arr.flatten())


def run_VAE(adata, output_dir):

    X = adata.X
    if np.issubdtype(X.dtype, np.integer) or X.max() > 100:  # Rough check for count data
        X = np.log1p(X)  # Apply log1p for count data
    y = adata.obs['labels'].values
    n_features = X.shape[1]
    features = torch.tensor(X, dtype=torch.float32)
    labels = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    n_hidden = 5
    n_latent = 5

    class VAE(nn.Module):
        def __init__(self, num_features):
            super().__init__()

            self.encoder = nn.Sequential(
                nn.Linear(num_features, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, 2 * n_latent),
            )

            self.decoder = nn.Sequential(
                nn.Linear(n_latent, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, num_features),
                # nn.Sigmoid()
            )

            self.DPD = nn.Sequential(
                nn.Linear(n_latent, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, 1),
                nn.Sigmoid(),
            )

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu

        def forward(self, x):
            mu_logvar = self.encoder(x)
            mu = mu_logvar[:, :n_latent]
            logvar = mu_logvar[:, n_latent:]
            z = self.reparameterize(mu, logvar)
            y = self.DPD(z)
            reconstructed = self.decoder(z)
            return reconstructed, y, mu, logvar

    model = VAE(n_features)
    recon_criterion = nn.MSELoss()
    dpd_criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # train
    model.train()
    losses = []
    re_losses = []
    kl_losses = []
    dpd_losses = []
    for epoch in range(200):
        for data, targets in dataloader:
            optimizer.zero_grad()
            recon_batch, y_dpd, mu, logvar = model(data)
            # reconstructed loss
            re_loss = recon_criterion(recon_batch, data)
            re_losses.append(re_loss.item())

            # kl loss
            kl_loss = (
                -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.shape[0]
            )
            kl_losses.append(kl_loss.item())

            # dpd loss
            dpd_loss = dpd_criterion(y_dpd, targets)
            dpd_losses.append(dpd_loss.item())

            # total loss
            if epoch <= 100:
                loss = re_loss + kl_loss * 0.1 + dpd_loss * 0.1
            else:
                loss = re_loss + kl_loss * 0.1 + dpd_loss * 0.1

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    model.eval()

    # Grad
    features.requires_grad = True
    _, y_prob, _, _ = model(features)
    loss = dpd_criterion(y_prob, labels)
    loss.backward()
    grads = features.grad.abs()
    grad_features_importance = grads.mean(dim=0)
    # grad_df = var_df.copy()
    grad_df = grad_features_importance.detach().numpy()
    # if sort_by_weight:
    #     shap_df = shap_df.sort_values(by="weight", ascending=False)
    #     shap_df = cumulative_weight_sum_rate(shap_df)
    #     grad_df = grad_df.sort_values(by="weight", ascending=False)
    #     grad_df = cumulative_weight_sum_rate(grad_df)
    arr = np.array(grad_df).reshape(1, -1)  # 需要转换成 2D
    normalized_arr = sk_normalize(arr, norm='l1', axis=1)
    return normalized_arr.flatten()


def run_RF(adata, output_dir):
    X = adata.X
    if np.issubdtype(X.dtype, np.integer) or X.max() > 100:  # Rough check for count data
        X = np.log1p(X)  # Apply log1p for count data
    # X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = adata.obs['labels'].values
    rf = RandomForestClassifier()
    rf.fit(X, y)
    arr = np.array(rf.feature_importances_.flatten()).reshape(1, -1)
    normalized_arr = sk_normalize(arr, norm='l1', axis=1)
    return normalized_arr.flatten()


def run_SVM(adata, output_dir):
    X = adata.X
    if np.issubdtype(X.dtype, np.integer) or X.max() > 100:  # Rough check for count data
        X = np.log1p(X)  # Apply log1p for count data
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = adata.obs['labels'].values
    svm = SVC(kernel='linear')
    svm.fit(X, y)
    arr = np.array(svm.coef_.flatten()).reshape(1, -1)
    normalized_arr = sk_normalize(arr, norm='l1', axis=1)
    return normalized_arr.flatten()


def run_MI(adata, output_dir):
    X = adata.X
    if np.issubdtype(X.dtype, np.integer) or X.max() > 100:  # Rough check for count data
        X = np.log1p(X)  # Apply log1p for count data
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = adata.obs['labels'].values
    arr =  np.array(mutual_info_classif(X, y).flatten()).reshape(1, -1)
    normalized_arr = sk_normalize(arr, norm='l1', axis=1)
    return normalized_arr.flatten()


def run_VELORAMA(adata, output_dir):
    import ray
    from ray import tune
    # Explicit imports (NO wildcard import)
    from velorama.train import train_model
    from velorama.utils import (
        construct_dag,
        calculate_diffusion_lags,
        move_files,
        load_gc_interactions,
        estimate_interactions,
    )

    target_genes = adata.var_names.tolist()
    reg_genes = adata.var_names.tolist()

    X_orig = adata.X.copy()
    if np.issubdtype(X_orig.dtype, np.integer) or X_orig.max() > 100:  # Rough check for count data
        X_orig = np.log1p(X_orig)  # Apply log1p for count data
    std = X_orig.std(0)
    std[std == 0] = 1
    X = torch.FloatTensor(X_orig - X_orig.mean(0)) / std
    X = X.to(torch.float32)

    Y_orig = adata.X.copy()
    std = Y_orig.std(0)
    std[std == 0] = 1
    Y = torch.FloatTensor(Y_orig - Y_orig.mean(0)) / std
    Y = Y.to(torch.float32)

    adata.uns['iroot'] = 0
    sc.pp.scale(adata)

    reg_target = 0  # 有值的情况下，就是有提供regulator和target信息，没有的话就是所有基因都是regulator和target
    dynamics = 'pseudotime'
    ptloc = 'pseudotime'
    proba = 1
    n_neighbors = 30
    velo_mode = 'stochastic'
    time_series = 0
    n_comps = 20
    lag = 5
    name = 'velorama_run'
    seed = 42
    hidden = 32
    penalty = 'H'
    save_dir = output_dir
    lam_start = -2
    lam_end = 1
    num_lambdas = 19

    # A 邻接矩阵，
    # AX AY 结果是一个 (lag × cells × genes) 的张量，表示不同时间步的扩散特征：计算 A * X，表示通过邻接矩阵传播 X。计算 A^2 * X，即将 A 再次传播。持续进行 lag 次，存储 A^t * X。
    A = construct_dag(adata, dynamics=dynamics, ptloc=ptloc, proba=proba,
                      n_neighbors=n_neighbors, velo_mode=velo_mode,
                      use_time=time_series, n_comps=n_comps)
    A = torch.FloatTensor(A)
    AX = calculate_diffusion_lags(A, X, lag)
    AY = None

    dir_name = '{}.seed{}.h{}.{}.lag{}.{}'.format(name, seed, hidden, penalty, lag, dynamics)

    if not os.path.exists(os.path.join(save_dir, dir_name)):
        os.makedirs(os.path.join(save_dir, dir_name), exist_ok=True)

    ray.init(object_store_memory=6 * 1024 ** 3, ignore_reinit_error=True)  # 可以调大
    # ray.init(local_mode=True)  # 就可以设置断点啦！

    lam_list = np.logspace(lam_start, lam_end, num=num_lambdas).tolist()

    config = {'name': name,
              'AX': AX,
              'AY': AY,
              'Y': Y,
              'seed': seed,
              'lr': 0.01,
              'lam': tune.grid_search(lam_list),
              'lam_ridge': 0.0,
              'penalty': penalty,
              'lag': lag,
              'hidden': [hidden],
              'max_iter': 200,
              'device': 'cpu',
              'lookback': 5,
              'check_every': 10,
              'verbose': True,
              'dynamics': dynamics,
              'results_dir': save_dir,
              'dir_name': dir_name,
              'reg_target': reg_target}
    resources_per_trial = {"cpu": 1, "gpu": 0, "memory": 1 * 1024 ** 3}  # 可以调大
    analysis = tune.run(train_model, resources_per_trial=resources_per_trial, config=config, storage_path=save_dir)

    target_dir = os.path.join(save_dir, dir_name)
    base_dir = save_dir
    move_files(base_dir, target_dir)

    # aggregate results
    lam_list = [np.round(lam, 4) for lam in lam_list]
    all_lags = load_gc_interactions(name, save_dir, lam_list, hidden_dim=hidden, lag=lag, penalty=penalty, dynamics=dynamics, seed=seed, ignore_lag=False)  #形状为[lam_count, TG_count, TF_count, lag]

    gc_mat = estimate_interactions(all_lags, lag=lag)  # tg_count x tf_count
    gc_df = pd.DataFrame(gc_mat.cpu().data.numpy(), index=target_genes, columns=reg_genes)
    # gc_df.to_csv(os.path.join(save_dir, '{}.{}.velorama.interactions.tsv'.format(name, dynamics)), sep='\t')  # 行是TG，列是TF

    ray.shutdown()
    return gc_df.mean(axis=0).values


def run_SCRIBE(adata, output_dir):
    from Scribe.read_export import load_anndata
    # Data is a matrix. The rows is an index of GINE_ID with RUN_ID(Multiple index) and columns is cell number
    adata = adata.copy()
    sc.pp.log1p(adata)
    adata.uns['iroot'] = 0
    sc.pp.scale(adata)
    sc.pp.neighbors(adata)
    sc.tl.dpt(adata)
    adata.obs['dpt_groups'] = ['0' if i < adata.obs['dpt_pseudotime'].median() else '1' for i in adata.obs['dpt_pseudotime']]

    model = load_anndata(adata)
    model.rdi(delays=[1,2,3], number_of_processes=1, uniformization=False, differential_mode=False)  # dict_keys([1, 2, 3, 'MAX'])

    edges = []
    values = []
    for id1 in adata.var_names:
        for id2 in adata.var_names:
            if id1 == id2: continue
            edges.append(id1.lower() + "\t" + id2.lower())
            values.append(model.rdi_results["MAX"].loc[id1, id2])

    edges_values = [[edges[i], values[i]] for i in range(len(edges))]
    df = pd.DataFrame(edges_values, columns=['Edge', 'Value'])
    df[['Source', 'Target']] = df['Edge'].str.split('\t', expand=True)
    df_sorted = df[['Source', 'Target', 'Value']].sort_values(by='Value', ascending=False)
    df_mean = df_sorted.groupby("Source")["Value"].mean().reset_index()
    df_mean = df_mean.set_index("Source").loc[list(adata.var_names)]

    return np.array(df_mean['Value'])


def run_DCI(adata, output_dir):
    # pip install causaldag
    from causaldag import dci
    from collections import Counter
    import itertools as itr
    import scipy
    adata = adata.copy()
    sc.pp.log1p(adata)

    # from causaldag.datasets import create_synthetic_difference
    # X1, X2, true_difference = create_synthetic_difference(nnodes=8, nsamples=10000)
    X1 = adata.X[adata.obs['labels'] == 0].astype(float)
    X2 = adata.X[adata.obs['labels'] == 1].astype(float)
    X1 += np.random.normal(0, 1e-6, size=X1.shape)
    X2 += np.random.normal(0, 1e-6, size=X2.shape)
    p = X1.shape[1]
    if scipy.sparse.issparse(adata.X):
        X_full = adata.X.toarray()
    else:
        X_full = adata.X
    corr = np.corrcoef(X_full.T)
    threshold = 0.3  # 可调
    candidate_edges = [(i, j) for i in range(p) for j in range(i + 1, p) if abs(corr[i, j]) >= threshold]
    print(f"[INFO] Filtered candidate edges: {len(candidate_edges)}")
    difference_matrix = dci(X1, X2, difference_ug_method='constraint', difference_ug=candidate_edges, alpha_ug=0.05, alpha_skeleton=0.1, max_set_size=2)
    # difference_matrix = dci(X1, X2, difference_ug_method='constraint', difference_ug=list(itr.combinations(range(p), 2)), alpha_ug=0.02, alpha_skeleton=0.1)
    # difference_matrix = dci(X1, X2, difference_ug_method='constraint', difference_ug=list(itr.combinations(range(p), 2)), alpha_ug=0.02, alpha_skeleton=0.1)
    # alpha_ug: significance level (lower → sparser edges), default 0.01
    # alpha_skeleton: threshold controlling sparsity of the skeleton graph, default 0.1
    # alpha_orient: threshold controlling edge orientation (lower → more directed edges)
    # max_set_size: maximum conditioning set size (smaller → faster)
    ddag_edges = set(zip(*np.where(difference_matrix != 0)))
    print("len(ddag_edges)", len(ddag_edges))
    count_dict = Counter([node for edge in ddag_edges for node in edge])
    count_df = pd.DataFrame(count_dict.items(), columns=['node', 'count']).sort_values('count', ascending=False)
    full_df = pd.DataFrame({'node': adata.var_names})  # 创建完整的node列表
    count_df['node'] = count_df['node'].map(full_df['node'])
    count_df = full_df.merge(count_df, on='node', how='left').fillna({'count': 0})
    arr = np.array(count_df['count'].values).reshape(1, -1)
    normalized_arr = sk_normalize(arr, norm='l1', axis=1)
    return normalized_arr.flatten()


def run_GENIE3(adata, output_dir):
    from GENIE3 import GENIE3, get_link_list
    import tempfile
    import os
    adata = adata.copy()
    sc.pp.log1p(adata)
    adata1 = adata[adata.obs['labels'] == 1]
    adata0 = adata[adata.obs['labels'] == 0]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_path = tmp_file.name
    X = adata0.X
    gene_names = list(adata.var_names)
    VIM = GENIE3(X , gene_names=gene_names)
    get_link_list(VIM, gene_names=gene_names, file_name=tmp_path)
    df0 = pd.read_csv(tmp_path, sep='\t', header=None)
    os.remove(tmp_path)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_path = tmp_file.name
    X = adata1.X
    gene_names = list(adata.var_names)
    VIM = GENIE3(X, gene_names=gene_names)
    get_link_list(VIM, gene_names=gene_names, file_name=tmp_path)
    df1 = pd.read_csv(tmp_path, sep='\t', header=None)
    os.remove(tmp_path)

    merged = pd.merge(df1, df0, on=[0, 1], suffixes=('_net1', '_net0'))
    merged['Diff_Strength'] = abs(merged['2_net1'] - merged['2_net0'])
    diff_network = merged[[0, 1, 'Diff_Strength']]
    arr = diff_network.groupby(0)['Diff_Strength'].mean()
    arr_sorted = arr.reindex(adata.var_names)
    normalized_arr = minmax_scale(arr_sorted)

    return normalized_arr


def select_single_feature(score_array, threshold=None, topk=None):
    """
    Select binary labels (1 for selected, 0 for not) from a 1D score array,
    using either a threshold or top-k strategy.
    """
    score_array = np.asarray(score_array)
    n = len(score_array)

    if topk is not None:
        topk_indices = np.argsort(score_array)[-topk:]
        selected = np.zeros(n, dtype=int)
        selected[topk_indices] = 1
        return selected

    elif threshold is not None:
        return (score_array >= threshold).astype(int)

    else:
        raise ValueError("Either threshold or topk must be specified.")


def calculate_metrics(weight_df, label_col="is_causal", score_col="weight", threshold=None, topk=None):
    from sklearn.metrics import (roc_curve, auc, confusion_matrix, accuracy_score, matthews_corrcoef,
                                 f1_score, precision_score, recall_score, precision_recall_curve)

    true_label = weight_df[label_col].values
    score_array = weight_df[score_col].values

    # Get predicted label by top-k or threshold
    pred_label = select_single_feature(score_array, threshold=threshold, topk=topk)

    # AUROC
    fpr, tpr, _ = roc_curve(true_label, score_array)
    roc_auc = auc(fpr, tpr)

    # AUPR
    precision_curve, recall_curve, _ = precision_recall_curve(true_label, score_array)
    aupr = auc(recall_curve, precision_curve)

    # Confusion matrix
    cm = confusion_matrix(true_label, pred_label)
    TN, FP = cm[0, 0], cm[0, 1]
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    # Other metrics
    acc = accuracy_score(true_label, pred_label)
    mcc = matthews_corrcoef(true_label, pred_label)
    precision = precision_score(true_label, pred_label, pos_label=1, zero_division=0)
    recall = recall_score(true_label, pred_label, pos_label=1, zero_division=0)
    f1 = f1_score(true_label, pred_label, pos_label=1, zero_division=0)

    return {
        "AUROC": roc_auc,
        "AUPR": aupr,
        "F1": f1,
        "ACC": acc,
        "MCC": mcc,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity
    }


def evaluate_prediction(true_label, pred_dict, topk=10):
    """
    Evaluate each prediction array in pred_dict using standard metrics.
    """
    results = []
    for submethod, pred in pred_dict.items():
        metrics = calculate_metrics(true_label, pred, topk=topk)
        results.append({
            'SubMethod': submethod,
            'AUROC': metrics[0],
            'AUPR': metrics[1],
            'F1': metrics[2],
            'ACC': metrics[3],
            'MCC': metrics[4],
            'Precision': metrics[5],
            'Recall': metrics[6],
            'Specificity': metrics[7],
        })
    return results


def plot_layerwise_metrics(df, output_dir, causal_strength=0.4, p_zero=0.2):
    """
    Generate Boxplot, Violinplot, and Barplot (mean±std) for each metric (AUROC, AUPR)
    with solid box color, improved clarity, and Nature Methods-compatible visuals.
    """
    sns.set_theme(style="white")
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "Arial",
    })

    df = df.copy()
    df['Method'] = df['Method'].replace({'CauTrigger_SHAP': 'CauTrigger', 'VAEgrad': 'VAE'})
    # method_order = ['CauTrigger', 'GENIE3', 'SCRIBE', 'PC', 'VAE', 'DCI', 'MI', 'RF', 'SVM']
    method_order = ['CauTrigger', 'CauTrigger_SHAP_ratio0.5', 'CauTrigger_SHAP_ratio0.6', 'CauTrigger_SHAP_ratio0.7', 'GENIE3', 'SCRIBE', 'PC', 'VAE', 'DCI', 'MI', 'RF', 'SVM']
    method_order = [m for m in method_order if m in df['Method'].unique()]

    df['Method'] = pd.Categorical(df['Method'], categories=method_order, ordered=True)
    method_count = len(method_order)

    layer_levels = sorted(df["Layer"].dropna().unique().tolist())
    df["Layer"] = pd.Categorical(df["Layer"], categories=layer_levels, ordered=True)

    layer_palette = {
        'layer1': '#3E4A89',  # deep blue-purple → downstream / closer to phenotype
        'layer2': '#74A9CF',  # light blue → mid-level regulatory factors
        'layer3': '#D9EF8B',  # light yellow-green → highest upstream regulators
        'all': '#5B9BD5',  # clean modern blue for overall score
    }

    metrics = ['AUROC', 'AUPR']

    for metric in metrics:
        cs_tag = f"cs{int(causal_strength * 100)}"
        p_tag = f"p{p_zero}"
        metric_tag = metric.lower()

        # === Boxplot ===
        plt.figure(figsize=(7, 3.5))
        ax = sns.boxplot(
            data=df,
            x="Method",
            y=metric,
            hue="Layer",
            palette=layer_palette,
            width=0.6,
            fliersize=0,
            linewidth=0.6,
        )

        sns.stripplot(
            data=df,
            x="Method",
            y=metric,
            hue="Layer",
            dodge=True,
            color='gray',
            size=2,
            jitter=0.25,
            alpha=0.3,
            edgecolor=None,
            linewidth=0,
            legend=False
        )

        # compute layer information
        layer_list = list(df["Layer"].cat.categories)
        n_layers = len(layer_list)
        group_width = 0.6
        box_width = group_width / n_layers

        # significance annotation
        y_offset = 0.02
        for i, method in enumerate(method_order):
            for j, layer in enumerate(layer_list):
                if method == "CauTrigger":
                    continue

                base_vals = df[(df["Method"] == "CauTrigger") & (df["Layer"] == layer)][metric]
                test_vals = df[(df["Method"] == method) & (df["Layer"] == layer)][metric]

                if len(base_vals) < 2 or len(test_vals) < 2:
                    continue

                stat, pval = ttest_ind(base_vals, test_vals, equal_var=False)

                if pval < 0.0001:
                    sig = "****"
                elif pval < 0.001:
                    sig = "***"
                elif pval < 0.01:
                    sig = "**"
                elif pval < 0.05:
                    sig = "*"
                else:
                    sig = "ns"

                x_base = i
                offset = -group_width / 2 + box_width / 2 + j * box_width
                x = x_base + offset

                y = test_vals.max()
                ax.annotate(
                    sig,
                    xy=(x, y + y_offset),
                    ha='center',
                    va='bottom',
                    fontsize=6,
                    color='black'
                )

        # separation lines between methods
        for i in range(1, method_count):
            ax.axvline(i - 0.5, linestyle='--', color='lightgray', linewidth=0.3, zorder=0)

        # legend handling
        handles, labels = ax.get_legend_handles_labels()
        unique_layers = list(dict.fromkeys(zip(labels, handles)))
        labels, handles = zip(*unique_layers)
        plt.legend(
            handles,
            labels,
            title="Layer",
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.18),
            ncol=len(labels),
            fontsize=11,
            title_fontsize=11,
            columnspacing=1.2,
            handlelength=1.5
        )

        plt.ylabel(metric, fontsize=13)
        plt.xlabel("")
        plt.xticks(rotation=30, ha='right', fontsize=11)
        plt.yticks(fontsize=11)
        plt.ylim(0, 1.05)
        sns.despine()
        plt.tight_layout()
        fname = f"{metric_tag}-boxplot-{cs_tag}-{p_tag}"
        plt.savefig(os.path.join(output_dir, f"{fname}.pdf"))
        plt.savefig(os.path.join(output_dir, f"{fname}.png"))
        plt.close()

        # === Violinplot ===
        plt.figure(figsize=(7, 3.5))
        ax = sns.violinplot(
            data=df,
            x="Method",
            y=metric,
            hue="Layer",
            palette=layer_palette,
            inner="quartile",
            cut=0,
            scale="width",
            bw=0.4,
            width=0.7,
            linewidth=1.0,
            dodge=True,
            saturation=0.8
        )

        # compute layer-level grouping info
        layer_list = list(df["Layer"].cat.categories)
        n_layers = len(layer_list)
        group_width = 0.7
        violin_width = group_width / n_layers

        # add significance stars
        y_offset = 0.02
        for i, method in enumerate(method_order):
            for j, layer in enumerate(layer_list):
                if method == "CauTrigger":
                    continue

                base_vals = df[(df["Method"] == "CauTrigger") & (df["Layer"] == layer)][metric]
                test_vals = df[(df["Method"] == method) & (df["Layer"] == layer)][metric]

                if len(base_vals) < 2 or len(test_vals) < 2:
                    continue

                stat, pval = ttest_ind(base_vals, test_vals, equal_var=False)

                if pval < 0.0001:
                    sig = "****"
                elif pval < 0.001:
                    sig = "***"
                elif pval < 0.01:
                    sig = "**"
                elif pval < 0.05:
                    sig = "*"
                else:
                    sig = "ns"

                # Compute the center position of the violin for this method-layer pair
                x_base = i
                offset = -group_width / 2 + violin_width / 2 + j * violin_width
                x = x_base + offset

                # Get the maximum value of this group
                y = test_vals.max()

                ax.annotate(
                    sig,
                    xy=(x, y + y_offset),
                    ha='center',
                    va='bottom',
                    fontsize=6,
                    color='black'
                )

        # Vertical separation lines between methods
        for i in range(1, method_count):
            ax.axvline(i - 0.5, linestyle='--', color='lightgray', linewidth=0.5, zorder=0)

        # Plot styling settings
        plt.ylabel(metric, fontsize=13)
        plt.xlabel("")
        plt.xticks(rotation=30, ha='right', fontsize=11)
        plt.yticks(fontsize=11)
        plt.ylim(0, 1.05)
        plt.legend(
            title="Layer",
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.18),
            ncol=len(layer_list),
            fontsize=11,
            title_fontsize=11,
            columnspacing=1.2,
            handlelength=1.5
        )
        sns.despine()
        plt.tight_layout()
        fname = f"{metric_tag}-violinplot-{cs_tag}-{p_tag}"
        plt.savefig(os.path.join(output_dir, f"{fname}.pdf"))
        plt.savefig(os.path.join(output_dir, f"{fname}.png"))
        plt.close()

        # === Barplot ===
        summary_df = df.groupby(["Method", "Layer"])[metric].agg(["mean", "std"]).reset_index()
        plt.figure(figsize=(7, 3.5))

        ax = sns.barplot(
            data=summary_df,
            x="Method",
            y="mean",
            hue="Layer",
            palette=layer_palette,
            errorbar=None,
            width=0.7
        )

        # Retrieve categorical layer information
        layer_list = list(df["Layer"].cat.categories)
        n_layers = len(layer_list)
        bar_width = 0.7 / n_layers
        group_width = 0.7

        # Vertical offset for significance markers
        y_offset = 0.02

        # Draw significance markers
        for i, method in enumerate(method_order):
            for j, layer in enumerate(layer_list):
                if method == "CauTrigger":
                    continue

                base_vals = df[(df["Method"] == "CauTrigger") & (df["Layer"] == layer)][metric]
                test_vals = df[(df["Method"] == method) & (df["Layer"] == layer)][metric]

                if len(base_vals) < 2 or len(test_vals) < 2:
                    continue
                stat, pval = ttest_ind(base_vals, test_vals, equal_var=False)

                # Determine significance level
                if pval < 0.0001:
                    sig = "****"
                elif pval < 0.001:
                    sig = "***"
                elif pval < 0.01:
                    sig = "**"
                elif pval < 0.05:
                    sig = "*"
                else:
                    sig = "ns"

                # Compute the center position of the bar for this method-layer pair
                x_base = i  # method position x axis
                offset = -group_width / 2 + bar_width / 2 + j * bar_width
                x = x_base + offset

                # Get the height of the bar at this position
                row = summary_df[
                    (summary_df["Method"] == method) &
                    (summary_df["Layer"] == layer)
                    ]
                mean = row["mean"].values[0]
                std = row["std"].values[0]
                y = mean + std

                ax.annotate(
                    sig,
                    xy=(x, y + y_offset),
                    ha='center',
                    va='bottom',
                    fontsize=6,
                    color='black'
                )

        # Add error bars
        for i, method in enumerate(method_order):
            for j, layer in enumerate(layer_list):
                row = summary_df[
                    (summary_df["Method"] == method) &
                    (summary_df["Layer"] == layer)
                    ]
                if row.empty:
                    continue

                mean = row["mean"].values[0]
                std = row["std"].values[0]

                x_base = i
                offset = -group_width / 2 + bar_width / 2 + j * bar_width
                x = x_base + offset

                ax.errorbar(
                    x=x,
                    y=mean,
                    yerr=std,
                    fmt='none',
                    ecolor='black',
                    elinewidth=1.5,
                    capsize=4,
                    capthick=1.5
                )
        # split lines
        for i in range(1, method_count):
            ax.axvline(i - 0.5, linestyle='--', color='lightgray', linewidth=0.5, zorder=0)

        # Apply plot style settings
        max_y = summary_df["mean"].max() + summary_df["std"].max() + 0.15
        plt.ylim(0, max(1.05, max_y))
        plt.ylabel(f"{metric} (Mean ± SD)", fontsize=13)
        plt.xlabel("")
        plt.xticks(rotation=30, ha='right', fontsize=11)
        plt.yticks(fontsize=11)
        plt.legend(
            title="Layer",
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.18),
            ncol=len(layer_list),
            fontsize=11,
            title_fontsize=11,
            columnspacing=1.2,
            handlelength=1.5
        )
        sns.despine()
        plt.tight_layout()
        fname = f"{metric_tag}-barplot-{cs_tag}-{p_tag}"
        plt.savefig(os.path.join(output_dir, f"{fname}.pdf"))
        plt.savefig(os.path.join(output_dir, f"{fname}.png"))
        plt.close()


def run_known_benchmark(
    algorithms,
    data_dir,
    output_dir,
    n_layers,
    n_datasets=3,
    seed_list=None,
    save_adata=True,
    rerun=False,
    **generation_args
):
    """
    Run benchmark evaluation for causal discovery methods on synthetic multi-layer datasets
    (known hierarchical structure). Supports both baseline methods and recursive multi-layer
    methods like CauTrigger.
    """
    # --- internal dispatch function ---
    def _run_ct_known(adata, out, mode="SHAP", topk_ratio=None):
        if n_layers == 2:
            return run_ct_2l_known(adata, out, mode=mode, topk_ratio=topk_ratio)
        elif n_layers == 3:
            return run_ct_3l_known(adata, out, mode=mode, topk_ratio=topk_ratio)
        else:
            raise NotImplementedError(f"{n_layers} layers not supported!")

    algorithm_functions = {
        "CauTrigger_SHAP": lambda adata, out: _run_ct_known(adata, out, mode="SHAP"),
        "CauTrigger_SHAP_ratio0.5": lambda adata, out: _run_ct_known(adata, out, mode="SHAP", topk_ratio=0.5),
        "CauTrigger_SHAP_ratio0.6": lambda adata, out: _run_ct_known(adata, out, mode="SHAP", topk_ratio=0.6),
        "CauTrigger_SHAP_ratio0.7": lambda adata, out: _run_ct_known(adata, out, mode="SHAP", topk_ratio=0.7),

        # Other baseline methods
        "PC": run_PC,
        "VAEgrad": run_VAE,
        "SVM": run_SVM,
        "RF": run_RF,
        "MI": run_MI,
        "DCI": run_DCI,
        # "NLBayes": run_NLBAYES,
        "GENIE3": run_GENIE3,
        # "GRNBOOST2": run_GRNBOOST2,
        "SCRIBE": run_SCRIBE,
        "VELORAMA": run_VELORAMA,
    }

    print(f"[INFO] Running benchmark on {n_datasets} datasets with algorithms: {algorithms}")

    if seed_list is None:
        seed_list = list(range(n_datasets))
    else:
        if len(seed_list) < n_datasets:
            max_seed = max(seed_list)
            seed_list += list(range(max_seed + 1, max_seed + 1 + (n_datasets - len(seed_list))))
        elif len(seed_list) > n_datasets:
            seed_list = seed_list[:n_datasets]

    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    for i, seed in enumerate(seed_list):
        set_seed(seed)
        # --- dynamic dataset generator ---
        if n_layers == 2:
            adata = generate_two_layer_synthetic_data(seed=seed, **generation_args)
        elif n_layers == 3:
            adata = generate_three_layer_synthetic_data(seed=seed, **generation_args)
        else:
            raise NotImplementedError(f"{n_layers} layers not supported for generation!")

        print(f"[INFO] Dataset {i + 1}: Generated with seed {seed}")

        if save_adata:
            dataset_dir = os.path.join(data_dir, f'dataset{i + 1}')
            os.makedirs(dataset_dir, exist_ok=True)
            adata.write(os.path.join(dataset_dir, 'adata.h5ad'))

        layer_names = sorted(adata.var['layer'].unique())
        for algo in algorithms:
            assert algo in algorithm_functions, f"[ERROR] Algorithm '{algo}' not registered in algorithm_functions!"
            if algo.startswith("CauTrigger"):
                # --- Check if both layer1 and layer2 results exist ---
                exist_all = True
                for layer_name in layer_names:
                    weight_path = os.path.join(output_dir, f'weights_dataset{i}_{layer_name}_{algo}.csv')
                    if not os.path.exists(weight_path):
                        exist_all = False
                        break

                if not rerun and exist_all:
                    print(f"[INFO] {algo} on dataset {i + 1} already exists. Loading...")
                    for layer_name in layer_names:
                        weight_path = os.path.join(output_dir, f'weights_dataset{i}_{layer_name}_{algo}.csv')
                        df = pd.read_csv(weight_path, index_col=0)
                        metrics = calculate_metrics(df, score_col="weight", label_col="is_causal", topk=20)
                        row = {
                            "Method": algo,
                            "Layer": layer_name,
                            "Dataset": i,
                            "Seed": seed,
                            "ScoreType": "weight",
                            **metrics
                        }
                        all_results.append(row)
                    continue

                # --- If not exist or rerun, re-run ---
                print(f"[INFO] Evaluating {algo} on recursive {n_layers}-layer setting (Dataset {i + 1})...")
                func = algorithm_functions[algo]
                pred_dict = func(adata, output_dir)

                for layer_name, df in pred_dict.items():
                    assert "is_causal" in df.columns, f"[ERROR] 'is_causal' column missing in CauTrigger output for {layer_name}"
                    weight_path = os.path.join(output_dir, f'weights_dataset{i}_{layer_name}_{algo}.csv')
                    df.to_csv(weight_path)
                    print(f"[INFO] Saved weights to: {weight_path}")

                    metrics = calculate_metrics(df, score_col="weight", label_col="is_causal", topk=20)
                    row = {
                        "Method": algo,
                        "Layer": layer_name,
                        "Dataset": i,
                        "Seed": seed,
                        "ScoreType": "weight",
                        **metrics
                    }
                    all_results.append(row)

            else:
                for layer_name in layer_names:
                    layer_vars = adata.var_names[adata.var['layer'] == layer_name]
                    sub_adata = adata[:, layer_vars]

                    weight_path = os.path.join(output_dir, f'weights_dataset{i}_{layer_name}_{algo}.csv')
                    # if algo != "DCI" and not rerun and os.path.exists(weight_path):
                    if not rerun and os.path.exists(weight_path):
                        print(f"[INFO] {algo} on {layer_name} (Dataset {i + 1}) already exists. Loading...")
                        weight_df = pd.read_csv(weight_path, index_col=0)
                        metrics = calculate_metrics(weight_df, score_col="weight", label_col="is_causal", topk=20)
                        row = {
                            "Method": algo,
                            "Layer": layer_name,
                            "Dataset": i,
                            "Seed": seed,
                            "ScoreType": "weight",
                            **metrics
                        }
                        all_results.append(row)
                        continue

                    print(f"[INFO] Evaluating {algo} on {layer_name} (Dataset {i + 1})...")
                    func = algorithm_functions[algo]
                    pred = func(sub_adata, output_dir)

                    pred_dict = {"weight": pred} if not isinstance(pred, dict) else pred
                    weight_df = pd.DataFrame(pred_dict, index=layer_vars)
                    weight_df["is_causal"] = adata.var.loc[layer_vars, "is_causal"].values

                    weight_df.to_csv(weight_path)
                    print(f"[INFO] Saved weights to: {weight_path}")

                    metrics = calculate_metrics(weight_df, score_col="weight", label_col="is_causal", topk=20)
                    row = {
                        "Method": algo,
                        "Layer": layer_name,
                        "Dataset": i,
                        "Seed": seed,
                        "ScoreType": "weight",
                        **metrics
                    }
                    all_results.append(row)

        del adata
        gc.collect()

    # Save final metrics
    df = pd.DataFrame(all_results)
    metrics_path = os.path.join(output_dir, 'Layerwise_Benchmark_Metrics.csv')
    df.to_csv(metrics_path, index=False)
    print(f"[INFO] Saved final evaluation to: {metrics_path}")

    # Draw plots
    plot_layerwise_metrics(
        df,
        output_dir,
        causal_strength=generation_args.get("causal_strength", 0.4),
        p_zero=generation_args.get("p_zero", 0.2)
    )


def run_unknown_benchmark(
    algorithms,
    data_dir,
    output_dir,
    n_layers,
    n_datasets=3,
    seed_list=None,
    save_adata=True,
    rerun=False,
    **generation_args
):
    """
    Run benchmark evaluation for causal discovery methods on synthetic multi-layer datasets
    (unknown hierarchical structure). Supports both baseline methods (flat) and recursive
    multi-layer methods like CauTrigger.
    """
    # --- internal dispatch function ---
    def _run_ct_unknown(adata, out, mode="SHAP", topk_ratio=0.5):
        if n_layers == 2:
            return run_ct_2l_unknown(adata, out, mode=mode, topk_ratio=topk_ratio)
        elif n_layers == 3:
            return run_ct_3l_unknown(adata, out, mode=mode, topk_ratio=topk_ratio)
        else:
            raise NotImplementedError(f"{n_layers} layers not supported!")

    algorithm_functions = {
        # CauTrigger scoring variants
        "CauTrigger_SHAP": lambda adata, out: _run_ct_unknown(adata, out, mode="SHAP"),
        "CauTrigger_SHAP_ratio0.5": lambda adata, out: _run_ct_unknown(adata, out, mode="SHAP", topk_ratio=0.5),
        "CauTrigger_SHAP_ratio0.6": lambda adata, out: _run_ct_unknown(adata, out, mode="SHAP", topk_ratio=0.6),
        "CauTrigger_SHAP_ratio0.7": lambda adata, out: _run_ct_unknown(adata, out, mode="SHAP", topk_ratio=0.7),

        # Other baseline methods
        "PC": run_PC,
        "VAEgrad": run_VAE,
        "SVM": run_SVM,
        "RF": run_RF,
        "MI": run_MI,
        "DCI": run_DCI,
        # "NLBayes": run_NLBAYES,
        "GENIE3": run_GENIE3,
        # "GRNBOOST2": run_GRNBOOST2,
        "SCRIBE": run_SCRIBE,
        "VELORAMA": run_VELORAMA,
    }

    print(f"[INFO] Running benchmark on {n_datasets} datasets with algorithms: {algorithms}")

    if seed_list is None:
        seed_list = list(range(n_datasets))
    else:
        if len(seed_list) < n_datasets:
            max_seed = max(seed_list)
            seed_list += list(range(max_seed + 1, max_seed + 1 + (n_datasets - len(seed_list))))
        elif len(seed_list) > n_datasets:
            seed_list = seed_list[:n_datasets]

    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    stepwise_results = []

    for i, seed in enumerate(seed_list):
        set_seed(seed)
        if n_layers == 2:
            adata = generate_two_layer_synthetic_data(seed=seed, **generation_args)
        elif n_layers == 3:
            adata = generate_three_layer_synthetic_data(seed=seed, **generation_args)
        else:
            raise NotImplementedError(f"{n_layers} layers not supported!")

        print(f"[INFO] Dataset {i + 1}: Generated with seed {seed}")

        if save_adata:
            dataset_dir = os.path.join(data_dir, f'dataset{i + 1}')
            os.makedirs(dataset_dir, exist_ok=True)
            adata.write(os.path.join(dataset_dir, 'adata.h5ad'))

        for algo in algorithms:
            assert algo in algorithm_functions, f"[ERROR] Algorithm '{algo}' not registered in algorithm_functions!"

            if algo.startswith("CauTrigger"):
                layer_name = "all"
                weight_path = os.path.join(output_dir, f'weights_dataset{i}_{layer_name}_{algo}.csv')

                if not rerun and os.path.exists(weight_path):
                    print(f"[INFO] {algo} on {layer_name} (Dataset {i + 1}) already exists. Loading...")
                    weight_df = pd.read_csv(weight_path, index_col=0)
                    metrics = calculate_metrics(weight_df, score_col="weight", label_col="is_causal", topk=20)
                    row = {
                        "Method": algo,
                        "Layer": layer_name,
                        "Dataset": i,
                        "Seed": seed,
                        "ScoreType": "weight",
                        **metrics
                    }
                    all_results.append(row)
                    continue

                print(f"[INFO] Evaluating {algo} on {layer_name} (Dataset {i + 1})...")
                func = algorithm_functions[algo]
                pred_dict = func(adata, output_dir)

                # Save and evaluate each step's result: steps and all
                for step_key in pred_dict.keys():
                    df = pred_dict[step_key]
                    assert "is_causal" in df.columns, f"[ERROR] Missing 'is_causal' in {step_key} result."

                    step_weight_path = os.path.join(output_dir, f'weights_dataset{i}_{step_key}_{algo}.csv')
                    df.to_csv(step_weight_path)
                    print(f"[INFO] Saved {step_key} weights to: {step_weight_path}")

                    score_type = f"weight_{step_key}" if step_key != "all" else "weight"
                    metrics = calculate_metrics(df, score_col="weight", label_col="is_causal", topk=20)
                    row = {
                        "Method": algo,
                        "Layer": step_key,
                        "Dataset": i,
                        "Seed": seed,
                        "ScoreType": score_type,
                        **metrics
                    }

                    if step_key == "all":
                        all_results.append(row)
                    else:
                        stepwise_results.append(row)

            else:
                layer_name = 'all'
                sub_adata = adata.copy()

                weight_path = os.path.join(output_dir, f'weights_dataset{i}_{layer_name}_{algo}.csv')
                if not rerun and os.path.exists(weight_path):
                # if algo != "DCI" and not rerun and os.path.exists(weight_path):

                    print(f"[INFO] {algo} on {layer_name} (Dataset {i + 1}) already exists. Loading...")
                    weight_df = pd.read_csv(weight_path, index_col=0)
                    metrics = calculate_metrics(weight_df, score_col="weight", label_col="is_causal", topk=20)
                    row = {
                        "Method": algo,
                        "Layer": layer_name,
                        "Dataset": i,
                        "Seed": seed,
                        "ScoreType": "weight",
                        **metrics
                    }
                    all_results.append(row)
                    continue

                print(f"[INFO] Evaluating {algo} on {layer_name} (Dataset {i + 1})...")
                func = algorithm_functions[algo]
                pred = func(sub_adata, output_dir)

                pred_dict = {"weight": pred} if not isinstance(pred, dict) else pred
                weight_df = pd.DataFrame(pred_dict, index=sub_adata.var_names)
                weight_df["is_causal"] = sub_adata.var["is_causal"].values

                weight_df.to_csv(weight_path)
                print(f"[INFO] Saved weights to: {weight_path}")

                metrics = calculate_metrics(weight_df, score_col="weight", label_col="is_causal", topk=20)
                row = {
                    "Method": algo,
                    "Layer": layer_name,
                    "Dataset": i,
                    "Seed": seed,
                    "ScoreType": "weight",
                    **metrics
                }
                all_results.append(row)

        del adata
        gc.collect()

    # Save final metrics
    df = pd.DataFrame(all_results)
    metrics_path = os.path.join(output_dir, 'Layerwise_Benchmark_Metrics.csv')
    df.to_csv(metrics_path, index=False)
    print(f"[INFO] Saved final evaluation to: {metrics_path}")

    # Draw plots
    plot_layerwise_metrics(
        df,
        output_dir,
        causal_strength=generation_args.get("causal_strength", 0.4),
        p_zero=generation_args.get("p_zero", 0.2)
    )


def plot_aggregate_layerwise_metrics(
    root_output_dir,
    hierarchy_prefix,
    causal_strength_list,
    p_zero_list,
    spurious_mode='semi_hrc',
    n_hidden=10,
    activation='linear',
    simulate_single_cell=True
):
    """
    Aggregate layer-wise benchmark metrics across different parameter combinations
    and generate 3×3 boxplots for AUROC/AUPR. A unified legend is placed below the
    title for clean visual comparison (Nature Methods style).
    """

    sns.set_theme(style="white")
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "Arial",
    })

    all_dfs = []

    for cs in causal_strength_list:
        for pz in p_zero_list:
            case_name = "_".join([
                hierarchy_prefix,
                spurious_mode,
                f"hidden{n_hidden}",
                activation,
                f"cs{int(cs * 100):02d}",
                f"p{pz}",
                "sc" if simulate_single_cell else "bulk",
            ])
            csv_path = os.path.join(root_output_dir, case_name, "Layerwise_Benchmark_Metrics.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df["ParamCombo"] = f"Causal Strength = {cs}, Sparsity = {pz}"
                all_dfs.append(df)

    if not all_dfs:
        print("[WARN] No matching benchmark files found.")
        return

    df = pd.concat(all_dfs, ignore_index=True)
    df['Method'] = df['Method'].replace({'CauTrigger_SHAP': 'CauTrigger', 'VAEgrad': 'VAE'})
    method_order = ['CauTrigger', 'CauTrigger_SHAP_ratio0.5', 'CauTrigger_SHAP_ratio0.6', 'CauTrigger_SHAP_ratio0.7',
                    'GENIE3', 'SCRIBE', 'PC', 'VAE', 'DCI', 'MI', 'RF', 'SVM']
    method_order = [m for m in method_order if m in df['Method'].unique()]
    df['Method'] = pd.Categorical(df['Method'], categories=method_order, ordered=True)

    layer_levels = sorted(df["Layer"].dropna().unique().tolist())
    df["Layer"] = pd.Categorical(df["Layer"], categories=layer_levels, ordered=True)

    layer_palette = {
        'layer1': '#3E4A89',
        'layer2': '#74A9CF',
        'layer3': '#D9EF8B',
        'all': '#5B9BD5',
    }

    tag_parts = [
        hierarchy_prefix,
        spurious_mode,
        f"hidden{n_hidden}",
        activation,
        "sc" if simulate_single_cell else "bulk",
    ]
    base_tag = "_".join(tag_parts)

    for metric in ["AUROC", "AUPR"]:
        fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharey=True)
        param_combos = sorted(df['ParamCombo'].unique())

        for ax, combo in zip(axes.flatten(), param_combos):
            subdf = df[df['ParamCombo'] == combo]
            sns.boxplot(
                data=subdf,
                x="Method",
                y=metric,
                hue="Layer",
                palette = layer_palette,
                ax=ax,
                width=0.6,
                fliersize=0,
                linewidth=1,
            )
            ax.set_title(combo, fontsize=14)
            ax.set_xlabel("")
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=30)
            ax.legend_.remove()
            ax.grid(False)
            # === Significance annotation (pairwise t-test vs. CauTrigger) ===
            method_order = df['Method'].cat.categories.tolist()
            layer_list = list(df["Layer"].cat.categories)
            n_layers = len(layer_list)
            group_width = 0.6  #
            box_width = group_width / n_layers
            y_offset = 0.02

            for i, method in enumerate(method_order):
                for j, layer in enumerate(layer_list):
                    if method == "CauTrigger":
                        continue

                    base_vals = subdf[(subdf["Method"] == "CauTrigger") & (subdf["Layer"] == layer)][metric]
                    test_vals = subdf[(subdf["Method"] == method) & (subdf["Layer"] == layer)][metric]

                    if len(base_vals) < 2 or len(test_vals) < 2:
                        continue

                    stat, pval = ttest_ind(base_vals, test_vals, equal_var=False)
                    if pval < 0.0001:
                        sig = "****"
                    elif pval < 0.001:
                        sig = "***"
                    elif pval < 0.01:
                        sig = "**"
                    elif pval < 0.05:
                        sig = "*"
                    else:
                        sig = "ns"

                    x_base = i
                    offset = -group_width / 2 + box_width / 2 + j * box_width
                    x = x_base + offset
                    # Estimate the upper whisker (Q3 + 1.5×IQR) to place the significance star
                    q1 = test_vals.quantile(0.25)
                    q3 = test_vals.quantile(0.75)
                    iqr = q3 - q1
                    upper = q3 + 1.5 * iqr
                    y = min(upper, test_vals.max())

                    ax.annotate(
                        sig,
                        xy=(x, y + y_offset),
                        ha='center',
                        va='bottom',
                        fontsize=6,
                        color='black'
                    )
            # n_methods = df['Method'].nunique()
            # for i in range(1, n_methods):
            #     ax.axvline(x=i - 0.5, linestyle='--', color='lightgray', linewidth=0.6, zorder=0)

        # Set the main title; no vertical offset is needed due to legend placement
        fig.suptitle(f"Overall {metric} across methods", fontsize=18, y=1.02)

        # Add a unified legend below the title
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=2,
            frameon=False
        )

        plt.tight_layout()
        out_prefix = f"{base_tag}_boxplot_{metric}"
        fig.savefig(os.path.join(root_output_dir, f"{out_prefix}.pdf"), bbox_inches='tight')
        fig.savefig(os.path.join(root_output_dir, f"{out_prefix}.png"), bbox_inches='tight')
        print(f"[INFO] Saved: {out_prefix}.pdf/.png")
        plt.show()

        plt.close(fig)


def compare_known_vs_unknown_layers(
    base_dir_unknown,
    base_dir_known,
    n_layers,
    method="CauTrigger_SHAP",
    topk=20,
    n_datasets=10,
    output_path=None
):
    layer_pairs = [(f"step{i + 1}", f"layer{i + 1}") for i in range(n_layers)]
    records = []
    for i in range(n_datasets):
        for u_layer, k_layer in layer_pairs:
            path_u = os.path.join(base_dir_unknown, f"weights_dataset{i}_{u_layer}_{method}.csv")
            path_k = os.path.join(base_dir_known, f"weights_dataset{i}_{k_layer}_{method}.csv")

            if not (os.path.exists(path_u) and os.path.exists(path_k)):
                print(f"[WARN] Missing file for dataset {i}, layer {u_layer} or {k_layer}")
                continue

            df_u = pd.read_csv(path_u, index_col=0)
            df_k = pd.read_csv(path_k, index_col=0)

            m_u = calculate_metrics(df_u, score_col="weight", label_col="is_causal", topk=topk)
            m_k = calculate_metrics(df_k, score_col="weight", label_col="is_causal", topk=topk)

            records.append({"Dataset": i, "Layer": u_layer, "Mode": "unknown", **m_u})
            records.append({"Dataset": i, "Layer": k_layer, "Mode": "known", **m_k})

    df_compare = pd.DataFrame(records)
    if output_path:
        df_compare.to_csv(output_path, index=False)
    return df_compare


def plot_hierarchy_comparison(df, output_dir, metric="AUROC", cs=None, p_zero=None):
    """
    Compare inferred (pseudo) vs prior (true) layer-level performance using boxplot.
    Outputs one plot per metric, with styling aligned to Nature Methods.
    """
    # === Global style ===
    sns.set_theme(style="white")
    plt.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "Arial"
    })

    # === Preprocess DataFrame ===
    df = df.copy()
    df["Layer"] = df["Layer"].replace({
        "step1": "Layer 1",
        "layer1": "Layer 1",
        "step2": "Layer 2",
        "layer2": "Layer 2",
        "step3": "Layer 3",
        "layer3": "Layer 3"
    })
    # df["Layer"] = pd.Categorical(df["Layer"], categories=["Layer 1", "Layer 2", "Layer 3"], ordered=True)
    unique_layers = sorted(df["Layer"].unique(), key=lambda x: int(x.split()[-1]))
    df["Layer"] = pd.Categorical(df["Layer"], categories=unique_layers, ordered=True)

    df["Setting"] = df["Mode"].replace({
        "unknown": "inferred",
        "known": "known"
    })
    df["Setting"] = pd.Categorical(df["Setting"], categories=["known", "inferred"], ordered=True)

    setting_palette = {
        "inferred": "#2C2C79",  # deep blue
        "known": "#9DC3E6"      # light blue
    }

    # === Plot ===
    plt.figure(figsize=(6, 4))
    ax = sns.boxplot(
        data=df,
        x="Layer",
        y=metric,
        hue="Setting",
        palette=setting_palette,
        width=0.6,
        fliersize=0,
        gap=0.15,  # hue box space
        linewidth=1
    )
    sns.stripplot(
        data=df,
        x="Layer",
        y=metric,
        hue="Setting",
        dodge=True,
        color="gray",
        alpha=0.5,
        size=4,
        jitter=0.2,
        edgecolor="none",
        linewidth=0,
        legend=False
    )
    layers = df["Layer"].cat.categories.tolist()
    settings = df["Setting"].cat.categories.tolist()
    n_settings = len(settings)
    group_width = 0.6
    box_width = group_width / n_settings
    y_offset = 0.02

    # Extract top y-coordinates of the boxes
    box_tops = {}
    for patch, layer in zip(ax.artists, [(l, s) for l in layers for s in settings]):
        l, s = layer
        y_top = patch.get_y() + patch.get_height()
        box_tops[(l, s)] = y_top

    for i, layer in enumerate(layers):
        group1 = df[(df["Layer"] == layer) & (df["Setting"] == settings[0])][metric]
        group2 = df[(df["Layer"] == layer) & (df["Setting"] == settings[1])][metric]

        if len(group1) < 2 or len(group2) < 2:
            continue

        stat, pval = ttest_ind(group1, group2, equal_var=False)

        if pval < 0.0001:
            sig = "****"
        elif pval < 0.001:
            sig = "***"
        elif pval < 0.01:
            sig = "**"
        elif pval < 0.05:
            sig = "*"
        else:
            sig = "ns"

        # Compute x positions of the two boxes (known vs inferred)
        x_base = i
        offset1 = -group_width / 2 + box_width / 2
        offset2 = +group_width / 2 - box_width / 2
        x1 = x_base + offset1
        x2 = x_base + offset2

        # Get y-positions of the box tops for significance annotation
        y1 = box_tops.get((layer, settings[0]), group1.max())
        y2 = box_tops.get((layer, settings[1]), group2.max())
        y = max(y1, y2) + y_offset

        # Draw connecting lines and significance stars
        ax.plot([x1, x1, x2, x2], [y, y + y_offset, y + y_offset, y], lw=1, c="k")
        ax.text((x1 + x2) / 2, y + y_offset * 1.2, sig, ha='center', va='bottom', fontsize=10)

    # === Adjust legend ===
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = list(dict.fromkeys(labels))  # preserve order
    unique_handles = [handles[labels.index(l)] for l in unique_labels]
    ax.legend(
        unique_handles, unique_labels,
        title=None,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=len(unique_labels),
        handlelength=1.5,
        columnspacing=1.5
    )

    # === Plot aesthetics ===
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1.05)
    sns.despine()
    plt.tight_layout()

    # === Save ===
    cs_tag = f"cs{int(cs * 100)}" if cs is not None else ""
    p_tag = f"p{p_zero}" if p_zero is not None else ""
    fname = f"{metric}_HierarchyComparison_{cs_tag}_{p_tag}".strip("_")
    plt.savefig(os.path.join(output_dir, f"{fname}.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f"{fname}.png"), bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved {fname}.pdf/png to: {output_dir}")


def plot_aggregate_hierarchy_comparison(
    root_output_dir,
    hierarchy_prefix,
    causal_strength_list,
    p_zero_list,
    metric="AUROC",
    spurious_mode="semi_hrc",
    n_hidden=10,
    activation="linear",
    simulate_single_cell=True
):
    """
    Aggregate layer-wise performance from multiple parameter combinations and
    visualize the comparison between prior (known) and inferred hierarchical
    assignments across different causal strengths and sparsity levels.
    """
    sns.set_theme(style="white")
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "Arial",
    })

    all_dfs = []

    for cs in causal_strength_list:
        for pz in p_zero_list:
            tag_parts = [
                hierarchy_prefix,
                spurious_mode,
                f"hidden{n_hidden}",
                activation,
                f"cs{int(cs * 100):02d}",
                f"p{pz}",
                "sc" if simulate_single_cell else "bulk",
            ]
            case_name = "_".join(tag_parts)
            csv_path = os.path.join(root_output_dir, case_name, "compare_known_unknown.csv")

            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df["ParamCombo"] = f"Causal Strength = {cs}, Sparsity = {pz}"
                all_dfs.append(df)

    if not all_dfs:
        print("[WARN] No compare_known_unknown.csv files found.")
        return

    df = pd.concat(all_dfs, ignore_index=True)

    df["Layer"] = df["Layer"].replace({
        "step1": "Layer 1",
        "layer1": "Layer 1",
        "step2": "Layer 2",
        "layer2": "Layer 2",
        "step3": "Layer 3",
        "layer3": "Layer 3"
    })
    unique_layers = sorted(df["Layer"].dropna().unique(), key=lambda x: int(x.split()[-1]))
    df["Layer"] = pd.Categorical(df["Layer"], categories=unique_layers, ordered=True)
    df["Setting"] = df["Mode"].replace({
        "unknown": "inferred",
        "known": "known"
    })
    df["Setting"] = pd.Categorical(df["Setting"], categories=["known", "inferred"], ordered=True)

    setting_palette = {
        "inferred": "#2C2C79",  # deep blue
        "known": "#9DC3E6"      # light blue
    }

    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharey=True)
    param_combos = sorted(df['ParamCombo'].unique())

    for ax, combo in zip(axes.flatten(), param_combos):
        subdf = df[df['ParamCombo'] == combo]
        sns.boxplot(
            data=subdf,
            x="Layer",
            y=metric,
            hue="Setting",
            palette=setting_palette,
            width=0.5,
            fliersize=0,
            linewidth=1,
            gap=0.15,  # hue box 之间留空
            # dodge=0.6,
            ax=ax
        )
        sns.stripplot(
            data=subdf,
            x="Layer",
            y=metric,
            hue="Setting",
            dodge=True,
            color="gray",
            alpha=0.4,
            size=3,
            edgecolor="none",
            linewidth=0,
            ax=ax,
            legend=False
        )
        ax.set_title(combo, fontsize=14)
        ax.set_xlabel("")
        ax.tick_params(axis='x', labelsize=14)
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.1)
        ax.grid(False)
        ax.legend_.remove()

        # ==== significance annotation ====
        # compute positions between paired boxes
        layers = subdf["Layer"].cat.categories.tolist()
        settings = subdf["Setting"].cat.categories.tolist()
        n_settings = len(settings)
        group_width = 0.6
        box_width = group_width / n_settings
        y_offset = 0.02  # vertical offset for significance markers

        box_tops = {}
        for patch, layer in zip(ax.artists, [(l, s) for l in layers for s in settings]):
            l, s = layer
            y_top = patch.get_y() + patch.get_height()
            box_tops[(l, s)] = y_top

        for i, layer in enumerate(layers):
            group1 = subdf[(subdf["Layer"] == layer) & (subdf["Setting"] == settings[0])][metric]
            group2 = subdf[(subdf["Layer"] == layer) & (subdf["Setting"] == settings[1])][metric]

            if len(group1) < 2 or len(group2) < 2:
                continue

            stat, pval = ttest_ind(group1, group2, equal_var=False)

            if pval < 0.0001:
                sig = "****"
            elif pval < 0.001:
                sig = "***"
            elif pval < 0.01:
                sig = "**"
            elif pval < 0.05:
                sig = "*"
            else:
                sig = "ns"

            # compute x-positions of the two box groups
            x_base = i
            offset1 = -group_width / 2 + box_width / 2
            offset2 = +group_width / 2 - box_width / 2
            x1 = x_base + offset1
            x2 = x_base + offset2

            # get top y-values of the paired boxplots
            y1 = box_tops.get((layer, settings[0]), group1.max())
            y2 = box_tops.get((layer, settings[1]), group2.max())
            y = max(y1, y2) + y_offset

            # draw significance bars and annotate stars
            ax.plot([x1, x1, x2, x2], [y, y + y_offset, y + y_offset, y], lw=1, c="k")
            ax.text((x1 + x2) / 2, y + y_offset * 1.2, sig, ha='center', va='bottom', fontsize=10)

    fig.suptitle(f"Layer-wise {metric}: Known vs Inferred", fontsize=18, y=1.02)

    # unified legend shared across subplots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique_labels = list(dict.fromkeys(labels))
    unique_handles = [handles[labels.index(l)] for l in unique_labels]
    fig.legend(
        unique_handles, unique_labels,
        title=None,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(unique_labels),
        frameon=False
    )

    plt.tight_layout()
    tag_parts = [
        hierarchy_prefix,
        spurious_mode,
        f"hidden{n_hidden}",
        activation,
        "sc" if simulate_single_cell else "bulk"
    ]
    base_tag = "_".join(tag_parts)
    fname = f"{base_tag}_PriorVsInferred_{metric}"

    fig.savefig(os.path.join(root_output_dir, f"{fname}.pdf"), bbox_inches='tight')
    fig.savefig(os.path.join(root_output_dir, f"{fname}.png"), bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"[INFO] Saved: {fname}.pdf/.png")
