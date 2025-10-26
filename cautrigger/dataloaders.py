from math import ceil, floor
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import anndata
from anndata import AnnData
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
from scipy.stats import truncnorm
import warnings

warnings.filterwarnings("ignore")


def data_splitter(
        adata: AnnData,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        use_gpu: bool = False,
):
    """Split indices in train/test/val sets."""
    n_train, n_val = validate_data_split(adata.n_obs, train_size, validation_size)
    random_state = np.random.RandomState(seed=42)
    permutation = random_state.permutation(adata.n_obs)
    val_idx = permutation[:n_val]
    train_idx = permutation[n_val: (n_val + n_train)]
    test_idx = permutation[(n_val + n_train):]

    train_adata = adata[train_idx]
    val_adata = adata[val_idx]
    if test_idx.shape[0] == 0:
        return train_adata, val_adata
    else:
        test_adata = adata[test_idx]
        return train_adata, val_adata, test_adata


def validate_data_split(
        n_samples: int, train_size: float, validation_size: Optional[float] = None
):
    """
    Check data splitting parameters and return n_train and n_val.

    Parameters
    ----------
    n_samples
        Number of samples to split
    train_size
        Size of train set. Need to be: 0 < train_size <= 1.
    validation_size
        Size of validation set. Need to be 0 <= validation_size < 1
    """
    if train_size > 1.0 or train_size <= 0.0:
        raise ValueError("Invalid train_size. Must be: 0 < train_size <= 1")

    n_train = ceil(train_size * n_samples)

    if validation_size is None:
        n_val = n_samples - n_train
    elif validation_size >= 1.0 or validation_size < 0.0:
        raise ValueError("Invalid validation_size. Must be 0 <= validation_size < 1")
    elif (train_size + validation_size) > 1:
        raise ValueError("train_size + validation_size must be between 0 and 1")
    else:
        n_val = floor(n_samples * validation_size)

    if n_train == 0:
        raise ValueError(
            "With n_samples={}, train_size={} and validation_size={}, the "
            "resulting train set will be empty. Adjust any of the "
            "aforementioned parameters.".format(n_samples, train_size, validation_size)
        )

    return n_train, n_val


def batch_sampler(
        adata: AnnData,
        batch_size: int,
        shuffle: bool = False,
        drop_last: Union[bool, int] = False,
):
    """
    Custom torch Sampler that returns a list of indices of size batch_size.

    Parameters
    ----------
    adata
        adata to sample from
    batch_size
        batch size of each iteration
    shuffle
        if ``True``, shuffles indices before sampling
    drop_last
        if int, drops the last batch if its length is less than drop_last.
        if drop_last == True, drops last non-full batch.
        if drop_last == False, iterate over all batches.
    """
    if drop_last > batch_size:
        raise ValueError(
            "drop_last can't be greater than batch_size. "
            + "drop_last is {} but batch_size is {}.".format(drop_last, batch_size)
        )

    last_batch_len = adata.n_obs % batch_size
    if (drop_last is True) or (last_batch_len < drop_last):
        drop_last_n = last_batch_len
    elif (drop_last is False) or (last_batch_len >= drop_last):
        drop_last_n = 0
    else:
        raise ValueError("Invalid input for drop_last param. Must be bool or int.")

    if shuffle is True:
        idx = torch.randperm(adata.n_obs).tolist()
    else:
        idx = torch.arange(adata.n_obs).tolist()

    if drop_last_n != 0:
        idx = idx[:-drop_last_n]

    adata_iter = [
        adata[idx[i: i + batch_size]] for i in range(0, len(idx), batch_size)
    ]
    return adata_iter


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


def generate_synthetic_jersey_nb(
        # simulating data
        n_samples=200,
        oversampling_factor=5,
        n_up_features=100,
        n_down_features=100,
        n_causal=10,
        n_hidden=5,
        n_latent=5,
        noise_scale=0.1,
        causal_strength=0.5,
        direct_ratio=0.0,
        is_linear=False,
        dist='nb',
        nb_shape=10.0,
        p_zero=0.5
):
    if not 0 <= causal_strength <= 10:
        raise ValueError("causal_strength must be between 0 and 10")

    total_samples = n_samples * oversampling_factor
    pos_samples = n_samples // 2
    neg_samples = n_samples - pos_samples
    # 1. Generate Z and Xc
    z = np.random.standard_normal(size=(total_samples, n_latent))

    # mean_values1 = np.random.uniform(low=0.5, high=1.5, size=n_causal)
    # std_devs1 = np.random.uniform(low=1, high=2, size=n_causal)
    # Xc = np.random.normal(loc=mean_values1, scale=std_devs1, size=(total_samples, n_causal))
    c = np.random.standard_normal(size=(total_samples, n_latent))
    weights1_c = np.random.standard_normal(size=(n_latent, n_causal))
    weights2_c = np.random.standard_normal(size=(n_causal, n_causal))
    Xc = np.dot(np.dot(c, weights1_c), weights2_c)

    # Xc = np.random.exponential(scale=mean_values1, size=(total_samples, n_causal))

    # 2. Generate Xs and Xd1 from Z
    # 2.1. Construct weight layers
    # Xs
    weights1 = np.random.standard_normal(size=(n_latent, n_hidden))
    weights2 = np.random.standard_normal(size=(n_hidden, (n_up_features - n_causal)))
    # Xd1
    weights3 = np.random.standard_normal(size=(n_latent, n_hidden))
    weights4 = np.random.standard_normal(size=(n_hidden, n_down_features))

    # 2.2. Generate Xs and Xd1
    noise1 = np.random.laplace(scale=0.5, size=(total_samples, n_down_features))
    Xs = np.dot(np.dot(z, weights1), weights2)
    # y2 = np.dot(np.dot(z, weights3), weights4) + noise1
    if is_linear:
        Xd1 = np.dot(np.dot(z, weights3), weights4) + noise1
    else:
        Xd1 = apply_activation(np.dot(apply_activation(np.dot(z, weights3), "tanh"), weights4), "tanh") + noise1

    # 3、Xc生成Xd2，将Xd1和Xd2直接叠加
    # Xc2 = np.zeros((total_samples, n_causal))
    # for i in range(n_causal):
    #     related_feature = Xs.dot(np.random.rand(int(n_features - n_causal), 1))
    #     Xc2[:, i] = related_feature.squeeze()
    # Xc2 = zscore_normalization(Xc2)
    # Xc += Xc2

    # 3. Generate Xd2 from Xc and combine Xd1 and Xd2
    weights7 = np.random.standard_normal(size=(n_causal, n_hidden))
    weights8 = np.random.standard_normal(size=(n_hidden, n_down_features))
    noise3 = np.random.laplace(scale=0.5, size=(total_samples, n_down_features))

    if is_linear:
        # Xd2 = Xc.dot(np.random.rand(n_causal, n_down_features))
        Xd2 = np.dot(np.dot(Xc, weights7), weights8) + noise3
    else:
        Xd2 = apply_activation(np.dot(apply_activation(np.dot(Xc, weights7), "tanh"), weights8), "tanh") + noise3

    rate = causal_strength
    Xd = rate * Xd2 + (1 - rate) * Xd1
    Xd = zscore_normalization(Xd)

    # 4. Generate y from Xd (and Xc)
    weights5 = np.random.standard_normal(size=(n_down_features, n_hidden))
    weights6 = np.random.standard_normal(size=(n_hidden, 1))
    noise2 = np.random.laplace(scale=0.5, size=(total_samples, 1))
    # y1 = np.dot(np.dot(Xc, weights5), weights6) + noise2
    if is_linear:
        y1 = np.dot(np.dot(Xd, weights5), weights6) + noise2
    else:
        y1 = apply_activation(np.dot(apply_activation(np.dot(Xd, weights5), "tanh"), weights6), "tanh") + noise2

    # direct effect from Xc to y
    weights9 = np.random.standard_normal(size=(n_causal, n_hidden))
    weights10 = np.random.standard_normal(size=(n_hidden, 1))
    noise4 = np.random.laplace(scale=0.5, size=(total_samples, 1))
    if is_linear:
        y2 = np.dot(np.dot(Xc, weights9), weights10) + noise4
    else:
        y2 = apply_activation(np.dot(apply_activation(np.dot(Xc, weights9), "tanh"), weights10), "tanh") + noise4

    y = (1 - direct_ratio) * y1 + direct_ratio * y2
    # y = y1

    # 5. Combine and normalize data
    data = np.hstack((Xc, Xs, Xd))
    noise = np.random.laplace(scale=noise_scale, size=(total_samples, n_up_features + n_down_features))
    data = zscore_normalization(data + noise)
    data = (data - data.min()) / (data.max() - data.min() + 1e-6)  # 归一化到 [0,1]

    if dist == 'zinb':
        prob_zero = np.random.uniform(0, 1,
                                size=(data.shape[0], data.shape[1])) < p_zero  # Generate a random array and compare with p_zero
        # Negative binomial part
        r = nb_shape
        neg_binom_part = np.random.negative_binomial(r, r / (data + r))  # NxJ
        # Apply the zero inflation: wherever prob_zero is True, set the value to zero
        data = np.where(prob_zero, 0, neg_binom_part)
    elif dist == 'nb':
        r = nb_shape
        data = np.random.negative_binomial(r, r / (data + r))


    # 6. Construct labels and return AnnData object
    sorted_indices = np.argsort(y, axis=0).flatten()
    top_indices = sorted_indices[-pos_samples:]
    down_indices = sorted_indices[:neg_samples]
    indices = np.concatenate([top_indices, down_indices])

    labels = np.where(y > np.median(y), 1, 0)
    # threshold = np.percentile(y, 25)
    # labels = np.where(y <= threshold, 1, 0)
    # indices = np.random.permutation(total_samples)

    feature_names = ["f_" + str(i + 1) for i in range(n_up_features)]
    feature_types = np.repeat(['causal', 'spurious'], [n_causal, n_up_features - n_causal])

    adata = AnnData(data[indices, 0:n_up_features], dtype=data.dtype)
    adata.obs["labels"] = labels[indices]
    adata.var["feat_type"] = feature_types
    adata.var["feat_label"] = (feature_types == 'causal').astype(int)
    adata.var.index = feature_names
    adata.obsm["X_down"] = data[indices, n_up_features:]

    return adata


def generate_synthetic_jersey_multi(
        n_samples=200,
        oversampling_factor=5,
        n_up_features=100,
        n_down_features=100,
        n_causal1=10,
        n_causal2=20,
        n_hidden=5,
        n_latent=5,
        noise_scale=0.1,
        causal_strength=0.5,
        is_linear=True,
):
    if not 0 <= causal_strength <= 10:
        raise ValueError("causal_strength must be between 0 and 10")

    total_samples = n_samples * oversampling_factor
    pos_samples = n_samples // 2
    neg_samples = n_samples - pos_samples

    # 1. Generate Z and Xc1
    z = np.random.standard_normal(size=(total_samples, n_latent))
    c = np.random.standard_normal(size=(total_samples, n_latent))
    weights1_c = np.random.standard_normal(size=(n_latent, n_causal1))
    weights2_c = np.random.standard_normal(size=(n_causal1, n_causal1))
    Xc1 = np.dot(np.dot(c, weights1_c), weights2_c)


    # 2. Generate Xs and Xc21 from Z
    # Xs
    weights1 = np.random.standard_normal(size=(n_latent, n_hidden))
    weights2 = np.random.standard_normal(size=(n_hidden, (n_up_features - n_causal1 - n_causal2)))
    # Xc21
    weights3 = np.random.standard_normal(size=(n_latent, n_hidden))
    weights4 = np.random.standard_normal(size=(n_hidden, n_causal2))

    noise1 = np.random.laplace(scale=0.5, size=(total_samples, n_causal2))
    Xs = np.dot(np.dot(z, weights1), weights2)
    if is_linear:
        Xc21 = np.dot(np.dot(z, weights3), weights4) + noise1
    else:
        Xc21 = apply_activation(np.dot(apply_activation(np.dot(z, weights3), "tanh"), weights4), "tanh") + noise1


    # 3. Generate Xc22 from Xc1 and combine Xc21 and Xc22
    weights7 = np.random.standard_normal(size=(n_causal1, n_hidden))
    weights8 = np.random.standard_normal(size=(n_hidden, n_causal2))
    noise3 = np.random.laplace(scale=0.5, size=(total_samples, n_causal2))

    if is_linear:
        Xc22 = np.dot(np.dot(Xc1, weights7), weights8) + noise3
    else:
        Xc22 = apply_activation(np.dot(apply_activation(np.dot(Xc1, weights7), "tanh"), weights8), "tanh") + noise3

    rate = causal_strength
    Xd = rate * Xc22 + (1 - rate) * Xc21
    Xc2 = zscore_normalization(Xd)

    # 3.5 Generate Xd from Xc2
    weights9 = np.random.standard_normal(size=(n_causal2, n_hidden))
    weights10 = np.random.standard_normal(size=(n_hidden, n_down_features))
    noise4 = np.random.laplace(scale=0.5, size=(total_samples, n_down_features))
    Xd = np.dot(np.dot(Xc2, weights9), weights10) + noise4


    # 4. Generate y from Xd
    weights5 = np.random.standard_normal(size=(n_down_features, n_hidden))
    weights6 = np.random.standard_normal(size=(n_hidden, 1))
    noise2 = np.random.laplace(scale=0.5, size=(total_samples, 1))
    # y1 = np.dot(np.dot(Xc, weights5), weights6) + noise2
    if is_linear:
        y1 = np.dot(np.dot(Xd, weights5), weights6) + noise2
    else:
        y1 = apply_activation(np.dot(apply_activation(np.dot(Xd, weights5), "tanh"), weights6), "tanh") + noise2

    # # direct effect from Xc to y
    # weights9 = np.random.standard_normal(size=(n_causal, n_hidden))
    # weights10 = np.random.standard_normal(size=(n_hidden, 1))
    # noise4 = np.random.laplace(scale=0.5, size=(total_samples, 1))
    # if is_linear:
    #     y2 = np.dot(np.dot(Xc, weights9), weights10) + noise4
    # else:
    #     y2 = apply_activation(np.dot(apply_activation(np.dot(Xc, weights9), "tanh"), weights10), "tanh") + noise4
    #
    # y = (1 - direct_ratio) * y1 + direct_ratio * y2
    y = y1

    # 5. Combine and normalize data
    data = np.hstack((Xc1, Xc2, Xs, Xd))
    noise = np.random.laplace(scale=noise_scale, size=(total_samples, n_up_features + n_down_features))
    data = zscore_normalization(data) + noise

    # 6. Construct labels and return AnnData object
    sorted_indices = np.argsort(y, axis=0).flatten()
    top_indices = sorted_indices[-pos_samples:]
    down_indices = sorted_indices[:neg_samples]
    indices = np.concatenate([top_indices, down_indices])

    labels = np.where(y > np.median(y), 1, 0)
    # threshold = np.percentile(y, 25)
    # labels = np.where(y <= threshold, 1, 0)
    # indices = np.random.permutation(total_samples)

    feature_names = ["f_" + str(i + 1) for i in range(n_up_features)]
    feature_types = np.repeat(['causal1', 'causal2', 'spurious'], [n_causal1, n_causal2, n_up_features - n_causal1 - n_causal2])

    adata = AnnData(data[indices, 0:n_up_features], dtype=data.dtype)
    adata.obs["labels"] = labels[indices]
    adata.var["feat_type"] = feature_types
    adata.var["feat_label"] = (feature_types == 'causal').astype(int)
    adata.var.index = feature_names
    adata.obsm["X_down"] = data[indices, n_up_features:]
    adata.obsm["X_causal1"] = data[indices, :n_causal1]
    adata.obsm["X_causal2"] = data[indices, n_causal1:n_causal1+n_causal2]

    return adata


def generate_synthetic_zhang_multi_1(
        n_samples=200,
        oversampling_factor=5,
        n_causal_features_3=50,
        n_causal_features_2=50,
        n_causal_features_1=50,
        n_spurious_features_3=100,
        n_spurious_features_2=100,
        n_spurious_features_1=100,
        n_hidden=5,
        n_latent=5,
        noise_scale=0.1,
        dist='nb',
        nb_shape=10.0,
        p_zero=0.5
):

    total_samples = n_samples * oversampling_factor
    pos_samples = n_samples // 2
    neg_samples = n_samples - pos_samples

    # 1. Generate Z and Xc3
    z = np.random.standard_normal(size=(total_samples, n_latent))
    c = np.random.standard_normal(size=(total_samples, n_latent))
    weights1_c = np.random.standard_normal(size=(n_latent, n_causal_features_3))
    weights2_c = np.random.standard_normal(size=(n_causal_features_3, n_causal_features_3))
    Xc3 = np.dot(np.dot(c, weights1_c), weights2_c)


    # 2. Generate Z -> Xs3 -> 2 -> 1
    # Xs3
    weights1 = np.random.standard_normal(size=(n_latent, n_hidden))
    weights2 = np.random.standard_normal(size=(n_hidden, n_spurious_features_3))
    Xs3 = np.dot(np.dot(z, weights1), weights2)

    #  Xs2
    weights3 = np.random.standard_normal(size=(n_spurious_features_3, n_hidden))
    weights4 = np.random.standard_normal(size=(n_hidden, n_spurious_features_2))
    Xs2 = np.dot(np.dot(Xs3, weights3), weights4)

    #  Xs1
    weights5 = np.random.standard_normal(size=(n_spurious_features_2, n_hidden))
    weights6 = np.random.standard_normal(size=(n_hidden, n_spurious_features_1))
    Xs1 = np.dot(np.dot(Xs2, weights5), weights6)

    # Xc3 -> Xc2 -> Xc1
    weights7 = np.random.standard_normal(size=(n_causal_features_3, n_hidden))
    weights8 = np.random.standard_normal(size=(n_hidden, n_causal_features_2))
    Xc2 = np.dot(np.dot(Xc3, weights7), weights8)

    weights9 = np.random.standard_normal(size=(n_causal_features_2, n_hidden))
    weights10 = np.random.standard_normal(size=(n_hidden, n_causal_features_1))
    Xc1 = np.dot(np.dot(Xc2, weights9), weights10)

    # generate y from Xc1 and z
    tmp = np.concatenate([Xc1, z], axis=1)
    weights11 = np.random.standard_normal(size=(n_latent + n_causal_features_1, n_hidden))
    weights12 = np.random.standard_normal(size=(n_hidden, 1))
    Y = np.dot(np.dot(tmp, weights11), weights12)


    # 5. Combine and normalize data
    data = np.hstack((Xc3, Xc2, Xc1, Xs3, Xs2, Xs1))
    noise = np.random.laplace(scale=noise_scale, size=(total_samples, n_causal_features_3 + n_causal_features_2 + n_causal_features_1 + n_spurious_features_3 + n_spurious_features_2 + n_spurious_features_1))
    data = zscore_normalization(data + noise)
    data = (data - data.min()) / (data.max() - data.min() + 1e-6)  # 归一化到 [0,1]

    if dist == 'zinb':
        prob_zero = np.random.uniform(0, 1,
                                      size=(data.shape[0],
                                            data.shape[1])) < p_zero  # Generate a random array and compare with p_zero
        # Negative binomial part
        r = nb_shape
        neg_binom_part = np.random.negative_binomial(r, r / (data + r))  # NxJ
        # Apply the zero inflation: wherever prob_zero is True, set the value to zero
        data = np.where(prob_zero, 0, neg_binom_part)
    elif dist == 'nb':
        r = nb_shape
        data = np.random.negative_binomial(r, r / (data + r))

    # 6. Construct labels and return AnnData object
    sorted_indices = np.argsort(Y, axis=0).flatten()
    top_indices = sorted_indices[-pos_samples:]
    down_indices = sorted_indices[:neg_samples]
    indices = np.concatenate([top_indices, down_indices])

    labels = np.where(Y > np.median(Y), 1, 0)
    # threshold = np.percentile(y, 25)
    # labels = np.where(y <= threshold, 1, 0)
    # indices = np.random.permutation(total_samples)


    feature_names = ["f_" + str(i + 1) for i in range(n_causal_features_3+n_spurious_features_3+n_causal_features_2+n_spurious_features_2+n_causal_features_1+n_spurious_features_1)]
    feature_types = np.repeat(['causal3', 'causal2', 'causal1', 'spurious3', 'spurious2', 'spurious1'],
                              [n_causal_features_3, n_causal_features_2, n_causal_features_1, n_spurious_features_3, n_spurious_features_2, n_spurious_features_1])

    adata = AnnData(data[indices, :])
    adata.obs["labels"] = labels[indices]
    adata.var["feat_type"] = feature_types
    adata.var.index = feature_names

    return adata


def generate_synthetic_zhang_multi_2(
        n_samples=200,
        oversampling_factor=5,
        n_causal_features_3=50,
        n_causal_features_2=50,
        n_causal_features_1=50,
        n_spurious_features_3=100,
        n_spurious_features_2=100,
        n_spurious_features_1=100,
        n_hidden=5,
        n_latent=5,
        noise_scale=0.1,
        dist='nb',
        nb_shape=10.0,
        p_zero=0.5,
):

    total_samples = n_samples * oversampling_factor
    pos_samples = n_samples // 2
    neg_samples = n_samples - pos_samples

    # 1. Generate Z and Xc3
    z = np.random.standard_normal(size=(total_samples, n_latent))
    c = np.random.standard_normal(size=(total_samples, n_latent))
    weights1_c = np.random.standard_normal(size=(n_latent, n_causal_features_3))
    weights2_c = np.random.standard_normal(size=(n_causal_features_3, n_causal_features_3))
    Xc3 = np.dot(np.dot(c, weights1_c), weights2_c)


    # 2. Generate Z -> Xs3  2  1
    # Xs3
    weights1 = np.random.standard_normal(size=(n_latent, n_hidden))
    weights2 = np.random.standard_normal(size=(n_hidden, n_spurious_features_3))
    Xs3 = np.dot(np.dot(z, weights1), weights2)

    #  Xs2
    weights3 = np.random.standard_normal(size=(n_latent, n_hidden))
    weights4 = np.random.standard_normal(size=(n_hidden, n_spurious_features_2))
    Xs2 = np.dot(np.dot(z, weights3), weights4)

    #  Xs1
    weights5 = np.random.standard_normal(size=(n_latent, n_hidden))
    weights6 = np.random.standard_normal(size=(n_hidden, n_spurious_features_1))
    Xs1 = np.dot(np.dot(z, weights5), weights6)

    # Xc3 -> Xc2 -> Xc1
    weights7 = np.random.standard_normal(size=(n_causal_features_3, n_hidden))
    weights8 = np.random.standard_normal(size=(n_hidden, n_causal_features_2))
    Xc2 = np.dot(np.dot(Xc3, weights7), weights8)

    weights9 = np.random.standard_normal(size=(n_causal_features_2, n_hidden))
    weights10 = np.random.standard_normal(size=(n_hidden, n_causal_features_1))
    Xc1 = np.dot(np.dot(Xc2, weights9), weights10)

    # generate y from Xc1 and z
    tmp = np.concatenate([Xc1, z], axis=1)
    weights11 = np.random.standard_normal(size=(n_latent + n_causal_features_1, n_hidden))
    weights12 = np.random.standard_normal(size=(n_hidden, 1))
    Y = np.dot(np.dot(tmp, weights11), weights12)


    # 5. Combine and normalize data
    # data3 = np.hstack((Xc3, Xs3))
    # noise = np.random.laplace(scale=noise_scale, size=(total_samples, n_causal_features_3 + n_spurious_features_3))
    # data3 = zscore_normalization(data3) + noise
    #
    # data2 = np.hstack((Xc2, Xs2))
    # noise = np.random.laplace(scale=noise_scale, size=(total_samples, n_causal_features_2 + n_spurious_features_2))
    # data2 = zscore_normalization(data2) + noise
    #
    # data1 = np.hstack((Xc1, Xs1))
    # noise = np.random.laplace(scale=noise_scale, size=(total_samples, n_causal_features_1 + n_spurious_features_1))
    # data1 = zscore_normalization(data1) + noise
    data = np.hstack((Xc3, Xc2, Xc1, Xs3, Xs2, Xs1))
    noise = np.random.laplace(scale=noise_scale, size=(total_samples,
                                                       n_causal_features_3 + n_causal_features_2 + n_causal_features_1 + n_spurious_features_3 + n_spurious_features_2 + n_spurious_features_1))
    data = zscore_normalization(data + noise)
    data = (data - data.min()) / (data.max() - data.min() + 1e-6)  # 归一化到 [0,1]

    if dist == 'zinb':
        prob_zero = np.random.uniform(0, 1,
                                      size=(data.shape[0],
                                            data.shape[1])) < p_zero  # Generate a random array and compare with p_zero
        # Negative binomial part
        r = nb_shape
        neg_binom_part = np.random.negative_binomial(r, r / (data + r))  # NxJ
        # Apply the zero inflation: wherever prob_zero is True, set the value to zero
        data = np.where(prob_zero, 0, neg_binom_part)
    elif dist == 'nb':
        r = nb_shape
        data = np.random.negative_binomial(r, r / (data + r))



    # 6. Construct labels and return AnnData object
    sorted_indices = np.argsort(Y, axis=0).flatten()
    top_indices = sorted_indices[-pos_samples:]
    down_indices = sorted_indices[:neg_samples]
    indices = np.concatenate([top_indices, down_indices])

    labels = np.where(Y > np.median(Y), 1, 0)
    # threshold = np.percentile(y, 25)
    # labels = np.where(y <= threshold, 1, 0)
    # indices = np.random.permutation(total_samples)


    # feature_names = ["f_" + str(i + 1) for i in range(n_causal_features_3+n_spurious_features_3)]
    # feature_types = np.repeat(['causal', 'spurious'], [n_causal_features_3, n_spurious_features_3])
    #
    # adata = AnnData(data3[indices, :])
    # adata.obs["labels"] = labels[indices]
    # adata.var["feat_type"] = feature_types
    # adata.var["feat_label"] = (feature_types == 'causal').astype(int)
    # adata.var.index = feature_names
    # adata.obsm["layer2"] = data2[indices, :]
    # adata.uns["layer2_feature_type"] = np.repeat(['causal', 'spurious'], [n_causal_features_2, n_spurious_features_2])
    # adata.obsm["layer1"] = data1[indices, :]
    # adata.uns["layer1_feature_type"] = np.repeat(['causal', 'spurious'], [n_causal_features_1, n_spurious_features_1])

    feature_names = ["f_" + str(i + 1) for i in range(n_causal_features_3+n_spurious_features_3+n_causal_features_2+n_spurious_features_2+n_causal_features_1+n_spurious_features_1)]
    feature_types = np.repeat(['causal3', 'causal2', 'causal1', 'spurious3', 'spurious2', 'spurious1'],
                              [n_causal_features_3, n_causal_features_2, n_causal_features_1, n_spurious_features_3, n_spurious_features_2, n_spurious_features_1])

    adata = AnnData(data[indices, :])
    adata.obs["labels"] = labels[indices]
    adata.var["feat_type"] = feature_types
    adata.var.index = feature_names

    return adata


def generate_synthetic_zhang_multi_3(
        n_samples=200,
        oversampling_factor=5,
        n_causal_features_3=50,
        n_causal_features_2=50,
        n_causal_features_1=50,
        n_spurious_features_3=100,
        n_spurious_features_2=100,
        n_spurious_features_1=100,
        n_hidden=5,
        n_latent=5,
        noise_scale=0.1,
        dist='nb',
        nb_shape=10.0,
        p_zero=0.5,
):

    total_samples = n_samples * oversampling_factor
    pos_samples = n_samples // 2
    neg_samples = n_samples - pos_samples

    # 1. Generate Xs3 and Xc3
    c = np.random.standard_normal(size=(total_samples, n_latent))
    weights1_c = np.random.standard_normal(size=(n_latent, n_causal_features_3))
    weights2_c = np.random.standard_normal(size=(n_causal_features_3, n_causal_features_3))
    Xc3 = np.dot(np.dot(c, weights1_c), weights2_c)


    # Xs3
    z = np.random.standard_normal(size=(total_samples, n_latent))
    weights1 = np.random.standard_normal(size=(n_latent, n_hidden))
    weights2 = np.random.standard_normal(size=(n_hidden, n_spurious_features_3))
    Xs3 = np.dot(np.dot(z, weights1), weights2)

    #  Xs2
    z = np.random.standard_normal(size=(total_samples, n_latent))
    weights3 = np.random.standard_normal(size=(n_latent, n_hidden))
    weights4 = np.random.standard_normal(size=(n_hidden, n_spurious_features_2))
    Xs2 = np.dot(np.dot(z, weights3), weights4)

    #  Xs1
    z = np.random.standard_normal(size=(total_samples, n_latent))
    weights5 = np.random.standard_normal(size=(n_latent, n_hidden))
    weights6 = np.random.standard_normal(size=(n_hidden, n_spurious_features_1))
    Xs1 = np.dot(np.dot(z, weights5), weights6)

    # Xc3 -> Xc2 -> Xc1
    weights7 = np.random.standard_normal(size=(n_causal_features_3, n_hidden))
    weights8 = np.random.standard_normal(size=(n_hidden, n_causal_features_2))
    Xc2 = np.dot(np.dot(Xc3, weights7), weights8)

    weights9 = np.random.standard_normal(size=(n_causal_features_2, n_hidden))
    weights10 = np.random.standard_normal(size=(n_hidden, n_causal_features_1))
    Xc1 = np.dot(np.dot(Xc2, weights9), weights10)

    # generate y from Xc1
    weights11 = np.random.standard_normal(size=(n_causal_features_1, n_hidden))
    weights12 = np.random.standard_normal(size=(n_hidden, 1))
    Y = np.dot(np.dot(Xc1, weights11), weights12)


    # 5. Combine and normalize data
    data = np.hstack((Xc3, Xc2, Xc1, Xs3, Xs2, Xs1))
    noise = np.random.laplace(scale=noise_scale, size=(total_samples,
                                                       n_causal_features_3 + n_causal_features_2 + n_causal_features_1 + n_spurious_features_3 + n_spurious_features_2 + n_spurious_features_1))
    data = zscore_normalization(data + noise)
    data = (data - data.min()) / (data.max() - data.min() + 1e-6)  # 归一化到 [0,1]

    if dist == 'zinb':
        prob_zero = np.random.uniform(0, 1,
                                      size=(data.shape[0],
                                            data.shape[1])) < p_zero  # Generate a random array and compare with p_zero
        # Negative binomial part
        r = nb_shape
        neg_binom_part = np.random.negative_binomial(r, r / (data + r))  # NxJ
        # Apply the zero inflation: wherever prob_zero is True, set the value to zero
        data = np.where(prob_zero, 0, neg_binom_part)
    elif dist == 'nb':
        r = nb_shape
        data = np.random.negative_binomial(r, r / (data + r))


    # 6. Construct labels and return AnnData object
    sorted_indices = np.argsort(Y, axis=0).flatten()
    top_indices = sorted_indices[-pos_samples:]
    down_indices = sorted_indices[:neg_samples]
    indices = np.concatenate([top_indices, down_indices])

    labels = np.where(Y > np.median(Y), 1, 0)
    # threshold = np.percentile(y, 25)
    # labels = np.where(y <= threshold, 1, 0)
    # indices = np.random.permutation(total_samples)


    feature_names = ["f_" + str(i + 1) for i in range(n_causal_features_3+n_spurious_features_3+n_causal_features_2+n_spurious_features_2+n_causal_features_1+n_spurious_features_1)]
    feature_types = np.repeat(['causal3', 'causal2', 'causal1', 'spurious3', 'spurious2', 'spurious1'],
                              [n_causal_features_3, n_causal_features_2, n_causal_features_1, n_spurious_features_3, n_spurious_features_2, n_spurious_features_1])

    adata = AnnData(data[indices, :])
    adata.obs["labels"] = labels[indices]
    adata.var["feat_type"] = feature_types
    adata.var.index = feature_names

    return adata