import logging
import os.path
import warnings
from collections.abc import Iterable as IterableClass
from functools import partial
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union, Literal

import random
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from anndata import AnnData
from tqdm import tqdm
import shap
from torch.distributions import Normal, Poisson
from scipy.linalg import norm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients
from scipy.sparse import load_npz
from statsmodels.stats.multitest import multipletests

from cautrigger.dataloaders import data_splitter, batch_sampler
from cautrigger.module import DualVAE1L, DualVAE2L, DualVAE3L
from cautrigger.causaleffect import joint_uncond_v2, beta_info_flow_v2, joint_uncond_single_dim_v2
from cautrigger.utils import set_seed


class CauTrigger1L(nn.Module):
    r"""First-layer causal generative model for perturbation-response modelling.
    CauTrigger1L wraps a single-layer DualVAE1L module and provides training and
    interpretation utilities for modelling how features signals causally affect 
    state. The model is designed for scenarios where all features are in one layer
    (first-layer decomposition).

    Parameters
    ----------
    adata
        Annotated data matrix containing upstream features in ``adata.X`` and any
        downstream or auxiliary arrays in ``adata.obsm`` as required by the module.
    n_latent
        Dimensionality of the latent space (default: 10).
    n_causal
        Number of causal latent factors (default: 2).
    n_state
        Number of discrete states the model may represent (default: 2).
    **model_kwargs
        Additional keyword arguments forwarded to the underlying DualVAE1L module.
    """

    def __init__(
            self,
            adata: AnnData,
            n_latent: int = 10,
            n_causal: int = 2,  # Number of casual factors
            n_state: int = 2,  # Number of states
            **model_kwargs,
    ):
        super(CauTrigger1L, self).__init__()
        self.adata = adata
        self.train_adata = None
        self.val_adata = None
        self.n_latent = n_latent
        self.n_causal = n_causal
        self.n_state = n_state
        self.batch_size = None
        self.ce_params = None
        self.history = {}

        self.module = DualVAE1L(
            n_input_up=adata.X.shape[1],
            n_latent=n_latent,
            n_causal=n_causal,
            n_state=n_state,
            **model_kwargs,
        )

    def train(
            self,
            max_epochs: Optional[int] = 400,
            lr: float = 5e-4,
            use_gpu: Optional[Union[str, int, bool]] = None,
            train_size: float = 1.0,
            validation_size: Optional[float] = None,
            batch_size: int = 128,
            early_stopping: bool = False,
            weight_decay: float = 1e-6,
            n_x: int = 5,
            n_alpha: int = 25,
            n_beta: int = 100,
            recons_weight: float = 1.0,
            kl_weight: float = 0.02,
            up_weight: float = 1.0,
            down_weight: float = 1.0,
            feat_l1_weight: float = 0.05,
            dpd_weight: float = 3.0,
            fide_kl_weight: float = 0.05,
            causal_weight: float = 1.0,
            down_fold: float = 1.0,
            causal_fold: float = 1.0,
            spurious_fold: float = 1.0,
            stage_training: bool = True,
            weight_scheme: str = None,
            im_factor: Optional[float] = None,
            drop_last: int = 10,
            **kwargs,
    ):
        r"""Trains the model using fractal variational autoencoder.
        
        Parameters
        ----------
        max_epochs
            Maximum number of training epochs
        lr
            Learning rate for optimizer
        use_gpu
            Whether to use GPU for training
        train_size
            Proportion of data to use for training
        validation_size
            Proportion of data to use for validation
        batch_size
            Number of samples per batch
        early_stopping
            Whether to use early stopping
        weight_decay
            Weight decay for optimizer
        n_x
            Number of samples for causal effect computation
        n_alpha
            Monte-carlo samples per causal factor
        n_beta
            Monte-carlo samples per noncausal factor
        recons_weight
            Weight for reconstruction loss
        kl_weight
            Weight for KL divergence loss
        up_weight
            Weight for upstream reconstruction
        down_weight
            Weight for downstream reconstruction
        feat_l1_weight
            Weight for feature L1 loss
        dpd_weight
            Weight for DPD loss
        fide_kl_weight
            Weight for fidelity KL loss
        causal_weight
            Weight for causal loss
        down_fold
            Downstream loss scaling factor
        causal_fold
            Causal loss scaling factor
        spurious_fold
            Spurious loss scaling factor
        stage_training
            Whether to use staged training
        weight_scheme
            Weight update scheme
        im_factor
            Imbalance factor for loss computation
        drop_last
            Number of samples to drop from last batch
        **kwargs
            Additional arguments
        """
        # set_seed(42)
        # torch.autograd.set_detect_anomaly(True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.module.to(device)
        train_adata, val_adata = data_splitter(
            self.adata,
            train_size=train_size,
            validation_size=validation_size,
            use_gpu=use_gpu,
        )
        self.train_adata, self.val_adata = train_adata, val_adata
        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])
        ce_params = {
            'N_alpha': n_alpha,
            'N_beta': n_beta,
            'K': self.n_causal,
            'L': self.n_latent - self.n_causal,
            'z_dim': self.n_latent,
            'M': self.n_state}
        self.ce_params = ce_params
        loss_weights = {
            'up_rec_loss': up_weight * recons_weight,
            'down_rec_loss': down_weight * recons_weight,
            'up_kl_loss': kl_weight,
            'feat_l1_loss_up': feat_l1_weight,
            'dpd_loss': dpd_weight,
            'fide_kl_loss': fide_kl_weight,
            'causal_loss': causal_weight,
        }

        self.batch_size = batch_size
        optimizer = optim.Adam(self.module.parameters(), lr=lr, weight_decay=weight_decay)
        epoch_losses = {'total_loss': [], 'up_rec_loss1': [], 'up_rec_loss2': [], 'down_rec_loss': [], 'up_kl_loss': [],
                        'feat_l1_loss_up': [], 'feat_l1_loss_down': [], 'dpd_loss': [], 'fide_kl_loss': [],
                        'causal_loss': []}
        self.module.train()
        for epoch in tqdm(range(max_epochs), desc="training", disable=False):
            train_adata_batch = batch_sampler(train_adata, batch_size, shuffle=True, drop_last=drop_last)
            batch_losses = {'total_loss': [], 'up_rec_loss1': [], 'up_rec_loss2': [], 'down_rec_loss': [],
                            'up_kl_loss': [], 'feat_l1_loss_up': [], 'feat_l1_loss_down': [], 'dpd_loss': [],
                            'fide_kl_loss': [], 'causal_loss': []}
            if stage_training:
                # loss_weights = self.module.update_loss_weights_sc(epoch, max_epochs, loss_weights)
                loss_weights = self.module.update_loss_weights(epoch, max_epochs, scheme=weight_scheme)
            for train_batch in train_adata_batch:
                inputs_up = torch.tensor(train_batch.X, dtype=torch.float32, device=device)
                # inputs_down = torch.tensor(train_batch.obsm['X_down'], dtype=torch.float32, device=device)
                labels = torch.tensor(train_batch.obs['labels'], dtype=torch.float32, device=device)
                model_outputs = self.module(inputs_up)
                loss_dict = self.module.compute_loss(model_outputs, inputs_up, labels, imb_factor=im_factor)

                causal_loss_list = []
                for idx in np.random.permutation(train_batch.shape[0])[:n_x]:
                    if loss_weights["causal_loss"] == 0:
                        causal_loss_list = [torch.tensor(0.0, device=device)]
                        break
                    _causal_loss1, _ = joint_uncond_v2(ce_params, self.module, inputs_up, idx, alpha_vi=True,
                                                       beta_vi=True, device=device)
                    _causal_loss2, _ = beta_info_flow_v2(ce_params, self.module, inputs_up, idx, alpha_vi=True,
                                                         beta_vi=False, device=device)
                    _causal_loss = _causal_loss1 * causal_fold - _causal_loss2 * spurious_fold
                    # _causal_loss = _causal_loss1 - _causal_loss2 * 3.0
                    causal_loss_list += [_causal_loss]
                up_rec_loss1 = loss_dict['up_rec_loss1'].mean()
                up_rec_loss2 = loss_dict['up_rec_loss2'].mean()
                down_rec_loss = loss_dict['down_rec_loss'].mean()
                up_kl_loss = loss_dict['up_kl_loss'].mean()
                feat_l1_loss_up = loss_dict['feat_l1_loss_up'].mean()
                feat_l1_loss_down = loss_dict['feat_l1_loss_down'].mean()
                dpd_loss = loss_dict['dpd_loss'].mean()
                fide_kl_loss = loss_dict['fide_kl_loss'].mean()
                causal_loss = torch.stack(causal_loss_list).mean()
                if self.module.feature_mapper_up.attention:
                    loss_weights["feat_l1_loss_up"] = 0.001
                total_loss = loss_weights['up_rec_loss'] * up_rec_loss1 + \
                             loss_weights['up_rec_loss'] * up_rec_loss2 + \
                             loss_weights['down_rec_loss'] * down_rec_loss + \
                             loss_weights['up_kl_loss'] * up_kl_loss + \
                             loss_weights['feat_l1_loss_up'] * feat_l1_loss_up + \
                             loss_weights['feat_l1_loss_down'] * feat_l1_loss_down * down_fold + \
                             loss_weights['dpd_loss'] * dpd_loss + \
                             loss_weights['fide_kl_loss'] * fide_kl_loss + \
                             loss_weights['causal_loss'] * causal_loss

                optimizer.zero_grad()
                # with torch.autograd.detect_anomaly():
                #     total_loss.backward()
                total_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.module.parameters(), max_norm=1.0)
                optimizer.step()

                # update batch losses
                batch_losses['total_loss'].append(total_loss.item())
                batch_losses['up_rec_loss1'].append(up_rec_loss1.item())
                batch_losses['up_rec_loss2'].append(up_rec_loss2.item())
                batch_losses['down_rec_loss'].append(down_rec_loss.item())
                batch_losses['up_kl_loss'].append(up_kl_loss.item())
                batch_losses['feat_l1_loss_up'].append(feat_l1_loss_up.item())
                batch_losses['feat_l1_loss_down'].append(feat_l1_loss_down.item())
                batch_losses['dpd_loss'].append(dpd_loss.item())
                batch_losses['fide_kl_loss'].append(fide_kl_loss.item())
                batch_losses['causal_loss'].append(causal_loss.item())

            # update epochs losses
            epoch_losses['total_loss'].append(np.mean(batch_losses['total_loss']))
            epoch_losses['up_rec_loss1'].append(np.mean(batch_losses['up_rec_loss1']))
            epoch_losses['up_rec_loss2'].append(np.mean(batch_losses['up_rec_loss2']))
            epoch_losses['down_rec_loss'].append(np.mean(batch_losses['down_rec_loss']))
            epoch_losses['up_kl_loss'].append(np.mean(batch_losses['up_kl_loss']))
            epoch_losses['feat_l1_loss_up'].append(np.mean(batch_losses['feat_l1_loss_up']))
            epoch_losses['feat_l1_loss_down'].append(np.mean(batch_losses['feat_l1_loss_down']))
            epoch_losses['dpd_loss'].append(np.mean(batch_losses['dpd_loss']))
            epoch_losses['fide_kl_loss'].append(np.mean(batch_losses['fide_kl_loss']))
            epoch_losses['causal_loss'].append(np.mean(batch_losses['causal_loss']))

            if epoch % 20 == 0 or epoch == (max_epochs - 1):
                total_loss = np.mean(batch_losses['total_loss'])
                logging.info(f"Epoch {epoch} training loss: {total_loss:.4f}")

        self.history = epoch_losses

    def pretrain_attention(
            self,
            prior_probs: Optional[np.ndarray] = None,
            max_epochs: Optional[int] = 50,
            pretrain_lr: float = 1e-3,
            batch_size: int = 128,
            use_gpu: Optional[Union[str, int, bool]] = None,
            train_size: float = 1.0,
            validation_size: Optional[float] = None
    ):
        r"""Pretrain attention network.

        Parameters
        ----------
        prior_probs
            Prior probabilities for attention weights
        max_epochs
            Maximum number of pretraining epochs
        pretrain_lr
            Learning rate for pretraining
        batch_size
            Number of samples per batch
        use_gpu
            Whether to use GPU for pretraining
        train_size
            Proportion of data to use for pretraining
        validation_size
            Proportion of data to use for validation
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.module.to(device)
        train_adata, _ = data_splitter(
            self.adata,
            train_size=train_size,
            validation_size=validation_size,
            use_gpu=use_gpu,
        )

        if prior_probs is None:
            prior_probs = np.ones(self.module.feature_mapper_up.n_features) * 0.5
        elif not isinstance(prior_probs, np.ndarray):
            prior_probs = np.array(prior_probs)

        prior_probs_tensor = torch.tensor(prior_probs, dtype=torch.float32).view(1, -1).to(device)

        criterion = torch.nn.MSELoss()
        pretrain_optimizer = torch.optim.Adam(self.module.feature_mapper_up.att_net.parameters(), lr=pretrain_lr)

        self.module.train()
        for epoch in tqdm(range(max_epochs), desc="pretraining", disable=False):
            train_adata_batch = batch_sampler(train_adata, batch_size, shuffle=True)
            for train_batch in train_adata_batch:
                inputs_up = torch.tensor(train_batch.X, dtype=torch.float32, device=device)

                attention_scores = self.module.feature_mapper_up.att_net(inputs_up)
                # Repeat prior_probs_tensor to match the batch size
                repeated_prior_probs = prior_probs_tensor.repeat(attention_scores.size(0), 1)

                loss = criterion(torch.sigmoid(attention_scores), repeated_prior_probs)

                pretrain_optimizer.zero_grad()
                loss.backward()
                pretrain_optimizer.step()

        print("Pretraining attention net completed.")

    def plot_train_losses(self, fig_size=(8, 8)):
        r"""Plot training loss curves for all recorded losses during training.

        This method visualizes the evolution of each loss component over epochs
        using subplots. It requires that the model has been trained and that
        training history is available in the `self.history` attribute.
    
        Parameters
        ----------
        fig_size : tuple of int, optional
            Figure size (width, height) in inches. Default is (8, 8).
        """
        # Set figure size
        fig = plt.figure(figsize=fig_size)
        if self.history is None:
            raise ValueError("You should train the model first!")
        epoch_losses = self.history
        # Plot a subplot of each loss
        for i, loss_name in enumerate(epoch_losses.keys()):
            # Gets the value of the current loss
            loss_values = epoch_losses[loss_name]
            # Create subplot
            ax = fig.add_subplot(3, 4, i + 1)
            # Draw subplot
            ax.plot(range(len(loss_values)), loss_values)
            # Set the subplot title
            ax.set_title(loss_name)
            # Set the subplot x-axis and y-axis labels
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')

        # adjust the distance and edges between sub-graphs
        plt.tight_layout()
        # show figure
        plt.show()

    def get_up_feature_weights(
            self,
            adata: Optional[AnnData] = None,
            method: Optional[str] = "SHAP",
            n_bg_samples: Optional[int] = 100,
            grad_source: Optional[str] = "prob",
            normalize: Optional[bool] = True,
            sort_by_weight: Optional[bool] = True,
            class_idx: Optional[int] = None,
            background_data: Optional[torch.Tensor] = None,
            return_background: Optional[bool] = False,
    ):
        r"""Compute and return feature importance weights for the upstream feature mapper.
    
        This method supports multiple strategies to estimate feature contributions:
        - **"Model"**: uses internal attention or learned weights from the model.
        - **"SHAP"**: computes SHAP values using DeepExplainer.
        - **"Grad"**: computes input gradients w.r.t. a specified output (e.g., probability or logit).
        - **"Ensemble"**: averages normalized absolute weights from all three methods above.
    
        The resulting weights are aggregated across samples (by mean), optionally normalized,
        and returned both as a sorted DataFrame aligned with `self.adata.var` and as a full
        sample-by-feature weight matrix.
    
        Parameters
        ----------
        adata : AnnData, optional
            AnnData object containing the data to compute feature weights on.
        method : str, optional
            Method to compute feature weights. One of {"Model", "SHAP", "Grad", "Ensemble"}.
            Default is "SHAP".
        n_bg_samples : int, optional
            Number of background samples used for SHAP explanation. Only relevant if
            `method="SHAP"`. Default is 100.
        grad_source : str, optional
            Target output for gradient computation when `method="Grad"`. Options are:
            - "prob": gradients w.r.t. predicted probabilities,
            - "logit": gradients w.r.t. logits,
            - "loss": gradients w.r.t. the DPD loss.
            Default is "prob".
        normalize : bool, optional
            Whether to normalize the final feature weights to sum to 1. Default is True.
        sort_by_weight : bool, optional
            Whether to sort the returned DataFrame by weight in descending order.
            Default is True.
        class_idx : int, optional
            Class index for which to compute SHAP values (only used when labels are present
            and `method="SHAP"`). If None, SHAP values are averaged over all classes or
            computed on the full dataset. Default is None.
    
        Returns
        -------
        weights_df : pandas.DataFrame
            DataFrame with the same index as `self.adata.var`, containing a new column
            `'weight'` with the computed feature importance scores. Sorted by weight if
            `sort_by_weight=True`.
        weights_full : numpy.ndarray
            Full sample-by-feature matrix of absolute weights before aggregation.
            Shape: `(n_samples, n_features)`.
        """
        if self.module.training:
            self.module.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adata = adata if adata is not None else self.adata

        adata_batch = batch_sampler(adata, self.batch_size, shuffle=False)
        def compute_shap_weights(key="prob", class_idx=None, background_data=None):
            # key = "prob" or "logit"
            if background_data is None:
                idx = np.random.permutation(adata.shape[0])[0:n_bg_samples]
                background_data = torch.tensor(adata.X[idx], dtype=torch.float32).to(device)
            else:
                # background_data from predefined matrix
                background_data = background_data.to(device)

            model = ShapModel(self.module, key).to(device)
            explainer = shap.DeepExplainer(model, background_data)

            if class_idx is not None:
                adata_subset = adata[adata.obs['labels'] == class_idx].copy()
                inputs_up = torch.tensor(adata_subset.X, dtype=torch.float32).to(device)
            else:
                inputs_up = torch.tensor(adata.X, dtype=torch.float32, device=device)
            # shap_value = explainer.shap_values(inputs_up)
            shap_value = explainer.shap_values(inputs_up, check_additivity=False)
            if shap_value.ndim == 3 and shap_value.shape[2] > 1:
                shap_value = shap_value[..., class_idx] if class_idx is not None else shap_value.mean(axis=2, keepdims=True)

            return shap_value, background_data

        def compute_grad_weights(grad_source="prob"):
            grad_weights_full = []
            for data in adata_batch:
                inputs_up = torch.tensor(data.X, dtype=torch.float32, device=device)
                # inputs_down = torch.tensor(data.obsm['X_down'], dtype=torch.float32, device=device)
                labels = torch.tensor(data.obs['labels'], dtype=torch.float32, device=device)

                inputs_up.requires_grad = True
                model_outputs = self.module(inputs_up, use_mean=True)

                if grad_source == "loss":
                    loss_dict = self.module.compute_loss(model_outputs, inputs_up, labels)
                    dpd_loss = loss_dict['dpd_loss']
                    dpd_loss.sum().backward()  # mean()
                elif grad_source == "prob":
                    prob = model_outputs["alpha_dpd"]["prob"]  # prob
                    prob.sum().backward()
                elif grad_source == 'logit':
                    prob = model_outputs["alpha_dpd"]["logit"]
                    prob.sum().backward()
                grad_weights_full.append(inputs_up.grad.cpu().numpy())

            return np.concatenate(grad_weights_full, axis=0)

        def compute_model_weights():
            if self.module.feature_mapper_up.attention:
                attention_weights_full = []
                for data in adata_batch:
                    inputs = torch.tensor(data.X, dtype=torch.float32, device=device)
                    model_outputs = self.module(inputs, use_mean=True)
                    att_w = model_outputs["feat_w_up"].cpu().detach().numpy()
                    attention_weights_full.append(att_w)
                weight_matrix = np.concatenate(attention_weights_full, axis=0)
            else:
                weight_vector = torch.sigmoid(self.module.feature_mapper_up.weight).cpu().detach().numpy()
                # Expand weight vector to a matrix with the same weight vector repeated for each sample in adata_batch
                weight_matrix = np.tile(weight_vector, (len(self.adata), 1))
            return weight_matrix

        weights_full = None
        if method == "Model":
            weights_full = compute_model_weights()
        elif method == "SHAP":
            weights_full, bg_used = compute_shap_weights(class_idx=class_idx, background_data=background_data)
        elif method == "Grad":
            weights_full = compute_grad_weights(grad_source=grad_source)
        elif method == "Ensemble":
            model_weights = np.abs(compute_model_weights())
            shap_weights, bg_used = compute_shap_weights(class_idx=class_idx, background_data=background_data)  # ← 传进去
            grad_weights = np.abs(compute_grad_weights())

            # Normalize each set of weights
            model_sum = np.sum(model_weights, axis=1, keepdims=True)
            model_weights = np.where(model_sum != 0, model_weights / model_sum, 0)

            shap_sum = np.sum(shap_weights, axis=1, keepdims=True)
            shap_weights = np.where(shap_sum != 0, shap_weights / shap_sum, 0)

            grad_sum = np.sum(grad_weights, axis=1, keepdims=True)
            grad_weights = np.where(grad_sum != 0, grad_weights / grad_sum, 0)

            # Combine the weights
            weights_full = (model_weights + shap_weights + grad_weights) / 3

        # Get the mean of the weights for each feature
        weights = np.mean(np.abs(weights_full), axis=0)
        weights_signed = np.mean(weights_full, axis=0)

        # Normalize the weights if required
        if normalize:
            weights = weights / np.sum(weights)
            weights_signed = weights_signed / np.sum(np.abs(weights_signed))

        # Create a new DataFrame with the weights
        weights_df = self.adata.var.copy()
        weights_df['weight'] = weights
        weights_df['weight_signed'] = weights_signed  # add new column

        # Sort the DataFrame by weight if required
        if sort_by_weight:
            weights_df = weights_df.sort_values(by='weight', ascending=False)

        if return_background and method == "SHAP":
            return weights_df, weights_full, bg_used
        else:
            return weights_df, weights_full

    def get_up_significance(
            self,
            adata: Optional[AnnData] = None,
            method: str = "SHAP",  # "SHAP" or "Grad"
            test_mode: str = "permutation",  # "permutation" or "sign_test"
            perm_mode: str = "global",  # "global" or "per_feature" (only for permutation)
            n_perm: int = 100,
            n_bg_samples: int = 100,
            grad_source: str = "prob",
            normalize: bool = False,
            class_idx: Optional[int] = None,
            target_genes: Optional[List[str]] = None,
            fdr_correct: bool = True,
            use_signed: bool = True,
            show_progress: bool = True,
            random_state: Optional[int] = 42,
    ):
        r"""
        Compute significance of upstream feature weights using:
        - Grad  → Binomial sign-consistency test
        - SHAP  → Binomial sign-consistency test or permutation test

        Parameters
        ----------
        adata : AnnData, optional
            Input AnnData object (default: self.adata)
        method : str
            "SHAP" or "Grad"
        test_mode : str
            For SHAP: "sign_test" or "permutation" (ignored for Grad)
        perm_mode : str
            "global" or "per_feature" shuffle strategy (for permutation test)
        n_perm : int
            Number of permutations (for permutation test)
        n_bg_samples : int
            Number of background samples for SHAP
        grad_source : str
            Source for gradient-based attribution ("prob", "logit", or "loss")
        normalize : bool
            Whether to normalize weights across features
        class_idx : int, optional
            Target class index for class-specific analysis
        target_genes : list of str, optional
            Genes/features to test; if None, use all
        fdr_correct : bool
            Apply Benjamini–Hochberg correction
        use_signed : bool
            Whether to use signed weights for p-value calculation
        show_progress : bool
            Display progress bar
        random_state : int, optional
            Random seed

        Returns
        -------
        df_result : pd.DataFrame
            DataFrame with ['weight', 'weight_signed', 'pvalue', ('qvalue')]
        perm_matrix : np.ndarray or None
            Permutation matrix for SHAP (None for Grad or sign_test)
        """
        if random_state is not None:
            np.random.seed(random_state)

        adata = adata if adata is not None else self.adata

        # =======================================================
        # Grad → Binomial Sign-Consistency Test
        # =======================================================
        if method == "Grad":
            df_obs, weights_full = self.get_up_feature_weights(
                adata=adata,
                method=method,
                n_bg_samples=n_bg_samples,
                grad_source=grad_source,
                normalize=normalize,
                sort_by_weight=False,
                class_idx=class_idx,
            )

            n = weights_full.shape[0]
            k = (weights_full > 0).sum(axis=0)

            pvals = np.array([
                stats.binomtest(int(kk), n, p=0.5, alternative="two-sided").pvalue
                for kk in k
            ])

            mean_grad = np.mean(weights_full, axis=0).astype(float)

            df_result = pd.DataFrame({
                "weight": np.abs(mean_grad),
                "weight_signed": mean_grad,
                "pvalue": pvals,
            }, index=df_obs.index)

            if fdr_correct:
                df_result["qvalue"] = multipletests(pvals, method="fdr_bh")[1]

            print(
                "[Note] Grad-based significance computed via two-sided Binomial test on gradient sign consistency (no permutation).")
            return df_result, None

        # =======================================================
        # SHAP → Binomial Sign-Consistency Test
        # =======================================================
        elif method == "SHAP" and test_mode == "sign_test":
            df_obs, shap_matrix = self.get_up_feature_weights(
                adata=adata,
                method=method,
                n_bg_samples=n_bg_samples,
                grad_source=grad_source,
                normalize=normalize,
                sort_by_weight=False,
                class_idx=class_idx,
            )

            n = shap_matrix.shape[0]
            k = (shap_matrix > 0).sum(axis=0)

            pvals = np.array([
                stats.binomtest(int(kk), n, p=0.5, alternative="two-sided").pvalue
                for kk in k
            ])

            # Remove trailing dimension if exists
            mean_shap = np.mean(shap_matrix, axis=0).astype(float).squeeze()

            df_result = pd.DataFrame({
                "weight": np.abs(mean_shap),
                "weight_signed": mean_shap,
                "pvalue": pvals,
            }, index=df_obs.index)

            if fdr_correct:
                df_result["qvalue"] = multipletests(pvals, method="fdr_bh")[1]

            print(
                "[Note] SHAP-based significance computed via two-sided Binomial test on contribution sign consistency (no permutation).")
            return df_result, None

        # =======================================================
        # SHAP → Permutation-Based Test
        # =======================================================
        elif method == "SHAP" and test_mode == "permutation":
            df_obs, _, bg_data = self.get_up_feature_weights(
                adata=adata,
                method=method,
                n_bg_samples=n_bg_samples,
                grad_source=grad_source,
                normalize=normalize,
                sort_by_weight=False,
                class_idx=class_idx,
                return_background=True,
            )

            var_names = df_obs.index.tolist()
            if target_genes is None:
                target_genes = var_names

            perm_matrix = np.zeros((len(target_genes), n_perm))
            iterator = tqdm(range(n_perm), desc="Permuting", disable=not show_progress)

            # ---------- Global permutation ----------
            if perm_mode == "global":
                for i in iterator:
                    adata_perm = adata.copy()
                    X = adata_perm.X.copy()
                    for j in range(X.shape[1]):
                        np.random.shuffle(X[:, j])
                    adata_perm.X = X

                    df_perm, _ = self.get_up_feature_weights(
                        adata=adata_perm,
                        method=method,
                        n_bg_samples=n_bg_samples,
                        grad_source=grad_source,
                        normalize=normalize,
                        sort_by_weight=False,
                        class_idx=class_idx,
                        background_data=bg_data,
                    )
                    col = "weight_signed" if (use_signed and "weight_signed" in df_perm.columns) else "weight"
                    perm_matrix[:, i] = df_perm.loc[target_genes, col].values

            # ---------- Per-feature permutation ----------
            elif perm_mode == "per_feature":
                for g_idx, gene in enumerate(tqdm(target_genes, desc="Target genes", disable=not show_progress)):
                    if gene not in var_names:
                        continue
                    j = var_names.index(gene)
                    for i in range(n_perm):
                        adata_perm = adata.copy()
                        X = adata_perm.X.copy()
                        np.random.shuffle(X[:, j])
                        adata_perm.X = X

                        df_perm, _ = self.get_up_feature_weights(
                            adata=adata_perm,
                            method=method,
                            n_bg_samples=n_bg_samples,
                            grad_source=grad_source,
                            normalize=normalize,
                            sort_by_weight=False,
                            class_idx=class_idx,
                            background_data=bg_data,
                        )
                        col = "weight_signed" if (use_signed and "weight_signed" in df_perm.columns) else "weight"
                        perm_matrix[g_idx, i] = df_perm.loc[gene, col]

            else:
                raise ValueError("perm_mode must be 'global' or 'per_feature'.")

            # ---------- Compute empirical p-values ----------
            pvals = np.zeros(len(target_genes))
            for k, gene in enumerate(target_genes):
                obs_val = (
                    df_obs.loc[gene, "weight_signed"]
                    if (use_signed and "weight_signed" in df_obs.columns)
                    else abs(df_obs.loc[gene, "weight"])
                )
                null_dist = perm_matrix[k, :]

                if use_signed and "weight_signed" in df_obs.columns:
                    if obs_val >= 0:
                        pvals[k] = (1 + np.sum(null_dist >= obs_val)) / (n_perm + 1)
                    else:
                        pvals[k] = (1 + np.sum(null_dist <= obs_val)) / (n_perm + 1)
                else:
                    pvals[k] = (1 + np.sum(np.abs(null_dist) >= abs(obs_val))) / (n_perm + 1)

            df_result = pd.DataFrame({
                "weight": df_obs.loc[target_genes, "weight"].values,
                "weight_signed": df_obs.loc[target_genes, "weight_signed"].values,
                "pvalue": pvals,
            }, index=target_genes)

            if fdr_correct:
                df_result["qvalue"] = multipletests(pvals, method="fdr_bh")[1]

            print(
                f"[Note] SHAP-based significance computed via permutation test ({perm_mode} mode, {n_perm} permutations).")
            return df_result, perm_matrix

        # =======================================================
        # Unsupported combinations
        # =======================================================
        else:
            raise ValueError(
                f"Unsupported configuration: method='{method}', test_mode='{test_mode}'. "
                "Supported: Grad(binomial), SHAP(sign_test/permutation)."
            )

    @torch.no_grad()
    def get_model_output(
            self,
            adata: Optional[AnnData] = None,
            batch_size: Optional[int] = None,
    ):
        r"""Obtain model predictions and latent representations for a given dataset.

        This method runs the trained model in evaluation mode and returns:
        - Concatenated latent embeddings from two latent spaces,
        - Logits and predicted probabilities from the downstream classifier,
        - Binary class predictions (thresholded at 0.5).
    
        If no `adata` is provided, the method uses the internal `self.adata`.
    
        Parameters
        ----------
        adata : AnnData, optional
            Annotated data matrix to generate outputs for. If None, defaults to `self.adata`.
            Default is None.
        batch_size : int, optional
            Number of samples per batch during inference. If None, uses `self.batch_size`.
            Default is None.
    
        Returns
        -------
        output : dict
            Dictionary containing the following keys:
            - `'latent'`: numpy.ndarray of shape `(n_samples, n_latent1 + n_latent2)`,  
              concatenated latent vectors from both latent modules.
            - `'logits'`: numpy.ndarray of shape `(n_samples,)` or `(n_samples, n_classes)`,  
              raw classifier logits.
            - `'probs'`: numpy.ndarray of same shape as `'logits'`,  
              predicted probabilities after sigmoid/softmax activation.
            - `'preds'`: numpy.ndarray of shape `(n_samples,)`,  
              binary predictions (1 if probability > 0.5, else 0).

        """
        if self.module.training:
            self.module.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        latent = []
        logits = []
        probs = []
        preds = []
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            inputs = torch.tensor(data.X, dtype=torch.float32, device=device)
            model_outputs = self.module(inputs, use_mean=True)
            latent_z = torch.cat([model_outputs["latent1"]["z"], model_outputs["latent2"]["z"]], dim=1)
            latent.append(latent_z.cpu().numpy())
            # latent.append(model_outputs['latent_up']['qz_m'].cpu().numpy())
            logits.append(model_outputs['alpha_dpd']['logit'].cpu().numpy())
            probs.append(model_outputs["alpha_dpd"]["prob"].cpu().numpy())
            preds.append(np.int_(model_outputs['alpha_dpd']['prob'].cpu().numpy() > 0.5))

        output = dict(latent=np.concatenate(latent, axis=0),
                      logits=np.concatenate(logits, axis=0),
                      probs=np.concatenate(probs, axis=0),
                      preds=np.concatenate(preds, axis=0))

        return output

    @torch.no_grad()
    def compute_information_flow(
            self,
            adata: Optional[AnnData] = None,
            dims: Optional[List[int]] = None,
            zero_floor: bool = False,
            plot_info_flow: Optional[bool] = True,
            skip_single_info: Optional[bool] = True,
            save_fig: Optional[bool] = False,
            save_dir: Optional[str] = None,
    ):
        r"""
        Compute information flow for latent dimensions.
        
        Parameters
        ----------
        adata
            AnnData object with input data
        dims
            Dimensions to compute information flow for
        zero_floor
            Whether to subtract minimum value
        plot_info_flow
            Whether to plot information flow
        skip_single_info
            Whether to skip single dimension plots
        save_fig
            Whether to save figures
        save_dir
            Directory to save figures
            
        Returns
        ----------
        info_flow
            Information flow for each dimension
        info_flow_cat
            Categorical information flow (causal vs spurious)
        """
        if self.module.training:
            self.module.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adata = adata if adata is not None else self.adata
        ce_params = self.ce_params
        if dims is None:
            dims = list(range(self.module.n_latent))

        # Calculate information flow
        info_flow = pd.DataFrame(index=range(adata.shape[0]), columns=dims)
        for i in range(adata.shape[0]):
            for j in dims:
                # Get the latent space of the current sample
                inputs = torch.tensor(adata.X, dtype=torch.float32, device=device)
                # Calculate the information flow
                info = joint_uncond_single_dim_v2(ce_params, self.module, inputs, i, j, alpha_vi=False, beta_vi=True,
                                                  device=device)
                info_flow.loc[i, j] = info.item()
        info_flow.set_index(adata.obs_names, inplace=True)
        if zero_floor:
            info_flow = info_flow - info_flow.min().min()
        info_flow = info_flow.apply(lambda x: x / (np.linalg.norm(x, ord=1) + 1e-8), axis=1)

        # Calculate information flow for causal and spurious dimensions
        dims = ['causal', 'spurious']
        info_flow_cat = pd.DataFrame(index=range(adata.shape[0]), columns=dims)
        for i in range(adata.shape[0]):
            # Get the latent space of the current sample
            inputs = torch.tensor(adata.X, dtype=torch.float32, device=device)
            # Calculate the information flow
            info_c, _ = joint_uncond_v2(ce_params, self.module, inputs, i, alpha_vi=False, beta_vi=True, device=device)
            info_s, _ = beta_info_flow_v2(ce_params, self.module, inputs, i, alpha_vi=True, beta_vi=False,
                                          device=device)
            info_flow_cat.loc[i, 'causal'] = -info_c.item()
            info_flow_cat.loc[i, 'spurious'] = -info_s.item()
        info_flow_cat.set_index(adata.obs_names, inplace=True)
        if zero_floor:
            info_flow_cat = info_flow_cat - info_flow_cat.min().min()
        info_flow_cat = info_flow_cat.apply(lambda x: x / (np.linalg.norm(x, ord=1) + 1e-8), axis=1)

        if plot_info_flow and not skip_single_info:
            # plot the information flow
            plt.figure(figsize=(10, 5))
            ax = sns.boxplot(data=info_flow, palette="pastel")
            plt.xlabel("Dimensions")
            plt.ylabel("Information Measurements")
            if save_fig:
                plt.savefig(os.path.join(save_dir, "info_flow_1l.png"))
                plt.savefig(os.path.join(save_dir, "info_flow_1l.pdf"))
            plt.show()
            plt.close()

        if plot_info_flow:
            plt.figure(figsize=(10, 5))
            ax = sns.boxplot(data=info_flow_cat, palette="pastel")
            plt.xlabel("Dimensions")
            plt.ylabel("Information Measurements")
            if save_fig:
                plt.savefig(os.path.join(save_dir, "info_flow_1l_cat.png"))
                plt.savefig(os.path.join(save_dir, "info_flow_1l_cat.pdf"))
            plt.show()
            plt.close()

        return info_flow, info_flow_cat


class CauTrigger2L(nn.Module):
    r"""Second-layer hierarchical causal decomposition model.
    CauTrigger2L wraps a DualVAE2L backbone and supports a second-stage causal
    decomposition (e.g., upstream -> downstream -> state).

    Parameters
    ----------
    adata
        Annotated data matrix. Expects upstream features in ``adata.X`` and the
        first downstream representation stored in ``adata.obsm['X_down']``.
    n_latent
        Latent dimension (default: 10).
    n_causal
        Number of causal factors (default: 2).
    n_state
        Number of discrete states (default: 2).
    **model_kwargs
        Forwarded to the DualVAE2L constructor.
    """

    def __init__(
            self,
            adata: AnnData,
            n_latent: int = 10,
            n_causal: int = 2,  # Number of casual factors
            n_state: int = 2,  # Number of state
            **model_kwargs,
    ):
        super(CauTrigger2L, self).__init__()
        self.adata = adata
        self.train_adata = None
        self.val_adata = None
        self.n_latent = n_latent
        self.n_causal = n_causal
        self.n_state = n_state
        self.batch_size = None
        self.ce_params = None
        self.history = {}

        self.module = DualVAE2L(
            n_input_up=adata.X.shape[1],
            n_input_down=adata.obsm['X_down'].shape[1],
            n_latent=n_latent,
            n_causal=n_causal,
            n_state=n_state,
            **model_kwargs,
        )

    def train(
            self,
            max_epochs: Optional[int] = 400,
            lr: float = 5e-4,
            use_gpu: Optional[Union[str, int, bool]] = None,
            train_size: float = 1.0,
            validation_size: Optional[float] = None,
            batch_size: int = 128,
            early_stopping: bool = False,
            weight_decay: float = 1e-6,
            n_x: int = 5,
            n_alpha: int = 25,
            n_beta: int = 100,
            recons_weight: float = 1.0,
            kl_weight: float = 0.02,
            up_weight: float = 1.0,
            down_weight: float = 1.0,
            feat_l1_weight: float = 0.05,
            dpd_weight: float = 3.0,
            fide_kl_weight: float = 0.05,
            causal_weight: float = 1.0,
            down_fold: float = 1.0,
            causal_fold: float = 1.0,
            spurious_fold: float = 1.0,
            stage_training: bool = True,
            weight_scheme: str = None,
            im_factor: Optional[float] = None,
            drop_last: int = 10,
            **kwargs,
    ):
        r"""Trains the model using fractal variational autoencoder.
        
        Parameters
        ----------
        max_epochs
            Maximum number of training epochs
        lr
            Learning rate for optimizer
        use_gpu
            Whether to use GPU for training
        train_size
            Proportion of data to use for training
        validation_size
            Proportion of data to use for validation
        batch_size
            Number of samples per batch
        early_stopping
            Whether to use early stopping
        weight_decay
            Weight decay for optimizer
        n_x
            Number of samples for causal effect computation
        n_alpha
            Monte-carlo samples per causal factor
        n_beta
            Monte-carlo samples per noncausal factor
        recons_weight
            Weight for reconstruction loss
        kl_weight
            Weight for KL divergence loss
        up_weight
            Weight for upstream reconstruction
        down_weight
            Weight for downstream reconstruction
        feat_l1_weight
            Weight for feature L1 loss
        dpd_weight
            Weight for DPD loss
        fide_kl_weight
            Weight for fidelity KL loss
        causal_weight
            Weight for causal loss
        down_fold
            Downstream loss scaling factor
        causal_fold
            Causal loss scaling factor
        spurious_fold
            Spurious loss scaling factor
        stage_training
            Whether to use staged training
        weight_scheme
            Weight update scheme
        im_factor
            Imbalance factor for loss computation
        drop_last
            Number of samples to drop from last batch
        **kwargs
            Additional arguments
        """
        # set_seed(42)
        # torch.autograd.set_detect_anomaly(True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.module.to(device)
        train_adata, val_adata = data_splitter(
            self.adata,
            train_size=train_size,
            validation_size=validation_size,
            use_gpu=use_gpu,
        )
        self.train_adata, self.val_adata = train_adata, val_adata
        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])
        ce_params = {
            'N_alpha': n_alpha,
            'N_beta': n_beta,
            'K': self.n_causal,
            'L': self.n_latent - self.n_causal,
            'z_dim': self.n_latent,
            'M': self.n_state}
        self.ce_params = ce_params
        loss_weights = {
            'up_rec_loss': up_weight * recons_weight,
            'down_rec_loss': down_weight * recons_weight,
            'up_kl_loss': kl_weight,
            'feat_l1_loss_up': feat_l1_weight,
            'dpd_loss': dpd_weight,
            'fide_kl_loss': fide_kl_weight,
            'causal_loss': causal_weight,
        }

        self.batch_size = batch_size
        optimizer = optim.Adam(self.module.parameters(), lr=lr, weight_decay=weight_decay)
        epoch_losses = {'total_loss': [], 'up_rec_loss1': [], 'up_rec_loss2': [], 'down_rec_loss': [], 'up_kl_loss': [],
                        'feat_l1_loss_up': [], 'feat_l1_loss_down': [], 'dpd_loss': [], 'fide_kl_loss': [],
                        'causal_loss': []}
        self.module.train()
        for epoch in tqdm(range(max_epochs), desc="training", disable=False):
            train_adata_batch = batch_sampler(train_adata, batch_size, shuffle=True, drop_last=drop_last)
            batch_losses = {'total_loss': [], 'up_rec_loss1': [], 'up_rec_loss2': [], 'down_rec_loss': [],
                            'up_kl_loss': [], 'feat_l1_loss_up': [], 'feat_l1_loss_down': [], 'dpd_loss': [],
                            'fide_kl_loss': [], 'causal_loss': []}
            if stage_training:
                # loss_weights = self.module.update_loss_weights_sc(epoch, max_epochs, loss_weights)
                loss_weights = self.module.update_loss_weights(epoch, max_epochs, scheme=weight_scheme)
            for train_batch in train_adata_batch:
                inputs_up = torch.tensor(train_batch.X, dtype=torch.float32, device=device)
                inputs_down = torch.tensor(train_batch.obsm['X_down'], dtype=torch.float32, device=device)
                labels = torch.tensor(train_batch.obs['labels'], dtype=torch.float32, device=device)
                model_outputs = self.module(inputs_up)
                loss_dict = self.module.compute_loss(model_outputs, inputs_up, inputs_down, labels,
                                                     imb_factor=im_factor)

                causal_loss_list = []
                for idx in np.random.permutation(train_batch.shape[0])[:n_x]:
                    if loss_weights["causal_loss"] == 0:
                        causal_loss_list = [torch.tensor(0.0, device=device)]
                        break
                    _causal_loss1, _ = joint_uncond_v2(ce_params, self.module, inputs_up, idx, alpha_vi=True,
                                                       beta_vi=True, device=device)
                    _causal_loss2, _ = beta_info_flow_v2(ce_params, self.module, inputs_up, idx, alpha_vi=True,
                                                         beta_vi=False, device=device)
                    _causal_loss = _causal_loss1 * causal_fold - _causal_loss2 * spurious_fold
                    # _causal_loss = _causal_loss1 - _causal_loss2 * 3.0
                    causal_loss_list += [_causal_loss]
                up_rec_loss1 = loss_dict['up_rec_loss1'].mean()
                up_rec_loss2 = loss_dict['up_rec_loss2'].mean()
                down_rec_loss = loss_dict['down_rec_loss'].mean()
                up_kl_loss = loss_dict['up_kl_loss'].mean()
                feat_l1_loss_up = loss_dict['feat_l1_loss_up'].mean()
                feat_l1_loss_down = loss_dict['feat_l1_loss_down'].mean()
                dpd_loss = loss_dict['dpd_loss'].mean()
                fide_kl_loss = loss_dict['fide_kl_loss'].mean()
                causal_loss = torch.stack(causal_loss_list).mean()
                if self.module.feature_mapper_up.attention:
                    loss_weights["feat_l1_loss_up"] = 0.001
                total_loss = loss_weights['up_rec_loss'] * up_rec_loss1 + \
                             loss_weights['up_rec_loss'] * up_rec_loss2 + \
                             loss_weights['down_rec_loss'] * down_rec_loss + \
                             loss_weights['up_kl_loss'] * up_kl_loss + \
                             loss_weights['feat_l1_loss_up'] * feat_l1_loss_up + \
                             loss_weights['feat_l1_loss_down'] * feat_l1_loss_down * down_fold + \
                             loss_weights['dpd_loss'] * dpd_loss + \
                             loss_weights['fide_kl_loss'] * fide_kl_loss + \
                             loss_weights['causal_loss'] * causal_loss

                optimizer.zero_grad()
                # with torch.autograd.detect_anomaly():
                #     total_loss.backward()
                total_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.module.parameters(), max_norm=1.0)
                optimizer.step()

                # update batch losses
                batch_losses['total_loss'].append(total_loss.item())
                batch_losses['up_rec_loss1'].append(up_rec_loss1.item())
                batch_losses['up_rec_loss2'].append(up_rec_loss2.item())
                batch_losses['down_rec_loss'].append(down_rec_loss.item())
                batch_losses['up_kl_loss'].append(up_kl_loss.item())
                batch_losses['feat_l1_loss_up'].append(feat_l1_loss_up.item())
                batch_losses['feat_l1_loss_down'].append(feat_l1_loss_down.item())
                batch_losses['dpd_loss'].append(dpd_loss.item())
                batch_losses['fide_kl_loss'].append(fide_kl_loss.item())
                batch_losses['causal_loss'].append(causal_loss.item())

            # update epochs losses
            epoch_losses['total_loss'].append(np.mean(batch_losses['total_loss']))
            epoch_losses['up_rec_loss1'].append(np.mean(batch_losses['up_rec_loss1']))
            epoch_losses['up_rec_loss2'].append(np.mean(batch_losses['up_rec_loss2']))
            epoch_losses['down_rec_loss'].append(np.mean(batch_losses['down_rec_loss']))
            epoch_losses['up_kl_loss'].append(np.mean(batch_losses['up_kl_loss']))
            epoch_losses['feat_l1_loss_up'].append(np.mean(batch_losses['feat_l1_loss_up']))
            epoch_losses['feat_l1_loss_down'].append(np.mean(batch_losses['feat_l1_loss_down']))
            epoch_losses['dpd_loss'].append(np.mean(batch_losses['dpd_loss']))
            epoch_losses['fide_kl_loss'].append(np.mean(batch_losses['fide_kl_loss']))
            epoch_losses['causal_loss'].append(np.mean(batch_losses['causal_loss']))

            if epoch % 20 == 0 or epoch == (max_epochs - 1):
                total_loss = np.mean(batch_losses['total_loss'])
                logging.info(f"Epoch {epoch} training loss: {total_loss:.4f}")

        self.history = epoch_losses

    def pretrain_attention(
            self,
            prior_probs: Optional[np.ndarray] = None,
            max_epochs: Optional[int] = 50,
            pretrain_lr: float = 1e-3,
            batch_size: int = 128,
            use_gpu: Optional[Union[str, int, bool]] = None,
            train_size: float = 1.0,
            validation_size: Optional[float] = None
    ):
        r"""Pretrain attention network.

        Parameters
        ----------
        prior_probs
            Prior probabilities for attention weights
        max_epochs
            Maximum number of pretraining epochs
        pretrain_lr
            Learning rate for pretraining
        batch_size
            Number of samples per batch
        use_gpu
            Whether to use GPU for pretraining
        train_size
            Proportion of data to use for pretraining
        validation_size
            Proportion of data to use for validation
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.module.to(device)
        train_adata, _ = data_splitter(
            self.adata,
            train_size=train_size,
            validation_size=validation_size,
            use_gpu=use_gpu,
        )

        if prior_probs is None:
            prior_probs = np.ones(self.module.feature_mapper_up.n_features) * 0.5
        elif not isinstance(prior_probs, np.ndarray):
            prior_probs = np.array(prior_probs)

        prior_probs_tensor = torch.tensor(prior_probs, dtype=torch.float32).view(1, -1).to(device)

        criterion = torch.nn.MSELoss()
        pretrain_optimizer = torch.optim.Adam(self.module.feature_mapper_up.att_net.parameters(), lr=pretrain_lr)

        self.module.train()
        for epoch in tqdm(range(max_epochs), desc="pretraining", disable=False):
            train_adata_batch = batch_sampler(train_adata, batch_size, shuffle=True)
            for train_batch in train_adata_batch:
                inputs_up = torch.tensor(train_batch.X, dtype=torch.float32, device=device)

                attention_scores = self.module.feature_mapper_up.att_net(inputs_up)
                # Repeat prior_probs_tensor to match the batch size
                repeated_prior_probs = prior_probs_tensor.repeat(attention_scores.size(0), 1)

                loss = criterion(torch.sigmoid(attention_scores), repeated_prior_probs)

                pretrain_optimizer.zero_grad()
                loss.backward()
                pretrain_optimizer.step()

        print("Pretraining attention net completed.")

    def plot_train_losses(self, fig_size=(8, 8)):
        r"""Plot training loss curves for all recorded losses during training.

        This method visualizes the evolution of each loss component over epochs
        using subplots. It requires that the model has been trained and that
        training history is available in the `self.history` attribute.
    
        Parameters
        ----------
        fig_size : tuple of int, optional
            Figure size (width, height) in inches. Default is (8, 8).
        """
        # Set figure size
        fig = plt.figure(figsize=fig_size)
        if self.history is None:
            raise ValueError("You should train the model first!")
        epoch_losses = self.history
        # Plot a subplot of each loss
        for i, loss_name in enumerate(epoch_losses.keys()):
            # Gets the value of the current loss
            loss_values = epoch_losses[loss_name]
            # Create subplot
            ax = fig.add_subplot(3, 4, i + 1)
            # Draw subplot
            ax.plot(range(len(loss_values)), loss_values)
            # Set the subplot title
            ax.set_title(loss_name)
            # Set the subplot x-axis and y-axis labels
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')

        # adjust the distance and edges between sub-graphs
        plt.tight_layout()
        # show figure
        plt.show()

    def get_up_feature_weights(
            self,
            adata: Optional[AnnData] = None,
            method: Optional[str] = "SHAP",
            n_bg_samples: Optional[int] = 100,
            grad_source: Optional[str] = "prob",
            normalize: Optional[bool] = True,
            sort_by_weight: Optional[bool] = True,
            class_idx: Optional[int] = None,
            background_data: Optional[torch.Tensor] = None,
            return_background: Optional[bool] = False,
    ):
        r"""Compute and return feature importance weights for the upstream feature mapper.
    
        This method supports multiple strategies to estimate feature contributions:
        - **"Model"**: uses internal attention or learned weights from the model.
        - **"SHAP"**: computes SHAP values using DeepExplainer.
        - **"Grad"**: computes input gradients w.r.t. a specified output (e.g., probability or logit).
        - **"Ensemble"**: averages normalized absolute weights from all three methods above.
    
        The resulting weights are aggregated across samples (by mean), optionally normalized,
        and returned both as a sorted DataFrame aligned with `self.adata.var` and as a full
        sample-by-feature weight matrix.
    
        Parameters
        ----------
        method : str, optional
            Method to compute feature weights. One of {"Model", "SHAP", "Grad", "Ensemble"}.
            Default is "SHAP".
        n_bg_samples : int, optional
            Number of background samples used for SHAP explanation. Only relevant if
            `method="SHAP"`. Default is 100.
        grad_source : str, optional
            Target output for gradient computation when `method="Grad"`. Options are:
            - "prob": gradients w.r.t. predicted probabilities,
            - "logit": gradients w.r.t. logits,
            - "loss": gradients w.r.t. the DPD loss.
            Default is "prob".
        normalize : bool, optional
            Whether to normalize the final feature weights to sum to 1. Default is True.
        sort_by_weight : bool, optional
            Whether to sort the returned DataFrame by weight in descending order.
            Default is True.
        class_idx : int, optional
            Class index for which to compute SHAP values (only used when labels are present
            and `method="SHAP"`). If None, SHAP values are averaged over all classes or
            computed on the full dataset. Default is None.
    
        Returns
        -------
        weights_df : pandas.DataFrame
            DataFrame with the same index as `self.adata.var`, containing a new column
            `'weight'` with the computed feature importance scores. Sorted by weight if
            `sort_by_weight=True`.
        weights_full : numpy.ndarray
            Full sample-by-feature matrix of absolute weights before aggregation.
            Shape: `(n_samples, n_features)`.
        """
        if self.module.training:
            self.module.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adata = adata if adata is not None else self.adata

        adata_batch = batch_sampler(adata, self.batch_size, shuffle=False)

        def compute_shap_weights(key="prob", class_idx=None, background_data=None):
            # key = "prob" or "logit"
            if background_data is None:
                idx = np.random.permutation(adata.shape[0])[0:n_bg_samples]
                background_data = torch.tensor(adata.X[idx], dtype=torch.float32).to(device)
            else:
                # background_data from predefined matrix
                background_data = background_data.to(device)

            model = ShapModel(self.module, key).to(device)
            explainer = shap.DeepExplainer(model, background_data)

            if class_idx is not None:
                adata_subset = adata[adata.obs['labels'] == class_idx].copy()
                inputs_up = torch.tensor(adata_subset.X, dtype=torch.float32).to(device)
            else:
                inputs_up = torch.tensor(adata.X, dtype=torch.float32, device=device)
            # shap_value = explainer.shap_values(inputs_up)
            shap_value = explainer.shap_values(inputs_up, check_additivity=False)
            if shap_value.ndim == 3 and shap_value.shape[2] > 1:
                shap_value = shap_value[..., class_idx] if class_idx is not None else shap_value.mean(axis=2, keepdims=True)

            return shap_value, background_data

        def compute_grad_weights(grad_source="prob"):
            grad_weights_full = []
            for data in adata_batch:
                self.module.zero_grad(set_to_none=True)
                inputs_up = torch.tensor(data.X, dtype=torch.float32, device=device)
                inputs_down = torch.tensor(data.obsm['X_down'], dtype=torch.float32, device=device)
                labels = torch.tensor(data.obs['labels'], dtype=torch.float32, device=device)

                inputs_up.requires_grad = True
                model_outputs = self.module(inputs_up, use_mean=True)

                if grad_source == "loss":
                    loss_dict = self.module.compute_loss(model_outputs, inputs_up, inputs_down, labels)
                    dpd_loss = loss_dict['dpd_loss']
                    dpd_loss.sum().backward()  # mean()
                elif grad_source == "prob":
                    prob = model_outputs["alpha_dpd"]["prob"]  # prob
                    prob.sum().backward()
                elif grad_source == 'logit':
                    prob = model_outputs["alpha_dpd"]["logit"]
                    prob.sum().backward()
                grad_weights_full.append(inputs_up.grad.cpu().numpy())

            return np.concatenate(grad_weights_full, axis=0)

        def compute_model_weights():
            if self.module.feature_mapper_up.attention:
                attention_weights_full = []
                for data in adata_batch:
                    inputs = torch.tensor(data.X, dtype=torch.float32, device=device)
                    model_outputs = self.module(inputs, use_mean=True)
                    att_w = model_outputs["feat_w_up"].cpu().detach().numpy()
                    attention_weights_full.append(att_w)
                weight_matrix = np.concatenate(attention_weights_full, axis=0)
            else:
                weight_vector = torch.sigmoid(self.module.feature_mapper_up.weight).cpu().detach().numpy()
                # Expand weight vector to a matrix with the same weight vector repeated for each sample in adata_batch
                weight_matrix = np.tile(weight_vector, (len(adata), 1))
            return weight_matrix

        weights_full = None
        bg_used = None
        if method == "Model":
            weights_full = compute_model_weights()
        elif method == "SHAP":
            weights_full, bg_used = compute_shap_weights(class_idx=class_idx, background_data=background_data)
        elif method == "Grad":
            weights_full = compute_grad_weights(grad_source=grad_source)
        elif method == "Ensemble":
            model_weights = np.abs(compute_model_weights())
            shap_weights, bg_used = compute_shap_weights(class_idx=class_idx, background_data=background_data)  # feed bg used
            shap_weights = np.abs(shap_weights)
            grad_weights = np.abs(compute_grad_weights())

            # Normalize each set of weights
            model_sum = np.sum(model_weights, axis=1, keepdims=True)
            model_weights = np.where(model_sum != 0, model_weights / model_sum, 0)

            shap_sum = np.sum(shap_weights, axis=1, keepdims=True)
            shap_weights = np.where(shap_sum != 0, shap_weights / shap_sum, 0)

            grad_sum = np.sum(grad_weights, axis=1, keepdims=True)
            grad_weights = np.where(grad_sum != 0, grad_weights / grad_sum, 0)

            # Combine the weights
            weights_full = (model_weights + shap_weights + grad_weights) / 3

        # Get the mean of the weights for each feature
        weights = np.mean(np.abs(weights_full), axis=0)
        weights_signed = np.mean(weights_full, axis=0)

        # Normalize the weights if required
        if normalize:
            weights = weights / np.sum(weights)
            weights_signed = weights_signed / np.sum(np.abs(weights_signed))

        # Create a new DataFrame with the weights
        weights_df = adata.var.copy()
        weights_df['weight'] = weights
        weights_df['weight_signed'] = weights_signed  # add new column

        # Sort the DataFrame by weight if required
        if sort_by_weight:
            weights_df = weights_df.sort_values(by='weight', ascending=False)

        if return_background and method == "SHAP":
            return weights_df, weights_full, bg_used
        else:
            return weights_df, weights_full

    def get_up_significance(
            self,
            adata: Optional[AnnData] = None,
            method: str = "SHAP",  # "SHAP" or "Grad"
            test_mode: str = "permutation",  # "permutation" or "sign_test"
            perm_mode: str = "global",  # "global" or "per_feature" (only for permutation)
            n_perm: int = 100,
            n_bg_samples: int = 100,
            grad_source: str = "prob",
            normalize: bool = False,
            class_idx: Optional[int] = None,
            target_genes: Optional[List[str]] = None,
            fdr_correct: bool = True,
            use_signed: bool = True,
            show_progress: bool = True,
            random_state: Optional[int] = 42,
    ):
        r"""
        Compute significance of upstream feature weights using:
        - Grad  → Binomial sign-consistency test
        - SHAP  → Binomial sign-consistency test or permutation test

        Parameters
        ----------
        adata : AnnData, optional
            Input AnnData object (default: self.adata)
        method : str
            "SHAP" or "Grad"
        test_mode : str
            For SHAP: "sign_test" or "permutation" (ignored for Grad)
        perm_mode : str
            "global" or "per_feature" shuffle strategy (for permutation test)
        n_perm : int
            Number of permutations (for permutation test)
        n_bg_samples : int
            Number of background samples for SHAP
        grad_source : str
            Source for gradient-based attribution ("prob", "logit", or "loss")
        normalize : bool
            Whether to normalize weights across features
        class_idx : int, optional
            Target class index for class-specific analysis
        target_genes : list of str, optional
            Genes/features to test; if None, use all
        fdr_correct : bool
            Apply Benjamini–Hochberg correction
        use_signed : bool
            Whether to use signed weights for p-value calculation
        show_progress : bool
            Display progress bar
        random_state : int, optional
            Random seed

        Returns
        -------
        df_result : pd.DataFrame
            DataFrame with ['weight', 'weight_signed', 'pvalue', ('qvalue')]
        perm_matrix : np.ndarray or None
            Permutation matrix for SHAP (None for Grad or sign_test)
        """
        if random_state is not None:
            np.random.seed(random_state)

        adata = adata if adata is not None else self.adata

        # =======================================================
        # Grad → Binomial Sign-Consistency Test
        # =======================================================
        if method == "Grad":
            df_obs, weights_full = self.get_up_feature_weights(
                adata=adata,
                method=method,
                n_bg_samples=n_bg_samples,
                grad_source=grad_source,
                normalize=normalize,
                sort_by_weight=False,
                class_idx=class_idx,
            )

            n = weights_full.shape[0]
            k = (weights_full > 0).sum(axis=0)

            pvals = np.array([
                stats.binomtest(int(kk), n, p=0.5, alternative="two-sided").pvalue
                for kk in k
            ])

            mean_grad = np.mean(weights_full, axis=0).astype(float)

            df_result = pd.DataFrame({
                "weight": np.abs(mean_grad),
                "weight_signed": mean_grad,
                "pvalue": pvals,
            }, index=df_obs.index)

            if fdr_correct:
                df_result["qvalue"] = multipletests(pvals, method="fdr_bh")[1]

            print(
                "[Note] Grad-based significance computed via two-sided Binomial test on gradient sign consistency (no permutation).")
            return df_result, None

        # =======================================================
        # SHAP → Binomial Sign-Consistency Test
        # =======================================================
        elif method == "SHAP" and test_mode == "sign_test":
            df_obs, shap_matrix = self.get_up_feature_weights(
                adata=adata,
                method=method,
                n_bg_samples=n_bg_samples,
                grad_source=grad_source,
                normalize=normalize,
                sort_by_weight=False,
                class_idx=class_idx,
            )

            n = shap_matrix.shape[0]
            k = (shap_matrix > 0).sum(axis=0)

            pvals = np.array([
                stats.binomtest(int(kk), n, p=0.5, alternative="two-sided").pvalue
                for kk in k
            ])

            # Remove trailing dimension if exists
            mean_shap = np.mean(shap_matrix, axis=0).astype(float).squeeze()

            df_result = pd.DataFrame({
                "weight": np.abs(mean_shap),
                "weight_signed": mean_shap,
                "pvalue": pvals,
            }, index=df_obs.index)

            if fdr_correct:
                df_result["qvalue"] = multipletests(pvals, method="fdr_bh")[1]

            print(
                "[Note] SHAP-based significance computed via two-sided Binomial test on contribution sign consistency (no permutation).")
            return df_result, None

        # =======================================================
        # SHAP → Permutation-Based Test
        # =======================================================
        elif method == "SHAP" and test_mode == "permutation":
            df_obs, _, bg_data = self.get_up_feature_weights(
                adata=adata,
                method=method,
                n_bg_samples=n_bg_samples,
                grad_source=grad_source,
                normalize=normalize,
                sort_by_weight=False,
                class_idx=class_idx,
                return_background=True,
            )

            var_names = df_obs.index.tolist()
            if target_genes is None:
                target_genes = var_names

            perm_matrix = np.zeros((len(target_genes), n_perm))
            iterator = tqdm(range(n_perm), desc="Permuting", disable=not show_progress)

            # ---------- Global permutation ----------
            if perm_mode == "global":
                for i in iterator:
                    adata_perm = adata.copy()
                    X = adata_perm.X.copy()
                    for j in range(X.shape[1]):
                        np.random.shuffle(X[:, j])
                    adata_perm.X = X

                    df_perm, _ = self.get_up_feature_weights(
                        adata=adata_perm,
                        method=method,
                        n_bg_samples=n_bg_samples,
                        grad_source=grad_source,
                        normalize=normalize,
                        sort_by_weight=False,
                        class_idx=class_idx,
                        background_data=bg_data,
                    )
                    col = "weight_signed" if (use_signed and "weight_signed" in df_perm.columns) else "weight"
                    perm_matrix[:, i] = df_perm.loc[target_genes, col].values

            # ---------- Per-feature permutation ----------
            elif perm_mode == "per_feature":
                for g_idx, gene in enumerate(tqdm(target_genes, desc="Target genes", disable=not show_progress)):
                    if gene not in var_names:
                        continue
                    j = var_names.index(gene)
                    for i in range(n_perm):
                        adata_perm = adata.copy()
                        X = adata_perm.X.copy()
                        np.random.shuffle(X[:, j])
                        adata_perm.X = X

                        df_perm, _ = self.get_up_feature_weights(
                            adata=adata_perm,
                            method=method,
                            n_bg_samples=n_bg_samples,
                            grad_source=grad_source,
                            normalize=normalize,
                            sort_by_weight=False,
                            class_idx=class_idx,
                            background_data=bg_data,
                        )
                        col = "weight_signed" if (use_signed and "weight_signed" in df_perm.columns) else "weight"
                        perm_matrix[g_idx, i] = df_perm.loc[gene, col]

            else:
                raise ValueError("perm_mode must be 'global' or 'per_feature'.")

            # ---------- Compute empirical p-values ----------
            pvals = np.zeros(len(target_genes))
            for k, gene in enumerate(target_genes):
                obs_val = (
                    df_obs.loc[gene, "weight_signed"]
                    if (use_signed and "weight_signed" in df_obs.columns)
                    else abs(df_obs.loc[gene, "weight"])
                )
                null_dist = perm_matrix[k, :]

                if use_signed and "weight_signed" in df_obs.columns:
                    if obs_val >= 0:
                        pvals[k] = (1 + np.sum(null_dist >= obs_val)) / (n_perm + 1)
                    else:
                        pvals[k] = (1 + np.sum(null_dist <= obs_val)) / (n_perm + 1)
                else:
                    pvals[k] = (1 + np.sum(np.abs(null_dist) >= abs(obs_val))) / (n_perm + 1)

            df_result = pd.DataFrame({
                "weight": df_obs.loc[target_genes, "weight"].values,
                "weight_signed": df_obs.loc[target_genes, "weight_signed"].values,
                "pvalue": pvals,
            }, index=target_genes)

            if fdr_correct:
                df_result["qvalue"] = multipletests(pvals, method="fdr_bh")[1]

            print(
                f"[Note] SHAP-based significance computed via permutation test ({perm_mode} mode, {n_perm} permutations).")
            return df_result, perm_matrix

        # =======================================================
        # Unsupported combinations
        # =======================================================
        else:
            raise ValueError(
                f"Unsupported configuration: method='{method}', test_mode='{test_mode}'. "
                "Supported: Grad(binomial), SHAP(sign_test/permutation)."
            )

    def get_2to1_ig(
            self,
            adata=None,
            key='prob',
            celltype=None,
            baseline=None,
    ):
        r"""Compute Integrated Gradients (IG) attributions from UP features to DOWN features.
        This method calculates feature-wise attribution scores using Integrated Gradients
        to understand how each up influences each down in the model.
    
        Parameters
        ----------
        adata : AnnData, optional
            Annotated data matrix to compute attributions for. If None, defaults to `self.adata`.
            Default is None.
        key : str, optional
            Model output key to attribute to. Typically `'prob'` (probability) or `'logit'`.
            Default is `'prob'`.
        celltype : str or None, optional
            Specific cell type to subset for attribution. If None or `'all'`, uses all cells.
            If specified, must exist in `adata.obs['celltype']`. Default is None.
        baseline : torch.Tensor or None, optional
            Baseline input for Integrated Gradients (same shape as input). If None, uses a
            zero tensor as baseline. Default is None.
    
        Returns
        -------
        ig_scores : numpy.ndarray
            Integrated Gradients attribution scores with shape `(n_cells, n_up, n_down)`,
        """
        if self.module.training:
            self.module.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adata = adata if adata is not None else self.adata
        if celltype is None or celltype == 'all':
            inputs_up = torch.tensor(adata.X, dtype=torch.float32).to(device)
            if baseline is None:
                baseline = torch.zeros_like(inputs_up)
            ig_model = MtoPModel(self.module, output_indices=None, key=key).to(device)
            ig = IntegratedGradients(ig_model)
            n_re_features = ig_model(inputs_up).shape[1]
            all_attributions = []
            for re_idx in range(n_re_features):
                attribution, delta = ig.attribute(
                    inputs_up,
                    baselines=baseline,
                    target=re_idx,
                    return_convergence_delta=True
                )
                all_attributions.append(attribution.detach().cpu().numpy())
            all_attributions = np.stack(all_attributions, axis=0)
            ig_scores = np.transpose(all_attributions, (1, 2, 0))
        elif celltype in adata.obs['celltype'].unique():
            adata_subset = adata[adata.obs['celltype'] == celltype].copy()
            inputs_up = torch.tensor(adata_subset.X, dtype=torch.float32).to(device)
            if baseline is None:
                baseline = torch.zeros_like(inputs_up)
            ig_model = MtoPModel(self.module, output_indices=None, key=key).to(device)
            ig = IntegratedGradients(ig_model)
            n_re_features = ig_model(inputs_up).shape[1]
            all_attributions = []
            for re_idx in range(n_re_features):
                attribution, delta = ig.attribute(
                    inputs_up,
                    baselines=baseline,
                    target=re_idx,
                    return_convergence_delta=True
                )
                all_attributions.append(attribution.detach().cpu().numpy())
            all_attributions = np.stack(all_attributions, axis=0)
            ig_scores = np.transpose(all_attributions, (1, 2, 0))
        else:
            raise ValueError(f"Cell type '{celltype}' not found in adata.obs['celltype'].")
        return ig_scores

    def get_input2z_ig(self, adata=None, key='prob', baseline=None):
        """
        Compute Integrated Gradients for input to latent space.
        """
        if self.module.training:
            self.module.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adata = adata if adata is not None else self.adata
        inputs_up = torch.tensor(adata.X, dtype=torch.float32).to(device)
        if baseline is None:
            baseline = torch.zeros_like(inputs_up)
        ig_model = MtoZModel(self.module, output_indices=None, key=key).to(device)
        ig = IntegratedGradients(ig_model)
        n_latent_features = ig_model(inputs_up).shape[1]
        all_attributions = []
        for latent_idx in range(n_latent_features):
            attribution, delta = ig.attribute(
                inputs_up,
                baselines=baseline,
                target=latent_idx,
                return_convergence_delta=True
            )
            all_attributions.append(attribution.detach().cpu().numpy())
        all_attributions = np.stack(all_attributions, axis=0)
        ig_scores = np.transpose(all_attributions, (1, 2, 0))
        return ig_scores

    @torch.no_grad()
    def get_model_output(
            self,
            adata: Optional[AnnData] = None,
            batch_size: Optional[int] = None,
    ):
        r"""Obtain model predictions and latent representations for a given dataset.

        This method runs the trained model in evaluation mode and returns:
        - Concatenated latent embeddings from two latent spaces,
        - Logits and predicted probabilities from the downstream classifier,
        - Binary class predictions (thresholded at 0.5).
    
        If no `adata` is provided, the method uses the internal `self.adata`.
    
        Parameters
        ----------
        adata : AnnData, optional
            Annotated data matrix to generate outputs for. If None, defaults to `self.adata`.
            Default is None.
        batch_size : int, optional
            Number of samples per batch during inference. If None, uses `self.batch_size`.
            Default is None.
    
        Returns
        -------
        output : dict
            Dictionary containing the following keys:
            - `'latent'`: numpy.ndarray of shape `(n_samples, n_latent1 + n_latent2)`,  
              concatenated latent vectors from both latent modules.
            - `'logits'`: numpy.ndarray of shape `(n_samples,)` or `(n_samples, n_classes)`,  
              raw classifier logits.
            - `'probs'`: numpy.ndarray of same shape as `'logits'`,  
              predicted probabilities after sigmoid/softmax activation.
            - `'preds'`: numpy.ndarray of shape `(n_samples,)`,  
              binary predictions (1 if probability > 0.5, else 0).

        """
        if self.module.training:
            self.module.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        latent = []
        logits = []
        probs = []
        preds = []
        x_down_rec_alpha = []
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            inputs = torch.tensor(data.X, dtype=torch.float32, device=device)
            model_outputs = self.module(inputs, use_mean=True)
            latent_z = torch.cat([model_outputs["latent1"]["z"], model_outputs["latent2"]["z"]], dim=1)
            latent.append(latent_z.cpu().numpy())
            # latent.append(model_outputs['latent_up']['qz_m'].cpu().numpy())
            logits.append(model_outputs['alpha_dpd']['logit'].cpu().numpy())
            probs.append(model_outputs["alpha_dpd"]["prob"].cpu().numpy())
            preds.append(np.int_(model_outputs['alpha_dpd']['prob'].cpu().numpy() > 0.5))
            x_down_rec_alpha.append(model_outputs["x_down_rec_alpha"].cpu().numpy())

        output = dict(latent=np.concatenate(latent, axis=0),
                      logits=np.concatenate(logits, axis=0),
                      probs=np.concatenate(probs, axis=0),
                      preds=np.concatenate(preds, axis=0),
                      x_down_rec_alpha=np.concatenate(x_down_rec_alpha, axis=0))

        return output

    @torch.no_grad()
    def compute_information_flow(
            self,
            adata: Optional[AnnData] = None,
            dims: Optional[List[int]] = None,
            zero_floor: bool = False,
            plot_info_flow: Optional[bool] = True,
            skip_single_info: Optional[bool] = True,
            save_fig: Optional[bool] = False,
            save_dir: Optional[str] = None,
    ):
        r"""
        Compute information flow for latent dimensions.
        
        Parameters
        ----------
        adata
            AnnData object with input data
        dims
            Dimensions to compute information flow for
        zero_floor
            Whether to subtract minimum value
        plot_info_flow
            Whether to plot information flow
        skip_single_info
            Whether to skip single dimension plots
        save_fig
            Whether to save figures
        save_dir
            Directory to save figures
            
        Returns
        ----------
        info_flow
            Information flow for each dimension
        info_flow_cat
            Categorical information flow (causal vs spurious)
        """
        if self.module.training:
            self.module.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adata = adata if adata is not None else self.adata
        ce_params = self.ce_params
        if dims is None:
            dims = list(range(self.module.n_latent))

        # Calculate information flow
        info_flow = pd.DataFrame(index=range(adata.shape[0]), columns=dims)
        for i in range(adata.shape[0]):
            for j in dims:
                # Get the latent space of the current sample
                inputs = torch.tensor(adata.X, dtype=torch.float32, device=device)
                # Calculate the information flow
                info = joint_uncond_single_dim_v2(ce_params, self.module, inputs, i, j, alpha_vi=False, beta_vi=True,
                                                  device=device)
                info_flow.loc[i, j] = info.item()
        info_flow.set_index(adata.obs_names, inplace=True)
        if zero_floor:
            info_flow = info_flow - info_flow.min().min()
        info_flow = info_flow.apply(lambda x: x / (np.linalg.norm(x, ord=1) + 1e-8), axis=1)

        # Calculate information flow for causal and spurious dimensions
        dims = ['causal', 'spurious']
        info_flow_cat = pd.DataFrame(index=range(adata.shape[0]), columns=dims)
        for i in range(adata.shape[0]):
            # Get the latent space of the current sample
            inputs = torch.tensor(adata.X, dtype=torch.float32, device=device)
            # Calculate the information flow
            info_c, _ = joint_uncond_v2(ce_params, self.module, inputs, i, alpha_vi=False, beta_vi=True, device=device)
            info_s, _ = beta_info_flow_v2(ce_params, self.module, inputs, i, alpha_vi=True, beta_vi=False,
                                          device=device)
            info_flow_cat.loc[i, 'causal'] = -info_c.item()
            info_flow_cat.loc[i, 'spurious'] = -info_s.item()
        info_flow_cat.set_index(adata.obs_names, inplace=True)
        if zero_floor:
            info_flow_cat = info_flow_cat - info_flow_cat.min().min()
        info_flow_cat = info_flow_cat.apply(lambda x: x / (np.linalg.norm(x, ord=1) + 1e-8), axis=1)

        if plot_info_flow and not skip_single_info:
            # plot the information flow
            plt.figure(figsize=(10, 5))
            ax = sns.boxplot(data=info_flow, palette="pastel")
            plt.xlabel("Dimensions")
            plt.ylabel("Information Measurements")
            if save_fig:
                plt.savefig(os.path.join(save_dir, "info_flow_2l.png"))
                plt.savefig(os.path.join(save_dir, "info_flow_2l.pdf"))
            plt.show()
            plt.close()

        if plot_info_flow:
            plt.figure(figsize=(10, 5))
            ax = sns.boxplot(data=info_flow_cat, palette="pastel")
            plt.xlabel("Dimensions")
            plt.ylabel("Information Measurements")
            if save_fig:
                plt.savefig(os.path.join(save_dir, "info_flow_2l_cat.png"))
                plt.savefig(os.path.join(save_dir, "info_flow_2l_cat.pdf"))
            plt.show()
            plt.close()

        return info_flow, info_flow_cat


class CauTrigger3L(nn.Module):
    r"""Third-layer hierarchical causal decomposition model.
    CauTrigger3L wraps a DualVAE3L module and supports third-stage causal decomposition
    (eg. x3 → xc2 → xc1 → y). It is intended for modelling complex cascades where
    effects propagate through multiple intermediate regulatory layers (for instance,
    multi-omic cascades).

    Parameters
    ----------
    adata
        Annotated data matrix. Expects upstream features in ``adata.X`` and
        downstream representations in ``adata.obsm['X_down1']`` and ``adata.obsm['X_down2']``.
    n_latent
        Latent dimensionality (default: 10).
    n_causal
        Number of causal latent factors (default: 2).
    n_state
        Number of discrete states (default: 2).
    **model_kwargs
        Extra args passed to DualVAE3L.
    """
    def __init__(
            self,
            adata: AnnData,
            n_latent: int = 10,
            n_causal: int = 2,  # Number of casual factors
            n_state: int = 2,  # Number of states
            **model_kwargs,
    ):
        super(CauTrigger3L, self).__init__()
        self.adata = adata
        self.train_adata = None
        self.val_adata = None
        self.n_latent = n_latent
        self.n_causal = n_causal
        self.n_state = n_state
        self.batch_size = None
        self.ce_params = None
        self.history = {}

        self.module = DualVAE3L(
            n_input_up=adata.X.shape[1],
            n_input_down1=adata.obsm['X_down1'].shape[1],
            n_input_down2=adata.obsm['X_down2'].shape[1],
            n_latent=n_latent,
            n_causal=n_causal,
            n_state=n_state,
            **model_kwargs,
        )

    def train(
            self,
            max_epochs: Optional[int] = 400,
            lr: float = 5e-4,
            use_gpu: Optional[Union[str, int, bool]] = None,
            train_size: float = 1.0,
            validation_size: Optional[float] = None,
            batch_size: int = 128,
            early_stopping: bool = False,
            weight_decay: float = 1e-6,
            n_x: int = 5,
            n_alpha: int = 25,
            n_beta: int = 100,
            recons_weight: float = 1.0,
            kl_weight: float = 0.02,
            up_weight: float = 1.0,
            down_weight: float = 1.0,
            feat_l1_weight: float = 0.05,
            dpd_weight: float = 3.0,
            fide_kl_weight: float = 0.05,
            causal_weight: float = 1.0,
            down_fold: float = 1.0,
            causal_fold: float = 1.0,
            spurious_fold: float = 1.0,
            stage_training: bool = True,
            weight_scheme: str = None,
            im_factor: Optional[float] = None,
            drop_last: Union[bool, int] = False,
            **kwargs,
    ):
        r"""Trains the model using fractal variational autoencoder.
        
        Parameters
        ----------
        max_epochs
            Maximum number of training epochs
        lr
            Learning rate for optimizer
        use_gpu
            Whether to use GPU for training
        train_size
            Proportion of data to use for training
        validation_size
            Proportion of data to use for validation
        batch_size
            Number of samples per batch
        early_stopping
            Whether to use early stopping
        weight_decay
            Weight decay for optimizer
        n_x
            Number of samples for causal effect computation
        n_alpha
            Monte-carlo samples per causal factor
        n_beta
            Monte-carlo samples per noncausal factor
        recons_weight
            Weight for reconstruction loss
        kl_weight
            Weight for KL divergence loss
        up_weight
            Weight for upstream reconstruction
        down_weight
            Weight for downstream reconstruction
        feat_l1_weight
            Weight for feature L1 loss
        dpd_weight
            Weight for DPD loss
        fide_kl_weight
            Weight for fidelity KL loss
        causal_weight
            Weight for causal loss
        down_fold
            Downstream loss scaling factor
        causal_fold
            Causal loss scaling factor
        spurious_fold
            Spurious loss scaling factor
        stage_training
            Whether to use staged training
        weight_scheme
            Weight update scheme
        im_factor
            Imbalance factor for loss computation
        drop_last
            Number of samples to drop from last batch
        **kwargs
            Additional arguments
        """
        # set_seed(42)
        # torch.autograd.set_detect_anomaly(True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.module.to(device)
        train_adata, val_adata = data_splitter(
            self.adata,
            train_size=train_size,
            validation_size=validation_size,
            use_gpu=use_gpu,
        )
        self.train_adata, self.val_adata = train_adata, val_adata
        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])
        ce_params = {
            'N_alpha': n_alpha,
            'N_beta': n_beta,
            'K': self.n_causal,
            'L': self.n_latent - self.n_causal,
            'z_dim': self.n_latent,
            'M': self.n_state}
        self.ce_params = ce_params
        loss_weights = {
            'up_rec_loss': up_weight * recons_weight,
            'down_rec_loss': down_weight * recons_weight,
            'up_kl_loss': kl_weight,
            'feat_l1_loss_up': feat_l1_weight,
            'feat_l1_loss_down': feat_l1_weight * down_fold,
            'dpd_loss': dpd_weight,
            'fide_kl_loss': fide_kl_weight,
            'causal_loss': causal_weight,
        }

        self.batch_size = batch_size
        optimizer = optim.Adam(self.module.parameters(), lr=lr, weight_decay=weight_decay)
        epoch_losses = {'total_loss': [], 'up_rec_loss1': [], 'up_rec_loss2': [], 'down1_rec_loss': [],
                        'down2_rec_loss': [],
                        'up_kl_loss': [], 'feat_l1_loss_up': [], 'feat_l1_loss_down': [], 'dpd_loss': [],
                        'fide_kl_loss': [], 'causal_loss': []}
        self.module.train()
        for epoch in tqdm(range(max_epochs), desc="training", disable=False):
            train_adata_batch = batch_sampler(train_adata, batch_size, shuffle=True, drop_last=drop_last)
            batch_losses = {'total_loss': [], 'up_rec_loss1': [], 'up_rec_loss2': [], 'down1_rec_loss': [],
                            'down2_rec_loss': [],
                            'up_kl_loss': [], 'feat_l1_loss_up': [], 'feat_l1_loss_down': [], 'dpd_loss': [],
                            'fide_kl_loss': [], 'causal_loss': []}
            if stage_training:
                # loss_weights = self.module.update_loss_weights_sc(epoch, max_epochs, loss_weights)
                loss_weights = self.module.update_loss_weights(epoch, max_epochs, scheme=weight_scheme)
            for train_batch in train_adata_batch:
                inputs_up = torch.tensor(train_batch.X, dtype=torch.float32, device=device)
                inputs_down1 = torch.tensor(train_batch.obsm['X_down1'], dtype=torch.float32, device=device)
                inputs_down2 = torch.tensor(train_batch.obsm['X_down2'], dtype=torch.float32, device=device)
                labels = torch.tensor(train_batch.obs['labels'], dtype=torch.float32, device=device)
                model_outputs = self.module(inputs_up)
                loss_dict = self.module.compute_loss(model_outputs, inputs_up, inputs_down1, inputs_down2, labels,
                                                     imb_factor=im_factor)

                causal_loss_list = []
                for idx in np.random.permutation(train_batch.shape[0])[:n_x]:
                    if loss_weights["causal_loss"] == 0:
                        causal_loss_list = [torch.tensor(0.0, device=device)]
                        break
                    _causal_loss1, _ = joint_uncond_v2(ce_params, self.module, inputs_up, idx, alpha_vi=True,
                                                       beta_vi=True, device=device)
                    _causal_loss2, _ = beta_info_flow_v2(ce_params, self.module, inputs_up, idx, alpha_vi=True,
                                                         beta_vi=False, device=device)
                    _causal_loss = _causal_loss1 * causal_fold - _causal_loss2 * spurious_fold
                    # _causal_loss = _causal_loss1 - _causal_loss2 * 3.0
                    causal_loss_list += [_causal_loss]
                up_rec_loss1 = loss_dict['up_rec_loss1'].mean()
                up_rec_loss2 = loss_dict['up_rec_loss2'].mean()
                down1_rec_loss = loss_dict['down1_rec_loss'].mean()
                down2_rec_loss = loss_dict['down2_rec_loss'].mean()
                up_kl_loss = loss_dict['up_kl_loss'].mean()
                feat_l1_loss_up = loss_dict['feat_l1_loss_up'].mean()
                feat_l1_loss_down = loss_dict['feat_l1_loss_down'].mean()
                dpd_loss = loss_dict['dpd_loss'].mean()
                fide_kl_loss = loss_dict['fide_kl_loss'].mean()
                causal_loss = torch.stack(causal_loss_list).mean()
                if self.module.feature_mapper_up.attention:
                    loss_weights["feat_l1_loss_up"] = 0.001
                total_loss = loss_weights['up_rec_loss'] * up_rec_loss1 + \
                             loss_weights['up_rec_loss'] * up_rec_loss2 + \
                             loss_weights['down_rec_loss'] * down1_rec_loss * down1_rec_loss + \
                             loss_weights['down_rec_loss'] * down2_rec_loss + \
                             loss_weights['up_kl_loss'] * up_kl_loss + \
                             loss_weights['feat_l1_loss_up'] * feat_l1_loss_up + \
                             loss_weights['feat_l1_loss_down'] * feat_l1_loss_down * down_fold + \
                             loss_weights['dpd_loss'] * dpd_loss + \
                             loss_weights['fide_kl_loss'] * fide_kl_loss + \
                             loss_weights['causal_loss'] * causal_loss

                optimizer.zero_grad()
                # with torch.autograd.detect_anomaly():
                #     total_loss.backward()
                total_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.module.parameters(), max_norm=1.0)
                optimizer.step()

                # update batch losses
                batch_losses['total_loss'].append(total_loss.item())
                batch_losses['up_rec_loss1'].append(up_rec_loss1.item())
                batch_losses['up_rec_loss2'].append(up_rec_loss2.item())
                batch_losses['down1_rec_loss'].append(down1_rec_loss.item())
                batch_losses['down2_rec_loss'].append(down2_rec_loss.item())
                batch_losses['up_kl_loss'].append(up_kl_loss.item())
                batch_losses['feat_l1_loss_up'].append(feat_l1_loss_up.item())
                batch_losses['feat_l1_loss_down'].append(feat_l1_loss_down.item())
                batch_losses['dpd_loss'].append(dpd_loss.item())
                batch_losses['fide_kl_loss'].append(fide_kl_loss.item())
                batch_losses['causal_loss'].append(causal_loss.item())

            # update epochs losses
            epoch_losses['total_loss'].append(np.mean(batch_losses['total_loss']))
            epoch_losses['up_rec_loss1'].append(np.mean(batch_losses['up_rec_loss1']))
            epoch_losses['up_rec_loss2'].append(np.mean(batch_losses['up_rec_loss2']))
            epoch_losses['down1_rec_loss'].append(np.mean(batch_losses['down1_rec_loss']))
            epoch_losses['down2_rec_loss'].append(np.mean(batch_losses['down2_rec_loss']))
            epoch_losses['up_kl_loss'].append(np.mean(batch_losses['up_kl_loss']))
            epoch_losses['feat_l1_loss_up'].append(np.mean(batch_losses['feat_l1_loss_up']))
            epoch_losses['feat_l1_loss_down'].append(np.mean(batch_losses['feat_l1_loss_down']))
            epoch_losses['dpd_loss'].append(np.mean(batch_losses['dpd_loss']))
            epoch_losses['fide_kl_loss'].append(np.mean(batch_losses['fide_kl_loss']))
            epoch_losses['causal_loss'].append(np.mean(batch_losses['causal_loss']))

            if epoch % 20 == 0 or epoch == (max_epochs - 1):
                total_loss = np.mean(batch_losses['total_loss'])
                logging.info(f"Epoch {epoch} training loss: {total_loss:.4f}")

        self.history = epoch_losses

    def pretrain_attention(
            self,
            prior_probs: Optional[np.ndarray] = None,
            max_epochs: Optional[int] = 50,
            pretrain_lr: float = 1e-3,
            batch_size: int = 128,
            use_gpu: Optional[Union[str, int, bool]] = None,
            train_size: float = 1.0,
            validation_size: Optional[float] = None
    ):
        r"""Pretrain attention network.

        Parameters
        ----------
        prior_probs
            Prior probabilities for attention weights
        max_epochs
            Maximum number of pretraining epochs
        pretrain_lr
            Learning rate for pretraining
        batch_size
            Number of samples per batch
        use_gpu
            Whether to use GPU for pretraining
        train_size
            Proportion of data to use for pretraining
        validation_size
            Proportion of data to use for validation
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.module.to(device)
        train_adata, _ = data_splitter(
            self.adata,
            train_size=train_size,
            validation_size=validation_size,
            use_gpu=use_gpu,
        )

        if prior_probs is None:
            prior_probs = np.ones(self.module.feature_mapper_up.n_features) * 0.5
        elif not isinstance(prior_probs, np.ndarray):
            prior_probs = np.array(prior_probs)

        prior_probs_tensor = torch.tensor(prior_probs, dtype=torch.float32).view(1, -1).to(device)

        criterion = torch.nn.MSELoss()
        pretrain_optimizer = torch.optim.Adam(self.module.feature_mapper_up.att_net.parameters(), lr=pretrain_lr)

        self.module.train()
        for epoch in tqdm(range(max_epochs), desc="pretraining", disable=False):
            train_adata_batch = batch_sampler(train_adata, batch_size, shuffle=True)
            for train_batch in train_adata_batch:
                inputs_up = torch.tensor(train_batch.X, dtype=torch.float32, device=device)

                attention_scores = self.module.feature_mapper_up.att_net(inputs_up)
                # Repeat prior_probs_tensor to match the batch size
                repeated_prior_probs = prior_probs_tensor.repeat(attention_scores.size(0), 1)

                loss = criterion(torch.sigmoid(attention_scores), repeated_prior_probs)

                pretrain_optimizer.zero_grad()
                loss.backward()
                pretrain_optimizer.step()

        print("Pretraining attention net completed.")

    def plot_train_losses(self, fig_size=(8, 8)):
        r"""Plot training loss curves for all recorded losses during training.

        This method visualizes the evolution of each loss component over epochs
        using subplots. It requires that the model has been trained and that
        training history is available in the `self.history` attribute.
    
        Parameters
        ----------
        fig_size : tuple of int, optional
            Figure size (width, height) in inches. Default is (8, 8).
        """
        # Set figure size
        fig = plt.figure(figsize=fig_size)
        if self.history is None:
            raise ValueError("You should train the model first!")
        epoch_losses = self.history
        # Plot a subplot of each loss
        for i, loss_name in enumerate(epoch_losses.keys()):
            # Gets the value of the current loss
            loss_values = epoch_losses[loss_name]
            # Create subplot
            ax = fig.add_subplot(3, 4, i + 1)
            # Draw subplot
            ax.plot(range(len(loss_values)), loss_values)
            # Set the subplot title
            ax.set_title(loss_name)
            # Set the subplot x-axis and y-axis labels
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')

        # adjust the distance and edges between sub-graphs
        plt.tight_layout()
        # show figure
        plt.show()

    def get_up_feature_weights(
            self,
            method: Optional[str] = "SHAP",
            n_bg_samples: Optional[int] = 100,
            grad_source: Optional[str] = "prob",
            normalize: Optional[bool] = True,
            sort_by_weight: Optional[bool] = False,
            class_idx: Optional[int] = None,
    ):
        r"""Obtain model predictions and latent representations for a given dataset.

        This method runs the trained model in evaluation mode and returns:
        - Concatenated latent embeddings from two latent spaces,
        - Logits and predicted probabilities from the downstream classifier,
        - Binary class predictions (thresholded at 0.5).
    
        If no `adata` is provided, the method uses the internal `self.adata`.
    
        Parameters
        ----------
        adata : AnnData, optional
            Annotated data matrix to generate outputs for. If None, defaults to `self.adata`.
            Default is None.
        batch_size : int, optional
            Number of samples per batch during inference. If None, uses `self.batch_size`.
            Default is None.
    
        Returns
        -------
        output : dict
            Dictionary containing the following keys:
            - `'latent'`: numpy.ndarray of shape `(n_samples, n_latent1 + n_latent2)`,  
              concatenated latent vectors from both latent modules.
            - `'logits'`: numpy.ndarray of shape `(n_samples,)` or `(n_samples, n_classes)`,  
              raw classifier logits.
            - `'probs'`: numpy.ndarray of same shape as `'logits'`,  
              predicted probabilities after sigmoid/softmax activation.
            - `'preds'`: numpy.ndarray of shape `(n_samples,)`,  
              binary predictions (1 if probability > 0.5, else 0).

        """
        if self.module.training:
            self.module.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def compute_shap_weights(key="prob", class_idx=None):
            # key = "prob" or "logit"
            idx = np.random.permutation(self.adata.shape[0])[0:n_bg_samples]
            background_data = torch.tensor(self.adata.X[idx], dtype=torch.float32).to(device)

            model = ShapModel(self.module, key).to(device)
            explainer = shap.DeepExplainer(model, background_data)

            if class_idx is not None:
                adata_subset = self.adata[self.adata.obs['labels'] == class_idx].copy()
                inputs_up = torch.tensor(adata_subset.X, dtype=torch.float32).to(device)
            else:
                inputs_up = torch.tensor(self.adata.X, dtype=torch.float32, device=device)
            # shap_value = explainer.shap_values(inputs_up)
            shap_value = explainer.shap_values(inputs_up, check_additivity=False)
            if shap_value.ndim == 3 and shap_value.shape[2] > 1:
                shap_value = shap_value[..., class_idx] if class_idx is not None else shap_value.mean(axis=2, keepdims=True)

            return shap_value

        def compute_grad_weights(grad_source="prob"):
            inputs_up = torch.tensor(self.adata.X, dtype=torch.float32, device=device)
            inputs_down1 = torch.tensor(self.adata.obsm['X_down1'], dtype=torch.float32, device=device)
            inputs_down2 = torch.tensor(self.adata.obsm['X_down2'], dtype=torch.float32, device=device)
            labels = torch.tensor(self.adata.obs['labels'], dtype=torch.float32, device=device)

            inputs_up.requires_grad = True
            model_outputs = self.module(inputs_up, use_mean=True)

            if grad_source == "loss":
                loss_dict = self.module.compute_loss(model_outputs, inputs_up, inputs_down1, inputs_down2, labels)
                dpd_loss = loss_dict['dpd_loss']
                dpd_loss.sum().backward()  # mean()
            elif grad_source == "prob":
                prob = model_outputs["alpha_dpd"]["prob"]  # prob
                prob.sum().backward()
            elif grad_source == 'logit':
                prob = model_outputs["alpha_dpd"]["logit"]
                prob.sum().backward()

            return inputs_up.grad.cpu().numpy()

        def compute_model_weights():
            inputs = torch.tensor(self.adata.X, dtype=torch.float32, device=device)
            model_outputs = self.module(inputs, use_mean=True)

            if self.module.feature_mapper_up.attention:
                return model_outputs["feat_w_up"].cpu().detach().numpy()
            else:
                weight_vector = torch.sigmoid(self.module.feature_mapper_up.weight).cpu().detach().numpy()
                return np.tile(weight_vector, (self.adata.shape[0], 1))

        weights_full = None
        if method == "Model":
            weights_full = compute_model_weights()
        elif method == "SHAP":
            weights_full = compute_shap_weights(class_idx=class_idx)
        elif method == "Grad":
            weights_full = compute_grad_weights(grad_source=grad_source)
        elif method == "Ensemble":
            model_weights = np.abs(compute_model_weights())
            shap_weights = np.abs(compute_shap_weights())
            grad_weights = np.abs(compute_grad_weights())

            # Normalize each set of weights
            def normalize_w(w):
                return w / np.sum(w, axis=1, keepdims=True)

            model_weights = normalize_w(model_weights)
            shap_weights = normalize_w(shap_weights)
            grad_weights = normalize_w(grad_weights)

            # Combine the weights
            weights_full = (model_weights + shap_weights + grad_weights) / 3

        # Get the mean of the weights for each feature
        weights = np.mean(np.abs(weights_full), axis=0)

        # Normalize the weights if required
        if normalize:
            weights = weights / np.sum(weights)

        # Create a new DataFrame with the weights
        weights_df = self.adata.var.copy()
        weights_df['weight'] = weights

        # Sort the DataFrame by weight if required
        if sort_by_weight:
            weights_df = weights_df.sort_values(by='weight', ascending=False)

        return weights_df, weights_full

    def get_up_significance(self, *args, **kwargs):
        raise NotImplementedError(
            "get_up_significance is implemented for CauTrigger1L/2L. "
            "For 3L, enable it after aligning get_up_feature_weights (adata/background_data/return_background, weight_signed) "
            "and verifying Grad loss inputs."
        )

    def get_3to2_shap(
            self,
            adata=None,
            n_bg_samples=5,
            key='prob',
            celltype = None,
            explainer_type='gradient',
            data_dir = None,
    ):
        """Compute SHAP values from TFs to REs using either DeepExplainer or GradientExplainer.
        """
        if self.module.training:
            self.module.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adata = adata if adata is not None else self.adata
        idx = np.random.permutation(adata.shape[0])[:n_bg_samples]
        background_data_tf = torch.tensor(adata.X[idx], dtype=torch.float32).to(device)
        if celltype is None or celltype == 'all':
            inputs_up = torch.tensor(adata.X, dtype=torch.float32).to(device)
            shap_model_tf_re = TFtoREModel(self.module, output_indices=None, key=key).to(device)
            if explainer_type == 'gradient':
                explainer_tf_re = shap.GradientExplainer(shap_model_tf_re, background_data_tf)
            elif explainer_type == 'deep':
                explainer_tf_re = shap.DeepExplainer(shap_model_tf_re, background_data_tf)
            else:
                raise ValueError(f"Unsupported explainer_type: {explainer_type}. Use 'gradient' or 'deep'.")
            try:
                shap_value_tf_re = explainer_tf_re.shap_values(inputs_up, check_additivity=False)
            except TypeError:
                shap_value_tf_re = explainer_tf_re.shap_values(inputs_up)
            shap_values_tf_re_all = shap_value_tf_re

        elif celltype in adata.obs['celltype'].unique() and data_dir is not None:
            deg_df = pd.read_csv(os.path.join(data_dir, 'markers_tftg.txt'), index_col=0)
            degs = np.array(deg_df.loc[deg_df['cluster'] == celltype, 'gene'])
            deg_tg = np.intersect1d(degs, adata.uns['X_down2_var_names'])
            sparse_matrix = load_npz(os.path.join(data_dir, "TG_RE_interaction_matrix.npz"))
            sparse_matrix_index = pd.read_csv(os.path.join(data_dir, "TG_RE_interaction_matrix_rows.txt")).values.flatten()
            sparse_matrix_column = pd.read_csv(os.path.join(data_dir, "TG_RE_interaction_matrix_cols.txt")).values.flatten()
            tg_to_row = {tg: i for i, tg in enumerate(sparse_matrix_index)}
            rows = [tg_to_row[tg] for tg in deg_tg if tg in tg_to_row]
            interacted_re_indices = set()
            for r in rows:
                re_indices = sparse_matrix[r].nonzero()[1]
                interacted_re_indices.update(re_indices)
            interacted_re_names = [sparse_matrix_column[i] for i in sorted(interacted_re_indices)]
            adata_subset = adata[adata.obs['celltype'] == celltype].copy()
            inputs_up = torch.tensor(adata_subset.X, dtype=torch.float32).to(device)
            RES = adata.uns['X_down1_var_names']
            indices = [np.where(RES == b)[0][0] for b in interacted_re_names]
            shap_model_tf_re = TFtoREModel(self.module, output_indices=indices, key=key).to(device)
            if explainer_type == 'gradient':
                explainer_tf_re = shap.GradientExplainer(shap_model_tf_re, background_data_tf)
            elif explainer_type == 'deep':
                explainer_tf_re = shap.DeepExplainer(shap_model_tf_re, background_data_tf)
            else:
                raise ValueError(f"Unsupported explainer_type: {explainer_type}. Use 'gradient' or 'deep'.")
            try:
                shap_value_tf_re = explainer_tf_re.shap_values(inputs_up, check_additivity=False)
            except TypeError:
                shap_value_tf_re = explainer_tf_re.shap_values(inputs_up)
            shap_values_tf_re_all = shap_value_tf_re
        else:
            raise ValueError(f"Cell type '{celltype}' not found in adata.obs['labels'] or not provide data_dir.")
        return shap_values_tf_re_all

    def get_3to2_ig(
            self,
            adata=None,
            key='prob',
            celltype=None,
            baseline=None,
    ):
        """Compute IG values from TFs to REs.
        """
        if self.module.training:
            self.module.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adata = adata if adata is not None else self.adata
        if celltype is None or celltype == 'all':
            inputs_up = torch.tensor(adata.X, dtype=torch.float32).to(device)
            if baseline is None:
                baseline = torch.zeros_like(inputs_up)
            ig_model = TFtoREModel(self.module, output_indices=None, key=key).to(device)
            ig = IntegratedGradients(ig_model)
            n_re_features = ig_model(inputs_up).shape[1]
            all_attributions = []
            for re_idx in range(n_re_features):
                attribution, delta = ig.attribute(
                    inputs_up,
                    baselines=baseline,
                    target=re_idx,
                    return_convergence_delta=True
                )
                all_attributions.append(attribution.detach().cpu().numpy())
            all_attributions = np.stack(all_attributions, axis=0)
            ig_scores = np.transpose(all_attributions, (1, 2, 0))
        elif celltype in adata.obs['celltype'].unique():
            adata_subset = adata[adata.obs['celltype'] == celltype].copy()
            inputs_up = torch.tensor(adata_subset.X, dtype=torch.float32).to(device)
            if baseline is None:
                baseline = torch.zeros_like(inputs_up)
            ig_model = TFtoREModel(self.module, output_indices=None, key=key).to(device)
            ig = IntegratedGradients(ig_model)
            n_re_features = ig_model(inputs_up).shape[1]
            all_attributions = []
            for re_idx in range(n_re_features):
                attribution, delta = ig.attribute(
                    inputs_up,
                    baselines=baseline,
                    target=re_idx,
                    return_convergence_delta=True
                )
                all_attributions.append(attribution.detach().cpu().numpy())
            all_attributions = np.stack(all_attributions, axis=0)
            ig_scores = np.transpose(all_attributions, (1, 2, 0))
        else:
            raise ValueError(f"Cell type '{celltype}' not found in adata.obs['celltype'].")
        return ig_scores

    def get_3to1_ig(
            self,
            adata=None,
            key='prob',
            celltype=None,
            baseline=None,
    ):
        """Compute IG values from TFs to TGs.
        """
        if self.module.training:
            self.module.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adata = adata if adata is not None else self.adata
        if celltype is None or celltype == 'all':
            inputs_up = torch.tensor(adata.X, dtype=torch.float32).to(device)
            if baseline is None:
                baseline = torch.zeros_like(inputs_up)
            ig_model = TFtoTGModel(self.module, output_indices=None, key=key).to(device)
            ig = IntegratedGradients(ig_model)
            n_re_features = ig_model(inputs_up).shape[1]
            all_attributions = []
            for re_idx in range(n_re_features):
                attribution, delta = ig.attribute(
                    inputs_up,
                    baselines=baseline,
                    target=re_idx,
                    return_convergence_delta=True
                )
                all_attributions.append(attribution.detach().cpu().numpy())
            all_attributions = np.stack(all_attributions, axis=0)
            ig_scores = np.transpose(all_attributions, (1, 2, 0))
        elif celltype in adata.obs['celltype'].unique():
            adata_subset = adata[adata.obs['celltype'] == celltype].copy()
            inputs_up = torch.tensor(adata_subset.X, dtype=torch.float32).to(device)
            if baseline is None:
                baseline = torch.zeros_like(inputs_up)
            ig_model = TFtoTGModel(self.module, output_indices=None, key=key).to(device)
            ig = IntegratedGradients(ig_model)
            n_re_features = ig_model(inputs_up).shape[1]
            all_attributions = []
            for re_idx in range(n_re_features):
                attribution, delta = ig.attribute(
                    inputs_up,
                    baselines=baseline,
                    target=re_idx,
                    return_convergence_delta=True
                )
                all_attributions.append(attribution.detach().cpu().numpy())
            all_attributions = np.stack(all_attributions, axis=0)
            ig_scores = np.transpose(all_attributions, (1, 2, 0))
        else:
            raise ValueError(f"Cell type '{celltype}' not found in adata.obs['celltype'].")
        return ig_scores

    @torch.no_grad()
    def get_model_output(
            self,
            adata: Optional[AnnData] = None,
            batch_size: Optional[int] = None,
    ):
        r"""Obtain model predictions and latent representations for a given dataset.

        This method runs the trained model in evaluation mode and returns:
        - Concatenated latent embeddings from two latent spaces,
        - Logits and predicted probabilities from the downstream classifier,
        - Binary class predictions (thresholded at 0.5).
    
        If no `adata` is provided, the method uses the internal `self.adata`.
    
        Parameters
        ----------
        adata : AnnData, optional
            Annotated data matrix to generate outputs for. If None, defaults to `self.adata`.
            Default is None.
        batch_size : int, optional
            Number of samples per batch during inference. If None, uses `self.batch_size`.
            Default is None.
    
        Returns
        -------
        output : dict
            Dictionary containing the following keys:
            - `'latent'`: numpy.ndarray of shape `(n_samples, n_latent1 + n_latent2)`,  
              concatenated latent vectors from both latent modules.
            - `'logits'`: numpy.ndarray of shape `(n_samples,)` or `(n_samples, n_classes)`,  
              raw classifier logits.
            - `'probs'`: numpy.ndarray of same shape as `'logits'`,  
              predicted probabilities after sigmoid/softmax activation.
            - `'preds'`: numpy.ndarray of shape `(n_samples,)`,  
              binary predictions (1 if probability > 0.5, else 0).

        """
        if self.module.training:
            self.module.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        latent = []
        logits = []
        probs = []
        preds = []
        x_down1_rec_alpha = []
        x_down2_rec_alpha = []
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            inputs = torch.tensor(data.X, dtype=torch.float32, device=device)
            model_outputs = self.module(inputs, use_mean=True)
            latent_z = torch.cat([model_outputs["latent1"]["z"], model_outputs["latent2"]["z"]], dim=1)
            latent.append(latent_z.cpu().numpy())
            # latent.append(model_outputs['latent_up']['qz_m'].cpu().numpy())
            logits.append(model_outputs['alpha_dpd']['logit'].cpu().numpy())
            probs.append(model_outputs["alpha_dpd"]["prob"].cpu().numpy())
            preds.append(np.int_(model_outputs['alpha_dpd']['prob'].cpu().numpy() > 0.5))
            x_down1_rec_alpha.append(model_outputs["x_down1_rec_alpha"].cpu().numpy())
            x_down2_rec_alpha.append(model_outputs["x_down2_rec_alpha"].cpu().numpy())

        output = dict(latent=np.concatenate(latent, axis=0),
                      logits=np.concatenate(logits, axis=0),
                      probs=np.concatenate(probs, axis=0),
                      preds=np.concatenate(preds, axis=0),
                      x_down1_rec_alpha=np.concatenate(x_down1_rec_alpha, axis=0),
                      x_down2_rec_alpha=np.concatenate(x_down2_rec_alpha, axis=0)
                      )

        return output

    @torch.no_grad()
    def compute_information_flow(
            self,
            adata: Optional[AnnData] = None,
            dims: Optional[List[int]] = None,
            zero_floor: bool = False,
            plot_info_flow: Optional[bool] = True,
            skip_single_info: Optional[bool] = True,
            save_fig: Optional[bool] = False,
            save_dir: Optional[str] = None,
    ):
        r"""
        Compute information flow for latent dimensions.
        
        Parameters
        ----------
        adata
            AnnData object with input data
        dims
            Dimensions to compute information flow for
        zero_floor
            Whether to subtract minimum value
        plot_info_flow
            Whether to plot information flow
        skip_single_info
            Whether to skip single dimension plots
        save_fig
            Whether to save figures
        save_dir
            Directory to save figures
            
        Returns
        ----------
        info_flow
            Information flow for each dimension
        info_flow_cat
            Categorical information flow (causal vs spurious)
        """
        if self.module.training:
            self.module.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adata = adata if adata is not None else self.adata
        ce_params = self.ce_params
        if dims is None:
            dims = list(range(self.module.n_latent))

        # Calculate information flow
        info_flow = pd.DataFrame(index=range(adata.shape[0]), columns=dims)
        for i in range(adata.shape[0]):
            for j in dims:
                # Get the latent space of the current sample
                inputs = torch.tensor(adata.X, dtype=torch.float32, device=device)
                # Calculate the information flow
                info = joint_uncond_single_dim_v2(ce_params, self.module, inputs, i, j, alpha_vi=False, beta_vi=True,
                                                  device=device)
                info_flow.loc[i, j] = info.item()
        info_flow.set_index(adata.obs_names, inplace=True)
        if zero_floor:
            info_flow = info_flow - info_flow.min().min()
        info_flow = info_flow.apply(lambda x: x / (np.linalg.norm(x, ord=1) + 1e-8), axis=1)

        # Calculate information flow for causal and spurious dimensions
        dims = ['causal', 'spurious']
        info_flow_cat = pd.DataFrame(index=range(adata.shape[0]), columns=dims)
        for i in range(adata.shape[0]):
            # Get the latent space of the current sample
            inputs = torch.tensor(adata.X, dtype=torch.float32, device=device)
            # Calculate the information flow
            info_c, _ = joint_uncond_v2(ce_params, self.module, inputs, i, alpha_vi=False, beta_vi=True, device=device)
            info_s, _ = beta_info_flow_v2(ce_params, self.module, inputs, i, alpha_vi=True, beta_vi=False,
                                          device=device)
            info_flow_cat.loc[i, 'causal'] = -info_c.item()
            info_flow_cat.loc[i, 'spurious'] = -info_s.item()
        info_flow_cat.set_index(adata.obs_names, inplace=True)
        if zero_floor:
            info_flow_cat = info_flow_cat - info_flow_cat.min().min()
        info_flow_cat = info_flow_cat.apply(lambda x: x / (np.linalg.norm(x, ord=1) + 1e-8), axis=1)

        if plot_info_flow and not skip_single_info:
            # plot the information flow
            plt.figure(figsize=(10, 5))
            ax = sns.boxplot(data=info_flow, palette="pastel")
            plt.xlabel("Dimensions")
            plt.ylabel("Information Measurements")
            if save_fig:
                plt.savefig(os.path.join(save_dir, "info_flow_3l.png"))
                plt.savefig(os.path.join(save_dir, "info_flow_3l.pdf"))
            plt.show()
            plt.close()
        if plot_info_flow:
            plt.figure(figsize=(10, 5))
            ax = sns.boxplot(data=info_flow_cat, palette="pastel")
            plt.xlabel("Dimensions")
            plt.ylabel("Information Measurements")
            if save_fig:
                plt.savefig(os.path.join(save_dir, "info_flow_3l_cat.png"))
                plt.savefig(os.path.join(save_dir, "info_flow_3l_cat.pdf"))
            plt.show()
            plt.close()

        return info_flow, info_flow_cat


class TFtoREModel(nn.Module):
    def __init__(self, original_model, output_indices=None, key='prob'):
        super().__init__()
        self.model = original_model
        self.output_indices = output_indices
        self.key = key

    def forward(self, x_up):
        latent1, latent2, _ = self.model.encode_x_up(x_up, use_mean=True)
        n_causal = self.model.n_causal

        alpha_z = torch.zeros_like(torch.cat((latent1["z"], latent2["z"]), dim=1))
        alpha_z[:, :n_causal] = latent1["z"]
        alpha_z[:, n_causal:] = latent2["z"].mean(dim=0, keepdim=True)

        x_down1_rec = self.model.decoder_down1(alpha_z)
        if self.output_indices is not None:
            x_down1_rec = x_down1_rec[:, self.output_indices]
        return x_down1_rec


class TFtoTGModel(nn.Module):
    def __init__(self, original_model, output_indices=None, key='prob'):
        super().__init__()
        self.model = original_model
        self.output_indices = output_indices
        self.key = key

    def forward(self, x_up):
        latent1, latent2, _ = self.model.encode_x_up(x_up, use_mean=True)
        n_causal = self.model.n_causal

        alpha_z = torch.zeros_like(torch.cat((latent1["z"], latent2["z"]), dim=1))
        alpha_z[:, :n_causal] = latent1["z"]
        alpha_z[:, n_causal:] = latent2["z"].mean(dim=0, keepdim=True)

        x_down1_rec = self.model.decoder_down1(alpha_z)
        x_down1_rec, feat_w_down1 = self.model.feature_mapper_down1(x_down1_rec, mode="causal")
        x_down2_rec = self.model.decoder_down2(x_down1_rec)

        if self.output_indices is not None:
            x_down2_rec = x_down2_rec[:, self.output_indices]
        return x_down2_rec


class MtoPModel(nn.Module):
    def __init__(self, original_model, output_indices=None, key='prob'):
        super().__init__()
        self.model = original_model
        self.output_indices = output_indices
        self.key = key

    def forward(self, x_up):
        latent1, latent2, _ = self.model.encode_x_up(x_up, use_mean=True)
        n_causal = self.model.n_causal

        alpha_z = torch.zeros_like(torch.cat((latent1["z"], latent2["z"]), dim=1))
        alpha_z[:, :n_causal] = latent1["z"]
        alpha_z[:, n_causal:] = latent2["z"].mean(dim=0, keepdim=True)

        x_down1_rec = self.model.decoder_down(alpha_z)
        if self.output_indices is not None:
            x_down1_rec = x_down1_rec[:, self.output_indices]
        return x_down1_rec


class MtoZModel(nn.Module):
    def __init__(self, original_model, output_indices=None, key='prob'):
        super().__init__()
        self.model = original_model
        self.output_indices = output_indices
        self.key = key

    def forward(self, x_up):
        latent1, latent2, _ = self.model.encode_x_up(x_up, use_mean=True)
        alpha_z = torch.cat((latent1["z"], latent2["z"]), dim=1)

        return alpha_z


class ShapModel(nn.Module):
    def __init__(self, original_model, key='prob'):
        super().__init__()
        self.original_model = original_model
        self.key = key

    def forward(self, x):
        model_outputs = self.original_model(x, use_mean=True)
        output = model_outputs["alpha_dpd"][self.key]
        return output
