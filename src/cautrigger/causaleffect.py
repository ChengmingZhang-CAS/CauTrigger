import numpy as np
import torch
import torch.nn.functional as F


def joint_uncond_v2(params, model, data, index, alpha_vi=False, beta_vi=True, eps=1e-8, device=None):
    r"""Estimate the negative unconditional mutual information -I(α; Ŷ) via Monte Carlo sampling.

    This function computes a sample-based estimate of the causal effect of the **causal latent factors**
    (denoted α, dimension K) on the model's predicted output Ŷ. It marginalizes over both causal and
    non-causal (spurious) latent variables using variational posterior statistics or standard normal priors.

    The mutual information is approximated as:
        I(α; Ŷ) ≈ E_{α}[KL(p(Ŷ|α) || p(Ŷ))] = H(Ŷ) - H(Ŷ|α)
    and this function returns **-I(α; Ŷ)**.

    Parameters
    ----------
    params : dict
        Dictionary containing simulation parameters:
        - 'N_alpha' (int): Number of Monte Carlo samples for causal latents (α).
        - 'N_beta' (int): Number of Monte Carlo samples for non-causal latents (β).
        - 'K' (int): Dimensionality of causal latent space.
        - 'L' (int): Dimensionality of non-causal (spurious) latent space.
        - 'M' (int): Number of output classes (dimension of classifier logits/probabilities).
    model : torch.nn.Module
        Trained model with components: `feature_mapper_up`, `encoder1`, `encoder2`,
        `decoder_down` (or `decoder_down1/2`), `feature_mapper_down`, and `dpd_model`.
    data : torch.Tensor
        Input data tensor of shape `(n_samples, n_features)`.
    index : int
        Index of the input sample in `data` to evaluate.
    alpha_vi : bool, optional
        If True, sample α from the empirical mean/variance of its inferred posterior;
        otherwise, use standard normal prior (μ=0, σ=1). Default is False.
    beta_vi : bool, optional
        If True, sample β from the empirical mean/variance of its inferred posterior;
        otherwise, use standard normal prior. Default is True.
    eps : float, optional
        Small constant for numerical stability in log-probability clamping. Default is 1e-8.
    device : torch.device or str, optional
        Device to perform computation on (e.g., 'cuda' or 'cpu'). If None, uses model's device.

    Returns
    -------
    neg_causal_effect : torch.Tensor
        Scalar tensor representing the estimated **-I(α; Ŷ)**.
    info : None
        Placeholder for compatibility; always returns None.
    """
    I = 0.0
    q = torch.zeros(params['M'], device=device)
    feat = data[index].repeat(params['N_alpha'] * params['N_beta'], 1)
    x1, _ = model.feature_mapper_up(feat, mode='causal')
    latent1 = model.encoder1(x1)
    x2, _ = model.feature_mapper_up(feat, mode='spurious')
    latent2 = model.encoder2(x2)
    mu = torch.cat((latent1["qz_m"], latent2["qz_m"]), dim=-1)
    std = torch.cat((latent1["qz_v"].sqrt(), latent2["qz_v"].sqrt()), dim=-1)

    # x_up_w, _ = model.feature_mapper_up(feat)
    # latent = model.encoder(x_up_w)
    # mu, std = latent["qz_m"], latent["qz_v"].sqrt()
    # print(std.abs().max().cpu().detach().numpy(), std.abs().min().cpu().detach().numpy())
    if alpha_vi:
        alpha_mu = mu[:, :params['K']].mean(0)
        alpha_std = std[:, :params['K']].mean(0)
    else:
        alpha_mu = 0
        alpha_std = 1

    if beta_vi:
        beta_mu = mu[:, params['K']:].mean(0)
        beta_std = std[:, params['K']:].mean(0)
    else:
        beta_mu = 0
        beta_std = 1

    alpha = torch.randn((params['N_alpha'], params['K']), device=device).mul(alpha_std).add_(alpha_mu).repeat(1, params[
        'N_beta']).view(params['N_alpha'] * params['N_beta'], params['K'])
    beta = torch.randn((params['N_alpha'] * params['N_beta'], params['L']), device=device).mul(beta_std).add_(beta_mu)
    zs = torch.cat([alpha, beta], dim=-1)
    if hasattr(model, 'decoder_down1'):
        # 3-layer
        x_down1_rec = model.decoder_down1(zs)
        x_down1_rec, _ = model.feature_mapper_down1(x_down1_rec, mode="causal")
        x_down_rec = model.decoder_down2(x_down1_rec)
        x_down_w, _ = model.feature_mapper_down2(x_down_rec, mode="causal")
    else:
        # 2-layer
        x_down_rec = model.decoder_down(zs)
        x_down_w, _ = model.feature_mapper_down(x_down_rec, mode="causal")

    logit, prob = model.dpd_model(x_down_w).values()
    if params['M'] == 2:
        yhat = torch.cat((prob, 1 - prob), dim=1).view(params['N_alpha'], params['N_beta'], params['M'])
    else:
        yhat = prob.view(params['N_alpha'], params['N_beta'], params['M'])
    # yhat = torch.cat((prob, 1 - prob), dim=1).view(params['N_alpha'], params['N_beta'], params['M'])

    p = yhat.mean(1)
    p = torch.clamp(p, eps, 1 - eps)
    I = torch.sum(torch.xlogy(p, p), dim=1).mean()
    # I = torch.sum(torch.mul(p, torch.log(p)), dim=1).mean()
    # I = torch.sum(torch.mul(p, F.log_softmax(p, dim=1)), dim=1).mean()

    q = p.mean(0)
    q = torch.clamp(q, eps, 1 - eps)
    I = I - torch.sum(torch.xlogy(q, q))
    # I = I - torch.sum(torch.mul(q, torch.log(q)))
    # I = I - torch.sum(q * F.log_softmax(q.unsqueeze(0), dim=1))

    return -I, None


def beta_info_flow_v2(params, model, data, index, alpha_vi=True, beta_vi=False, eps=1e-8, device=None):
    r"""Estimate the negative mutual information -I(β; Ŷ) to quantify spurious information flow.

    This function evaluates how much information the **non-causal (spurious) latent factors** (β, dimension L)
    leak into the model's prediction Ŷ. It is analogous to `joint_uncond_v2` but swaps the roles of α and β
    in the sampling procedure.

    The estimate corresponds to **-I(β; Ŷ)**, where lower (more negative) values indicate stronger
    undesirable dependence on spurious features.

    Parameters
    ----------
    params : dict
        Same as in `joint_uncond_v2`.
    model : torch.nn.Module
        Same as in `joint_uncond_v2`.
    data : torch.Tensor
        Input data tensor of shape `(n_samples, n_features)`.
    index : int
        Index of the input sample in `data` to evaluate.
    alpha_vi : bool, optional
        If True, sample α from its empirical posterior; otherwise, use standard normal.
        Default is True (opposite of `joint_uncond_v2`).
    beta_vi : bool, optional
        If True, sample β from its empirical posterior; otherwise, use standard normal.
        Default is False.
    eps : float, optional
        Numerical stability constant for probability clamping. Default is 1e-8.
    device : torch.device or str, optional
        Computation device. If None, inferred from model.

    Returns
    -------
    neg_info_flow : torch.Tensor
        Scalar tensor representing the estimated **-I(β; Ŷ)**.
    info : None
        Placeholder for compatibility; always returns None.
    """
    I = 0.0
    q = torch.zeros(params['M'], device=device)
    feat = data[index].repeat(params['N_alpha'] * params['N_beta'], 1)
    x1, _ = model.feature_mapper_up(feat, mode='causal')
    latent1 = model.encoder1(x1)
    x2, _ = model.feature_mapper_up(feat, mode='spurious')
    latent2 = model.encoder2(x2)
    mu = torch.cat((latent1["qz_m"], latent2["qz_m"]), dim=-1)
    std = torch.cat((latent1["qz_v"].sqrt(), latent2["qz_v"].sqrt()), dim=-1)

    # x_up_w, _ = model.feature_mapper_up(feat)
    # latent = model.encoder(x_up_w)
    # mu, std = latent["qz_m"], latent["qz_v"].sqrt()
    if alpha_vi:
        alpha_mu = mu[:, :params['K']].mean(0)
        alpha_std = std[:, :params['K']].mean(0)
    else:
        alpha_mu = 0
        alpha_std = 1

    if beta_vi:
        beta_mu = mu[:, params['K']:].mean(0)
        beta_std = std[:, params['K']:].mean(0)
    else:
        beta_mu = 0
        beta_std = 1

    alpha = torch.randn((params['N_alpha'] * params['N_beta'], params['K']), device=device).mul(alpha_std).add_(alpha_mu)
    beta = torch.randn((params['N_alpha'], params['L']), device=device).mul(beta_std).add_(beta_mu).repeat(
        1, params['N_beta']).view(params['N_alpha'] * params['N_beta'], params['L'])

    zs = torch.cat([alpha, beta], dim=-1)
    if hasattr(model, 'decoder_down1'):
        # 3-layer
        x_down1_rec = model.decoder_down1(zs)
        x_down1_rec, _ = model.feature_mapper_down1(x_down1_rec, mode="causal")
        x_down_rec = model.decoder_down2(x_down1_rec)
        x_down_w, _ = model.feature_mapper_down2(x_down_rec, mode="causal")
    else:
        # 2-layer
        x_down_rec = model.decoder_down(zs)
        x_down_w, _ = model.feature_mapper_down(x_down_rec, mode="causal")
    logit, prob = model.dpd_model(x_down_w).values()

    if params['M'] == 2:
        yhat = torch.cat((prob, 1 - prob), dim=1).view(params['N_alpha'], params['N_beta'], params['M'])
    else:
        yhat = prob.view(params['N_alpha'], params['N_beta'], params['M'])
    # yhat = torch.cat((prob, 1 - prob), dim=1).view(params['N_alpha'], params['N_beta'], params['M'])
    p = yhat.mean(1)
    p = torch.clamp(p, eps, 1 - eps)
    I = torch.sum(torch.mul(p, torch.log(p)), dim=1).mean()
    # I = torch.sum(torch.mul(p, F.log_softmax(p, dim=1)), dim=1).mean()
    q = p.mean(0)
    q = torch.clamp(q, eps, 1 - eps)
    I = I - torch.sum(torch.mul(q, torch.log(q)))
    # I = I - torch.sum(q * F.log_softmax(q.unsqueeze(0), dim=1))
    return -I, None


def joint_uncond_single_dim_v2(params, model, data, index, dim, alpha_vi=False, beta_vi=True, eps=1e-6, device=None):
    r"""Estimate -I(z_i; Ŷ) for a single latent dimension i via conditional Monte Carlo sampling.

    This function computes the mutual information between **one specific latent dimension** `z_i`
    (where `i = dim`) and the model's output Ŷ, marginalizing over all other latent dimensions.
    It is useful for per-dimension causal attribution.

    The procedure:
        1. Sample z_i independently `N_alpha` times.
        2. For each z_i, sample the remaining (z_dim - 1) dimensions `N_beta` times.
        3. Estimate p(Ŷ | z_i) and p(Ŷ) to compute I(z_i; Ŷ).

    Parameters
    ----------
    params : dict
        Dictionary containing:
        - 'N_alpha' (int): Samples for the target latent dimension.
        - 'N_beta' (int): Samples for all other latent dimensions per fixed z_i.
        - 'K', 'L' (int): Causal and spurious latent dimensions (used to infer total `z_dim = K + L`).
        - 'M' (int): Number of output classes.
    model : torch.nn.Module
        Same architecture assumptions as in `joint_uncond_v2`.
    data : torch.Tensor
        Input data tensor.
    index : int
        Index of the sample in `data` to analyze.
    dim : int
        Zero-indexed latent dimension to evaluate (0 ≤ dim < K + L).
    alpha_vi : bool, optional
        If True, sample the target dimension from its empirical posterior mean/std;
        else use standard normal. Default is False.
    beta_vi : bool, optional
        If True, sample **all** latent dimensions (including non-target) from their joint empirical posterior;
        else use standard normal for background dimensions. Default is True.
    eps : float, optional
        Clamping epsilon for log-probabilities. Slightly larger (1e-6) for single-dim stability.
        Default is 1e-6.
    device : torch.device or str, optional
        Computation device.

    Returns
    -------
    mutual_info_estimate : torch.Tensor
        Scalar tensor representing **I(z_dim; Ŷ)** (note: **not negated**, unlike the other two functions).
    """
    I = 0.0
    q = torch.zeros(params['M'], device=device)
    feat = data[index].repeat(params['N_alpha'] * params['N_beta'], 1)
    x1, _ = model.feature_mapper_up(feat, mode='causal')
    latent1 = model.encoder1(x1)
    x2, _ = model.feature_mapper_up(feat, mode='spurious')
    latent2 = model.encoder2(x2)
    mu = torch.cat((latent1["qz_m"], latent2["qz_m"]), dim=-1)
    std = torch.cat((latent1["qz_v"].sqrt(), latent2["qz_v"].sqrt()), dim=-1)

    if alpha_vi:
        alpha_mu = mu[:, dim].mean(0)
        alpha_std = std[:, dim].mean(0)
    else:
        alpha_mu = 0
        alpha_std = 1

    if beta_vi:
        beta_mu = mu.mean(0)
        beta_std = std.mean(0)
    else:
        beta_mu = 0
        beta_std = 1

    alpha = torch.randn((params['N_alpha'], 1), device=device).mul(alpha_std).add_(alpha_mu).repeat(
        1, params['N_beta']).view(params['N_alpha'] * params['N_beta'], 1)
    zs = torch.randn((params['N_alpha'] * params['N_beta'], params['z_dim']), device=device).mul(beta_std).add_(beta_mu)
    zs[:, dim] = alpha[:, 0]
    if hasattr(model, 'decoder_down1'):
        # 3-layer
        x_down1_rec = model.decoder_down1(zs)
        x_down1_rec, _ = model.feature_mapper_down1(x_down1_rec, mode="causal")
        x_down_rec = model.decoder_down2(x_down1_rec)
        x_down_w, _ = model.feature_mapper_down2(x_down_rec, mode="causal")
    else:
        # 2-layer
        x_down_rec = model.decoder_down(zs)
        x_down_w, _ = model.feature_mapper_down(x_down_rec, mode="causal")
    # x_down_rec = model.decoder_down(zs)
    # x_down_w, _ = model.feature_mapper_down(x_down_rec)
    logit, prob = model.dpd_model(x_down_w).values()

    if params['M'] == 2:
        yhat = torch.cat((prob, 1 - prob), dim=1).view(params['N_alpha'], params['N_beta'], params['M'])
    else:
        yhat = prob.view(params['N_alpha'], params['N_beta'], params['M'])
    # yhat = torch.cat((prob, 1 - prob), dim=1).view(params['N_alpha'], params['N_beta'], params['M'])

    p = yhat.mean(1)
    p = torch.clamp(p, eps, 1 - eps)
    I = torch.sum(torch.mul(p, torch.log(p)), dim=1).mean()
    # I = torch.sum(torch.mul(p, F.log_softmax(p, dim=1)), dim=1).mean()
    q = p.mean(0)
    q = torch.clamp(q, eps, 1 - eps)
    I = I - torch.sum(torch.mul(q, torch.log(q)))
    # I = I - torch.sum(q * F.log_softmax(q.unsqueeze(0), dim=1))
    return I

