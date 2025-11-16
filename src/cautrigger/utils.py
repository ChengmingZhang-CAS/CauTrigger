import os
import numpy as np
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_features(df, threshold=None, topk=None, elbow=False):
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)
    selected_features = []
    for column in df.columns:
        sorted_column = df[column].sort_values(ascending=False)

        if elbow:
            weights = sorted_column.values
            cumulative_weights = weights.cumsum()
            diff = pd.Series(cumulative_weights).diff()
            elbow_point = diff.idxmax() + 1 if diff.idxmax() is not None else 1
            selected = pd.Series(0, index=df.index)
            selected[sorted_column.nlargest(elbow_point).index] = 1

        else:
            if threshold and not topk:
                cum_sum = sorted_column.cumsum()
                selected = (cum_sum <= threshold).astype(int)
                if selected.sum() == 0:
                    selected[sorted_column.index[0]] = 1
            elif topk:
                top_k_features = sorted_column.nlargest(topk).index
                selected = pd.Series(0, index=df.index)
                selected[top_k_features] = 1
            else:
                raise ValueError('Please pass valid argument!')

        selected = pd.Series(selected, name=column)
        selected_features.append(selected)
    selected_df = pd.concat(selected_features, axis=1)
    selected_df.columns = df.columns
    return selected_df.reindex(df.index)


# The method of drawing vector field map is borrowed from method CellOracle: "Dissecting cell identity via network inference and in silico gene perturbation". https://github.com/morris-lab/CellOracle

def plot_vector_field(adata,pert_Gene=None,pert_celltype=None,state_obs=None,embedding_name=None,
                      sampled_fraction=1, min_mass=0.008,scale=0.1, save_dir=None,smooth=0.8,n_grid=40,
                     n_suggestion=12,show=True,dot_size=None,run_suggest_mass_thresholds=False,direction=None,n_neighbors=None,palette=None):
    from scipy import sparse
    from sklearn.neighbors import NearestNeighbors
    from velocyto.estimation import colDeltaCorpartial
    from scipy.stats import norm as normal
    import math

    def estimate_transition_prob(adata, embedding_name,sampled_fraction, n_neighbors=None, sigma_corr=0.005):
        sampling_probs = (0.5, 0.1)
        X = adata.layers["imputed_count"].transpose().copy()
        delta_X = adata.layers["delta_X"].transpose().copy()
        embedding = adata.obsm[embedding_name].copy()
        if n_neighbors is None:
            n_neighbors = int(adata.shape[0] / 5)
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=4)
        nn.fit(embedding)
        embedding_knn = nn.kneighbors_graph(mode="connectivity")
        neigh_ixs = embedding_knn.indices.reshape((-1, n_neighbors + 1))
        p = np.linspace(sampling_probs[0], sampling_probs[1], neigh_ixs.shape[1])
        p = p / p.sum()
        sampling_ixs = np.stack([np.random.choice(neigh_ixs.shape[1],
                                                  size=(int(sampled_fraction * (n_neighbors + 1)),),
                                                  replace=False,
                                                  p=p) for i in range(neigh_ixs.shape[0])], 0)
        neigh_ixs = neigh_ixs[np.arange(neigh_ixs.shape[0])[:, None], sampling_ixs]
        nonzero = neigh_ixs.shape[0] * neigh_ixs.shape[1]
        embedding_knn = sparse.csr_matrix((np.ones(nonzero),
                                           neigh_ixs.ravel(),
                                           np.arange(0, nonzero + 1, neigh_ixs.shape[1])),
                                          shape=(neigh_ixs.shape[0],
                                                 neigh_ixs.shape[0]))
        adata.uns['neigh_ixs'] = neigh_ixs.copy()
        corrcoef = colDeltaCorpartial(X, delta_X, neigh_ixs)
        if np.any(np.isnan(corrcoef)):
            corrcoef[np.isnan(corrcoef)] = 1
        # transition_prob = np.exp(corrcoef / sigma_corr) * embedding_knn.A
        transition_prob = np.exp(corrcoef / sigma_corr) * embedding_knn.toarray()
        transition_prob /= transition_prob.sum(1)[:, None]
        adata.obsm['embedding_knn'] = embedding_knn.copy()
        adata.obsp['transition_prob'] = transition_prob.copy()

    def calculate_embedding_shift(adata, embedding_name):
        transition_prob = adata.obsp['transition_prob'].copy()
        embedding = adata.obsm[embedding_name].copy()
        embedding_knn = adata.obsm['embedding_knn'].copy()
        unitary_vectors = embedding.T[:, None, :] - embedding.T[:, :, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            unitary_vectors /= np.linalg.norm(unitary_vectors, ord=2, axis=0)
            np.fill_diagonal(unitary_vectors[0, ...], 0)
            np.fill_diagonal(unitary_vectors[1, ...], 0)
        delta_embedding = (transition_prob * unitary_vectors).sum(2)
        # delta_embedding -= (embedding_knn.A * unitary_vectors).sum(2) / embedding_knn.sum(1).A.T
        delta_embedding -= (embedding_knn.toarray() * unitary_vectors).sum(2) / np.array(embedding_knn.sum(1)).T
        delta_embedding = delta_embedding.T
        adata.obsm['delta_embedding'] = delta_embedding.copy()

    def calculate_p_mass(adata,state_obs,pert_celltype, embedding_name, smooth, n_grid, n_neighbors=None):
        steps = (n_grid, n_grid)
        embedding = adata.obsm[embedding_name].copy()
        adata_tmp = adata.copy()
        adata_tmp.obsm['delta_embedding'][~adata_tmp.obs[state_obs].isin(pert_celltype)] = 0
        delta_embedding = adata_tmp.obsm['delta_embedding'].copy()
        grs = []
        for dim_i in range(embedding.shape[1]):
            m, M = np.min(embedding[:, dim_i]), np.max(embedding[:, dim_i])
            m = m - 0.025 * np.abs(M - m)
            M = M + 0.025 * np.abs(M - m)
            gr = np.linspace(m, M, steps[dim_i])
            grs.append(gr)
        meshes_tuple = np.meshgrid(*grs)
        gridpoints_coordinates = np.vstack([i.flat for i in meshes_tuple]).T
        if n_neighbors is None:
            n_neighbors = int(adata.shape[0] / 5)
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(embedding)
        dists, neighs = nn.kneighbors(gridpoints_coordinates)
        std = np.mean([(g[1] - g[0]) for g in grs])
        gaussian_w = normal.pdf(loc=0, scale=smooth * std, x=dists)
        total_p_mass = gaussian_w.sum(1)
        UZ = (delta_embedding[neighs] * gaussian_w[:, :, None]).sum(1) / np.maximum(1, total_p_mass)[:,None]
        magnitude = np.linalg.norm(UZ, axis=1)
        flow_embedding = embedding
        flow_grid = gridpoints_coordinates
        flow = UZ
        flow_norm = UZ / np.percentile(magnitude, 99.5)
        flow_norm_magnitude = np.linalg.norm(flow_norm, axis=1)
        adata.uns['total_p_mass'] = total_p_mass.copy()
        adata.uns['flow_grid'] = flow_grid.copy()
        adata.uns['flow'] = flow.copy()

    def suggest_mass_thresholds(adata, embedding_name, n_suggestion,save_dir, s=1, n_col=4):
        embedding = adata.obsm[embedding_name].copy()
        total_p_mass = adata.uns['total_p_mass'].copy()
        flow_grid = adata.uns['flow_grid'].copy()
        min_ = total_p_mass.min()
        max_ = total_p_mass.max()
        suggestions = np.linspace(min_, max_ / 2, n_suggestion)
        n_rows = math.ceil(n_suggestion / n_col)
        fig, ax = plt.subplots(n_rows, n_col, figsize=[5 * n_col, 5 * n_rows])
        if n_rows == 1:
            ax = ax.reshape(1, -1)
        row = 0
        col = 0
        for i in range(n_suggestion):
            ax_ = ax[row, col]
            col += 1
            if col == n_col:
                col = 0
                row += 1
            idx = total_p_mass > suggestions[i]
            ax_.scatter(embedding[:, 0], embedding[:, 1], c="lightgray", s=s)
            ax_.scatter(flow_grid[idx, 0],
                        flow_grid[idx, 1],
                        c="black", s=s)
            ax_.set_title(f"min_mass: {suggestions[i]: .2g}")
            ax_.axis("off")
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/suggest_mass_thresholds.png", bbox_inches='tight')
        plt.show()

    def calculate_mass_filter(adata, embedding_name, min_mass, plot=False):
        embedding = adata.obsm[embedding_name].copy()
        total_p_mass = adata.uns['total_p_mass'].copy()
        flow_grid = adata.uns['flow_grid'].copy()
        mass_filter = (total_p_mass < min_mass)
        adata.uns['mass_filter'] = mass_filter.copy()
        if plot:
            fig, ax = plt.subplots(figsize=[5, 5])
            ax.scatter(embedding[:, 0], embedding[:, 1], c="lightgray", s=10)
            ax.scatter(flow_grid[~mass_filter, 0],
                       flow_grid[~mass_filter, 1],
                       c="black", s=0.5)
            ax.set_title("Grid points selected")
            ax.axis("off")
            plt.show()

    def plot_flow(adata, state_obs, embedding_name, dot_size, scale, pert_Gene, save_dir, show,direction,palette=None):
        fig, ax = plt.subplots()
        sc.pl.embedding(adata, basis=embedding_name, color=state_obs, ax=ax, show=False,size=dot_size, palette=palette)
        ax.set_title("")
        if ax.get_legend() is not None:
            ax.get_legend().set_visible(False)
        mass_filter = adata.uns['mass_filter'].copy()
        gridpoints_coordinates = adata.uns['flow_grid'].copy()
        flow = adata.uns['flow'].copy()
        ax.quiver(gridpoints_coordinates[~mass_filter, 0],
                           gridpoints_coordinates[~mass_filter, 1],
                           flow[~mass_filter, 0],
                           flow[~mass_filter, 1],
                           scale=scale)
        ax.axis("off")
        if direction == 'Activation':
            ax.set_title(f"{' and '.join(pert_Gene)} {direction}")
        elif direction =='both':
            ax.set_title(f"{pert_Gene[0]} Activation and {pert_Gene[1]} Knock Out")
        else:
            ax.set_title(f"{' and '.join(pert_Gene)} Knock Out")
        plt.tight_layout()
        if save_dir:
            if direction == 'Activation':
                plt.savefig(f"{save_dir}/{' and '.join(pert_Gene)} {direction}.png", bbox_inches='tight')
                plt.savefig(f"{save_dir}/{' and '.join(pert_Gene)} {direction}.pdf", bbox_inches='tight')
            elif direction =='both':
                plt.savefig(f"{save_dir}/{pert_Gene[0]} Activation and {pert_Gene[1]} Knock Out.png", bbox_inches='tight')
                plt.savefig(f"{save_dir}/{pert_Gene[0]} Activation and {pert_Gene[1]} Knock Out.pdf", bbox_inches='tight')
            else:
                plt.savefig(f"{save_dir}/{' and '.join(pert_Gene)} Knock Out.png", bbox_inches='tight')
                plt.savefig(f"{save_dir}/{' and '.join(pert_Gene)} Knock Out.pdf", bbox_inches='tight')
        if show:
            plt.show()

    estimate_transition_prob(adata,embedding_name=embedding_name, n_neighbors=n_neighbors, sampled_fraction=sampled_fraction)
    calculate_embedding_shift(adata,embedding_name=embedding_name)
    calculate_p_mass(adata,embedding_name=embedding_name,n_neighbors=n_neighbors,pert_celltype=pert_celltype, smooth=smooth, n_grid=n_grid, state_obs=state_obs)
    if run_suggest_mass_thresholds:
        suggest_mass_thresholds(adata, embedding_name=embedding_name,n_suggestion=n_suggestion,save_dir=save_dir)
        return adata
    calculate_mass_filter(adata,embedding_name=embedding_name, min_mass=min_mass, plot=False)
    plot_flow(adata,state_obs=state_obs, embedding_name=embedding_name,dot_size=dot_size, scale=scale,show=show,direction=direction,pert_Gene=pert_Gene, save_dir=save_dir, palette=palette)
    return adata


def pert_plot_vector_field(adata_TF, adata_down, model, pert_Gene,pert_celltype, run_suggest_mass_thresholds,fold,state_obs,dot_size=None, scale=0.1, min_mass=0.008,save_dir=None,embedding_name='X_tsne',n_neighbors=None,n_grid=40,palette=None,direction=None):
    """
    Perform in silico perturbation and visualize the resulting vector field to show cell state transitions. The method of drawing vector field map is borrowed from method CellOracle

    Parameters:
        - adata_TF [AnnData]: Annotated data object containing transcription factor expression data.
        - adata_down [AnnData]: Annotated data object for downstream analysis with embedding coordinates.
        - model [torch.nn.Module]: Trained CAUTrigger model with eval() and get_model_output() methods.
        - pert_Gene [list of str]: List of features to be perturbed.
        - pert_celltype [list of str]: List of cell types that will receive the perturbation.
        - run_suggest_mass_thresholds [bool]: Whether to run suggested mass threshold selection. 
                                             Must be set to True initially to determine an appropriate 'min_mass'.
        - fold [list of float]: Multiplication factors for each perturbed feature; length must match pert_Gene.
        - state_obs [str]: Column name in adata_TF.obs representing cell states/types.
        - dot_size [float, optional]: Size of dots in the plot. Defaults to scanpy's default if None.
        - scale [float]: Scaling factor for vector arrows. Larger values result in smaller arrows. Default is 0.1.
        - min_mass [float]: Threshold for filtering low-density grid points. Use 'run_suggest_mass_thresholds=True'
                           first to determine an appropriate value. Default is 0.008.
        - save_dir [str, optional]: Directory path to save plots. No saving occurs if None. Default is None.
        - embedding_name [str]: Key for embedding coordinates in adata_down.obsm. Default is 'X_tsne'.
        - n_neighbors [int, optional]: Number of neighbors for vector field calculation. If None, defaults to 
                                      int(adata.shape[0] / 5). Default is None.
        - n_grid [int]: Grid density for vector field visualization. Default is 40.
        - palette [dict, optional]: Color mapping dictionary for cell types. Default is None.
        - direction [str, optional]: Label indicating perturbation direction for figure title. Default is None.
    Returns:
        - model_output_pert [dict]: Model output after applying perturbations.
    """
    model.eval()
    with torch.no_grad():
        model_output = model.get_model_output(adata_TF)
    adata_pert = adata_TF.copy()
    pert_cell_idx = np.where(adata_TF.obs[state_obs].isin(pert_celltype))[0]
    for ind, gene in enumerate(pert_Gene):
        adata_pert.X[:, adata_pert.var_names.get_loc(gene)] = adata_pert.X[pert_cell_idx, adata_pert.var_names.get_loc(gene)].max() * fold[ind]
    model.eval()
    with torch.no_grad():
        model_output_pert = model.get_model_output(adata_pert)

    if "x_down2_rec_alpha" in model_output:
        down_key = "x_down2_rec_alpha"  # 3L model
    elif "x_down_rec_alpha" in model_output:
        down_key = "x_down_rec_alpha"  # 2L model
    else:
        raise KeyError("Cannot find x_down_rec_alpha or x_down2_rec_alpha in model output.")

    adata_down.layers["imputed_count"] = np.float64(np.exp(model_output[down_key]))
    adata_down.layers["simulated_count"] = np.float64(np.exp(model_output_pert[down_key]))
    adata_down.layers["delta_X"] = adata_down.layers["simulated_count"].copy() - adata_down.layers["imputed_count"].copy()
    ax = plot_vector_field(adata_down, embedding_name=embedding_name, state_obs=state_obs,
                                   pert_Gene=pert_Gene, pert_celltype=pert_celltype, scale=scale,min_mass=min_mass,
                           save_dir=save_dir,dot_size=dot_size,run_suggest_mass_thresholds=run_suggest_mass_thresholds,
                           n_neighbors=n_neighbors,n_grid=n_grid,palette=palette, direction=direction)
    return model_output_pert

def plot_stream(adata,pert_Gene=None,pert_celltype=None,state_obs=None,embedding_name=None,
                      sampled_fraction=1, min_mass=0.008,scale=0.1, save_dir=None,smooth=0.8,n_grid=40,
                     n_suggestion=12,show=True,dot_size=None,run_suggest_mass_thresholds=False,direction=None,n_neighbors=None,palette=None):
    from scipy import sparse
    from sklearn.neighbors import NearestNeighbors
    from velocyto.estimation import colDeltaCorpartial
    from scipy.stats import norm as normal
    import math

    def estimate_transition_prob(adata, embedding_name,sampled_fraction, n_neighbors=None, sigma_corr=0.005):
        sampling_probs = (0.5, 0.1)
        X = adata.layers["imputed_count"].transpose().copy()
        delta_X = adata.layers["delta_X"].transpose().copy()
        embedding = adata.obsm[embedding_name].copy()
        if n_neighbors is None:
            n_neighbors = int(adata.shape[0] / 5)
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=4)
        nn.fit(embedding)
        embedding_knn = nn.kneighbors_graph(mode="connectivity")
        neigh_ixs = embedding_knn.indices.reshape((-1, n_neighbors + 1))
        p = np.linspace(sampling_probs[0], sampling_probs[1], neigh_ixs.shape[1])
        p = p / p.sum()
        sampling_ixs = np.stack([np.random.choice(neigh_ixs.shape[1],
                                                  size=(int(sampled_fraction * (n_neighbors + 1)),),
                                                  replace=False,
                                                  p=p) for i in range(neigh_ixs.shape[0])], 0)
        neigh_ixs = neigh_ixs[np.arange(neigh_ixs.shape[0])[:, None], sampling_ixs]
        nonzero = neigh_ixs.shape[0] * neigh_ixs.shape[1]
        embedding_knn = sparse.csr_matrix((np.ones(nonzero),
                                           neigh_ixs.ravel(),
                                           np.arange(0, nonzero + 1, neigh_ixs.shape[1])),
                                          shape=(neigh_ixs.shape[0],
                                                 neigh_ixs.shape[0]))
        adata.uns['neigh_ixs'] = neigh_ixs.copy()
        corrcoef = colDeltaCorpartial(X, delta_X, neigh_ixs)
        if np.any(np.isnan(corrcoef)):
            corrcoef[np.isnan(corrcoef)] = 1
        # transition_prob = np.exp(corrcoef / sigma_corr) * embedding_knn.A
        transition_prob = np.exp(corrcoef / sigma_corr) * embedding_knn.toarray()
        transition_prob /= transition_prob.sum(1)[:, None]
        adata.obsm['embedding_knn'] = embedding_knn.copy()
        adata.obsp['transition_prob'] = transition_prob.copy()

    def calculate_embedding_shift(adata, embedding_name):
        transition_prob = adata.obsp['transition_prob'].copy()
        embedding = adata.obsm[embedding_name].copy()
        embedding_knn = adata.obsm['embedding_knn'].copy()
        unitary_vectors = embedding.T[:, None, :] - embedding.T[:, :, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            unitary_vectors /= np.linalg.norm(unitary_vectors, ord=2, axis=0)
            np.fill_diagonal(unitary_vectors[0, ...], 0)
            np.fill_diagonal(unitary_vectors[1, ...], 0)
        delta_embedding = (transition_prob * unitary_vectors).sum(2)
        # delta_embedding -= (embedding_knn.A * unitary_vectors).sum(2) / embedding_knn.sum(1).A.T
        delta_embedding -= (embedding_knn.toarray() * unitary_vectors).sum(2) / np.array(embedding_knn.sum(1)).T
        delta_embedding = delta_embedding.T
        adata.obsm['delta_embedding'] = delta_embedding.copy()

    def calculate_p_mass(adata,state_obs,pert_celltype, embedding_name, smooth, n_grid, n_neighbors=None):
        steps = (n_grid, n_grid)
        embedding = adata.obsm[embedding_name].copy()
        adata_tmp = adata.copy()
        adata_tmp.obsm['delta_embedding'][~adata_tmp.obs[state_obs].isin(pert_celltype)] = 0
        delta_embedding = adata_tmp.obsm['delta_embedding'].copy()
        grs = []
        for dim_i in range(embedding.shape[1]):
            m, M = np.min(embedding[:, dim_i]), np.max(embedding[:, dim_i])
            m = m - 0.025 * np.abs(M - m)
            M = M + 0.025 * np.abs(M - m)
            gr = np.linspace(m, M, steps[dim_i])
            grs.append(gr)
        meshes_tuple = np.meshgrid(*grs)
        gridpoints_coordinates = np.vstack([i.flat for i in meshes_tuple]).T
        if n_neighbors is None:
            n_neighbors = int(adata.shape[0] / 5)
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(embedding)
        dists, neighs = nn.kneighbors(gridpoints_coordinates)
        std = np.mean([(g[1] - g[0]) for g in grs])
        gaussian_w = normal.pdf(loc=0, scale=smooth * std, x=dists)
        total_p_mass = gaussian_w.sum(1)
        UZ = (delta_embedding[neighs] * gaussian_w[:, :, None]).sum(1) / np.maximum(1, total_p_mass)[:,None]
        magnitude = np.linalg.norm(UZ, axis=1)
        flow_embedding = embedding
        flow_grid = gridpoints_coordinates
        flow = UZ
        flow_norm = UZ / np.percentile(magnitude, 99.5)
        flow_norm_magnitude = np.linalg.norm(flow_norm, axis=1)
        adata.uns['total_p_mass'] = total_p_mass.copy()
        adata.uns['flow_grid'] = flow_grid.copy()
        adata.uns['flow'] = flow.copy()

    def suggest_mass_thresholds(adata, embedding_name, n_suggestion,save_dir, s=1, n_col=4):
        embedding = adata.obsm[embedding_name].copy()
        total_p_mass = adata.uns['total_p_mass'].copy()
        flow_grid = adata.uns['flow_grid'].copy()
        min_ = total_p_mass.min()
        max_ = total_p_mass.max()
        suggestions = np.linspace(min_, max_ / 2, n_suggestion)
        n_rows = math.ceil(n_suggestion / n_col)
        fig, ax = plt.subplots(n_rows, n_col, figsize=[5 * n_col, 5 * n_rows])
        if n_rows == 1:
            ax = ax.reshape(1, -1)
        row = 0
        col = 0
        for i in range(n_suggestion):
            ax_ = ax[row, col]
            col += 1
            if col == n_col:
                col = 0
                row += 1
            idx = total_p_mass > suggestions[i]
            ax_.scatter(embedding[:, 0], embedding[:, 1], c="lightgray", s=s)
            ax_.scatter(flow_grid[idx, 0],
                        flow_grid[idx, 1],
                        c="black", s=s)
            ax_.set_title(f"min_mass: {suggestions[i]: .2g}")
            ax_.axis("off")
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/suggest_mass_thresholds.png", bbox_inches='tight')
        plt.show()

    def calculate_mass_filter(adata, embedding_name, min_mass, plot=False):
        embedding = adata.obsm[embedding_name].copy()
        total_p_mass = adata.uns['total_p_mass'].copy()
        flow_grid = adata.uns['flow_grid'].copy()
        mass_filter = (total_p_mass < min_mass)
        adata.uns['mass_filter'] = mass_filter.copy()
        if plot:
            fig, ax = plt.subplots(figsize=[5, 5])
            ax.scatter(embedding[:, 0], embedding[:, 1], c="lightgray", s=10)
            ax.scatter(flow_grid[~mass_filter, 0],
                       flow_grid[~mass_filter, 1],
                       c="black", s=0.5)
            ax.set_title("Grid points selected")
            ax.axis("off")
            plt.show()

    def plot_flow(adata, state_obs, embedding_name, dot_size, scale, pert_Gene, save_dir, show,direction,palette=None):
        fig, ax = plt.subplots()
        sc.pl.embedding(adata, basis=embedding_name, color=state_obs, ax=ax, show=False,size=dot_size, palette=palette)
        ax.set_title("")
        if ax.get_legend() is not None:
            ax.get_legend().set_visible(False)
        mass_filter = adata.uns['mass_filter'].copy()
        gridpoints_coordinates = adata.uns['flow_grid'].copy()
        flow = adata.uns['flow'].copy()


        from scipy.interpolate import griddata
        
        # 原始稀疏坐标与流场
        xy = adata.uns['flow_grid'][~mass_filter]        # shape: [N, 2]
        uv = adata.uns['flow'][~mass_filter]             # shape: [N, 2]
        
        # 生成规则网格坐标
        x_lin = np.linspace(xy[:, 0].min(), xy[:, 0].max(), 40)
        y_lin = np.linspace(xy[:, 1].min(), xy[:, 1].max(), 40)
        X, Y = np.meshgrid(x_lin, y_lin)
        
        # 插值 U 和 V 分量
        U = griddata(xy, uv[:, 0], (X, Y), method='linear', fill_value=0)
        V = griddata(xy, uv[:, 1], (X, Y), method='linear', fill_value=0)
        
        # 可视化
        ax.streamplot(x_lin, y_lin, U, V,
                      color='black',
                      density=1.2,
                      linewidth=1,
                      arrowsize=1.5,
                      integration_direction='both')

        
        ax.axis("off")
        if direction == 'Activation':
            ax.set_title(f"{' and '.join(pert_Gene)} {direction}")
        elif direction =='both':
            ax.set_title(f"{pert_Gene[0]} Activation and {pert_Gene[1]} Knock Out")
        else:
            ax.set_title(f"{' and '.join(pert_Gene)} Knock Out")
        plt.tight_layout()
        if save_dir:
            if direction == 'Activation':
                plt.savefig(f"{save_dir}/{' and '.join(pert_Gene)} {direction}.png", bbox_inches='tight')
                plt.savefig(f"{save_dir}/{' and '.join(pert_Gene)} {direction}.pdf", bbox_inches='tight')
            elif direction =='both':
                plt.savefig(f"{save_dir}/{pert_Gene[0]} Activation and {pert_Gene[1]} Knock Out.png", bbox_inches='tight')
                plt.savefig(f"{save_dir}/{pert_Gene[0]} Activation and {pert_Gene[1]} Knock Out.pdf", bbox_inches='tight')
            else:
                plt.savefig(f"{save_dir}/{' and '.join(pert_Gene)} Knock Out.png", bbox_inches='tight')
                plt.savefig(f"{save_dir}/{' and '.join(pert_Gene)} Knock Out.pdf", bbox_inches='tight')
        if show:
            plt.show()

    estimate_transition_prob(adata,embedding_name=embedding_name, n_neighbors=n_neighbors, sampled_fraction=sampled_fraction)
    calculate_embedding_shift(adata,embedding_name=embedding_name)
    calculate_p_mass(adata,embedding_name=embedding_name,n_neighbors=n_neighbors,pert_celltype=pert_celltype, smooth=smooth, n_grid=n_grid, state_obs=state_obs)
    if run_suggest_mass_thresholds:
        suggest_mass_thresholds(adata, embedding_name=embedding_name,n_suggestion=n_suggestion,save_dir=save_dir)
        return adata
    calculate_mass_filter(adata,embedding_name=embedding_name, min_mass=min_mass, plot=False)
    plot_flow(adata,state_obs=state_obs, embedding_name=embedding_name,dot_size=dot_size, scale=scale,show=show,direction=direction,pert_Gene=pert_Gene, save_dir=save_dir,palette=palette)
    return adata


def pert_plot_stream(adata_TF, adata_down, model, pert_Gene,pert_celltype, run_suggest_mass_thresholds,fold,state_obs,dot_size=None, scale=0.1, min_mass=0.008,save_dir=None,embedding_name='X_tsne',n_neighbors=None,n_grid=40,palette=None,direction=None):
    """
    Perform in silico perturbation and visualize the resulting streamlines (trajectories) to show cell state transitions. This is generally same to the vector field map but change vectors to streamlines. The method of drawing vector field map is borrowed from method CellOracle

    Parameters:
        - adata_TF [AnnData]: Annotated data object containing transcription factor expression data.
        - adata_down [AnnData]: Annotated data object for downstream analysis with embedding coordinates.
        - model [torch.nn.Module]: Trained CAUTrigger model with eval() and get_model_output() methods.
        - pert_Gene [list of str]: List of features to be perturbed.
        - pert_celltype [list of str]: List of cell types that will receive the perturbation.
        - run_suggest_mass_thresholds [bool]: Whether to run suggested mass threshold selection. 
                                             Must be set to True initially to determine an appropriate 'min_mass'.
        - fold [list of float]: Multiplication factors for each perturbed feature; length must match pert_Gene.
        - state_obs [str]: Column name in adata_TF.obs representing cell states/types.
        - dot_size [float, optional]: Size of dots in the plot. Defaults to scanpy's default if None.
        - scale [float]: Scaling factor for vector arrows. Larger values result in smaller arrows. Default is 0.1.
        - min_mass [float]: Threshold for filtering low-density grid points. Use 'run_suggest_mass_thresholds=True'
                           first to determine an appropriate value. Default is 0.008.
        - save_dir [str, optional]: Directory path to save plots. No saving occurs if None. Default is None.
        - embedding_name [str]: Key for embedding coordinates in adata_down.obsm. Default is 'X_tsne'.
        - n_neighbors [int, optional]: Number of neighbors for vector field calculation. If None, defaults to 
                                      int(adata.shape[0] / 5). Default is None.
        - n_grid [int]: Grid density for vector field visualization. Default is 40.
        - palette [dict, optional]: Color mapping dictionary for cell types. Default is None.
        - direction [str, optional]: Label indicating perturbation direction for figure title. Default is None.
    Returns:
        - model_output_pert [dict]: Model output after applying  perturbations.
    """
    model.eval()
    with torch.no_grad():
        model_output = model.get_model_output(adata_TF)
    adata_pert = adata_TF.copy()
    pert_cell_idx = np.where(adata_TF.obs[state_obs].isin(pert_celltype))[0]
    for ind, gene in enumerate(pert_Gene):
        adata_pert.X[:, adata_pert.var_names.get_loc(gene)] = adata_pert.X[pert_cell_idx, adata_pert.var_names.get_loc(gene)].max() * fold[ind]
    model.eval()
    with torch.no_grad():
        model_output_pert = model.get_model_output(adata_pert)

    if "x_down2_rec_alpha" in model_output:
        down_key = "x_down2_rec_alpha"  # 3L model
    elif "x_down_rec_alpha" in model_output:
        down_key = "x_down_rec_alpha"  # 2L model
    else:
        raise KeyError("Cannot find x_down_rec_alpha or x_down2_rec_alpha in model output.")

    adata_down.layers["imputed_count"] = np.float64(np.exp(model_output[down_key]))
    adata_down.layers["simulated_count"] = np.float64(np.exp(model_output_pert[down_key]))
    adata_down.layers["delta_X"] = adata_down.layers["simulated_count"].copy() - adata_down.layers["imputed_count"].copy()
    ax = plot_stream(adata_down, embedding_name=embedding_name,direction=direction, state_obs=state_obs,
                                   pert_Gene=pert_Gene, pert_celltype=pert_celltype, scale=scale,min_mass=min_mass, 
                            save_dir=save_dir,dot_size=dot_size,run_suggest_mass_thresholds=run_suggest_mass_thresholds,n_neighbors=n_neighbors,n_grid=n_grid,
                           palette=palette)
    return model_output_pert
