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

def plot_vector_field(adata,state_pair=None,KO_Gene=None,state_obs=None,embedding_name=None,method=None,
                      sampled_fraction=1, min_mass=0.008,scale=0.1, save_dir=None,smooth=0.8,n_grid=40,
                     draw_single=None,n_suggestion=12,show=True,dot_size=None,run_suggest_mass_thresholds=False,direction=None,n_neighbors=None,):
    from scipy import sparse
    from sklearn.neighbors import NearestNeighbors
    from velocyto.estimation import colDeltaCorpartial
    from scipy.stats import norm as normal
    import math

    def replace_with_CauTrigger(adata, corrcoef, state_pair=state_pair, state_obs=state_obs, method=method):
        corrcoef_ = corrcoef.copy()
        if method == 'prob':
            probs = adata.obs['probs'].copy()
            probs_pert = adata.obs['probs_pert'].copy()
        if method == 'logits':
            probs = adata.obs['logits'].copy()
            probs_pert = adata.obs['logits_pert'].copy()
        neigh_ixs = adata.uns['neigh_ixs'].copy()
        target_index = np.where((adata.obs[state_obs] == state_pair[0]) | (adata.obs[state_obs] == state_pair[1]))[0]
        nontarget_neighbor_index = np.setdiff1d(np.unique(neigh_ixs[target_index, :].flatten()), target_index)
        nonneighbor_index = np.setdiff1d(np.setdiff1d(np.arange(probs.shape[0]), target_index), nontarget_neighbor_index)
        for i in target_index:
            cols = np.setdiff1d(np.union1d(target_index, nontarget_neighbor_index), i)
            cor =  - abs(probs_pert[i] - probs[cols])
            corrcoef_[i, cols] = cor
        corrcoef_[np.union1d(nontarget_neighbor_index, nonneighbor_index), :] = 0
        return corrcoef_

    def estimate_transition_prob(adata, state_pair=state_pair,state_obs=state_obs, embedding_name=embedding_name, n_neighbors=None,
                                 sampled_fraction=sampled_fraction, sigma_corr=0.005, replace_prob=None,draw_single=None):
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
        if replace_prob == 'prob':
            corrcoef = replace_with_CauTrigger(adata, corrcoef, state_pair=state_pair,state_obs=state_obs, method='prob')
        if replace_prob == 'logits':
            corrcoef = replace_with_CauTrigger(adata, corrcoef,state_pair=state_pair,state_obs=state_obs, method='logits')
        if np.any(np.isnan(corrcoef)):
            corrcoef[np.isnan(corrcoef)] = 1
        transition_prob = np.exp(corrcoef / sigma_corr) * embedding_knn.A
        transition_prob /= transition_prob.sum(1)[:, None]
        adata.obsm['embedding_knn'] = embedding_knn.copy()
        adata.obsp['transition_prob'] = transition_prob.copy()

    def calculate_embedding_shift(adata, embedding_name=embedding_name):
        transition_prob = adata.obsp['transition_prob'].copy()
        embedding = adata.obsm[embedding_name].copy()
        embedding_knn = adata.obsm['embedding_knn'].copy()
        unitary_vectors = embedding.T[:, None, :] - embedding.T[:, :, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            unitary_vectors /= np.linalg.norm(unitary_vectors, ord=2, axis=0)
            np.fill_diagonal(unitary_vectors[0, ...], 0)
            np.fill_diagonal(unitary_vectors[1, ...], 0)
        delta_embedding = (transition_prob * unitary_vectors).sum(2)
        delta_embedding -= (embedding_knn.A * unitary_vectors).sum(2) / embedding_knn.sum(1).A.T
        delta_embedding = delta_embedding.T
        adata.obsm['delta_embedding'] = delta_embedding.copy()

    def calculate_p_mass(adata, embedding_name=embedding_name, smooth=smooth, n_grid=n_grid, n_neighbors=None,draw_single=None):
        steps = (n_grid, n_grid)
        embedding = adata.obsm[embedding_name].copy()
        if draw_single:
            adata_tmp = adata.copy()
            adata_tmp.obsm['delta_embedding'][adata_tmp.obs[state_obs] != draw_single] = 0
            delta_embedding = adata_tmp.obsm['delta_embedding'].copy()
        else:
            delta_embedding = adata.obsm['delta_embedding'].copy()
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

    def suggest_mass_thresholds(adata, embedding_name=embedding_name, n_suggestion=n_suggestion,save_dir=save_dir, s=1, n_col=4):
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
        plt.savefig(f"{save_dir}/suggest_mass_thresholds.png", bbox_inches='tight')
        plt.show()

    def calculate_mass_filter(adata, embedding_name=embedding_name, min_mass=min_mass, plot=False):
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

    def plot_flow(adata, state_obs=state_obs, embedding_name=embedding_name, dot_size=dot_size, scale=scale, KO_Gene=KO_Gene, save_dir=save_dir, show=show,direction=direction):
        fig, ax = plt.subplots()
        sc.pl.embedding(adata, basis=embedding_name, color=state_obs, ax=ax, show=False,size=dot_size, palette=sns.color_palette())
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
            ax.set_title(f"{' and '.join(KO_Gene)} {direction}")
        elif direction =='both':
            ax.set_title(f"{KO_Gene[0]} Activation and {KO_Gene[1]} Knock Out")
        else:
            ax.set_title(f"{' and '.join(KO_Gene)} Knock Out")
        plt.tight_layout()
        if save_dir:
            if direction == 'Activation':
                plt.savefig(f"{save_dir}/{' and '.join(KO_Gene)} {direction}.png", bbox_inches='tight')
                plt.savefig(f"{save_dir}/{' and '.join(KO_Gene)} {direction}.pdf", bbox_inches='tight', backend='Cairo')
            elif direction =='both':
                plt.savefig(f"{save_dir}/{KO_Gene[0]} Activation and {KO_Gene[1]} Knock Out.png", bbox_inches='tight')
                plt.savefig(f"{save_dir}/{KO_Gene[0]} Activation and {KO_Gene[1]} Knock Out.pdf", bbox_inches='tight', backend='Cairo')
            else:
                plt.savefig(f"{save_dir}/{' and '.join(KO_Gene)} Knock Out.png", bbox_inches='tight')
                plt.savefig(f"{save_dir}/{' and '.join(KO_Gene)} Knock Out.pdf", bbox_inches='tight', backend='Cairo')
        if show:
            plt.show()

    estimate_transition_prob(adata,state_pair=state_pair,state_obs=state_obs,embedding_name=embedding_name, replace_prob=method, n_neighbors=n_neighbors, sampled_fraction=1)
    calculate_embedding_shift(adata,embedding_name=embedding_name)
    calculate_p_mass(adata,embedding_name=embedding_name,draw_single=draw_single,n_neighbors=n_neighbors)
    if run_suggest_mass_thresholds:
        suggest_mass_thresholds(adata, embedding_name=embedding_name,n_suggestion=n_suggestion)
        return adata
    calculate_mass_filter(adata,embedding_name=embedding_name, min_mass=min_mass, plot=False)
    plot_flow(adata,state_obs=state_obs, embedding_name=embedding_name,dot_size=dot_size, scale=scale,show=show,direction=direction)
    return adata


def pert_plot_up(adata_TF, adata_down, model, state_pair, KO_Gene, state_obs,run_suggest_mass_thresholds,fold=2,dot_size=None, method=None, scale=0.1, min_mass=0.008,save_dir=None,draw_single=None,embedding_name='X_tsne',n_neighbors=None,n_grid=40):
    model.eval()
    with torch.no_grad():
        model_output = model.get_model_output(adata_TF)
    adata_pert = adata_TF.copy()
    pert_cell_idx = np.where((adata_TF.obs[state_obs] == state_pair[0]) | (adata_TF.obs[state_obs] == state_pair[1]))[0]
    for ind, gene in enumerate(KO_Gene):
        adata_pert.X[pert_cell_idx, adata_pert.var_names.get_loc(gene)] = adata_pert.X[pert_cell_idx, adata_pert.var_names.get_loc(gene)].max() * fold[ind]
    model.eval()
    with torch.no_grad():
        model_output_pert = model.get_model_output(adata_pert)
    adata_all = sc.concat([adata_TF, adata_down],axis=1)
    adata_all.obs = adata_TF.obs.copy()
    adata_all.uns = adata_TF.uns.copy()
    adata_all.obsm = adata_TF.obsm.copy()
    adata_all.obsp = adata_TF.obsp.copy()
    adata_all.obs['probs'] = model_output['probs'].copy()
    adata_all.obs['logits'] = model_output['logits'].copy()
    adata_all.obs['probs_pert'] = model_output_pert['probs'].copy()
    adata_all.obs['logits_pert'] = model_output_pert['logits'].copy()
    adata_all.layers["imputed_count"] = np.float64(np.exp(adata_all.X.copy()))
    adata_all.layers["simulated_count"] = np.float64(np.exp(np.concatenate([model_output_pert['x_up_rec1'].copy(), model_output_pert['x_down_rec_alpha'].copy()],axis=1)))
    adata_all.layers["delta_X"] = adata_all.layers["simulated_count"].copy() - adata_all.layers["imputed_count"].copy()
    paul_for_co = adata_down.copy()
    paul_for_co.layers['pert_X'] = model_output_pert['x_down_rec_alpha'].copy()
    paul_for_co.obs['probs'] = model_output['probs'].copy()
    paul_for_co.obs['logits'] = model_output['logits'].copy()
    paul_for_co.obs['probs_pert'] = model_output_pert['probs'].copy()
    paul_for_co.obs['logits_pert'] = model_output_pert['logits'].copy()
    paul_for_co.layers["imputed_count"] = np.float64(paul_for_co.X.copy())
    paul_for_co.layers["simulated_count"] = np.float64(paul_for_co.layers['pert_X'].copy())
    paul_for_co.layers["delta_X"] = paul_for_co.layers["simulated_count"].copy() - paul_for_co.layers["imputed_count"].copy()
    non_pert_cell_idx = np.where(~((adata_TF.obs[state_obs] == state_pair[0]) | (adata_TF.obs[state_obs] == state_pair[1])))[0]
    paul_for_co.layers["delta_X"][non_pert_cell_idx, :] = 0
    ax = plot_vector_field(adata_all, state_pair=state_pair, state_obs=state_obs, embedding_name=embedding_name,direction='Activation',
                                  method=method, KO_Gene=KO_Gene, scale=scale,min_mass=min_mass, save_dir=save_dir,draw_single=draw_single,dot_size=dot_size,run_suggest_mass_thresholds=run_suggest_mass_thresholds,n_neighbors=n_neighbors,n_grid=n_grid)


def pert_plot_down(adata_TF, adata_down, model, state_pair, KO_Gene, state_obs,run_suggest_mass_thresholds,fold=2,dot_size=None, method=None, scale=0.1, min_mass=0.008,save_dir=None,draw_single=None,embedding_name='X_tsne',n_neighbors=None,n_grid=40):
    model.eval()
    with torch.no_grad():
        model_output = model.get_model_output(adata_TF)
    adata_pert = adata_TF.copy()
    pert_cell_idx = np.where((adata_TF.obs[state_obs] == state_pair[0]) | (adata_TF.obs[state_obs] == state_pair[1]))[0]
    for ind, gene in enumerate(KO_Gene):
        adata_pert.X[pert_cell_idx, adata_pert.var_names.get_loc(gene)] = 0
    model.eval()
    with torch.no_grad():
        model_output_pert = model.get_model_output(adata_pert)
    adata_all = sc.concat([adata_TF, adata_down],axis=1)
    adata_all.obs = adata_TF.obs.copy()
    adata_all.uns = adata_TF.uns.copy()
    adata_all.obsm = adata_TF.obsm.copy()
    adata_all.obsp = adata_TF.obsp.copy()
    adata_all.obs['probs'] = model_output['probs'].copy()
    adata_all.obs['logits'] = model_output['logits'].copy()
    adata_all.obs['probs_pert'] = model_output_pert['probs'].copy()
    adata_all.obs['logits_pert'] = model_output_pert['logits'].copy()
    adata_all.layers["imputed_count"] = np.float64(np.exp(adata_all.X.copy()))
    adata_all.layers["simulated_count"] = np.float64(np.exp(np.concatenate([model_output_pert['x_up_rec1'].copy(), model_output_pert['x_down_rec_alpha'].copy()],axis=1)))
    adata_all.layers["delta_X"] = adata_all.layers["simulated_count"].copy() - adata_all.layers["imputed_count"].copy()
    paul_for_co = adata_down.copy()
    paul_for_co.layers['pert_X'] = model_output_pert['x_down_rec_alpha'].copy()
    paul_for_co.obs['probs'] = model_output['probs'].copy()
    paul_for_co.obs['logits'] = model_output['logits'].copy()
    paul_for_co.obs['probs_pert'] = model_output_pert['probs'].copy()
    paul_for_co.obs['logits_pert'] = model_output_pert['logits'].copy()
    paul_for_co.layers["imputed_count"] = np.float64(paul_for_co.X.copy())
    paul_for_co.layers["simulated_count"] = np.float64(paul_for_co.layers['pert_X'].copy())
    paul_for_co.layers["delta_X"] = paul_for_co.layers["simulated_count"].copy() - paul_for_co.layers["imputed_count"].copy()
    non_pert_cell_idx = np.where(~((adata_TF.obs[state_obs] == state_pair[0]) | (adata_TF.obs[state_obs] == state_pair[1])))[0]
    paul_for_co.layers["delta_X"][non_pert_cell_idx, :] = 0
    ax = plot_vector_field(adata_all, state_pair=state_pair, state_obs=state_obs, embedding_name=embedding_name,direction='Activation',
                                  method=method, KO_Gene=KO_Gene, scale=scale,min_mass=min_mass, save_dir=save_dir,draw_single=draw_single,dot_size=dot_size,run_suggest_mass_thresholds=run_suggest_mass_thresholds,n_neighbors=n_neighbors,n_grid=n_grid)


def calculate_GIs(first_expr, second_expr, double_expr):
    from sklearn.linear_model import TheilSenRegressor
    from dcor import distance_correlation
    singles_expr = np.array([first_expr, second_expr]).T
    first_expr = first_expr.T
    second_expr = second_expr.T
    double_expr = double_expr.T
    results = {}
    results['ts'] = TheilSenRegressor(fit_intercept=False,
                                      max_subpopulation=1e5,
                                      max_iter=1000,
                                      random_state=1000)
    X = singles_expr
    y = double_expr
    results['ts'].fit(X, y.ravel())
    Zts = results['ts'].predict(X)
    results['c1'] = results['ts'].coef_[0]
    results['c2'] = results['ts'].coef_[1]
    results['mag'] = np.sqrt((results['c1'] ** 2 + results['c2'] ** 2))
    results['dcor'] = distance_correlation(singles_expr, double_expr)
    results['dcor_singles'] = distance_correlation(first_expr, second_expr)
    results['dcor_first'] = distance_correlation(first_expr, double_expr)
    results['dcor_second'] = distance_correlation(second_expr, double_expr)
    results['corr_fit'] = np.corrcoef(Zts.flatten(), double_expr.flatten())[0, 1]
    results['dominance'] = np.abs(np.log10(results['c1'] / results['c2']))
    results['eq_contr'] = np.min([results['dcor_first'], results['dcor_second']]) / np.max(
        [results['dcor_first'], results['dcor_second']])
    return results


def draw_regulon(regulator, RGMs, output_path):
    import networkx as nx
    from matplotlib.patches import FancyArrowPatch
    os.makedirs(output_path, exist_ok=True)
    df = RGMs[regulator]
    df = df.sort_values(by=2, ascending=True).drop_duplicates(subset=[0, 1], keep='first')
    G = nx.DiGraph()
    edge_colors = []
    for _, row in df.iterrows():
        G.add_edge(row[0], row[1], attribute=row[2])
        if row[2] == 'Activation':
            edge_colors.append('green')
        elif row[2] == 'Repression':
            edge_colors.append('red')
        else:
            edge_colors.append('grey')
    node_attributes = pd.Series(index=G.nodes(), dtype=str)
    for _, row in df.iterrows():
        node_attributes[row[0]] = row[2]
        node_attributes[row[1]] = row[2]
    groups = node_attributes.unique()
    grouped_nodes = {group: list(node_attributes[node_attributes == group].index) for group in groups}
    nodes_sorted = []
    for group in groups:
        nodes_sorted.extend(grouped_nodes[group])
    def circos_layout(nodes, radius):
        pos = dict()
        theta = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
        for node, angle in zip(nodes, theta):
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            pos[node] = (x, y)
        return pos
    pos = circos_layout(nodes_sorted, radius=1)
    plt.figure(figsize=(10, 8))
    plt.gca().set_facecolor('white')
    plt.grid(False)
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')
    for (u, v), color in zip(G.edges(), edge_colors):
        start = np.array(pos[u])
        end = np.array(pos[v])
        dx, dy = end - start
        length = np.sqrt(dx ** 2 + dy ** 2)
        unit_dx, unit_dy = dx / length, dy / length
        node_radius = 0.1
        start_arrow = start + node_radius * np.array([unit_dx, unit_dy])
        arrow = FancyArrowPatch(start_arrow, end-node_radius * np.array([unit_dx, unit_dy]), mutation_scale=15, color=color,
                                arrowstyle='-|>', lw=2)
        plt.gca().add_patch(arrow)
    handles = [
        plt.Line2D([0], [0], color='green', lw=2, label='Activation'),
        plt.Line2D([0], [0], color='red', lw=2, label='Repression'),
        plt.Line2D([0], [0], color='grey', lw=2, label='Unknown')
    ]
    plt.legend(handles=handles, title='Edge Attributes', loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{regulator}_regulon1.pdf'), format='pdf')
    plt.savefig(os.path.join(output_path, f'{regulator}_regulon1.png'), format='png')
    plt.gca().invert_xaxis()
    plt.savefig(os.path.join(output_path, f'{regulator}_regulon2.pdf'), format='pdf')
    plt.savefig(os.path.join(output_path, f'{regulator}_regulon2.png'), format='png')
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(output_path, f'{regulator}_regulon3.pdf'), format='pdf')
    plt.savefig(os.path.join(output_path, f'{regulator}_regulon3.png'), format='png')
    plt.gca().invert_xaxis()
    plt.savefig(os.path.join(output_path, f'{regulator}_regulon4.pdf'), format='pdf')
    plt.savefig(os.path.join(output_path, f'{regulator}_regulon4.png'), format='png')
