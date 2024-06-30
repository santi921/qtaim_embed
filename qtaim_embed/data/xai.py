from copy import deepcopy
import numpy as np 
import torch

def get_labelled_importance(feat_mask, dataset):
    """
    Get the importance of each feature in the dataset
    Takes:
        feat_mask: dict of feature masks
        dataset: HeteroGraphGraphLabelDataset
    Returns:
        feature_imp: dict of feature importance
    """
    importance_total = []
    feature_names = dataset.feature_names()
    feature_imp = {}
    for key in feat_mask.keys():
        feature_imp[key] = {}
        for i, feat in enumerate(feat_mask[key]):
            importance_total.append(feat)
            feature_imp[key][feature_names[key][i]] = float(feat)
    
    # normalize importance
    importance_total = np.array(importance_total)
    std_importance = np.std(importance_total)
    mean_importance = np.mean(importance_total)

    feature_imp = {key: {k: (v - mean_importance)/std_importance for k, v in feature_imp[key].items()} for key in feature_imp.keys()}
    return feature_imp


def get_top_features(feature_importance, n=10):
    """
    Get the top n features for each level of the model
    Takes:
    - feature_importance: dict of dicts with feature importance for each level
    - n: number of top features to return
    Returns:
    - top_features: dict of dicts with top n features for each level
    """
    top_features = {}
    for key in feature_importance.keys():
        top_features[key] = {k: v for k, v in sorted(feature_importance[key].items(), key=lambda item: item[1], reverse=True)[:n]}
    return top_features


def get_feats_w_qtaim(feature_importance):
    """
    Simple function to get only qtaim features
    Takes:
        feature_importance: dict of dicts with feature importance for each level
    """
    qtaim_feats = {}
    for key in feature_importance.keys():
        qtaim_feats[key] = {k: v for k, v in feature_importance[key].items() if "extra_" in k}
    return qtaim_feats


def convert_edge_mask_to_z_scores(edge_mask):
    """
    Convert edge mask to z-scores
    Takes: 
        edge_mask: dict of edge masks
    Returns:
        edge_mask_z: dict of edge masks with z-score values
    """

    edge_mask_z = {}
    feat_list = []
    feat_list = [feat for key in edge_mask.keys() for feat in edge_mask[key].numpy()]
    feat_list = np.array(feat_list)
    mean = np.mean(feat_list)
    std = np.std(feat_list)
    
    for key in edge_mask.keys():
        edge_mask_z[key] = []
        for i, feat in enumerate(edge_mask[key]):
            edge_mask_z[key].append((feat.numpy() - mean) / std)
        edge_mask_z[key] = torch.tensor(edge_mask_z[key])
    return edge_mask_z


def process_edge_mask(edge_mask_z, scale=True): 
    """
    Process edge mask to get the most important features for each edge type 
    Takes: 
        edge_mask_z: edge mask from the explainer
        scale: whether to scale the features or not
    Returns:
        edge_mask_processed: processed
    """

    a2b = edge_mask_z[('atom', 'a2b', 'bond')] 
    b2a = edge_mask_z[('bond', 'b2a', 'atom')] 
    
    a2b = torch.tensor([float(torch.max(a2b[i], a2b[i+1])) for i in range(int(list(a2b.shape)[0]/2))])
    b2a = torch.tensor([float(torch.max(b2a[i], b2a[i+1])) for i in range(int(list(b2a.shape)[0]/2))])
    b2g = edge_mask_z[('bond', 'b2g', 'global')]
    g2b = edge_mask_z[('global', 'g2b', 'bond')] 
    
    edge_mask_new = deepcopy(edge_mask_z)
    edge_mask_new[('atom', 'a2b', 'bond')] = a2b
    edge_mask_new[('bond', 'b2a', 'atom')] = b2a

    atom_feat_max_list = []
    bond_feat_max_list = []

    for ind, a2a in enumerate(edge_mask_z[('atom', 'a2a', 'atom')]):
        atom_fts = torch.tensor([a2a, edge_mask_z[('atom', 'a2g', 'global')][ind]])
        atom_fts = torch.tensor([i for i in atom_fts if i > 0])
        atom_feat_max_list.append(torch.sum(atom_fts))
        #atom_feat_max = float(torch.max())
        #atom_feat_max_list.append(atom_feat_max)
    

    for ind, b2b in enumerate(edge_mask_z[('bond', 'b2b', 'bond')]):
        bond_fts = torch.tensor([b2b, a2b[ind], b2a[ind], b2g[ind], g2b[ind]])
        # discard all feats < 0
        bond_fts = torch.tensor([i for i in bond_fts if i > 0])
        
        bond_fts_sum = torch.sum(bond_fts)
        bond_feat_max_list.append(bond_fts_sum)
        #bond_feat_max = float(torch.max(bond_fts))
        #bond_feat_max_list.append(bond_feat_max)
    

    atom = torch.tensor(atom_feat_max_list) 
    bond = torch.tensor(bond_feat_max_list)
    global_ft = edge_mask_z[('global', 'g2g', 'global')]
    combined = torch.cat([atom, bond, global_ft])
    mean = torch.mean(combined)
    std = torch.std(combined)
    if scale == True:
        edge_mask_processed = {
            "atom": (atom - mean) / std, 
            "bond": (bond - mean) / std,
            "global": (global_ft - mean) / std
        }
    else: 
        edge_mask_processed = {
            "atom": atom, 
            "bond": bond,
            "global": global_ft
        }
    return edge_mask_processed